"""Tests for src/ingest/odds_api.py (v1.5, Phase 1c revisit).

Zero network calls anywhere in this file -- api.the-odds-api.com is
blocked in this environment (verified, see the module docstring), so
everything here exercises pure planning math, snapshot-date derivation
against the repo's own cached schedules.parquet, manifest schema, and
normalize() against a checked-in synthetic fixture
(tests/fixtures/odds_api_sample_raw.json). No test in this file imports
``requests`` or sets ``ODDS_API_KEY``.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from src.ingest import odds_api as oa

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "odds_api_sample_raw.json"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return oa.load_config()


# --------------------------------------------------------------------------
# Config sanity
# --------------------------------------------------------------------------


def test_config_loads_and_has_required_keys(cfg: dict) -> None:
    for key in ("base_url", "region", "credit_budget", "sport_key", "featured_markets", "snapshot_seasons", "props_era_start", "candidate_futures_markets"):
        assert key in cfg, f"configs/odds_api.yaml missing key: {key}"
    assert cfg["featured_markets"] == ["h2h", "spreads", "totals"]
    assert cfg["credit_budget"] == 3000
    assert cfg["sport_key"] == "americanfootball_nfl"


# --------------------------------------------------------------------------
# Snapshot-date derivation (synthetic schedules, no dependency on the real
# cached data/raw/schedules.parquet)
# --------------------------------------------------------------------------


@pytest.fixture()
def synthetic_schedules(tmp_path) -> Path:
    df = pl.DataFrame(
        {
            "season": [2022, 2022, 2022, 2023, 2023, 2023],
            "week": [1, 1, 2, 1, 1, 1],
            "game_type": ["REG", "REG", "REG", "REG", "REG", "PRE"],
            "gameday": ["2022-09-08", "2022-09-11", "2022-09-18", "2023-09-07", "2023-09-10", "2023-08-15"],
        }
    )
    path = tmp_path / "schedules.parquet"
    df.write_parquet(path)
    return path


def test_snapshot_dates_use_earliest_reg_gameday_minus_4_days(synthetic_schedules: Path) -> None:
    dates = oa.snapshot_dates_for_seasons([2022, 2023], schedules_path=synthetic_schedules)
    # 2022's earliest REG gameday is 2022-09-08 (week 1, min gameday) -> minus 4 days = 2022-09-04.
    assert dates[2022] == "2022-09-04T12:00:00Z"
    # 2023's earliest REG gameday is 2023-09-07 (the PRE row must be excluded) -> minus 4 = 2023-09-03.
    assert dates[2023] == "2023-09-03T12:00:00Z"


def test_snapshot_dates_skip_seasons_absent_from_schedules(synthetic_schedules: Path) -> None:
    dates = oa.snapshot_dates_for_seasons([2022, 2099], schedules_path=synthetic_schedules)
    assert 2099 not in dates
    assert 2022 in dates


def test_snapshot_dates_format_is_iso8601_t_noon_utc(synthetic_schedules: Path) -> None:
    dates = oa.snapshot_dates_for_seasons([2022, 2023], schedules_path=synthetic_schedules)
    for iso in dates.values():
        assert iso.endswith("T12:00:00Z")
        date_part = iso.split("T")[0]
        assert len(date_part) == 10 and date_part.count("-") == 2


# Data-dependent: cross-check against the real cached schedules.parquet
# when present, matching the repo-wide "_skip_if_*" convention.
def test_snapshot_dates_against_real_schedules_if_cached() -> None:
    if not oa.SCHEDULES_PATH.exists():
        pytest.skip("data/raw/schedules.parquet not cached; run `python -m src.ingest.nflverse`")
    dates = oa.snapshot_dates_for_seasons(oa.load_config()["snapshot_seasons"])
    for season, iso in dates.items():
        assert iso.endswith("T12:00:00Z"), f"season {season}: snapshot {iso} not T12:00:00Z"


# --------------------------------------------------------------------------
# Dry-run planner math -- the docs' formula: 10 credits x regions x markets
# --------------------------------------------------------------------------


@pytest.fixture()
def snapshot_dates(cfg: dict) -> dict[int, str]:
    # Synthetic dates covering every configured snapshot season, independent of
    # whatever real schedules.parquet happens to be cached in this environment.
    return {season: f"{season}-09-0{4 if season % 2 else 3}T12:00:00Z" for season in cfg["snapshot_seasons"]}


def test_estimate_cost_matches_docs_formula() -> None:
    assert oa.estimate_cost(1, 3) == 30
    assert oa.estimate_cost(2, 3) == 60
    assert oa.estimate_cost(1, 1) == 10


def test_pull_team_plan_is_one_request_per_snapshot_all_featured_markets(cfg: dict, snapshot_dates) -> None:
    reqs = oa.team_snapshot_requests(cfg, snapshot_dates)
    assert len(reqs) == len(cfg["snapshot_seasons"]) == 6
    for r in reqs:
        assert sorted(r.markets) == sorted(cfg["featured_markets"])
        assert r.est_cost == oa.estimate_cost(1, 3) == 30


def test_pull_team_plan_total_projected_cost_is_180(cfg: dict, snapshot_dates) -> None:
    reqs = oa.team_snapshot_requests(cfg, snapshot_dates)
    assert sum(r.est_cost for r in reqs) == 180


def test_verify_futures_plan_is_one_request_displayed_at_10_credits(cfg: dict, snapshot_dates) -> None:
    req = oa.verify_futures_request(cfg, snapshot_dates)
    assert req is not None
    assert req.season == cfg["props_era_start"]
    assert set(req.markets) == set(cfg["candidate_futures_markets"])
    assert req.est_cost == 10
    # Worst-case ceiling (used for the pre-call hard-abort check) is the full formula.
    assert req.worst_case_cost == oa.estimate_cost(1, len(cfg["candidate_futures_markets"]))


def test_expensive_props_plan_totals_the_documented_estimate(cfg: dict, snapshot_dates) -> None:
    reqs = oa.expensive_props_requests(cfg, snapshot_dates)
    assert len(reqs) == len(oa.props_era_seasons(cfg))
    assert sum(r.est_cost for r in reqs) == oa.EXPENSIVE_PROPS_ESTIMATE == 1920


def test_props_era_seasons_excludes_pre_2023(cfg: dict) -> None:
    seasons = oa.props_era_seasons(cfg)
    assert all(s >= cfg["props_era_start"] for s in seasons)
    assert 2020 not in seasons and 2023 in seasons


def test_full_request_plan_has_every_group(cfg: dict) -> None:
    if not oa.SCHEDULES_PATH.exists():
        pytest.skip("data/raw/schedules.parquet not cached; run `python -m src.ingest.nflverse`")
    plan = oa.full_request_plan(cfg)
    assert set(plan) == {"verify_futures", "pull_team", "pull_props_cheap_if_futures_found", "pull_props_expensive_fallback"}
    assert len(plan["pull_team"]) == 6


def test_dry_run_print_plan_never_calls_network(cfg: dict, capsys) -> None:
    """print_plan must run to completion with no ODDS_API_KEY set and no network access."""
    if not oa.SCHEDULES_PATH.exists():
        pytest.skip("data/raw/schedules.parquet not cached; run `python -m src.ingest.nflverse`")
    import os

    assert os.environ.get(oa.API_KEY_ENV) is None, "test environment must not carry a real API key"
    oa.print_plan(cfg)
    out = capsys.readouterr().out
    assert "REDACTED" not in out or "apiKey=<ODDS_API_KEY not set>" in out or "ODDS_API_KEY not set" in out
    assert "Projected total" in out


def test_redact_url_hides_api_key() -> None:
    url = "https://api.the-odds-api.com/v4/historical/sports/x/odds?apiKey=SECRET123&regions=us"
    redacted = oa.redact_url(url, "SECRET123")
    assert "SECRET123" not in redacted
    assert "REDACTED" in redacted


# --------------------------------------------------------------------------
# Manifest schema
# --------------------------------------------------------------------------


def test_manifest_entry_schema(cfg: dict, snapshot_dates, tmp_path) -> None:
    req = oa.team_snapshot_requests(cfg, snapshot_dates)[0]
    url = oa.build_url(cfg, req, "SECRETKEY")
    entry = oa.manifest_entry(
        req,
        url=url,
        api_key="SECRETKEY",
        cost_credits=req.est_cost,
        headers={"x-requests-remaining": "2900", "x-requests-used": "100"},
        out_file=tmp_path / "raw" / "2023_h2h+spreads+totals_x.json",
    )
    for key in ("label", "url", "timestamp", "cost_credits", "headers", "season", "markets", "snapshot", "out_file"):
        assert key in entry
    assert "SECRETKEY" not in entry["url"]
    assert entry["headers"]["x-requests-remaining"] == "2900"


def test_append_and_load_manifest_roundtrip(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    entry = {"label": "test", "url": "https://x", "timestamp": "now", "cost_credits": 10, "headers": {}, "season": 2023, "markets": ["h2h"], "snapshot": "s", "out_file": "f.json"}
    oa.append_manifest(entry, path=path)
    oa.append_manifest(entry, path=path)
    loaded = oa.load_manifest(path)
    assert len(loaded) == 2
    assert loaded[0] == entry


# --------------------------------------------------------------------------
# CreditTracker hard-abort behavior
# --------------------------------------------------------------------------


def test_credit_tracker_aborts_when_projected_exceeds_budget() -> None:
    tracker = oa.CreditTracker(budget=100)
    tracker.check_before(50, "call1")
    tracker.spent_est += 50
    with pytest.raises(SystemExit):
        tracker.check_before(60, "call2")


def test_credit_tracker_aborts_when_exceeding_api_remaining_quota() -> None:
    tracker = oa.CreditTracker(budget=100_000)
    tracker.remaining_from_api = 5
    with pytest.raises(SystemExit):
        tracker.check_before(10, "call1")


def test_credit_tracker_record_updates_running_totals() -> None:
    tracker = oa.CreditTracker(budget=1000)
    tracker.record({"x-requests-remaining": "2990", "x-requests-used": "10"}, 10, "call1")
    assert tracker.spent_est == 10
    assert tracker.remaining_from_api == 2990
    assert tracker.used_from_api == 10
    assert len(tracker.calls) == 1


# --------------------------------------------------------------------------
# Normalize (fixture-driven, zero network)
# --------------------------------------------------------------------------


@pytest.fixture()
def raw_dir_with_fixture(tmp_path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    payload = json.loads(FIXTURE_PATH.read_text())
    (raw_dir / "2023_h2h+spreads+totals_2023-09-03T120000Z.json").write_text(json.dumps(payload))
    return raw_dir


def test_normalize_raw_event_takes_median_price_across_bookmakers() -> None:
    payload = json.loads(FIXTURE_PATH.read_text())
    event = payload["data"][0]  # Chiefs vs Lions, two bookmakers
    row = oa.normalize_raw_event(event, season=2023, snapshot=payload["timestamp"])
    assert row["home_team"] == "Kansas City Chiefs"
    assert row["away_team"] == "Detroit Lions"
    # DraftKings -280 / FanDuel -290 -> median (mean of the two, since n=2)
    assert row["home_ml"] == pytest.approx(-285.0)
    assert row["away_ml"] == pytest.approx(235.0)
    assert row["home_spread"] == pytest.approx(-6.25)
    assert row["total"] == pytest.approx(53.75)


def test_normalize_raw_event_tolerates_missing_markets() -> None:
    payload = json.loads(FIXTURE_PATH.read_text())
    event = payload["data"][1]  # Jets vs Bills, h2h only, one bookmaker
    row = oa.normalize_raw_event(event, season=2023, snapshot=payload["timestamp"])
    assert row["home_ml"] == pytest.approx(150.0)
    assert row["home_spread"] is None
    assert row["total"] is None


def test_normalize_raw_event_tolerates_no_bookmakers() -> None:
    payload = json.loads(FIXTURE_PATH.read_text())
    event = payload["data"][2]  # no bookmakers at all
    row = oa.normalize_raw_event(event, season=2023, snapshot=payload["timestamp"])
    assert row["home_ml"] is None and row["away_ml"] is None and row["total"] is None


def test_normalize_raw_dir_produces_one_row_per_event(raw_dir_with_fixture: Path) -> None:
    df = oa.normalize_raw_dir(raw_dir_with_fixture)
    assert df.height == 3
    assert set(df.columns) == set(oa._TEAM_LINES_SCHEMA)
    assert df.get_column("season").unique().to_list() == [2023]


def test_normalize_raw_dir_skips_futures_probe_files(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "2023_futures_probe_2023-09-03T120000Z.json").write_text(
        json.dumps({"timestamp": "t", "data": [{"id": "x", "bookmakers": []}]})
    )
    df = oa.normalize_raw_dir(raw_dir)
    assert df.height == 0


def test_normalize_raw_dir_empty_when_no_raw_files(tmp_path) -> None:
    df = oa.normalize_raw_dir(tmp_path / "nonexistent")
    assert df.height == 0
    assert set(df.columns) == set(oa._TEAM_LINES_SCHEMA)


def test_run_normalize_writes_parquet(raw_dir_with_fixture: Path, tmp_path) -> None:
    out_path = tmp_path / "team_lines.parquet"
    df = oa.run_normalize(raw_dir=raw_dir_with_fixture, out_path=out_path)
    assert out_path.exists()
    reloaded = pl.read_parquet(out_path)
    assert reloaded.height == df.height


# --------------------------------------------------------------------------
# Data-tracking guard: data/external/odds_api/** must be git-tracked (the
# .gitignore negation carve-out), since the raw JSON is the cross-machine
# contract this local-run split relies on.
# --------------------------------------------------------------------------


def test_raw_json_output_is_git_tracked() -> None:
    import subprocess

    probe = oa.RAW_DIR / ".gitkeep_test_probe.json"
    oa.RAW_DIR.mkdir(parents=True, exist_ok=True)
    probe.write_text("{}")
    try:
        result = subprocess.run(
            ["git", "check-ignore", str(probe)], cwd=oa.REPO_ROOT, capture_output=True, text=True
        )
        # git check-ignore exits 0 (and prints the path) when the path IS ignored;
        # exit 1 means "not ignored", which is what we want here.
        assert result.returncode == 1, f"{probe} is gitignored (exit {result.returncode}): {result.stdout}"
    finally:
        probe.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# No-network guard for run_verify_futures / run_pull_team / run_pull_props:
# they must refuse cleanly (SystemExit) with no ODDS_API_KEY, before any
# import of `requests` or network I/O happens.
# --------------------------------------------------------------------------


def test_network_subcommands_refuse_without_api_key(cfg: dict, monkeypatch) -> None:
    monkeypatch.delenv(oa.API_KEY_ENV, raising=False)
    with pytest.raises(SystemExit):
        oa._api_key()


def test_pull_props_refuses_expensive_fallback_without_flag(cfg: dict) -> None:
    # No manifest.json on disk in this environment (no network call has ever run here) --
    # verify-futures is therefore "not yet run", so the expensive fallback path applies,
    # and it must refuse without --allow-expensive, before touching the network at all
    # (no ODDS_API_KEY is set in this test environment either).
    if oa.MANIFEST_PATH.exists():
        pytest.skip("a real manifest.json exists in this environment; this guard needs a clean slate")
    with pytest.raises(SystemExit, match="allow-expensive"):
        oa.run_pull_props(cfg, allow_expensive=False)
