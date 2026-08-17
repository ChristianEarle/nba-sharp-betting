"""The Odds API ingest (v1.5, Phase 1c revisit).

v1 scoped Vegas/odds data out of training entirely (see README's "No Vegas
data anywhere in training" limitation) because this environment's egress
proxy blocks ``api.the-odds-api.com`` outright -- verified directly, not
assumed: every request to that host fails at the proxy layer before it
ever reaches the API. That has not changed for v1.5. **Nothing in this
module is executed against the network from this environment.** It is
code-complete, tested with zero network calls (``tests/test_odds_api.py``
runs entirely against a synthetic fixture and pure-math planner checks),
and meant to be run **locally**, where the proxy is not in the way -- see
the README's "v1.5 Odds API (local run required)" section for the exact
command sequence and expected costs.

Four CLI subcommands, in the order a local run should use them:

1. ``--dry-run`` -- prints the full request plan (every URL, API key
   REDACTED, per-request credit cost, projected total) with **zero**
   network calls. Safe to run anywhere, including this environment.
2. ``--verify-futures`` -- ONE historical odds call on the earliest
   props-era snapshot (``props_era_start``, default 2023), probing
   ``candidate_futures_markets`` (configs/odds_api.yaml) and reporting
   which season-long futures market keys the API actually returns data
   for. Costs ~10 credits (see "Credit accounting" below for why that's
   an estimate, not a promise). Requires ``ODDS_API_KEY``.
3. ``--pull-team`` -- featured markets (h2h/spreads/totals) for all six
   preseason snapshots (2020-2025), ~180 credits (6 x 10 x 3).
4. ``--pull-props`` -- IF ``--verify-futures`` already found season-long
   futures keys (recorded in the manifest), pulls those for the props-era
   snapshots -- cheap, one call per season. ELSE falls back to Week-1
   event-level player-prop odds for 2023-2025, which is expensive
   (~1,920 credits: 3 seasons x ~8 markets x ~4 divisional-window events
   x 10 x 1 region, order-of-magnitude -- an event-level pull's exact cost
   depends on how many games/markets a season actually returns) and only
   runs when ``--allow-expensive`` is also passed, refusing otherwise with
   a clear message.

A fifth, network-free mode:

5. ``--normalize`` -- parses whatever is already in
   ``data/external/odds_api/raw/*.json`` into
   ``data/external/odds_api/team_lines.parquet`` (season, snapshot,
   home/away teams, h2h prices, spread, total). Pure local, no network, no
   API key. Tolerates partial/missing markets per event -- a raw file
   missing a bookmaker or a market just leaves those columns null for that
   event, it never raises.

Credit accounting
------------------
the-odds-api's own historical-odds pricing: **10 credits x regions x
markets that came back with data** per request (a request specifying
markets that return nothing is not charged for those markets -- verified
against the API's own published pricing page, not assumed). This module
tracks two numbers throughout a run, both printed after every call:

- **Projected** (``CreditTracker.spent_est``) -- the *worst-case* cost of
  every call made so far, computed as ``10 * regions * markets_requested``
  (not markets returned) -- the number used for the pre-call hard-abort
  check, since the real post-call cost isn't known until the response
  headers arrive.
- **Actual**, read from every response's ``x-requests-remaining`` /
  ``x-requests-used`` headers (the API's own authoritative account
  ledger) after every call. Whichever of "projected total would exceed
  ``credit_budget``" or "planned cost exceeds the account's own
  ``x-requests-remaining``" trips first, the run hard-aborts *before*
  making that call -- never after.

Data contract
-------------
``data/external/`` is git-tracked by a negation in the repo's
``.gitignore`` (``data/*`` then ``!data/external/``); this module's own
subtree (``data/external/odds_api/**``) inherits that negation --
verified directly against ``git check-ignore`` while building this module
(see ``tests/test_odds_api.py::test_raw_json_output_is_git_tracked`` for
the standing guard). The raw JSON responses are the cross-machine
contract: this environment can read/normalize/test against them, but only
a local run (outside the proxy) can ever produce fresh ones.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "odds_api.yaml"
SCHEDULES_PATH = REPO_ROOT / "data" / "raw" / "schedules.parquet"

EXTERNAL_DIR = REPO_ROOT / "data" / "external" / "odds_api"
RAW_DIR = EXTERNAL_DIR / "raw"
MANIFEST_PATH = EXTERNAL_DIR / "manifest.json"
TEAM_LINES_PATH = EXTERNAL_DIR / "team_lines.parquet"

API_KEY_ENV = "ODDS_API_KEY"


def _filename_safe(snapshot: str) -> str:
    """ISO8601 timestamps carry ':' -- illegal in filenames on some filesystems (notably

    Windows). Raw-file/manifest ``file_stub``s use this instead of the raw timestamp string.
    """
    return snapshot.replace(":", "")

# the-odds-api's documented historical-odds multiplier: a historical call
# costs 10x what the equivalent live call would.
HISTORICAL_CREDIT_MULTIPLIER = 10

# Order-of-magnitude estimate for the Week-1 event-level props fallback
# (--pull-props with no verified futures market): 3 seasons x ~8 candidate
# prop markets x ~4 requests worth of Week-1 events x 10 credits x 1
# region. Documented as an estimate in both places it's surfaced (--dry-run
# output and the README) -- the real number depends on how many Week-1
# games and markets a season actually returns data for.
EXPENSIVE_PROPS_ESTIMATE = 1920


# --------------------------------------------------------------------------
# Config + snapshot-date derivation
# --------------------------------------------------------------------------


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def snapshot_dates_for_seasons(seasons: list[int], schedules_path: Path = SCHEDULES_PATH) -> dict[int, str]:
    """season -> ISO8601 snapshot timestamp ("preseason, a few days before kickoff").

    Each season's first REG-season gameday (min ``week``, then min
    ``gameday`` within that week -- the earliest kickoff of the season,
    including any Thursday-night opener), minus 4 days, at
    ``T12:00:00Z``. Derived from ``data/raw/schedules.parquet`` at runtime
    rather than hardcoded, so a schedules re-pull (e.g. a season's
    calendar shifting by a day) keeps these snapshot dates honest without
    an edit here.
    """
    df = pl.read_parquet(schedules_path)
    reg = df.filter(pl.col("game_type") == "REG")
    out: dict[int, str] = {}
    for season in seasons:
        sub = reg.filter(pl.col("season") == season)
        if sub.height == 0:
            continue
        first_week = sub.get_column("week").min()
        first_gameday = sub.filter(pl.col("week") == first_week).get_column("gameday").min()
        kickoff = date.fromisoformat(str(first_gameday))
        snapshot = kickoff - timedelta(days=4)
        out[season] = f"{snapshot.isoformat()}T12:00:00Z"
    return out


# --------------------------------------------------------------------------
# Request planning (pure, no network -- the --dry-run path)
# --------------------------------------------------------------------------


def estimate_cost(n_regions: int, n_markets: int) -> int:
    """Worst-case credit cost of one historical-odds request: 10 x regions x markets requested.

    The real post-call cost (billed only for markets that returned data)
    can be lower -- see the module docstring's "Credit accounting"
    section -- but the pre-call hard-abort check must use the worst case,
    since the real number isn't known until the response arrives.
    """
    return HISTORICAL_CREDIT_MULTIPLIER * n_regions * n_markets


@dataclass(frozen=True)
class PlannedRequest:
    """One planned (or executed) API call: everything --dry-run needs to print, plus enough

    to actually issue the call and to name/manifest its output file.
    """

    label: str  # human-readable, e.g. "pull-team season=2023"
    endpoint: str  # path relative to base_url, e.g. "/historical/sports/{sport}/odds"
    params: dict[str, str]
    markets: list[str]
    regions: list[str]
    season: int | None
    snapshot: str | None
    est_cost: int  # displayed / "typical" projection -- see worst_case_cost
    file_stub: str  # "<season>_<market>_<snapshot>" (no extension)
    worst_case_cost: int = 0  # ceiling used by the pre-call hard-abort check; 0 -> defaults to est_cost

    def __post_init__(self) -> None:
        if self.worst_case_cost == 0:
            object.__setattr__(self, "worst_case_cost", self.est_cost)


def _historical_odds_request(
    cfg: dict,
    *,
    label: str,
    season: int,
    snapshot: str,
    markets: list[str],
    file_stub_market: str,
    est_cost: int | None = None,
) -> PlannedRequest:
    """One historical-odds request plan. ``est_cost`` overrides the displayed/typical cost

    (used by verify-futures, whose real billed cost is expected to be near the historical
    multiplier's floor -- see that function's docstring); the pre-call hard-abort ceiling
    (``worst_case_cost``) is always the full ``10 x regions x markets_requested`` formula,
    regardless of what's displayed as the typical estimate.
    """
    regions = [cfg["region"]]
    params = {
        "regions": ",".join(regions),
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "date": snapshot,
    }
    worst_case = estimate_cost(len(regions), len(markets))
    return PlannedRequest(
        label=label,
        endpoint=f"/historical/sports/{cfg['sport_key']}/odds",
        params=params,
        markets=markets,
        regions=regions,
        season=season,
        snapshot=snapshot,
        est_cost=worst_case if est_cost is None else est_cost,
        worst_case_cost=worst_case,
        file_stub=f"{season}_{file_stub_market}_{_filename_safe(snapshot)}",
    )


def team_snapshot_requests(cfg: dict, snapshot_dates: dict[int, str]) -> list[PlannedRequest]:
    """--pull-team's plan: one combined h2h+spreads+totals call per configured preseason snapshot."""
    markets = list(cfg["featured_markets"])
    market_stub = "+".join(markets)
    reqs = []
    for season in cfg["snapshot_seasons"]:
        if season not in snapshot_dates:
            continue
        reqs.append(
            _historical_odds_request(
                cfg,
                label=f"pull-team season={season}",
                season=season,
                snapshot=snapshot_dates[season],
                markets=markets,
                file_stub_market=market_stub,
            )
        )
    return reqs


def verify_futures_request(cfg: dict, snapshot_dates: dict[int, str]) -> PlannedRequest | None:
    """--verify-futures' plan: ONE call, earliest props-era snapshot, every candidate market at once.

    ``est_cost`` (the number displayed and summed into --dry-run's headline projection) is
    pinned to the historical-odds floor (10 credits = 1 region x 1 market), per the brief's
    "Cost ~10 credits" -- the-odds-api only bills for markets that actually return data (see
    module docstring's "Credit accounting"), and season-long futures markets are expected to
    be rare-to-absent, so the realistic case is that few or none of the candidates bill at
    all. The pre-call hard-abort ceiling (``worst_case_cost``) is still the full
    ``10 x regions x len(candidates)`` in case every candidate turns out to exist.
    """
    season = cfg["props_era_start"]
    if season not in snapshot_dates:
        return None
    candidates = list(cfg["candidate_futures_markets"])
    return _historical_odds_request(
        cfg,
        label=f"verify-futures season={season}",
        season=season,
        snapshot=snapshot_dates[season],
        markets=candidates,
        file_stub_market="futures_probe",
        est_cost=HISTORICAL_CREDIT_MULTIPLIER,
    )


def props_era_seasons(cfg: dict) -> list[int]:
    return [s for s in cfg["snapshot_seasons"] if s >= cfg["props_era_start"]]


def cheap_futures_requests(cfg: dict, snapshot_dates: dict[int, str], verified_markets: list[str]) -> list[PlannedRequest]:
    """--pull-props' cheap path: one call per props-era season, the verified season-long futures keys only."""
    market_stub = "+".join(verified_markets)
    reqs = []
    for season in props_era_seasons(cfg):
        if season not in snapshot_dates:
            continue
        reqs.append(
            _historical_odds_request(
                cfg,
                label=f"pull-props (futures) season={season}",
                season=season,
                snapshot=snapshot_dates[season],
                markets=verified_markets,
                file_stub_market=market_stub,
            )
        )
    return reqs


def week1_window(season: int, schedules_path: Path = SCHEDULES_PATH) -> tuple[date, date]:
    """[first, last] gameday of a season's Week 1, from schedules.parquet."""
    df = pl.read_parquet(schedules_path)
    reg = df.filter((pl.col("game_type") == "REG") & (pl.col("season") == season))
    first_week = reg.get_column("week").min()
    wk = reg.filter(pl.col("week") == first_week)
    return (
        date.fromisoformat(str(wk.get_column("gameday").min())),
        date.fromisoformat(str(wk.get_column("gameday").max())),
    )


def historical_events_request(cfg: dict, season: int, snapshot: str) -> PlannedRequest:
    """Historical events lookup for one snapshot -- the API bills this endpoint at 1 credit."""
    return PlannedRequest(
        label=f"events-lookup season={season}",
        endpoint=f"/historical/sports/{cfg['sport_key']}/events",
        params={"date": snapshot},
        markets=[],
        regions=[],
        season=season,
        snapshot=snapshot,
        est_cost=1,
        file_stub=f"{season}_events_{_filename_safe(snapshot)}",
    )


def week1_events(cfg: dict, season: int, events_payload: dict) -> list[dict]:
    """Filter an events-lookup payload to the season's Week-1 games, by commence date."""
    start, end = week1_window(season)
    out = []
    for ev in events_payload.get("data", []):
        commence = ev.get("commence_time", "")
        if not commence:
            continue
        d = date.fromisoformat(commence[:10])
        if start <= d <= end:
            out.append(ev)
    return out


def event_props_request(cfg: dict, season: int, snapshot: str, event: dict) -> PlannedRequest:
    """Per-event historical player-prop odds for one Week-1 game."""
    markets = list(cfg["event_prop_markets"])
    regions = [cfg["region"]]
    event_id = event["id"]
    return PlannedRequest(
        label=f"week1-props season={season} {event.get('away_team','?')} @ {event.get('home_team','?')}",
        endpoint=f"/historical/sports/{cfg['sport_key']}/events/{event_id}/odds",
        params={
            "regions": ",".join(regions),
            "markets": ",".join(markets),
            "oddsFormat": "american",
            "date": snapshot,
        },
        markets=markets,
        regions=regions,
        season=season,
        snapshot=snapshot,
        est_cost=estimate_cost(len(regions), len(markets)),
        file_stub=f"{season}_week1_props_{event_id}_{_filename_safe(snapshot)}",
    )


def expensive_props_requests(cfg: dict, snapshot_dates: dict[int, str]) -> list[PlannedRequest]:
    """--pull-props' fallback path: Week-1 event-level player-prop odds, 2023-2025.

    One planned request per props-era season -- in a real local run this
    expands to one request per Week-1 event once the event list for that
    snapshot is known (an events lookup is required first; this planner
    represents the per-season aggregate for --dry-run's purposes, at the
    documented ``EXPENSIVE_PROPS_ESTIMATE`` order-of-magnitude total, not
    a promise of an exact per-event count).
    """
    seasons = props_era_seasons(cfg)
    if not seasons:
        return []
    per_season_est = EXPENSIVE_PROPS_ESTIMATE // len(seasons)
    reqs = []
    for season in seasons:
        if season not in snapshot_dates:
            continue
        reqs.append(
            PlannedRequest(
                label=f"pull-props (Week-1 event props, expensive) season={season}",
                endpoint=f"/historical/sports/{cfg['sport_key']}/events/{{eventId}}/odds",
                params={"regions": cfg["region"], "markets": "<candidate player-prop markets>", "date": snapshot_dates[season]},
                markets=["<Week-1 event-level player props>"],
                regions=[cfg["region"]],
                season=season,
                snapshot=snapshot_dates[season],
                est_cost=per_season_est,
                file_stub=f"{season}_week1_props_{_filename_safe(snapshot_dates[season])}",
            )
        )
    return reqs


def full_request_plan(cfg: dict) -> dict[str, list[PlannedRequest]]:
    """Every planned request, both --pull-props branches shown (verify-futures hasn't run yet at

    --dry-run time, so which branch a real run takes is unknown) -- the plan --dry-run prints.
    """
    snapshot_dates = snapshot_dates_for_seasons(cfg["snapshot_seasons"])
    # One probe per candidate market, mirroring run_verify_futures -- the plan
    # must show what the run will actually issue, not the old single batch call.
    verify_reqs = [
        r
        for r in (
            single_market_probe_request(cfg, snapshot_dates, m) for m in cfg["candidate_futures_markets"]
        )
        if r is not None
    ]
    return {
        "verify_futures": verify_reqs,
        "pull_team": team_snapshot_requests(cfg, snapshot_dates),
        "pull_props_cheap_if_futures_found": cheap_futures_requests(
            cfg, snapshot_dates, cfg["candidate_futures_markets"]
        ),
        "pull_props_expensive_fallback": expensive_props_requests(cfg, snapshot_dates),
    }


def redact_url(url: str, api_key: str) -> str:
    return url.replace(api_key, "REDACTED") if api_key else url


def build_url(cfg: dict, req: PlannedRequest, api_key: str) -> str:
    endpoint = req.endpoint.format(eventId="{eventId}")
    params = dict(req.params)
    params["apiKey"] = api_key
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{cfg['base_url']}{endpoint}?{query}"


def print_plan(cfg: dict) -> None:
    plan = full_request_plan(cfg)
    api_key = os.environ.get(API_KEY_ENV, "")
    total = 0
    print(f"Odds API request plan | sport={cfg['sport_key']} region={cfg['region']} budget={cfg['credit_budget']}")
    for group, reqs in plan.items():
        if not reqs:
            continue
        print(f"\n[{group}] {len(reqs)} request(s)")
        for r in reqs:
            url = build_url(cfg, r, api_key or "<ODDS_API_KEY not set>")
            print(f"  - {r.label}: cost~{r.est_cost}")
            print(f"      {redact_url(url, api_key)}")
            if group != "pull_props_expensive_fallback":
                # The expensive fallback's plan is a documented per-season estimate
                # (see expensive_props_requests' docstring), not summed into the
                # single projected total below -- it only runs with --allow-expensive
                # and only if verify-futures found nothing cheap.
                total += r.est_cost
    print(f"\nProjected total (verify-futures + pull-team + cheap-futures-props path): {total} credits")
    expensive_total = sum(r.est_cost for r in plan["pull_props_expensive_fallback"])
    print(
        f"If verify-futures finds no season-long futures keys, --pull-props --allow-expensive "
        f"instead costs an estimated {expensive_total} credits (Week-1 event-level props, 2023-2025)."
    )
    print(f"credit_budget = {cfg['credit_budget']}; hard-abort triggers before any call that would exceed it.")


# --------------------------------------------------------------------------
# Credit tracking (runtime, network subcommands only)
# --------------------------------------------------------------------------


def _norm_headers(headers: Any) -> dict[str, str]:
    """Lowercase-keyed copy of a response's headers.

    ``requests`` returns a case-insensitive mapping, but ``dict(resp.headers)``
    freezes whatever casing the server happened to send -- after which a
    ``.get("x-requests-remaining")`` silently misses a ``X-Requests-Remaining``.
    Normalising once here keeps every downstream lookup honest.
    """
    return {str(k).lower(): v for k, v in dict(headers).items()}


class OddsAPIError(RuntimeError):
    """A non-2xx response that the caller did not declare tolerable.

    Carries the status, an already-redacted URL and a body snippet, so the
    error can be surfaced without ever routing the API key through a
    traceback (``requests``' own HTTPError embeds the full request URL,
    api key included).
    """

    def __init__(self, status_code: int, redacted_url: str, body_snippet: str) -> None:
        self.status_code = status_code
        self.redacted_url = redacted_url
        self.body_snippet = body_snippet
        super().__init__(f"HTTP {status_code} for {redacted_url}: {body_snippet}")


@dataclass
class CreditTracker:
    budget: int
    spent_est: int = 0
    remaining_from_api: int | None = None
    used_from_api: int | None = None
    calls: list[dict] = field(default_factory=list)

    def check_before(self, planned_cost: int, label: str) -> None:
        projected = self.spent_est + planned_cost
        if projected > self.budget:
            raise SystemExit(
                f"HARD ABORT before '{label}': projected spend {projected} credits would exceed "
                f"credit_budget={self.budget}. Already spent (estimated): {self.spent_est}."
            )
        if self.remaining_from_api is not None and planned_cost > self.remaining_from_api:
            raise SystemExit(
                f"HARD ABORT before '{label}': planned cost {planned_cost} credits exceeds the "
                f"account's own remaining quota ({self.remaining_from_api}, per the last "
                "x-requests-remaining header)."
            )

    def observe_quota(self, headers: dict) -> None:
        """Update the API-reported ledger from ANY response, success or failure.

        Split out of ``record`` so a non-2xx response's quota headers are still
        captured. The previous ``raise_for_status()``-before-``record()`` ordering
        threw the response away on error, blinding credit reporting on exactly
        the failures where the remaining quota matters most.
        """
        remaining = headers.get("x-requests-remaining")
        used = headers.get("x-requests-used")
        if remaining is not None:
            self.remaining_from_api = int(remaining)
        if used is not None:
            self.used_from_api = int(used)

    def record(self, headers: dict, planned_cost: int, label: str) -> dict:
        self.observe_quota(headers)
        self.spent_est += planned_cost
        snap = {
            "label": label,
            "planned_cost": planned_cost,
            "running_projected_total": self.spent_est,
            "x_requests_remaining": self.remaining_from_api,
            "x_requests_used": self.used_from_api,
        }
        self.calls.append(snap)
        print(
            f"  -> cost~{planned_cost} | running projected total={self.spent_est} | "
            f"api remaining={self.remaining_from_api} used={self.used_from_api}"
        )
        return snap


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def append_manifest(entry: dict, path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = load_manifest(path)
    entries.append(entry)
    path.write_text(json.dumps(entries, indent=2, default=str))


def manifest_entry(
    req: PlannedRequest, *, url: str, api_key: str, cost_credits: int, headers: dict, out_file: Path
) -> dict:
    return {
        "label": req.label,
        "url": redact_url(url, api_key),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cost_credits": cost_credits,
        "headers": {
            "x-requests-remaining": headers.get("x-requests-remaining"),
            "x-requests-used": headers.get("x-requests-used"),
        },
        "season": req.season,
        "markets": req.markets,
        "snapshot": req.snapshot,
        "out_file": str(out_file.relative_to(REPO_ROOT)) if out_file.is_relative_to(REPO_ROOT) else str(out_file),
    }


# --------------------------------------------------------------------------
# Network calls (never exercised in this environment -- see module docstring)
# --------------------------------------------------------------------------


def _api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise SystemExit(
            f"{API_KEY_ENV} is not set. This subcommand makes a real network call to "
            "api.the-odds-api.com and needs your API key -- export it and re-run:\n"
            f"  export {API_KEY_ENV}=<your key>"
        )
    return key


def execute_request(
    cfg: dict,
    req: PlannedRequest,
    tracker: CreditTracker,
    *,
    out_dir: Path = RAW_DIR,
    tolerate_statuses: tuple[int, ...] = (),
) -> Path | None:
    """Issue one planned request, write the raw response verbatim, append the manifest, update the tracker.

    Returns the written path, or ``None`` when the response carried one of
    ``tolerate_statuses`` (used by the per-market futures probe, where a 422
    is the API's way of saying "no such market key" -- a result, not a fault).
    Any other non-2xx raises ``OddsAPIError``, which unlike ``requests``'
    HTTPError never carries the unredacted URL.

    Imports ``requests`` lazily so nothing in this module needs the
    dependency merely to be imported by tests / --dry-run / --normalize.
    """
    import requests

    api_key = _api_key()
    tracker.check_before(req.worst_case_cost, req.label)

    url = build_url(cfg, req, api_key)
    print(f"GET {req.label} -> {redact_url(url, api_key)}")
    resp = requests.get(url, timeout=60)

    # Read the account ledger BEFORE any status check -- see observe_quota's
    # docstring for why the old raise-first ordering was a reporting hole.
    headers = _norm_headers(resp.headers)
    tracker.observe_quota(headers)

    if resp.status_code != 200:
        print(
            f"  -> HTTP {resp.status_code} | api remaining={tracker.remaining_from_api} "
            f"used={tracker.used_from_api}"
        )
        if resp.status_code in tolerate_statuses:
            return None
        raise OddsAPIError(
            resp.status_code, redact_url(url, api_key), redact_url(resp.text[:300], api_key)
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{req.file_stub}.json"
    out_path.write_text(resp.text)

    tracker.record(headers, req.est_cost, req.label)
    # Pass the real response headers, not the tracker's snapshot: the snapshot
    # keys them with underscores (x_requests_remaining), while manifest_entry
    # reads the hyphenated header names -- so every manifest entry previously
    # recorded null quota numbers.
    append_manifest(manifest_entry(req, url=url, api_key=api_key, cost_credits=req.est_cost, headers=headers, out_file=out_path))
    return out_path


def single_market_probe_request(cfg: dict, snapshot_dates: dict[int, str], market: str) -> PlannedRequest | None:
    """One candidate futures market, probed on its own -- see run_verify_futures."""
    season = cfg["props_era_start"]
    if season not in snapshot_dates:
        return None
    return _historical_odds_request(
        cfg,
        label=f"verify-futures market={market}",
        season=season,
        snapshot=snapshot_dates[season],
        markets=[market],
        file_stub_market=f"futures_probe_{market}",
    )


def run_verify_futures(cfg: dict) -> list[str]:
    """Probe every candidate futures market in its OWN request.

    The original implementation sent all candidates as one comma-joined
    ``markets`` parameter. The API rejects the *entire* request with 422 if
    any single key is invalid, so that batch probe could not distinguish
    "no season-long futures markets exist" from "one bogus key poisoned an
    otherwise valid batch" -- it failed closed and reported nothing. One
    request per market makes each key's verdict independent: a 422 is
    tolerated and recorded as "key does not exist" (the API does not bill
    rejected requests) rather than aborting the run.

    Costs at most ``10 x len(candidates)`` credits, and in practice far less,
    since only markets that actually return data are billed.
    """
    snapshot_dates = snapshot_dates_for_seasons(cfg["snapshot_seasons"])
    candidates = list(cfg["candidate_futures_markets"])
    tracker = CreditTracker(budget=cfg["credit_budget"])

    hits: list[str] = []
    absent: list[str] = []
    empty: list[str] = []
    errored: list[tuple[str, int]] = []

    for market in candidates:
        req = single_market_probe_request(cfg, snapshot_dates, market)
        if req is None:
            raise SystemExit(
                f"no schedules data for props_era_start={cfg['props_era_start']}; cannot derive a snapshot date"
            )
        try:
            out_path = execute_request(cfg, req, tracker, tolerate_statuses=(422,))
        except OddsAPIError as exc:
            errored.append((market, exc.status_code))
            print(f"  -> {market}: HTTP {exc.status_code} -- {exc.body_snippet}")
            continue

        if out_path is None:
            absent.append(market)
            print(f"  -> {market}: rejected (422) -- market key does not exist")
            continue

        payload = json.loads(out_path.read_text())
        if market in _market_keys_present(payload):
            hits.append(market)
            print(f"  -> {market}: EXISTS and returned data")
        else:
            empty.append(market)
            print(f"  -> {market}: accepted but returned no data at this snapshot")

    print(f"\nverify-futures | snapshot season={cfg['props_era_start']} | candidates probed={len(candidates)}")
    print(f"  FOUND (season-long futures keys that exist): {hits if hits else 'none'}")
    print(f"  rejected as invalid keys: {absent}")
    print(f"  valid but empty at this snapshot: {empty}")
    if errored:
        print(f"  errored (NOT a verdict on the key): {errored}")
    print(
        f"  quota | x-requests-remaining={tracker.remaining_from_api} "
        f"x-requests-used={tracker.used_from_api} | projected spend this run={tracker.spent_est}"
    )
    return hits


def _market_keys_present(payload: dict) -> set[str]:
    """Every distinct market ``key`` present anywhere in a historical-odds response's bookmakers."""
    keys: set[str] = set()
    for event in payload.get("data", []):
        for bk in event.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if "key" in mk:
                    keys.add(mk["key"])
    return keys


def run_pull_team(cfg: dict) -> list[Path]:
    snapshot_dates = snapshot_dates_for_seasons(cfg["snapshot_seasons"])
    reqs = team_snapshot_requests(cfg, snapshot_dates)
    tracker = CreditTracker(budget=cfg["credit_budget"])
    paths = []
    for req in reqs:
        paths.append(execute_request(cfg, req, tracker))
    print(f"\npull-team done | {len(paths)} snapshot(s) written to {RAW_DIR}")
    return paths


def run_probe_props(cfg: dict) -> dict:
    """Measure the event-props path with ONE event before committing to the full pull.

    One events lookup (1 credit) + one Week-1 event's props call (worst case
    ``10 x len(event_prop_markets)``). Reports which prop markets actually
    carry historical data at the earliest props-era snapshot, the *billed*
    cost of the single event (measured from the x-requests-used delta -- the
    account's own ledger, not an estimate), and an exact extrapolation for
    the full 3-season pull.
    """
    season = cfg["props_era_start"]
    snapshot_dates = snapshot_dates_for_seasons(cfg["snapshot_seasons"])
    snapshot = snapshot_dates[season]
    tracker = CreditTracker(budget=cfg["credit_budget"])

    ev_path = execute_request(cfg, historical_events_request(cfg, season, snapshot), tracker)
    events = week1_events(cfg, season, json.loads(ev_path.read_text()))
    if not events:
        raise SystemExit(f"probe-props: events lookup returned no Week-1 events for {season}")
    print(f"probe-props | season={season} | Week-1 events found: {len(events)}")

    used_before = tracker.used_from_api
    target = sorted(events, key=lambda e: e.get("commence_time", ""))[0]
    out_path = execute_request(cfg, event_props_request(cfg, season, snapshot, target), tracker)
    used_after = tracker.used_from_api

    billed = (used_after - used_before) if (used_before is not None and used_after is not None) else None
    payload = json.loads(out_path.read_text())
    # Event-odds responses nest the event under "data" as a single object.
    ev_data = payload.get("data", {})
    present = set()
    for bk in ev_data.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if "key" in mk:
                present.add(mk["key"])
    requested = set(cfg["event_prop_markets"])

    n_events = len(events)
    seasons = props_era_seasons(cfg)
    extrapolated = None if billed is None else billed * n_events * len(seasons) + len(seasons)
    print(f"\nprobe-props | event: {target.get('away_team')} @ {target.get('home_team')}")
    print(f"  prop markets WITH historical data: {sorted(present & requested)}")
    print(f"  requested but no data: {sorted(requested - present)}")
    print(f"  billed for this one event (x-requests-used delta): {billed} credits")
    print(
        f"  extrapolation: {billed} x {n_events} Week-1 events x {len(seasons)} seasons "
        f"+ {len(seasons)} events-lookups = ~{extrapolated} credits for the full pull"
    )
    print(
        f"  quota | x-requests-remaining={tracker.remaining_from_api} "
        f"x-requests-used={tracker.used_from_api}"
    )
    return {
        "billed_per_event": billed,
        "n_week1_events": n_events,
        "markets_present": sorted(present & requested),
        "extrapolated_full_pull": extrapolated,
    }


def run_pull_props(cfg: dict, *, allow_expensive: bool) -> list[Path]:
    manifest = load_manifest()
    verified: list[str] = []
    for entry in manifest:
        if "futures_probe" in entry.get("out_file", "") or "verify-futures" in entry.get("label", ""):
            raw = json.loads((REPO_ROOT / entry["out_file"]).read_text())
            verified.extend(sorted(_market_keys_present(raw) & set(cfg["candidate_futures_markets"])))
    verified = sorted(set(verified))

    snapshot_dates = snapshot_dates_for_seasons(cfg["snapshot_seasons"])
    tracker = CreditTracker(budget=cfg["credit_budget"])

    if verified:
        print(f"pull-props | verify-futures previously found season-long futures keys: {verified} -- using the cheap path")
        reqs = cheap_futures_requests(cfg, snapshot_dates, verified)
    else:
        if not allow_expensive:
            raise SystemExit(
                "pull-props: --verify-futures found no season-long futures market (or has not been run "
                "yet). Falling back to Week-1 event-level player-prop odds costs an estimated "
                f"~{EXPENSIVE_PROPS_ESTIMATE} credits (2023-2025) -- refusing without --allow-expensive. "
                "Run `--verify-futures` first, or re-run with `--pull-props --allow-expensive` to accept "
                "the expensive fallback."
            )
        print(
            f"pull-props | no season-long futures key found -- falling back to the expensive "
            f"Week-1 event-level props path (~{EXPENSIVE_PROPS_ESTIMATE} credits, --allow-expensive was passed)"
        )
        paths: list[Path] = []
        for season in props_era_seasons(cfg):
            if season not in snapshot_dates:
                continue
            snapshot = snapshot_dates[season]
            ev_path = execute_request(cfg, historical_events_request(cfg, season, snapshot), tracker)
            events = sorted(
                week1_events(cfg, season, json.loads(ev_path.read_text())),
                key=lambda e: e.get("commence_time", ""),
            )
            print(f"pull-props | season={season}: {len(events)} Week-1 events")
            for event in events:
                out = execute_request(cfg, event_props_request(cfg, season, snapshot, event), tracker)
                if out is not None:
                    paths.append(out)
        print(f"\npull-props done | {len(paths)} file(s) written to {RAW_DIR}")
        return paths

    paths = []
    for req in reqs:
        out = execute_request(cfg, req, tracker)
        if out is not None:
            paths.append(out)
    print(f"\npull-props done | {len(paths)} file(s) written to {RAW_DIR}")
    return paths


# --------------------------------------------------------------------------
# Normalize (pure local -- no network, no API key)
# --------------------------------------------------------------------------


def _median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def normalize_raw_event(event: dict, season: int, snapshot: str) -> dict:
    """One event's bookmaker markets -> one tidy row: median price/point across bookmakers.

    Median (not "first bookmaker") so one outlier book doesn't skew the
    team-line snapshot -- a deliberate, simple choice, not a claim of
    consensus-line precision. Missing markets/outcomes leave their columns
    null rather than raising -- "tolerate partial data" per the brief.
    """
    home = event.get("home_team")
    away = event.get("away_team")
    h2h_home, h2h_away, spread_home, total_points = [], [], [], []

    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            key = mk.get("key")
            outcomes = mk.get("outcomes", [])
            if key == "h2h":
                for o in outcomes:
                    if o.get("name") == home and o.get("price") is not None:
                        h2h_home.append(float(o["price"]))
                    elif o.get("name") == away and o.get("price") is not None:
                        h2h_away.append(float(o["price"]))
            elif key == "spreads":
                for o in outcomes:
                    if o.get("name") == home and o.get("point") is not None:
                        spread_home.append(float(o["point"]))
            elif key == "totals":
                for o in outcomes:
                    if o.get("point") is not None:
                        total_points.append(float(o["point"]))

    return {
        "season": season,
        "snapshot": snapshot,
        "event_id": event.get("id"),
        "commence_time": event.get("commence_time"),
        "home_team": home,
        "away_team": away,
        "home_ml": _median(h2h_home),
        "away_ml": _median(h2h_away),
        "home_spread": _median(spread_home),
        "total": _median(total_points),
    }


def normalize_raw_dir(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    """Parse every raw/*.json in ``raw_dir`` into one tidy team_lines frame.

    Tolerates: a file that isn't team-level odds at all (skipped -- e.g. a
    verify-futures probe file, whose payload shape this function does not
    expect team h2h/spreads/totals markets in), a file with fewer than
    three markets present, an event missing a bookmaker entirely. Nothing
    here raises on partial data; a row simply carries nulls for whatever
    wasn't found.
    """
    rows: list[dict] = []
    if not raw_dir.exists():
        return pl.DataFrame(schema=_TEAM_LINES_SCHEMA)

    for path in sorted(raw_dir.glob("*.json")):
        # Team-lines files only: skip futures probes, per-event props files,
        # and events-lookup files (whose "data" is a bare event list with no
        # bookmakers -- normalize_raw_event would emit all-null odds rows).
        if "futures_probe" in path.stem or "week1_props" in path.stem or "_events_" in path.stem:
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        data = payload.get("data")
        if not isinstance(data, list):
            continue
        # file_stub convention: "<season>_<market(s)>_<snapshot>" -- season is
        # the leading underscore-delimited token, always an int.
        try:
            season = int(path.stem.split("_")[0])
        except ValueError:
            continue
        snapshot = payload.get("timestamp", "")
        for event in data:
            rows.append(normalize_raw_event(event, season, snapshot))

    if not rows:
        return pl.DataFrame(schema=_TEAM_LINES_SCHEMA)
    return pl.DataFrame(rows, schema_overrides=_TEAM_LINES_SCHEMA).sort(["season", "snapshot", "home_team"])


_TEAM_LINES_SCHEMA = {
    "season": pl.Int64,
    "snapshot": pl.Utf8,
    "event_id": pl.Utf8,
    "commence_time": pl.Utf8,
    "home_team": pl.Utf8,
    "away_team": pl.Utf8,
    "home_ml": pl.Float64,
    "away_ml": pl.Float64,
    "home_spread": pl.Float64,
    "total": pl.Float64,
}


def run_normalize(raw_dir: Path = RAW_DIR, out_path: Path = TEAM_LINES_PATH) -> pl.DataFrame:
    df = normalize_raw_dir(raw_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"normalize | wrote {out_path} | {df.height:,} rows x {df.width} cols (from {raw_dir})")
    return df


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The Odds API ingest (v1.5) -- see module docstring; local run required.")
    ap.add_argument("--dry-run", action="store_true", help="print the full request plan, zero network calls")
    ap.add_argument("--verify-futures", action="store_true", help="ONE historical call probing season-long futures keys (~10 credits)")
    ap.add_argument("--pull-team", action="store_true", help="featured markets, 6 preseason snapshots (~180 credits)")
    ap.add_argument("--pull-props", action="store_true", help="props-era snapshots (cheap if verify-futures found a key, else needs --allow-expensive)")
    ap.add_argument("--probe-props", action="store_true", help="ONE events lookup + ONE Week-1 event's props: measure real per-event cost before --pull-props")
    ap.add_argument("--allow-expensive", action="store_true", help="permit --pull-props' expensive Week-1 event-props fallback")
    ap.add_argument("--normalize", action="store_true", help="parse raw/*.json -> team_lines.parquet, no network, no key")
    args = ap.parse_args(argv)

    cfg = load_config()
    ran_anything = False

    if args.dry_run:
        print_plan(cfg)
        ran_anything = True

    if args.normalize:
        run_normalize()
        ran_anything = True

    if args.verify_futures:
        run_verify_futures(cfg)
        ran_anything = True

    if args.probe_props:
        run_probe_props(cfg)
        ran_anything = True

    if args.pull_team:
        run_pull_team(cfg)
        ran_anything = True

    if args.pull_props:
        run_pull_props(cfg, allow_expensive=args.allow_expensive)
        ran_anything = True

    if not ran_anything:
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
