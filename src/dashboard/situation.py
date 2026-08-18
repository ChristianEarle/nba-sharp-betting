"""Per-player roster-situation context (v2.5) -- the "why" behind the numbers.

The board's rationale strings name model features ("elite max single vacated
carry share"), which is honest but opaque: the reader wants the underlying
roster story -- WHO left the player's team, how much usage and how many
touchdowns walked out the door, who arrived to compete, and where the player
sits in his own team's positional pecking order. Every one of those facts is
already on disk; this module turns them into short plain-English lines
attached to each board row (payload key ``sit``).

Sources (2025 REG production + season-2026 offseason rosters -- pure
preseason facts, same leakage class as everything else the board shows;
presentation-layer only, never a model input):

- ``data/raw/player_stats.parquet``  2025 usage per (player, team):
  carries/rush TDs, targets/receiving TDs, pass attempts.
- ``data/raw/rosters.parquet``       season-2026 rows = who is on which
  roster now. A 2025 contributor with no 2026 row anywhere is "gone"
  (retired/unsigned) -- still a departure from his old team's usage pool.

Line types, per board player:

1. ARRIVAL      he changed teams: what he produced for the old team.
2. DEPARTURES   who left his 2026 team at his usage type (RB/QB ->
                carries, WR/TE -> targets), with the share of team usage
                and TDs now up for grabs. Top 2 by share, >=10% only.
3. COMPETITION  who arrived at his position with material 2025 usage.
4. PECKING      his market rank (consensus ECR) among his own team's
                same-position board rows -- "the 1" or "behind X".

Team codes are normalized through ``src.features.shared.normalize_team``
so 2025 stats teams and 2026 roster teams join cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from src.features.shared import normalize_team

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

STATS_SEASON = 2025
ROSTER_SEASON = 2026

# Usage column + TD column + human nouns, by board position.
_USAGE = {
    "QB": ("attempts", None, "pass attempts", None),
    "RB": ("carries", "rushing_tds", "carries", "rush TDs"),
    "WR": ("targets", "receiving_tds", "targets", "receiving TDs"),
    "TE": ("targets", "receiving_tds", "targets", "receiving TDs"),
}
_DEPARTURE_MIN_SHARE = 0.10
_COMPETITION_MIN_USAGE = {"QB": 200, "RB": 80, "WR": 50, "TE": 40}


def _load_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    ps = pl.read_parquet(RAW_DIR / "player_stats.parquet")
    ps = ps.filter(
        (pl.col("season") == STATS_SEASON)
        & (pl.col("season_type") == "REG")
        & pl.col("player_id").is_not_null()
    )
    ps = normalize_team(ps, "team")
    ros = pl.read_parquet(RAW_DIR / "rosters.parquet")
    ros = ros.filter((pl.col("season") == ROSTER_SEASON) & pl.col("gsis_id").is_not_null())
    ros = normalize_team(ros, "team")
    return ps, ros


def _usage_2025(ps: pl.DataFrame) -> pl.DataFrame:
    """(gsis, team) -> position_group + summed usage columns, 2025 REG."""
    return (
        ps.group_by(["player_id", "team"])
        .agg(
            pl.col("player_display_name").first().alias("name"),
            pl.col("position_group").first().alias("pos"),
            pl.col("attempts").sum().alias("attempts"),
            pl.col("carries").sum().alias("carries"),
            pl.col("rushing_tds").sum().alias("rushing_tds"),
            pl.col("targets").sum().alias("targets"),
            pl.col("receiving_tds").sum().alias("receiving_tds"),
        )
        .rename({"player_id": "gsis_id"})
    )


def _plural(n: int, noun: str) -> str:
    """'1 rush TD' / '8 rush TDs' -- the plural nouns in _USAGE end in 's'."""
    return f"{n} {noun[:-1] if n == 1 else noun}"


def _fmt_usage(row: dict, pos: str) -> str:
    usage_col, td_col, usage_noun, td_noun = _USAGE[pos]
    parts = [_plural(int(row[usage_col]), usage_noun)]
    if td_col and row[td_col]:
        parts.append(_plural(int(row[td_col]), td_noun))
    return " and ".join(parts)


def build_situations(board: pd.DataFrame) -> dict[str, list[str]]:
    """board (needs key/gsis_id/pos/team/player_name/consensus_ecr_pos_rank) -> {key: [lines]}."""
    ps, ros = _load_frames()
    usage = _usage_2025(ps)
    team_2026 = dict(
        zip(
            ros.get_column("gsis_id").to_list(),
            ros.get_column("team").to_list(),
        )
    )
    # Team totals per usage column, for shares.
    totals = usage.group_by("team").agg(
        pl.col("attempts").sum().alias("t_attempts"),
        pl.col("carries").sum().alias("t_carries"),
        pl.col("targets").sum().alias("t_targets"),
    )
    urows = usage.join(totals, on="team", how="left").to_dicts()

    # Index 2025 usage by gsis (primary = row with most of the player's own usage).
    by_gsis: dict[str, dict] = {}
    for r in urows:
        best = by_gsis.get(r["gsis_id"])
        if best is None or (r["carries"] + r["targets"] + r["attempts"]) > (
            best["carries"] + best["targets"] + best["attempts"]
        ):
            by_gsis[r["gsis_id"]] = r

    out: dict[str, list[str]] = {}
    board = board.copy()

    # Pecking order: market rank among same (team, pos) board rows.
    board["_ecr"] = pd.to_numeric(board.get("consensus_ecr_pos_rank"), errors="coerce")

    for _, b in board.iterrows():
        pos, team, key = b.get("pos"), b.get("team"), b.get("key")
        gsis = b.get("gsis_id")
        if pos not in _USAGE or not isinstance(team, str) or not key:
            continue
        usage_col, td_col, usage_noun, td_noun = _USAGE[pos]
        lines: list[str] = []

        # 1. ARRIVAL
        mine = by_gsis.get(gsis) if isinstance(gsis, str) else None
        if mine and mine["team"] != team:
            lines.append(f"Arrived from {mine['team']} -- had {_fmt_usage(mine, pos)} there in 2025.")

        # 2. DEPARTURES from his 2026 team
        total_key = f"t_{usage_col}"
        departed = []
        for r in urows:
            if r["team"] != team or r["pos"] != pos or r["gsis_id"] == gsis:
                continue
            now = team_2026.get(r["gsis_id"])
            if now == team:
                continue  # still on the roster
            share = (r[usage_col] / r[total_key]) if r[total_key] else 0.0
            if share >= _DEPARTURE_MIN_SHARE:
                departed.append((share, r, now))
        departed.sort(key=lambda x: -x[0])
        for share, r, now in departed[:2]:
            dest = f"to {now}" if now else "(unsigned)"
            tds = f" and {_plural(int(r[td_col]), td_noun)}" if td_col and r[td_col] else ""
            lines.append(
                f"{r['name']} left {dest} -- {share:.0%} of the team's 2025 {usage_noun}{tds} up for grabs."
            )

        # 3. NEW COMPETITION at his position
        arrivals = []
        for r in urows:
            if r["pos"] != pos or r["gsis_id"] == gsis or r["team"] == team:
                continue
            if team_2026.get(r["gsis_id"]) != team:
                continue
            if r[usage_col] >= _COMPETITION_MIN_USAGE[pos]:
                arrivals.append(r)
        arrivals.sort(key=lambda r: -r[usage_col])
        for r in arrivals[:2]:
            lines.append(
                f"New competition: {r['name']} arrived from {r['team']} ({_fmt_usage(r, pos)} in 2025)."
            )

        # 4. PECKING ORDER by market price among same team+pos board rows
        peers = board[(board["team"] == team) & (board["pos"] == pos) & board["_ecr"].notna()]
        my_ecr = b.get("_ecr")
        if pd.notna(my_ecr) and len(peers) > 1:
            better = peers[peers["_ecr"] < my_ecr]
            if better.empty:
                lines.append(f"Priced as {team}'s top {pos} by market rank.")
            else:
                ahead = better.sort_values("_ecr")["player_name"].tolist()
                names = ", ".join(ahead[:2]) + (" +more" if len(ahead) > 2 else "")
                lines.append(f"Priced {pos}{len(better) + 1} on {team}, behind {names}.")

        if lines:
            out[key] = lines
    return out
