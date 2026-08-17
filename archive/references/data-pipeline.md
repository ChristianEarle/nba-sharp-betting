# Data Pipeline — APIs, Sources & Model Inputs

Read this file when building or refining clawdbot's data collection, odds comparison,
injury monitoring, or any data-driven NBA analysis workflow.

---

## 1. DATA SOURCE HIERARCHY

Not all data sources are equal. Prioritize by reliability and timeliness:

### Tier 1: Primary Sources (Highest Trust)

| Source | Data Type | Access | Notes |
|--------|-----------|--------|-------|
| **Kalshi API** | Market prices, orderbook, volume | REST + WebSocket, free | Your trading venue — always current |
| **The Odds API** | Sportsbook lines from 20+ books | REST, free tier available | theoddsapi.com — best aggregated odds source |
| **NBA.com Official** | Injury reports, box scores, schedules | Web scraping / undocumented API | Authoritative for injuries and stats |
| **Pinnacle odds** | Sharpest line in the market | Via The Odds API or direct | Gold standard for true probability estimation |

### Tier 2: Rich Data Sources

| Source | Data Type | Access | Notes |
|--------|-----------|--------|-------|
| **balldontlie.io** | Player stats, game logs, historical data | REST API, free tier | Great for player prop modeling |
| **Basketball Reference** | Advanced stats, SRS, pace, ratings | Web scraping | bbref.com — gold standard for advanced metrics |
| **nba_api (Python)** | NBA.com stats endpoint wrapper | `pip install nba_api` | Programmatic access to NBA.com stats |
| **nbainjuries (Python)** | Structured injury report data | `pip install nbainjuries` | Historical + real-time injury data since 2021-22 |

### Tier 3: Supplementary Sources

| Source | Data Type | Access |
|--------|-----------|--------|
| **ESPN API** | Scores, schedules, some stats | Undocumented REST endpoints |
| **RotoWire** | Injury reports with fantasy context | Web, rotowire.com |
| **Action Network** | Sharp betting data, public % | actionnetwork.com |
| **SportsGrid** | Kalshi market movers, prediction market analysis | sportsgrid.com |

---

## 2. THE ODDS API — SPORTSBOOK LINE COMPARISON

This is clawdbot's primary tool for the sportsbook ↔ Kalshi arbitrage strategy.

### Setup
```python
import requests

ODDS_API_KEY = "your_key_here"  # from theoddsapi.com
BASE_URL = "https://api.the-odds-api.com/v4"

# Get NBA moneylines from all available books
def get_nba_odds():
    url = f"{BASE_URL}/sports/basketball_nba/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us,us2",       # US sportsbooks
        "markets": "h2h",          # moneyline (head-to-head)
        "oddsFormat": "decimal",   # easier to convert to probability
        "bookmakers": "pinnacle,fanduel,draftkings,betmgm,caesars"
    }
    response = requests.get(url, params=params)
    return response.json()
```

### Converting Odds to Probability

```python
def decimal_odds_to_probability(decimal_odds):
    """Convert decimal odds to implied probability."""
    return 1 / decimal_odds

def american_to_probability(american_odds):
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def remove_vig(prob_a, prob_b):
    """Remove vig from two-way market to get true probabilities."""
    total = prob_a + prob_b
    return prob_a / total, prob_b / total
```

### Building the Comparison Matrix

```python
def compare_kalshi_vs_sportsbooks(kalshi_price, sportsbook_odds):
    """
    kalshi_price: YES price in cents (e.g., 65 = 65¢ = 65% implied)
    sportsbook_odds: dict of book_name → decimal_odds for the YES team
    
    Returns edge for each sportsbook comparison.
    """
    kalshi_implied = kalshi_price / 100
    
    edges = {}
    for book, odds in sportsbook_odds.items():
        book_implied = decimal_odds_to_probability(odds)
        edge = book_implied - kalshi_implied
        edges[book] = {
            "book_implied": round(book_implied, 4),
            "kalshi_implied": kalshi_implied,
            "edge": round(edge, 4),
            "tradeable": edge >= 0.05  # minimum 5¢ edge
        }
    
    # Pinnacle gets extra weight — it's the sharpest line
    if "pinnacle" in edges:
        edges["pinnacle"]["weight"] = 1.5
    
    return edges
```

### Available Markets via The Odds API

| Market Key | Description | Kalshi Equivalent |
|-----------|-------------|-------------------|
| `h2h` | Moneyline (who wins) | Game winner YES/NO |
| `spreads` | Point spread | Kalshi spread markets (Will [Team] win by X+?) |
| `totals` | Over/under total points | Not directly on Kalshi (yet) |
| `player_points` | Player points over/under | Kalshi player props (points) |
| `player_rebounds` | Player rebounds over/under | Kalshi player props (rebounds) |
| `player_assists` | Player assists over/under | Kalshi player props (assists) |
| `player_threes` | Player 3-pointers over/under | Kalshi player props (threes) |

---

## 3. KALSHI API — MARKET DATA PIPELINE

### Fetching NBA Markets

```python
import requests

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

def get_nba_markets():
    """Fetch all open NBA markets from Kalshi."""
    # NBA markets typically use series tickers containing "NBA" or "KXNBA"
    url = f"{KALSHI_BASE}/markets"
    params = {
        "status": "open",
        "limit": 200
    }
    response = requests.get(url, params=params)
    markets = response.json().get("markets", [])
    
    # Filter for NBA-related markets
    nba_markets = [
        m for m in markets
        if "nba" in m.get("title", "").lower()
        or "basketball" in m.get("title", "").lower()
        or any(team in m.get("title", "") for team in NBA_TEAMS)
    ]
    return nba_markets

def get_orderbook(ticker, depth=10):
    """Get current orderbook for a market."""
    url = f"{KALSHI_BASE}/markets/{ticker}/orderbook"
    params = {"depth": depth}
    response = requests.get(url, params=params)
    return response.json()

# NBA team names for matching
NBA_TEAMS = [
    "Hawks", "Celtics", "Nets", "Hornets", "Bulls", "Cavaliers",
    "Mavericks", "Nuggets", "Pistons", "Warriors", "Rockets", "Pacers",
    "Clippers", "Lakers", "Grizzlies", "Heat", "Bucks", "Timberwolves",
    "Pelicans", "Knicks", "Thunder", "Magic", "76ers", "Suns",
    "Trail Blazers", "Kings", "Spurs", "Raptors", "Jazz", "Wizards"
]
```

### Real-Time Monitoring via WebSocket

```python
import asyncio
import websockets
import json

async def monitor_kalshi_markets(tickers):
    """Subscribe to real-time updates for specific NBA markets."""
    uri = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    
    async with websockets.connect(uri) as ws:
        # Authenticate (implement your RSA signing here)
        # ...
        
        # Subscribe to orderbook updates for target markets
        for ticker in tickers:
            subscribe_msg = {
                "type": "subscribe",
                "channels": ["orderbook_delta", "trade"],
                "market_tickers": [ticker]
            }
            await ws.send(json.dumps(subscribe_msg))
        
        # Process incoming updates
        async for message in ws:
            data = json.loads(message)
            process_market_update(data)

def process_market_update(data):
    """Process real-time market data and check for edge opportunities."""
    # Compare against current model probabilities
    # Alert if edge threshold is crossed
    # Log all updates for analysis
    pass
```

---

## 4. INJURY DATA PIPELINE

### Official NBA Injury Reports

```python
# Using nbainjuries package for structured injury data
from nbainjuries import injury
from datetime import datetime

def get_current_injuries():
    """Fetch current NBA injury report."""
    now = datetime.now()
    report = injury.get_reportdata(now, return_df=True)
    return report

# Report columns include:
# - Team
# - Player Name
# - Status (Out, Doubtful, Questionable, Probable, Available)
# - Reason (specific injury/illness)
# - Report datetime
```

### Injury Report Timing (Critical for Trading)

| Situation | Report Deadline | clawdbot Action |
|-----------|----------------|-----------------|
| Normal game day | 5 PM local, day before | Check report at 5:15 PM, compare to Kalshi prices |
| B2B (2nd game) | 1 PM local, game day | Check report at 1:15 PM — this is PRIME edge time |
| Game-day update | 11 AM - 1 PM local | Monitor continuously — status changes drive line movement |
| Late scratch | Up to tip-off | WebSocket monitoring essential — fastest responders win |

### Injury Impact Quantification

```python
def estimate_injury_impact(player_name, team):
    """
    Estimate point swing from a player's absence.
    Uses on/off net rating differential.
    """
    # Fetch player's on/off data from nba_api or Basketball Reference
    # on_court_net_rating - off_court_net_rating = impact per 100 possessions
    # Adjust by minutes played / 48 to get per-game impact
    # 
    # Example: Player averages 34 min/game
    #   On-court net rating: +8.5
    #   Off-court net rating: +2.1
    #   Impact = (8.5 - 2.1) × (34/48) = +4.5 points per game
    #
    # This player being OUT swings the line by ~4.5 points
    pass
```

---

## 5. SCHEDULE AND REST DATA

### NBA Schedule API

```python
# Using nba_api for schedule data
from nba_api.stats.endpoints import leaguegamefinder, scoreboardv2

def get_today_games():
    """Get today's NBA schedule."""
    from datetime import date
    scoreboard = scoreboardv2.ScoreboardV2(game_date=date.today().strftime("%Y-%m-%d"))
    games = scoreboard.get_data_frames()[0]
    return games

def get_team_schedule(team_id, season="2025-26"):
    """Get full season schedule for rest/B2B analysis."""
    games = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team_id,
        season_nullable=season
    )
    return games.get_data_frames()[0]
```

### Back-to-Back Detection

```python
from datetime import datetime, timedelta

def detect_b2b(schedule_df, game_date):
    """
    Check if a team is on the 2nd of a back-to-back.
    Returns B2B info dict or None.
    """
    yesterday = game_date - timedelta(days=1)
    
    yesterday_game = schedule_df[
        schedule_df['GAME_DATE'] == yesterday.strftime("%Y-%m-%d")
    ]
    
    if len(yesterday_game) > 0:
        return {
            "is_b2b": True,
            "previous_game": yesterday_game.iloc[0].to_dict(),
            "travel_distance": calculate_travel(
                yesterday_game.iloc[0]['MATCHUP'],
                game_date  # current game location
            ),
            "previous_was_ot": yesterday_game.iloc[0].get('OT', False)
        }
    return {"is_b2b": False}

def calculate_rest_days(schedule_df, game_date):
    """Calculate days since last game."""
    previous_games = schedule_df[
        schedule_df['GAME_DATE'] < game_date.strftime("%Y-%m-%d")
    ].sort_values('GAME_DATE', ascending=False)
    
    if len(previous_games) > 0:
        last_game = datetime.strptime(previous_games.iloc[0]['GAME_DATE'], "%Y-%m-%d")
        return (game_date - last_game).days
    return 99  # season opener
```

---

## 6. ADVANCED STATS FOR MODELING

### Team Ratings

```python
from nba_api.stats.endpoints import leaguedashteamstats

def get_team_ratings():
    """Fetch current team offensive/defensive ratings."""
    stats = leaguedashteamstats.LeagueDashTeamStats(
        measure_type_detailed_defense="Advanced",
        season="2025-26",
        per_mode_detailed="Per100Possessions"
    )
    df = stats.get_data_frames()[0]
    
    # Key columns: TEAM_NAME, OFF_RATING, DEF_RATING, NET_RATING, PACE
    return df[['TEAM_NAME', 'TEAM_ID', 'OFF_RATING', 'DEF_RATING', 
               'NET_RATING', 'PACE', 'W', 'L']]
```

### Player Props Modeling

```python
from nba_api.stats.endpoints import playergamelogs

def get_player_game_logs(player_id, season="2025-26"):
    """Fetch player game logs for prop modeling."""
    logs = playergamelogs.PlayerGameLogs(
        player_id_nullable=player_id,
        season_nullable=season
    )
    df = logs.get_data_frames()[0]
    
    # Key columns: PTS, REB, AST, FG3M, MIN, MATCHUP
    # Calculate rolling averages, B2B splits, home/away splits
    return df

def model_player_prop(player_id, prop_type, line, context):
    """
    Estimate probability of player going over the prop line.
    
    prop_type: "points", "rebounds", "assists", "threes"
    line: the over/under number (e.g., 24.5 for points)
    context: dict with game context (opponent, B2B, home/away, etc.)
    """
    logs = get_player_game_logs(player_id)
    
    # Base rate: what % of games does player exceed this line?
    stat_col = {"points": "PTS", "rebounds": "REB", 
                "assists": "AST", "threes": "FG3M"}[prop_type]
    
    base_rate = (logs[stat_col] > line).mean()
    
    # Apply situational adjustments
    adjustments = 0
    
    if context.get("is_b2b"):
        # B2B discount: -5% to -15% depending on position
        adjustments -= 0.08
    
    if context.get("rest_advantage"):
        adjustments += 0.03
    
    # Matchup adjustment (opponent defensive rating for this stat)
    # ...
    
    adjusted_probability = base_rate + adjustments
    return min(max(adjusted_probability, 0.05), 0.95)  # clamp to reasonable range
```

---

## 7. THE EDGE DETECTION ENGINE

### Full Pipeline: Game Evaluation

```python
def evaluate_game(home_team, away_team, game_date):
    """
    Full sharp evaluation pipeline for an NBA game.
    Returns trade recommendation if edge exists.
    """
    # 1. Get sportsbook consensus
    odds = get_nba_odds()  # from The Odds API
    game_odds = find_game_in_odds(odds, home_team, away_team)
    
    # 2. Calculate sharp implied probability (Pinnacle-weighted)
    pinnacle_prob = get_pinnacle_implied(game_odds)
    consensus_prob = get_consensus_implied(game_odds)
    
    # 3. Get Kalshi prices
    kalshi_markets = get_nba_markets()
    kalshi_game = find_game_in_kalshi(kalshi_markets, home_team, away_team)
    kalshi_yes_price = kalshi_game['yes_ask'] / 100  # convert cents to decimal
    
    # 4. Check situational factors
    schedule = get_team_schedule(away_team)
    b2b_info = detect_b2b(schedule, game_date)
    rest_days_home = calculate_rest_days(get_team_schedule(home_team), game_date)
    rest_days_away = calculate_rest_days(schedule, game_date)
    
    # 5. Check injuries
    injuries = get_current_injuries()
    home_injuries = injuries[injuries['Team'] == home_team]
    away_injuries = injuries[injuries['Team'] == away_team]
    
    # 6. Compute model probability
    model_prob = compute_model_probability(
        pinnacle_prob, consensus_prob,
        b2b_info, rest_days_home, rest_days_away,
        home_injuries, away_injuries
    )
    
    # 7. Calculate edge
    edge = model_prob - kalshi_yes_price
    
    # 8. Generate recommendation
    if abs(edge) >= 0.05:  # minimum 5¢ edge
        side = "YES" if edge > 0 else "NO"
        entry_price = kalshi_yes_price if side == "YES" else (1 - kalshi_yes_price)
        
        # Kelly sizing
        kelly_pct = calculate_kelly(
            model_prob if side == "YES" else (1 - model_prob),
            entry_price
        )
        position_size = kelly_pct * 0.25  # quarter-Kelly
        
        return {
            "action": "TRADE",
            "side": side,
            "ticker": kalshi_game['ticker'],
            "entry_price": entry_price,
            "model_probability": model_prob,
            "edge": abs(edge),
            "position_pct": min(position_size, 0.03),  # cap at 3%
            "edge_sources": compile_edge_sources(b2b_info, injuries, pinnacle_prob, kalshi_yes_price),
            "confluence_score": count_aligned_factors(b2b_info, rest_days_home, rest_days_away, injuries),
            "confidence": "HIGH" if abs(edge) >= 0.08 else "MODERATE"
        }
    
    return {"action": "PASS", "edge": edge, "reason": "Insufficient edge"}
```

---

## 8. DATA REFRESH SCHEDULE

| Data Type | Refresh Frequency | Method |
|-----------|-------------------|--------|
| Kalshi market prices | Real-time (WebSocket) or every 60s (REST) | WebSocket preferred |
| Sportsbook odds | Every 5 minutes during active hours | The Odds API polling |
| Injury reports | Every 15 minutes; continuous near deadlines | nbainjuries + NBA.com scrape |
| Team ratings/stats | Daily (morning) | nba_api batch pull |
| Player game logs | Daily (morning) | nba_api batch pull |
| Schedule/B2B data | Daily (morning) | nba_api batch pull |
| Referee assignments | Once per day (morning of game day) | NBA.com scrape |

### Staleness Rules

- **Kalshi prices**: stale after 120 seconds — never trade on stale data
- **Sportsbook odds**: stale after 10 minutes — re-fetch before any trade decision
- **Injury data**: stale after 30 minutes during game days; immediately stale near report deadlines
- **Stats/ratings**: acceptable for 24 hours (they don't change mid-day)

---

## 9. BACKTESTING FRAMEWORK

### Essential Backtesting Principles

1. **Use closing lines, not opening lines** — your model should beat the closing line
2. **Include Kalshi fees** in all P&L calculations
3. **Don't look ahead** — only use data available at the time of the simulated trade
4. **Test on out-of-sample data** — at minimum, hold out the most recent 20% of games
5. **Track CLV, not just win rate** — CLV is the true measure of edge

### Minimum Sample Sizes for Confidence

| Claim | Minimum Sample |
|-------|---------------|
| "My model has an edge" | 500+ trades |
| "This situational angle works" | 100+ instances |
| "This player prop model is profitable" | 200+ trades per prop type |
| "B2B fading is +EV" | 50+ B2B games per season |

Anything less is likely noise. Be honest about sample size limitations.
