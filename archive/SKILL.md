---
name: nba-sharp-betting
description: >
  Comprehensive NBA sharp betting knowledge for clawdbot's Kalshi sportsbetting interface.
  Use this skill whenever clawdbot needs to evaluate an NBA game, assess a Kalshi NBA market,
  identify edges, size positions, or make any NBA betting decision. Trigger on ANY mention of
  NBA betting, basketball markets, game analysis, player props, moneylines, spreads, totals,
  point spreads, injury impact, schedule spots, rest advantages, back-to-backs, line movement,
  sharp action, closing line value, Kalshi NBA markets, or any request to evaluate/trade NBA
  events on Kalshi. Also trigger when building or refining NBA trading strategies, backtesting
  NBA systems, or comparing sportsbook odds against Kalshi prices. NOTE: clawdbot trades
  game-day markets ONLY (moneylines, spreads, props, combos). NEVER trade futures markets.
---

# NBA Sharp Betting — clawdbot Knowledge Base

This skill contains everything clawdbot needs to operate as a sharp NBA bettor on Kalshi.
It covers market structure, edge identification, data sources, situational analysis,
position sizing, and operational discipline.

**Before evaluating any NBA market, read this file completely.**
**For deep-dive reference material, read the files in `references/`:**

| File | When to Read |
|------|-------------|
| `references/situational-edges.md` | When evaluating a specific game or schedule spot |
| `references/kalshi-mechanics.md` | When placing orders, sizing positions, or managing Kalshi-specific execution |
| `references/data-pipeline.md` | When fetching odds, stats, injuries, or building models |

---

## 1. SHARP BETTING PHILOSOPHY

Sharp betting is not about picking winners. It is about finding **mispriced contracts** —
situations where the true probability of an outcome differs meaningfully from the price
the market is offering. Everything flows from this principle.

### The Core Equation

```
Edge = True Probability − Implied Probability (from market price)
```

On Kalshi, a contract priced at 65¢ implies a 65% probability. If your model says the
true probability is 73%, you have an 8-cent edge. That's a trade. If your model says 66%,
that's noise — pass.

### What Separates Sharps from the Public

| Trait | Public Bettor | Sharp Bettor |
|-------|--------------|--------------|
| **Decision basis** | Gut, fandom, narratives | Models, data, situational analysis |
| **Timing** | Bets when convenient | Bets when lines are soft (openers, late injury news) |
| **Sizing** | Flat or emotional | Kelly Criterion or fractional Kelly |
| **Tracking** | Remembers wins, forgets losses | Logs every trade, reviews CLV |
| **Edge source** | "I know basketball" | Information asymmetry, speed, better models |
| **Losing streaks** | Tilts, chases, increases size | Stays disciplined, trusts process |
| **Win rate target** | Wants 70%+ | Happy with 53-57% at the right prices |

### The Golden Metric: Closing Line Value (CLV)

CLV is the single most important metric for evaluating a sharp bettor's skill over time.
It measures whether you consistently bet at prices better than where the market closes.

```
CLV = Your Entry Price − Closing Price
```

If you buy YES at 62¢ and the market closes at 67¢, you captured 5¢ of CLV. Over hundreds
of bets, positive CLV means you are finding real edges — even if short-term results are noisy.

**clawdbot should log CLV on every single trade.**

---

## 2. KALSHI NBA MARKET TYPES

As of late 2025, Kalshi offers several NBA market categories:

### Game Markets (Moneyline)
- Binary: "Will [Team] win [Game]?" → YES/NO
- Contract settles at $1.00 if correct, $0.00 if wrong
- Prices in cents represent implied probability
- Available for regular season, play-in, and playoff games

### Player Props (Launched Nov 2025)
- Points, rebounds, assists, three-pointers over/unders
- Limited to ~50 players at launch
- $10,000 per-trade limit per user
- Lower liquidity than game markets — wider bid/ask gaps, more edge opportunity

### Spread Markets
- "Will [Team] win by X or more points?" — YES/NO binary contracts
- Equivalent to traditional point spread betting
- Use sportsbook spread consensus (Pinnacle) as the baseline to compare against Kalshi prices
- Edges found the same way as moneylines: model implied probability vs. Kalshi price

**Kalshi Spread Symmetry Rule:** Kalshi lists spreads from one team's perspective, but the
opposing team's contract is the mirror. A "Team A +4" YES contract is equivalent to
"Team B -4" NO contract. Concretely:

- "Lakers +4" YES at 55¢ = "Celtics -4" NO at 55¢
- If you think the Lakers will keep it within 4, you can EITHER buy "Lakers +4" YES
  OR sell "Celtics -4" YES (buy NO) — same position, same payout
- **Always check both sides of the orderbook.** One side may have better liquidity or
  a tighter spread. If "Lakers +4" YES has a 5¢ wide spread but "Celtics -4" NO has
  a 2¢ spread, trade the Celtics side for better execution.
- When computing edge, make sure you're not double-counting: +4 for Team A and -4 for
  Team B are the SAME market, not two separate opportunities.

### Combo Markets (Parlays)
- Kalshi introduced "combos" in 2025 mimicking traditional parlays
- Correlation between legs is often mispriced — this is where sharp money looks

### ⛔ FUTURES MARKETS — DO NOT TRADE
- Championship, conference, division winners, MVP, awards, season win totals
- clawdbot must NEVER trade futures markets regardless of perceived edge
- Reasons: capital locked for months, model uncertainty too high over long horizons,
  opportunity cost of tied-up bankroll, and futures edges are nearly impossible to validate via CLV
- If a user asks about futures, acknowledge the market exists but decline to trade it

### Key Difference from Sportsbooks
- **You're trading against other users, not the house** — Kalshi is an exchange
- No vig baked into the line — the spread between YES bid and YES ask IS the cost
- You can buy AND sell at any time — exit positions before settlement
- Prices move based on supply/demand, not bookmaker risk management
- Fees are typically <2% per contract

---

## 3. THE SHARP EVALUATION FRAMEWORK

For every NBA game or market, clawdbot should run through this checklist:

### Step 1: Establish True Probability
Use multiple inputs weighted by reliability:

1. **Sharp sportsbook consensus** (Pinnacle, Circa, Bookmaker.eu) — most reliable baseline
2. **Power ratings model** — team strength adjusted for injuries, rest, venue
3. **Market-implied probability** from Kalshi — what the crowd thinks
4. **Situational factors** — see `references/situational-edges.md`

### Step 2: Identify the Edge
```
Edge = Model Probability − Kalshi Implied Probability
```

**Minimum edge thresholds for clawdbot:**
- Game moneylines: ≥ 5¢ edge (model says 70%, Kalshi says ≤65¢)
- Spreads: ≥ 5¢ edge (same logic as moneylines, binary contract)
- Player props: ≥ 8¢ edge (less liquid, need wider margin)
- Combos/parlays: ≥ 12¢ edge (compounding error risk)
- Futures: DO NOT TRADE — no exceptions

### Step 3: Validate the Edge Source
Ask: **WHY does this edge exist?**

Legitimate edge sources:
- Injury news not yet priced in (key player ruled out 90 min before tip)
- Schedule spot the market underweights (3rd game in 4 nights, cross-country travel)
- Rest disparity (one team on 3 days rest vs. opponent on back-to-back)
- Public bias toward big-market teams or nationally televised favorites
- Sportsbook line movement not yet reflected on Kalshi (lag arbitrage)
- Player prop inefficiency (Kalshi limited to ~50 players, less price discovery)

Suspicious "edges" — proceed with caution:
- "This team always covers at home" (small sample, likely regression)
- Model disagrees with every sharp book (your model is probably wrong)
- Edge only exists on one obscure metric (overfitting)

### Step 4: Size the Position
See Kelly Criterion section below. Never risk more than 3% of bankroll on a single game.

### Step 5: Execute and Log
- Place order on Kalshi (limit orders preferred over market orders)
- Log: market ticker, entry price, model probability, edge size, bankroll %, timestamp
- After settlement: log result, P&L, CLV

---

## 4. POSITION SIZING — KELLY CRITERION

The Kelly Criterion determines optimal bet size given your edge and the odds:

```
Kelly % = (bp − q) / b

Where:
  b = decimal odds − 1 (on Kalshi: (1 / price) − 1)
  p = your estimated true probability
  q = 1 − p
```

**Example:** Kalshi YES price is 60¢. Your model says true probability is 68%.
```
b = (1 / 0.60) − 1 = 0.667
p = 0.68
q = 0.32
Kelly % = (0.667 × 0.68 − 0.32) / 0.667 = (0.4536 − 0.32) / 0.667 = 20.0%
```

### CRITICAL: Use Fractional Kelly

Full Kelly is mathematically optimal but assumes perfect probability estimates.
**clawdbot should use quarter-Kelly (0.25×) to half-Kelly (0.5×) for all NBA bets.**

This accounts for model uncertainty and avoids ruin from inevitable estimation errors.

### Hard Limits for clawdbot

| Rule | Limit |
|------|-------|
| Max single game exposure | 3% of bankroll |
| Max single prop exposure | 2% of bankroll |
| Max daily exposure (all games) | 10% of bankroll |
| Max correlated exposure | 5% across related markets |
| Min edge to trade | 5¢ (moneylines), 8¢ (props) |

---

## 5. TIMING AND EXECUTION

### The Two Optimal Windows

**Window 1: Line Openers (24-48 hours before tip)**
- Lines are softest before sharp money arrives
- Best for strong model-based edges where you have high conviction
- Kalshi prices often lag sportsbook openers by hours
- This is where the Kalshi-vs-sportsbook lag arbitrage lives

**Window 2: Late News (60-90 minutes before tip)**
- Injury reports finalize (5 PM local day before; 1 PM for B2Bs)
- "Questionable" becomes "Out" — lines move 2-4 points
- Kalshi repricing lags sportsbooks by minutes to hours on injury news
- **This is clawdbot's biggest structural advantage** — monitor injury feeds,
  compare implied probability shift vs. Kalshi current price, trade the gap

**Dead Zone: 12-4 hours before tip**
- Sharps have already moved lines, but late injury news hasn't broken
- Worst prices available — avoid trading in this window

### Execution Rules for Kalshi

1. **Use limit orders, not market orders** — the spread on Kalshi is your cost
2. **Check orderbook depth** — thin books mean slippage, wide spreads mean less edge
3. **Consider selling NO instead of buying YES** (or vice versa) if the other side has better liquidity
4. **Monitor your position for exit opportunities** — if the line moves in your favor before
   settlement, you can sell for profit without waiting for the game result

---

## 6. THE PUBLIC BIAS PLAYBOOK

The NBA betting public consistently overvalues certain things. These biases create
systematic edges for sharp bettors:

### Biases to Exploit

| Public Bias | Sharp Counter |
|------------|---------------|
| Overbet favorites, especially big-market teams | Look for value on underdogs, especially in nationally televised games |
| Overbet overs (fans want high-scoring games) | Unders are systematically underbet |
| Recency bias (team on a hot streak gets inflated odds) | Regression to the mean — hot streaks end |
| Overvalue star player names regardless of matchup | Evaluate actual lineup + matchup context |
| Ignore schedule and rest factors | Rest/travel is one of the most reliable edges |
| Chase losses on late-night West Coast games | Late games have thinner markets and more mispricing |
| Overreact to a single blowout result | Sample size matters — one game is noise |

### Reverse Line Movement (RLM)

When 70%+ of bets are on one side but the line moves the OTHER direction, it signals
sharp money is on the unpopular side. On Kalshi, this manifests as:

- Public is buying YES heavily (volume is high on YES side)
- But YES price is dropping (or not rising as expected)
- This means large-size traders are selling YES / buying NO
- Follow the money direction, not the ticket count

---

## 7. CROSS-PLATFORM ARBITRAGE (SPORTSBOOK ↔ KALSHI)

This is clawdbot's most reliable edge source. Sportsbooks and Kalshi price the same
events independently, and repricing lag creates windows.

### How It Works

1. Monitor sharp sportsbook lines (Pinnacle, Circa) for NBA games
2. Convert sportsbook odds to implied probability
3. Compare against Kalshi YES/NO prices for the same game
4. If sportsbook says Team A is 72% to win but Kalshi YES is 65¢ → buy YES
5. If sportsbook says Team A is 60% but Kalshi YES is 68¢ → buy NO (or sell YES)

### Why the Lag Exists

- Sportsbooks have dedicated NBA odds-making teams and real-time adjustment
- Kalshi is user-driven — prices only move when users trade
- Injury news hits sportsbooks first (they have direct team contacts)
- Sharp money hits sportsbooks first (higher limits, established accounts)
- Kalshi reprices reactively, not proactively

### Required Data Sources

- The Odds API (theoddsapi.com) — aggregates real-time odds from 20+ books
- Pinnacle odds (the sharpest line in the market)
- Kalshi API — real-time YES/NO prices and orderbook depth

See `references/data-pipeline.md` for full API integration details.

---

## 8. KEY NBA METRICS FOR MODELING

### Team-Level

| Metric | What It Tells You | Source |
|--------|-------------------|--------|
| Net Rating | Points scored minus allowed per 100 possessions | NBA.com/stats |
| Offensive Rating (ORtg) | Points per 100 possessions on offense | NBA.com/stats |
| Defensive Rating (DRtg) | Points allowed per 100 possessions | NBA.com/stats |
| Pace | Possessions per 48 minutes | NBA.com/stats |
| SRS (Simple Rating System) | Point differential + strength of schedule | Basketball Reference |
| Home/Away splits | Performance delta by venue | NBA.com/stats |
| Last 10 games trend | Recent form vs. season baseline | NBA.com/stats |
| ATS record by situation | Performance vs. spread in specific scheduling spots | Historical tracking |

### Player-Level (for Props)

| Metric | What It Tells You |
|--------|-------------------|
| Usage Rate | % of team possessions used by player when on court |
| Minutes per game | Baseline for counting stat projections |
| Per-36 stats | Normalized production independent of minutes |
| On/Off splits | Team performance with/without the player |
| Matchup history | Performance vs. specific defenders or teams |
| B2B impact | How much production drops on second of back-to-back |
| Home/Away splits | Some players perform significantly differently by venue |

### The Injury Impact Hierarchy

Not all injuries are equal. Quantify impact by player's on/off differential:

```
Injury Impact = Team Net Rating (with player) − Team Net Rating (without player)
```

**Tier 1: Market-moving absences (3-6 point swing)**
- Top-10 players (franchise cornerstones)
- Elite two-way players who anchor both offense and defense

**Tier 2: Significant impact (1.5-3 point swing)**
- Starting-caliber players with limited replacement depth
- Key role players in specific matchups (e.g., only rim protector)

**Tier 3: Marginal impact (<1.5 points)**
- Bench players with capable replacements
- Players already in a reduced role due to rotation changes

---

## 9. BANKROLL MANAGEMENT

### The Foundation

- **Starting bankroll**: Whatever amount you deposit to Kalshi for NBA trading
- **Unit size**: 1% of current bankroll (recalculate weekly)
- **Bet size range**: 0.5 to 3 units depending on edge and confidence
- **Never chase losses** — stick to the system regardless of recent results

### Drawdown Protocol

| Drawdown Level | Action |
|---------------|--------|
| -5% of bankroll | Review recent trades for process errors. Continue if process is clean. |
| -10% of bankroll | Reduce unit size to 0.5%. Review model calibration. |
| -15% of bankroll | Pause all trading for 48 hours. Full model audit. |
| -20% of bankroll | Stop trading. Comprehensive system review before resuming. |

### Record-Keeping Requirements

clawdbot must log for every trade:
```
{
  "timestamp": "2026-02-28T19:30:00Z",
  "market_ticker": "KXNBA-26FEB28-LAL-BOS",
  "side": "YES",
  "entry_price": 0.62,
  "contracts": 50,
  "model_probability": 0.70,
  "edge": 0.08,
  "kelly_fraction": 0.25,
  "bankroll_pct": 0.02,
  "edge_source": "B2B fatigue + sportsbook lag",
  "sportsbook_implied": 0.71,
  "result": null,
  "pnl": null,
  "clv": null,
  "notes": "LAL on 2nd of B2B, traveled from DEN. Pinnacle moved to -220 BOS."
}
```

---

## 10. OPERATIONAL CHECKLIST — DAILY WORKFLOW

### Pre-Market (Morning)

1. Pull today's NBA schedule — identify all games
2. Check injury reports (5 PM previous day reports are now final)
3. Run power ratings model with current injury context
4. Identify schedule spots (B2Bs, rest advantages, travel) — see `references/situational-edges.md`
5. Pull sportsbook consensus lines from The Odds API
6. Pull Kalshi prices for all available NBA markets
7. Flag any game where model-vs-Kalshi edge ≥ 5¢

### Execution (Afternoon)

8. For flagged games: validate edge source, confirm it's not noise
9. Size positions using quarter-Kelly
10. Place limit orders on Kalshi
11. Set alerts for injury report updates (1 PM for B2B games, ongoing throughout day)

### Late Window (60-90 min before tip)

12. Monitor final injury updates
13. Compare sportsbook line movement to Kalshi prices
14. Execute any new edges created by late-breaking news
15. Adjust or exit existing positions if new information changes the picture

### Post-Settlement (Next Morning)

16. Record all results, P&L, and CLV
17. Update running performance metrics
18. Flag any trades where the process was wrong (even if the result was right)
19. Weekly: recalculate bankroll and unit size

---

## FINAL PRINCIPLES

1. **The market is smarter than you most of the time.** Your job is to find the small percentage of situations where it isn't.
2. **Process over results.** A good bet that loses is still a good bet. A bad bet that wins is still a bad bet.
3. **Information decays fast.** An edge that exists at 2 PM may be gone by 5 PM.
4. **Small edges compound.** 54% win rate at the right prices beats 65% win rate at bad prices.
5. **The biggest edge is discipline.** Most bettors beat themselves through over-betting, chasing, and abandoning systems during drawdowns.
6. **Always be honest about uncertainty.** If your model isn't confident, don't trade. Cash is a position.
