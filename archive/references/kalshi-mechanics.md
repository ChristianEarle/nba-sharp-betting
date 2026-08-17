# Kalshi Mechanics — Execution, Order Management & Platform Specifics

Read this file when placing orders, sizing positions, managing execution, or dealing
with any Kalshi-specific trading mechanics.

---

## 1. HOW KALSHI CONTRACTS WORK

### Binary Contract Structure

Every Kalshi NBA market is a binary contract:
- **YES** settles at $1.00 if the outcome occurs
- **NO** settles at $1.00 if the outcome does NOT occur
- **YES price + NO price ≈ $1.00** (the gap is the spread / market inefficiency)
- All prices are in cents (e.g., YES at 65¢ = 65% implied probability)

### Buying YES vs. Selling NO

These are economically equivalent but may have different execution characteristics:

| Action | Cost | Max Profit | When to Prefer |
|--------|------|-----------|----------------|
| Buy YES at 65¢ | 65¢ per contract | 35¢ (settle at $1) | When YES orderbook is deeper |
| Sell NO at 35¢ | 65¢ collateral | 35¢ (NO settles at $0) | When NO orderbook has better liquidity |

**clawdbot should check BOTH sides of the book** before placing any order.
Sometimes the YES bid/ask spread is wide but the NO side is tighter, or vice versa.

### The Orderbook

Kalshi's orderbook only shows bids (not asks) due to the reciprocal nature of binary markets:
- A YES bid at 60¢ is simultaneously a NO offer at 40¢
- A NO bid at 35¢ is simultaneously a YES offer at 65¢
- **Depth matters**: thin books (few contracts at each price level) mean slippage risk

---

## 2. ORDER TYPES

### Limit Orders (PREFERRED)

- Set your desired price — order sits in the book until matched or cancelled
- No slippage — you get exactly the price you specify
- Risk: may not get filled if the market moves away from your price
- **clawdbot should default to limit orders for all trades**

### Market Orders

- Execute immediately at the best available price
- Guaranteed fill but at potentially worse prices
- Use ONLY when speed is critical (e.g., injury news just broke, need to capture edge NOW)
- Check orderbook depth first — if only 10 contracts are available at the best price,
  a 100-contract market order will blow through multiple price levels

### Order Sizing Guidelines

| Orderbook Depth at Best Price | Max Order Size (clawdbot) |
|------------------------------|--------------------------|
| <50 contracts | 10-20 contracts (limit only) |
| 50-200 contracts | Up to 50 contracts |
| 200-500 contracts | Up to 100 contracts |
| 500+ contracts | Up to Kelly-sized position |

**Never size larger than 25% of visible orderbook depth** to avoid moving the price
against yourself.

---

## 3. KALSHI API ESSENTIALS

### Base URL
```
https://api.elections.kalshi.com/trade-api/v2
```
(Despite the "elections" subdomain, this serves ALL Kalshi markets including NBA.)

### Authentication
- RSA key pair authentication
- Generate key pair, upload public key to Kalshi account
- Sign requests with private key
- Tokens expire every 30 minutes — implement auto-refresh

### Key Endpoints for NBA Trading

**Market Discovery:**
```
GET /markets?status=open&category=sports
  → Returns all open sports markets with prices, volume, orderbook summary

GET /markets/{ticker}
  → Get specific market details (price, volume, open interest, close time)

GET /events/{event_ticker}?with_nested_markets=true
  → Get event with all associated markets (e.g., a game with moneyline + props)

GET /series?category=sports
  → Browse all sports series templates
```

**Orderbook:**
```
GET /markets/{ticker}/orderbook?depth=10
  → Current bids at each price level
  → Remember: YES bids = NO offers and vice versa
```

**Trading:**
```
POST /portfolio/orders
  → Place a new order (limit or market)
  → Body: { action: "buy"/"sell", ticker, type: "limit"/"market",
            side: "yes"/"no", count: N, yes_price/no_price: cents }

GET /portfolio/orders
  → View all active orders

DELETE /portfolio/orders/{order_id}
  → Cancel an open order

GET /portfolio/positions
  → View all current positions with P&L
```

**Market Data:**
```
GET /markets/{ticker}/candlesticks
  → Historical price data for backtesting

GET /trades?ticker={ticker}
  → Recent trade history (who's trading, at what prices)
```

### WebSocket (Real-Time)
```
wss://api.elections.kalshi.com/trade-api/ws/v2
  → Subscribe to live orderbook updates, trade feed, market status changes
  → Requires authentication
  → Essential for the late-news arbitrage strategy
```

### Rate Limits
- Check `GET /account/api-limits` for your current limits
- Implement exponential backoff on 429 responses
- Cache market data locally — don't re-fetch what hasn't changed

---

## 4. KALSHI FEE STRUCTURE

### Trading Fees
- Typically <2% per contract
- Fee is assessed on the trade, not on settlement
- Example: $100 trade → fee capped at ~$1.74

### Deposit/Withdrawal
- 2% fee on debit card deposits
- $2 fee for debit card withdrawals
- ACH transfers may have different fee structures
- Crypto deposit options may be available

### Fee Impact on Edge Calculation

**Always include fees when calculating true edge:**
```
True Edge = (Model Probability × $1.00) − Entry Price − Fee
```

If your edge is 5¢ but fees eat 2¢, your true edge is only 3¢.
For small edges, fees can eliminate profitability entirely.

**Minimum viable edge after fees: ~4¢ for moneylines, ~6¢ for props**

---

## 5. POSITION MANAGEMENT

### Monitoring Open Positions

clawdbot should check positions at these intervals:
- Every 15 minutes during active trading hours
- Every 5 minutes in the 2 hours before game tip
- Real-time via WebSocket during games (if live trading)

### Exit Strategies

**Pre-Game Exit (Sell Before Settlement)**

If the market price has moved in your favor by ≥50% of your edge, consider selling:
```
Example: Bought YES at 62¢ with model probability 70¢
         Market now at 67¢ (moved 5¢ of your 8¢ edge)
         Selling locks in 5¢ profit per contract with zero game risk
```

This is especially valuable when:
- New information has reduced your confidence in the original thesis
- You want to free up capital for a higher-conviction trade
- The game hasn't started yet and you've already captured most of the expected value

**Hold to Settlement**

Default behavior when:
- Your edge thesis remains intact
- No new information has changed the picture
- You're comfortable with the binary outcome risk

**Stop-Loss via Sell**

If the market moves significantly against you AND your thesis has been invalidated
(not just price noise):
- Sell at market to limit losses
- Example: you bought YES at 65¢, key player gets ruled out, now you believe
  true probability is 55% but market is still at 58¢ → sell at 58¢, take the 7¢ loss
  rather than risking a full 65¢ loss

### Hedging Correlated Positions

If you have multiple bets on the same game (moneyline + prop), recognize the correlation:
- If you bet Team A wins AND Player A over points, both positions lose if Team A gets blown out
- Consider the net exposure, not individual position sizes
- Max correlated exposure: 5% of bankroll

---

## 6. KALSHI-SPECIFIC MARKET QUIRKS

### Player Props Limitations (as of Nov 2025)
- Limited to ~50 players
- Only points, rebounds, assists, three-pointers
- $10,000 per-trade limit
- Lower liquidity → wider spreads → more edge opportunity BUT more execution risk
- Spreads can be 5-10¢ wide on thin props (vs. 1-3¢ on game moneylines)

### Combo Markets (Parlays)
- Kalshi introduced these in 2025
- Correlation between legs is often mispriced by the platform
- Example: Team A wins AND game goes under — these are positively correlated
  if Team A's path to winning involves defense, but Kalshi may price them independently
- Sharp opportunity: find combos where the correlation premium is too low

### Market Close Times
- Game markets typically close at tip-off or shortly before
- Some markets may allow live trading during the game
- Props usually close at tip-off
- Spread markets close at tip-off (same as moneylines)
- Futures: DO NOT TRADE — clawdbot ignores these entirely

### Settlement
- Games: settled based on official NBA game result
- Spreads: settled based on official final score and the spread line
- Props: settled based on official NBA box score statistics
- Settlement can take minutes to hours after the event concludes

### Kalshi Operational Risks

Based on recent history, be aware of:
- **Deposit delays during high-traffic events** — pre-fund your account before busy nights
- **Settlement disputes** — rare but have occurred (Jan 2026 NFL incident)
  Keep screenshots of your orders as records
- **Regulatory risk** — some states (MA) have challenged Kalshi's legality
  Know your state's status

---

## 7. LIVE TRADING ON KALSHI

If clawdbot expands to live/in-game trading:

### Fourth Quarter Patterns
- Close games feature slower pace, lower efficiency
- Home court advantage amplifies in crunch time
- Free throw rate increases — relevant for player prop totals
- Defensive intensity increases — scoring efficiency drops

### Live Price Movement
- Kalshi prices during live games can overreact to runs
- A team goes on a 10-0 run → market panic-sells their opponent
- If fundamentals haven't changed, this is a buying opportunity on the opponent
- **Key principle**: in-game variance is high. A 10-point lead in the 2nd quarter
  is far less predictive than a 10-point lead in the 4th quarter.

### Latency Considerations for Live
- Kalshi's live market updates lag the actual game by seconds
- TV broadcasts lag by 5-20 seconds depending on the feed
- Stat feeds (NBA.com, ESPN) update with a few seconds delay
- If you're watching a game and see something material, Kalshi prices may take
  30-120 seconds to fully adjust
- This is a small but real edge for attentive live traders
