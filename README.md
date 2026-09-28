# BTC/USDT Trading Bot — Setup & Usage Guide

This guide walks you through the whole process, start to finish, in the
order you'd actually do it. No prior assumption that you remember how any
of this works — just follow the steps in order.

---

## What this actually is

A program that watches the price of Bitcoin (in USDT) and decides when to
buy and sell, based on a set of rules (the "strategy"). **This project is
demo-only — it is not able to place real-money trades.** It can run in
these ways:

| Mode | What happens | Real money involved? |
|---|---|---|
| **Backtest** | Tests the strategy against *past* price data, instantly, to see how it would have done | No — it's just math on old data |
| **Paper trading** | Watches *real, live* prices right now and pretends to trade, on paper | No — completely simulated, no exchange account touched at all |
| **Demo** | Places real orders against Binance's Spot Demo Mode | No — simulated funds, but a genuine order execution flow |
| **Testnet** | Places real orders against Binance's Spot Testnet | No — simulated funds, independent/synthetic price data |

There is no "live" mode in this build — setting `TRADING_MODE=live` in
`.env` causes the app to refuse to start, on purpose. This project is
scoped to demonstration only.

**For your exhibition, you almost certainly want Backtest (and optionally
Paper or Demo trading).**

---

## Before you start

You need:
1. **Python installed** (3.9 or newer). To check, open a terminal and type:
   ```bash
   python3 --version
   ```
   If that gives an error, install Python from [python.org](https://python.org) first.
2. **This folder**, unzipped, somewhere on your computer.
3. A terminal (Command Prompt/PowerShell on Windows, Terminal on Mac) opened
   **inside this folder**. If you're not sure how: on Mac, right-click the
   folder → "New Terminal at Folder". On Windows, open the folder in File
   Explorer, click the address bar, type `cmd`, press Enter.

Every command below assumes you're sitting in this folder in your terminal.

---

## Step 1 — Install the required packages

Copy and paste this into your terminal:

```bash
pip install -r requirements.txt
```

This downloads all the libraries the bot needs (pandas, numpy, the Binance
connector, etc). It only needs to be done once.

**If `pip` doesn't work**, try `pip3` instead:
```bash
pip3 install -r requirements.txt
```

---

## Step 2 — Set up your settings file

Copy and paste:

```bash
cp .env.example .env
```

(On Windows, use `copy .env.example .env` instead.)

This creates a file called `.env` where all the settings live (which coin
to trade, how the bot connects to Binance, etc). **You don't need to edit
anything in it yet** — the defaults are already set to the safest option
(`paper` mode, no real money, no API keys needed).

---

## Step 3 — Get some price data to test with

You have two options. Option A is for a real result; Option B is a fallback
if you don't have internet access when you need to test.

### Option A — Real Bitcoin price history (recommended)

This downloads real historical BTC/USDT prices from Binance. Needs internet.

```bash
python3 fetch_historical_data.py --days 365 --out btc_data.csv
```

This takes a minute or two and creates a file called `btc_data.csv` with a
year of real price data.

### Option B — Use the sample data that's already included

If you don't have internet right now, there's already a ready-made data
file at `sample_data/sample_btcusdt_15m.csv`. This is **fake, computer-generated
data** — useful for checking the bot runs correctly, but not real Bitcoin
prices, so don't use it to judge how well the strategy actually performs.

---

## Step 4 — Run your first backtest

This is the main thing you'll demo. It runs the strategy against your price
data and tells you how it would have performed — profit, win rate, biggest
loss, etc.

**If you used real data (Option A):**
```bash
python3 main.py backtest --data btc_data.csv --plot
```

**If you used the sample data (Option B):**
```bash
python3 main.py backtest --data sample_data/sample_btcusdt_15m.csv --relaxed-demo --plot
```

> **Why `--relaxed-demo` only for the sample data?** The real strategy is
> very picky on purpose — it only trades when ~9 different conditions line
> up, which happens naturally on real Bitcoin data but rarely on the fake
> sample data. `--relaxed-demo` loosens those conditions just so the fake
> data actually produces some trades to look at. **Never use this flag with
> your real data** — it would give you a misleadingly rosy result.

**What you'll see:** a full report printed to your screen, and two new files:
- `backtest_report.txt` — the same report, saved as a text file
- `backtest_report_equity.png` — a chart of profit/loss over time (only if you used `--plot`)

---

## Step 5 — Understand the report

Here's what the key numbers mean:

- **ROI** — total return, as a percentage of your starting money
- **Win Rate** — percentage of trades that made money
- **Sharpe Ratio** — a measure of return vs. risk (higher is better; above 1 is generally considered good)
- **Max Drawdown** — the worst peak-to-trough drop in your account value (lower is better)
- **Profit Factor** — total money won ÷ total money lost (above 1 means profitable overall)

These are exactly the kind of numbers you'd put on a slide for your exhibition.

---

## Step 6 (optional) — Run it "live" in paper mode

This makes the bot watch **real, current** Bitcoin prices and make simulated
trading decisions in real time — good for a live demo where you want to show
it actually "thinking," without any money or API keys involved.

```bash
python3 main.py live
```

Leave this running and watch the terminal — it will print what it's doing
roughly once a minute (checking price, deciding, occasionally opening or
closing a simulated position). Press `Ctrl+C` to stop it at any time.

This uses `TRADING_MODE=paper` from your `.env` file, which is already the
default — you don't need to change anything.

---

## Going further: Demo and Testnet modes

You almost certainly don't need these for your exhibition, but if you want
to show a *real order* actually being placed (with simulated money — no
real financial risk either way), there are two options:

### Demo Mode (recommended of the two)

Demo Mode uses realistic, live market data and the exact same rules as
Binance's real exchange — the more representative option of the two.

1. Log into your real Binance.com account, then open **Binance Demo Trading**
2. Go to API Key Management directly at [demo.binance.com/en/my/settings/api-management](https://demo.binance.com/en/my/settings/api-management)
3. Generate an HMAC key and copy the Secret Key immediately (shown once)
4. Open `.env` in any text editor and fill in:
   ```
   BINANCE_API_KEY=your_key_here
   BINANCE_API_SECRET=your_secret_here
   TRADING_MODE=demo
   ```
5. Run `python3 main.py live` again — you'll see a `MODE: DEMO` banner at startup

### Testnet

Testnet uses its own separate, independent price data (not tied to real
BTC movement) — useful mainly for testing against upcoming API features,
less useful for a demo that's meant to show the bot reacting to the market.

1. Get free testnet API keys at [testnet.binance.vision](https://testnet.binance.vision) (log in with GitHub)
2. Fill in `.env` the same way as above, but with `TRADING_MODE=testnet`
3. Run `python3 main.py live` again — you'll see a `MODE: TESTNET` banner at startup

### About "live" mode

There isn't one, on purpose. This project is scoped to demonstration only —
setting `TRADING_MODE=live` in `.env` makes the app refuse to start, with
an explanation of why, rather than doing anything. No code path in this
codebase is able to place a real-money order, regardless of what's in `.env`.

If you ever extend this project beyond the scope of this assignment to
support real trading, standard practice is to only ever give an API key
permission to trade — never permission to withdraw funds.

---

## Troubleshooting

**"pip: command not found"** → try `pip3` instead of `pip`, or `python3 -m pip install -r requirements.txt`

**"No module named pandas" (or similar)** → Step 1 didn't complete successfully — run `pip install -r requirements.txt` again and check for red error text

**Backtest shows 0 trades** → if you're using real data, this can genuinely happen (the strategy is picky and won't force a trade if conditions aren't right) — try a longer date range with `--days 730`. If you're using the sample data, make sure you added `--relaxed-demo`

**"Configuration problem: TRADING_MODE=live is disabled..."** → this is expected and deliberate, not a bug — see "About live mode" above. Set `TRADING_MODE` to `paper`, `demo`, or `testnet` instead.

**The bot crashes with a Binance connection error in `demo` or `testnet` mode** → check your internet connection, and double check your API keys are pasted correctly into `.env` with no extra spaces

**Something else looks broken** → run the built-in tests, which check every part of the bot without needing internet:
```bash
python3 tests/test_orchestration.py
python3 tests/test_exchange_modes.py
```
If both print "ALL ... PASSED" at the end, the bot's logic — including the paper/demo/testnet mode routing and safety checks — is working correctly, and the issue is likely something environmental (internet, API keys, Python version).

---

## What's in this folder (quick reference)

| File | What it does |
|---|---|
| `main.py` | The command you actually run — `backtest` or `live` |
| `strategy.py` | The trading rules (when to buy/sell, position size, stop-loss) |
| `bot.py` | Runs the strategy live against Binance |
| `backtest.py` | Runs the strategy against historical data |
| `exchange.py` | Talks to Binance's servers — paper/demo/testnet only, refuses live |
| `risk_manager.py` | Keeps track of account balance, wins/losses |
| `logger.py` | Writes what's happening to the screen and log files |
| `performance_analytics.py` | Builds the performance report |
| `fetch_historical_data.py` | Downloads real price history |
| `config.py` | Reads your settings from `.env`, refuses to start in live mode |
| `.env` | Your settings (you created this in Step 2) — never commit this |
| `.gitignore` | Keeps `.env` and other secrets/generated files out of Git |
| `requirements.txt` | List of libraries to install |
| `sample_data/` | Fake practice data (see Step 3, Option B) |
| `tests/` | Scripts that check everything still works, including the mode safety checks |
