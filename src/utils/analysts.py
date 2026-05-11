"""Constants and utilities related to analysts configuration."""

from src.agents.aswath_damodaran import aswath_damodaran_agent
from src.agents.ben_graham import ben_graham_agent
from src.agents.bill_ackman import bill_ackman_agent
from src.agents.cathie_wood import cathie_wood_agent
from src.agents.charlie_munger import charlie_munger_agent
from src.agents.cliff_asness import cliff_asness_agent
from src.agents.ed_seykota import ed_seykota_agent
from src.agents.fundamentals import fundamentals_analyst_agent
from src.agents.growth_agent import growth_analyst_agent
from src.agents.howard_marks import howard_marks_agent
from src.agents.jim_simons import jim_simons_agent
from src.agents.joel_greenblatt import joel_greenblatt_agent
from src.agents.michael_burry import michael_burry_agent
from src.agents.mohnish_pabrai import mohnish_pabrai_agent
from src.agents.nassim_taleb import nassim_taleb_agent
from src.agents.news_sentiment import news_sentiment_agent
from src.agents.peter_lynch import peter_lynch_agent
from src.agents.phil_fisher import phil_fisher_agent
from src.agents.rakesh_jhunjhunwala import rakesh_jhunjhunwala_agent
from src.agents.ray_dalio import ray_dalio_agent
from src.agents.sentiment import sentiment_analyst_agent
from src.agents.stanley_druckenmiller import stanley_druckenmiller_agent
from src.agents.technicals import technical_analyst_agent
from src.agents.valuation import valuation_analyst_agent
from src.agents.warren_buffett import warren_buffett_agent

# Define analyst configuration - single source of truth
ANALYST_CONFIG = {
    "aswath_damodaran": {
        "display_name": "Aswath Damodaran",
        "description": "The Dean of Valuation",
        "investing_style": "Focuses on intrinsic value and financial metrics to assess investment opportunities through rigorous valuation analysis.",
        "pull_quote": "Start with the company story, connect it to FCFF drivers — growth, margins, reinvestment, risk — and let the DCF decide the price you should pay.",
        "agent_func": aswath_damodaran_agent,
        "type": "analyst",
        "strategy_group": "quality_compounders",
        "order": 0,
    },
    "ben_graham": {
        "display_name": "Ben Graham",
        "description": "The Father of Value Investing",
        "investing_style": "Emphasizes a margin of safety and invests in undervalued companies with strong fundamentals through systematic value analysis.",
        "pull_quote": "Margin of safety first: I apply seven systematic rules — Graham Number, NCAV, P/E ≤ 15 — and never deviate for narrative.",
        "agent_func": ben_graham_agent,
        "type": "analyst",
        "strategy_group": "deep_value",
        "order": 1,
    },
    "bill_ackman": {
        "display_name": "Bill Ackman",
        "description": "The Activist Investor",
        "investing_style": "Seeks to influence management and unlock value through strategic activism and contrarian investment positions.",
        "pull_quote": "Identify the specific lever — board change, spin-off, buyback — that can re-rate a high-quality franchise by 50–100%.",
        "agent_func": bill_ackman_agent,
        "type": "analyst",
        "strategy_group": "growth_and_catalyst",
        "order": 2,
    },
    "cathie_wood": {
        "display_name": "Cathie Wood",
        "description": "The Queen of Growth Investing",
        "investing_style": "Focuses on disruptive innovation and growth, investing in companies that are leading technological advancements and market disruption.",
        "pull_quote": "Bet on disruptive innovation with a massive TAM; accept the volatility — exponential growth rewards patience on a multi-year horizon.",
        "agent_func": cathie_wood_agent,
        "type": "analyst",
        "strategy_group": "growth_and_catalyst",
        "order": 3,
    },
    "charlie_munger": {
        "display_name": "Charlie Munger",
        "description": "The Rational Thinker",
        "investing_style": "Advocates for value investing with a focus on quality businesses and long-term growth through rational decision-making.",
        "pull_quote": "Invert always: ask what makes it a terrible investment first, then let a lollapalooza of reinforcing moat signals earn your conviction.",
        "agent_func": charlie_munger_agent,
        "type": "analyst",
        "strategy_group": "quality_compounders",
        "order": 4,
    },
    "michael_burry": {
        "display_name": "Michael Burry",
        "description": "The Big Short Contrarian",
        "investing_style": "Makes contrarian bets, often shorting overvalued markets and investing in undervalued assets through deep fundamental analysis.",
        "pull_quote": "Hunt value with hard numbers — FCF yield, EV/EBIT, balance sheet. Be contrarian: hatred in the press is your friend when fundamentals are solid.",
        "agent_func": michael_burry_agent,
        "type": "analyst",
        "strategy_group": "deep_value",
        "order": 5,
    },
    "mohnish_pabrai": {
        "display_name": "Mohnish Pabrai",
        "description": "The Dhandho Investor",
        "investing_style": "Focuses on value investing and long-term growth through fundamental analysis and a margin of safety.",
        "pull_quote": "Heads I win, tails I don't lose much: clone great investors, demand 8%+ FCF yield, keep it simple, and wait for the price to come to you.",
        "agent_func": mohnish_pabrai_agent,
        "type": "analyst",
        "strategy_group": "deep_value",
        "order": 6,
    },
    "nassim_taleb": {
        "display_name": "Nassim Taleb",
        "description": "The Black Swan Risk Analyst",
        "investing_style": "Focuses on tail risk, antifragility, and asymmetric payoffs. Uses barbell strategy, avoids fragile companies via negativa, and seeks convex positions with limited downside and unlimited upside.",
        "pull_quote": "Seek the antifragile: convex payoffs with bounded downside, avoid the fragile, and never mistake low volatility for safety.",
        "agent_func": nassim_taleb_agent,
        "type": "analyst",
        "strategy_group": "macro_and_cycle",
        "order": 7,
    },
    "peter_lynch": {
        "display_name": "Peter Lynch",
        "description": "The 10-Bagger Investor",
        "investing_style": "Invests in companies with understandable business models and strong growth potential using the 'buy what you know' strategy.",
        "pull_quote": "Buy what you know: if the PEG is cheap and the business is simple enough to explain to a twelve-year-old, it could be a ten-bagger.",
        "agent_func": peter_lynch_agent,
        "type": "analyst",
        "strategy_group": "growth_and_catalyst",
        "order": 8,
    },
    "phil_fisher": {
        "display_name": "Phil Fisher",
        "description": "The Scuttlebutt Investor",
        "investing_style": "Emphasizes investing in companies with strong management and innovative products, focusing on long-term growth through scuttlebutt research.",
        "pull_quote": "Exceptional management + sustained R&D-led growth + consistent margins: pay a fair price for a great business and hold it for decades.",
        "agent_func": phil_fisher_agent,
        "type": "analyst",
        "strategy_group": "quality_compounders",
        "order": 9,
    },
    "rakesh_jhunjhunwala": {
        "display_name": "Rakesh Jhunjhunwala",
        "description": "The Big Bull Of India",
        "investing_style": "Leverages macroeconomic insights to invest in high-growth sectors, particularly within emerging markets and domestic opportunities.",
        "pull_quote": "Structural tailwind + earnings quality + low leverage + market leadership + contrarian timing — that is how you find multi-decade compounders.",
        "agent_func": rakesh_jhunjhunwala_agent,
        "type": "analyst",
        "strategy_group": "macro_and_cycle",
        "order": 10,
    },
    "stanley_druckenmiller": {
        "display_name": "Stanley Druckenmiller",
        "description": "The Macro Investor",
        "investing_style": "Focuses on macroeconomic trends, making large bets on currencies, commodities, and interest rates through top-down analysis.",
        "pull_quote": "Macro regime first: liquidity and rates determine the tide; when the thesis is clear, size very large and cut the moment it changes.",
        "agent_func": stanley_druckenmiller_agent,
        "type": "analyst",
        "strategy_group": "macro_and_cycle",
        "order": 11,
    },
    "warren_buffett": {
        "display_name": "Warren Buffett",
        "description": "The Oracle of Omaha",
        "investing_style": "Seeks companies with strong fundamentals and competitive advantages through value investing and long-term ownership.",
        "pull_quote": "Buy wonderful businesses with durable moats and exceptional capital allocation at a price that gives a margin of safety, then never sell.",
        "agent_func": warren_buffett_agent,
        "type": "analyst",
        "strategy_group": "quality_compounders",
        "order": 12,
    },
    "technical_analyst": {
        "display_name": "Technical Analyst",
        "description": "Chart Pattern Specialist",
        "investing_style": "Focuses on chart patterns and market trends to make investment decisions, often using technical indicators and price action analysis.",
        "pull_quote": "Trend, momentum, mean-reversion, and volatility signals combined by weight — the tape tells the truth when charts are read without emotion.",
        "agent_func": technical_analyst_agent,
        "type": "analyst",
        "strategy_group": "quant_systematic",
        "order": 13,
    },
    "fundamentals_analyst": {
        "display_name": "Fundamentals Analyst",
        "description": "Financial Statement Specialist",
        "investing_style": "Delves into financial statements and economic indicators to assess the intrinsic value of companies through fundamental analysis.",
        "pull_quote": "Score profitability, growth, financial health, and efficiency systematically — when three of four dimensions align bullish, the signal is robust.",
        "agent_func": fundamentals_analyst_agent,
        "type": "analyst",
        "strategy_group": "sentiment_and_analytical",
        "order": 14,
    },
    "growth_analyst": {
        "display_name": "Growth Analyst",
        "description": "Growth Specialist",
        "investing_style": "Analyzes growth trends and valuation to identify growth opportunities through growth analysis.",
        "pull_quote": "Follow revenue acceleration and expanding margins — when growth is durable and valuation is still reasonable, the compounding does the work.",
        "agent_func": growth_analyst_agent,
        "type": "analyst",
        "strategy_group": "growth_and_catalyst",
        "order": 15,
    },
    "news_sentiment_analyst": {
        "display_name": "News Sentiment Analyst",
        "description": "News Sentiment Specialist",
        "investing_style": "Analyzes news sentiment to predict market movements and identify opportunities through news analysis.",
        "pull_quote": "Read the news flow quantitatively: positive sentiment breadth and rising coverage predict price momentum before it shows up in fundamentals.",
        "agent_func": news_sentiment_agent,
        "type": "analyst",
        "strategy_group": "sentiment_and_analytical",
        "order": 16,
    },
    "sentiment_analyst": {
        "display_name": "Sentiment Analyst",
        "description": "Market Sentiment Specialist",
        "investing_style": "Gauges market sentiment and investor behavior to predict market movements and identify opportunities through behavioral analysis.",
        "pull_quote": "Market psychology leaves measurable footprints — insider activity, put/call ratios, short interest — read the crowd to trade against its worst impulses.",
        "agent_func": sentiment_analyst_agent,
        "type": "analyst",
        "strategy_group": "sentiment_and_analytical",
        "order": 17,
    },
    "valuation_analyst": {
        "display_name": "Valuation Analyst",
        "description": "Company Valuation Specialist",
        "investing_style": "Specializes in determining the fair value of companies, using various valuation models and financial metrics for investment decisions.",
        "pull_quote": "Model DCF, owner earnings, EV/EBITDA, and P/E together — when multiple methods converge on undervaluation, the margin of safety is real.",
        "agent_func": valuation_analyst_agent,
        "type": "analyst",
        "strategy_group": "sentiment_and_analytical",
        "order": 18,
    },
    "ray_dalio": {
        "display_name": "Ray Dalio",
        "description": "The All-Weather Macro Investor",
        "investing_style": "Applies the debt-cycle framework, economic-regime analysis, and risk-parity principles to assess whether an equity is positioned for all-weather resilience.",
        "pull_quote": "Understand the debt cycle and economic regime — resilient balance sheets bought cheaply in early-cycle fear are the foundation of all-weather returns.",
        "agent_func": ray_dalio_agent,
        "type": "analyst",
        "strategy_group": "macro_and_cycle",
        "order": 19,
    },
    "howard_marks": {
        "display_name": "Howard Marks",
        "description": "The Credit Cycle Master",
        "investing_style": "Focuses on market cycles, second-level thinking, and credit quality. Seeks assets where fear-driven pricing offers a margin of safety and the risk premium adequately compensates.",
        "pull_quote": "Second-level thinking: ask what the price implies and what the crowd is missing — the best entries come when fear drives pricing below fair value.",
        "agent_func": howard_marks_agent,
        "type": "analyst",
        "strategy_group": "macro_and_cycle",
        "order": 20,
    },
    "cliff_asness": {
        "display_name": "Cliff Asness",
        "description": "The Quant Factor Investor",
        "investing_style": "Applies AQR's multi-factor framework: value, 12-1 momentum, quality (ROIC + margins + earnings stability), and low-volatility. Signals driven by factor-score composites, not narratives.",
        "pull_quote": "Value + momentum + quality + low-vol all aligned is the highest conviction signal; never let narrative override the factor scores.",
        "agent_func": cliff_asness_agent,
        "type": "analyst",
        "strategy_group": "quant_systematic",
        "order": 21,
    },
    "ed_seykota": {
        "display_name": "Ed Seykota",
        "description": "The Trend-Following Pioneer",
        "investing_style": "Systematic CTA: rides trends confirmed by 200-MA, Donchian breakouts, and momentum; sizes positions by ATR volatility; cuts losers mercilessly.",
        "pull_quote": "The trend is your friend until the bend: ride the Donchian breakout, size by ATR volatility, and cut losers the instant the trend reverses.",
        "agent_func": ed_seykota_agent,
        "type": "analyst",
        "strategy_group": "quant_systematic",
        "order": 22,
    },
    "joel_greenblatt": {
        "display_name": "Joel Greenblatt",
        "description": "The Magic Formula Investor",
        "investing_style": "Ranks stocks by Magic Formula: high ROIC (quality) combined with high earnings yield (cheap). Looks for special situations — spin-offs, restructurings, insider buying — as catalysts.",
        "pull_quote": "Magic Formula: rank by high ROIC plus high earnings yield, then wait for a special-situation catalyst — spin-off, insider buy — to unlock the value.",
        "agent_func": joel_greenblatt_agent,
        "type": "analyst",
        "strategy_group": "deep_value",
        "order": 23,
    },
    "jim_simons": {
        "display_name": "Jim Simons",
        "description": "The Quant / Stat-Arb Legend",
        "investing_style": "Renaissance-style statistical arbitrage: mean-reversion Z-scores, return autocorrelation, volume microstructure, and Hurst exponent regime detection. Pure data, no narrative.",
        "pull_quote": "No narrative, only data: Hurst below 0.5 means mean-reversion is our edge — Z-scores, autocorrelation, and volume microstructure decide every trade.",
        "agent_func": jim_simons_agent,
        "type": "analyst",
        "strategy_group": "quant_systematic",
        "order": 24,
    },
}

STRATEGY_GROUP_INFO: dict[str, dict[str, str]] = {
    "deep_value": {
        "display_name": "Deep Value",
        "description": "Find companies trading below intrinsic worth with a clear margin of safety, using hard numbers over narrative.",
    },
    "quality_compounders": {
        "display_name": "Quality Compounders",
        "description": "Long-term ownership of durable franchises with strong moats, exceptional management, and compounding economics.",
    },
    "growth_and_catalyst": {
        "display_name": "Growth & Catalyst",
        "description": "Disruptive innovation, activist catalysts, and outsized growth potential at a price the market hasn't yet recognised.",
    },
    "macro_and_cycle": {
        "display_name": "Macro & Cycle",
        "description": "Top-down regime analysis — debt cycles, liquidity flows, and tail-risk awareness — to time and size positions.",
    },
    "quant_systematic": {
        "display_name": "Quant & Systematic",
        "description": "Rule-based factor models, trend-following, and statistical arbitrage — data decides, narrative is noise.",
    },
    "sentiment_and_analytical": {
        "display_name": "Sentiment & Analytical",
        "description": "Bottom-up specialists in fundamentals, valuation, growth signals, and market-sentiment indicators.",
    },
}

# Derive ANALYST_ORDER from ANALYST_CONFIG for backwards compatibility
ANALYST_ORDER = [(config["display_name"], key) for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])]


def get_analyst_nodes():
    """Get the mapping of analyst keys to their (node_name, agent_func) tuples."""
    return {key: (f"{key}_agent", config["agent_func"]) for key, config in ANALYST_CONFIG.items()}


def get_agents_list():
    """Get the list of agents for API responses."""
    return [{"key": key, "display_name": config["display_name"], "description": config["description"], "investing_style": config["investing_style"], "order": config["order"]} for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])]


def get_strategy_groups() -> dict[str, list[str]]:
    """Return mapping of group_key -> list of analyst keys."""
    out: dict[str, list[str]] = {}
    for key, cfg in ANALYST_CONFIG.items():
        out.setdefault(cfg["strategy_group"], []).append(key)
    return out


def get_agent_to_group() -> dict[str, str]:
    """Return mapping of `{key}_agent` node name -> group_key for fast lookup."""
    return {f"{key}_agent": cfg["strategy_group"] for key, cfg in ANALYST_CONFIG.items()}


def get_strategy_group_info() -> dict[str, dict[str, str]]:
    """Return display metadata (display_name, description) for each strategy group."""
    return STRATEGY_GROUP_INFO
