"""Build script: generates docs/agents.json and docs/flow.json from ANALYST_CONFIG and copies the logo."""

import json
from pathlib import Path
import shutil

from src.utils.analysts import ANALYST_CONFIG, STRATEGY_GROUP_INFO, get_strategy_groups

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
ASSETS_SRC = REPO_ROOT / "assets" / "logo-detailed.jpeg"
ASSETS_DST = DOCS_DIR / "assets" / "logo.jpeg"

GROUP_ORDER = [
    "deep_value",
    "quality_compounders",
    "growth_and_catalyst",
    "macro_and_cycle",
    "quant_systematic",
    "sentiment_and_analytical",
]

# Keep in sync with --c-* CSS vars in docs/style.css
GROUP_COLORS: dict[str, str] = {
    "deep_value": "#4f6ef7",
    "quality_compounders": "#059669",
    "growth_and_catalyst": "#d97706",
    "macro_and_cycle": "#7c3aed",
    "quant_systematic": "#0891b2",
    "sentiment_and_analytical": "#e11d48",
}


def _initials(display_name: str) -> str:
    words = display_name.split()
    if len(words) == 1:
        return words[0][:2].upper()
    return "".join(w[0].upper() for w in words if w[0].isalpha())[:3]


def build() -> None:
    groups_map = get_strategy_groups()

    groups = []
    for group_key in GROUP_ORDER:
        info = STRATEGY_GROUP_INFO[group_key]
        agent_keys = groups_map.get(group_key, [])
        agents = sorted(
            [
                {
                    "key": key,
                    "display_name": ANALYST_CONFIG[key]["display_name"],
                    "description": ANALYST_CONFIG[key]["description"],
                    "investing_style": ANALYST_CONFIG[key]["investing_style"],
                    "pull_quote": ANALYST_CONFIG[key].get("pull_quote", ""),
                    "order": ANALYST_CONFIG[key]["order"],
                    "initials": _initials(ANALYST_CONFIG[key]["display_name"]),
                }
                for key in agent_keys
            ],
            key=lambda a: a["order"],
        )
        groups.append(
            {
                "key": group_key,
                "display_name": info["display_name"],
                "description": info["description"],
                "agents": agents,
            }
        )

    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "assets").mkdir(exist_ok=True)

    out_path = DOCS_DIR / "agents.json"
    out_path.write_text(json.dumps({"groups": groups}, indent=2, ensure_ascii=False))
    print(f"Written {out_path}")

    if ASSETS_SRC.exists():
        shutil.copy2(ASSETS_SRC, ASSETS_DST)
        print(f"Copied logo → {ASSETS_DST}")
    else:
        print(f"Warning: logo not found at {ASSETS_SRC}")

    build_flow()


def build_flow() -> None:
    groups_map = get_strategy_groups()

    groups = [
        {
            "key": group_key,
            "display_name": STRATEGY_GROUP_INFO[group_key]["display_name"],
            "description": STRATEGY_GROUP_INFO[group_key]["description"],
            "count": len(groups_map.get(group_key, [])),
            "color": GROUP_COLORS[group_key],
        }
        for group_key in GROUP_ORDER
    ]

    flow = {
        "stages": {
            "input": {"label": "Tickers + Portfolio"},
            "preflight": {"label": "Preflight", "note": "Regime classifier + conviction weights"},
            "debate": {"label": "Debate Node", "note": "LLM moderates contested tickers"},
            "risk": {"label": "Risk Management", "note": "Volatility & correlation limits"},
            "pm": {"label": "Portfolio Manager", "note": "Final action per ticker"},
            "decision": {"label": "Trading Decision"},
        },
        "groups": groups,
    }

    out_path = DOCS_DIR / "flow.json"
    out_path.write_text(json.dumps(flow, indent=2, ensure_ascii=False))
    print(f"Written {out_path}")


if __name__ == "__main__":
    build()
