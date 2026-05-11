# Third-Party Notices

This project incorporates material from the projects listed below.

---

## virattt/ai-hedge-fund

**Source:** https://github.com/virattt/ai-hedge-fund  
**License:** MIT

The following files contain LLM persona prompts, orchestration patterns, or
agent code that are derived from or closely adapted from ai-hedge-fund:

- `src/agents/aswath_damodaran.py` — verbatim prompt
- `src/agents/ben_graham.py` — verbatim prompt
- `src/agents/bill_ackman.py` — verbatim prompt
- `src/agents/cathie_wood.py` — verbatim prompt
- `src/agents/charlie_munger.py` — verbatim prompt
- `src/agents/michael_burry.py` — verbatim prompt
- `src/agents/mohnish_pabrai.py` — verbatim prompt
- `src/agents/nassim_taleb.py` — verbatim prompt
- `src/agents/peter_lynch.py` — verbatim prompt
- `src/agents/phil_fisher.py` — verbatim prompt
- `src/agents/rakesh_jhunjhunwala.py` — verbatim prompt
- `src/agents/stanley_druckenmiller.py` — verbatim prompt
- `src/agents/warren_buffett.py` — verbatim prompt
- `src/agents/portfolio_manager.py` — adapted from upstream prompt, extended with debate and position context
- `src/agents/fundamentals.py`, `src/agents/sentiment.py`, `src/agents/technicals.py`, `src/agents/valuation.py`, `src/agents/risk_manager.py`, `src/agents/growth_agent.py`, `src/agents/news_sentiment.py` — forked agent code, independently authored prompts
- `src/utils/analysts.py` — `ANALYST_CONFIG` registry pattern derived from upstream
- `src/graph/state.py` — `AgentState` shape derived from upstream
- `src/main.py` — orchestration entrypoint pattern derived from upstream (`run_quorai` was originally `run_hedge_fund`)

The MIT License text for this upstream is reproduced below:

```
MIT License

Copyright (c) 2024 Virat Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## TauricResearch/TradingAgents

**Source:** https://github.com/TauricResearch/TradingAgents  
**License:** Apache-2.0

The bull/bear debate primitive in `src/agents/debate_node.py` is conceptually
inspired by the bull-researcher / bear-researcher / research-manager pattern in
TradingAgents. No source code from TradingAgents is reproduced in this file.

For the full Apache License 2.0 text, see:
https://github.com/TauricResearch/TradingAgents/blob/main/LICENSE
