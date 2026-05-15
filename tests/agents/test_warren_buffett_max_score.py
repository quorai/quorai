"""Tests for R31: Buffett max_possible_score must include consistency component."""

import inspect
import re


class TestBuffettMaxScore:
    def test_max_possible_score_includes_consistency(self):
        """
        max_possible_score must include the consistency component max (3).
        Before the fix: consistency was summed into total_score but its max was omitted
        from max_possible_score, inflating score/max by ~10%.
        The sum block should have '+ 3' or 'consistency' contribution.
        """
        from src.agents import warren_buffett as wb_mod

        source = inspect.getsource(wb_mod)

        # Find the max_possible_score block by extracting lines between its assignment and end of block.
        lines = source.splitlines()
        in_block = False
        block_lines = []
        for line in lines:
            if "max_possible_score" in line and "=" in line and "(" in line:
                in_block = True
            if in_block:
                block_lines.append(line)
                # End of block: line with just ')' or matching close paren
                stripped = line.strip()
                if stripped == ")" or (stripped.startswith(")") and len(stripped) <= 2):
                    break

        block = "\n".join(block_lines)
        # The block must contain '+ 3' or 'consistency' to account for consistency max
        has_consistency = "+ 3" in block or "consistency" in block.lower()
        assert has_consistency, f"max_possible_score block does not include consistency component (expected '+ 3' or 'consistency'):\n{block}"

    def test_analyze_consistency_returns_score_key(self):
        """analyze_consistency must return a dict with a 'score' key (basic contract)."""
        from src.agents.warren_buffett import analyze_consistency

        result_empty = analyze_consistency([])
        assert "score" in result_empty
        assert result_empty["score"] == 0  # insufficient data → 0

    def test_total_score_numerator_count(self):
        """total_score should reference all 6 components including consistency."""
        from src.agents import warren_buffett as wb_mod

        source = inspect.getsource(wb_mod)
        m = re.search(r"total_score\s*=\s*(.+)", source)
        assert m, "Could not find total_score = ... line"
        total_line = m.group(1)
        assert "consistency" in total_line, "total_score must include consistency_analysis['score']"
