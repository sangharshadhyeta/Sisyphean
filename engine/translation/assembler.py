"""Response assembler — combines stage outputs into one clean response.

The outer loop (Claude Code) sees only this final assembled response.
All the internal planning, searching, and iteration is invisible to it.
"""
from __future__ import annotations

import logging

from engine.translation.executor import StageResult

logger = logging.getLogger(__name__)


def assemble(results: list[StageResult], outcome: str) -> str:
    """Combine stage results into a single coherent response.

    Strategy:
    - Single stage: return output directly, no headers
    - Multiple stages: label each stage output only if they're meaningfully different
    - Always include the final write/code output prominently
    """
    if not results:
        return "(No output produced)"

    # Single stage — return output directly
    if len(results) == 1:
        return results[0].output.strip()

    # Multiple stages — find the most important output
    # Write stages (code/doc) take priority as the primary output
    write_results = [r for r in results if r.stage.type in ("write_code", "write_doc")]
    research_results = [r for r in results if r.stage.type == "research"]
    other_results = [r for r in results if r.stage.type not in ("write_code", "write_doc", "research")]

    parts: list[str] = []

    # Research findings first if no write stage
    if not write_results and research_results:
        for r in research_results:
            if r.output.strip():
                parts.append(r.output.strip())

    # Write outputs are the primary deliverable
    for r in write_results:
        if r.output.strip():
            parts.append(r.output.strip())

    # Other stages (verify, reflect) as supplementary notes
    for r in other_results:
        if r.output.strip():
            parts.append(r.output.strip())

    # If we had both research and write, add research as context only if short
    if write_results and research_results:
        for r in research_results:
            brief = r.output.strip()
            if brief and len(brief) < 300:
                parts.insert(0, f"*Research: {brief}*")

    return "\n\n".join(p for p in parts if p.strip()) or "(No output produced)"


def assemble_search_only(results: list[StageResult]) -> str:
    """For search-intent tasks: format as an informative answer with sources."""
    if not results:
        return "(No search results)"

    parts: list[str] = []
    for r in results:
        if r.output.strip():
            parts.append(r.output.strip())
        if r.searches_done:
            logger.debug("Searches performed: %s", r.searches_done)

    return "\n\n".join(parts) or "(No output produced)"
