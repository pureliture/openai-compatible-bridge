"""Opt-in synthetic live regression using the production LFMSummarizer."""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import re
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openai_compatible_bridge.context_compaction import load_settings, _required_evidence_indexes
from openai_compatible_bridge.lfm_summary import LFMSummarizer
from openai_compatible_bridge.providers.ollama import OllamaChatClient


_EXIT_CODE_MENTION = re.compile(
    r"\b(?:(?:exit|return)[\s_-]*(?:code|status)|"
    r"(?:exited|returned)\s+(?:with\s+)?(?:exit[\s_-]*)?(?:code|status))\b",
    re.IGNORECASE,
)
_EXIT_CODE_VALUE = re.compile(
    r"\s*(?:(?:is|was|of)\s+|[:=]\s*)?(-?\d+)\b",
    re.IGNORECASE,
)
_NEGATED_EXIT_CLAIM = re.compile(
    r"\b(?:not|no|never|without|unknown|unclear|unspecified|unavailable|unreported)\b",
    re.IGNORECASE,
)


def _claim_clause(text: str, start: int, end: int) -> str:
    separators = ".!?;\n"
    left = max((text.rfind(char, 0, start) for char in separators), default=-1) + 1
    right_positions = [position for char in separators if (position := text.find(char, end)) >= 0]
    right = min(right_positions, default=len(text))
    return text[left:right]


def exit_claim_matches_source(source: str, summary: str) -> bool:
    """Require numeric exit claims to match an explicit source exit_code."""
    try:
        source_value = json.loads(source)
    except (TypeError, ValueError):
        source_value = None
    expected = None
    if isinstance(source_value, dict) and type(source_value.get("exit_code")) is int:
        expected = source_value["exit_code"]

    mentions = list(_EXIT_CODE_MENTION.finditer(summary))
    if expected is None:
        return not mentions
    if len(mentions) != 1:
        return False

    mention = mentions[0]
    value = _EXIT_CODE_VALUE.match(summary, mention.end())
    if value is None or int(value.group(1)) != expected:
        return False
    clause = _claim_clause(summary, mention.start(), value.end())
    if _NEGATED_EXIT_CLAIM.search(clause):
        return False
    if re.match(r"\s*(?:,|\bor\b|\band\b|/)\s*-?\d+\b", summary[value.end():], re.IGNORECASE):
        return False
    return True


async def exercise(output: Path) -> None:
    fixture = runpy.run_path(str(ROOT / 'tests/test_lfm_live_integration.py'))
    cases = {
        'catalog': (fixture['_source'](), ('sample-addon', 'exact', '12', 'pass')),
        'optional_warning': ('violet-module catalog uses exact-name matching.\n' * 60 +
                             'Warning: optional descriptions omitted.', ('violet-module', 'exact', 'optional', 'omitt')),
        'known_exit': (json.dumps({'output': '2 passed units\n' * 60, 'exit_code': 0}), ('2', 'pass')),
        'unknown_exit': (json.dumps({'output': '2 passed units\n' * 60, 'exit_code': None}), ('2', 'pass')),
    }
    settings = load_settings({'CONTEXT_COMPACTION_LFM_TIMEOUT_SECONDS': '60'})
    client = OllamaChatClient(base_url=os.environ['LFM_INTEGRATION_BASE_URL'])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to blend previous execution with this run.
    with output.open('x') as stream:
        try:
            for repetition in range(1, 11):
                for name, (source, facts) in cases.items():
                    raw = {}
                    async def generate(**kwargs):
                        raw['response'] = await client.generate(**kwargs)
                        return raw['response']
                    evidence = tuple(source.splitlines()[i] for i in _required_evidence_indexes(source.splitlines()))
                    summary = await LFMSummarizer(generate=generate, settings=settings).summarize(
                        source, evidence, lambda: None, invocation={})
                    text = summary['summary'].lower()
                    assert all(fact in text for fact in facts), (name, summary)
                    assert exit_claim_matches_source(source, summary['summary']), name
                    stream.write(json.dumps({'case': name, 'repetition': repetition,
                                             'summary': summary, **raw}) + '\n')
                    stream.flush()
                    print(name, repetition, summary, flush=True)
        finally:
            await client.close()

if __name__ == '__main__':
    asyncio.run(exercise(Path(sys.argv[1])))
    fixture = runpy.run_path(str(ROOT / 'tests/test_lfm_live_integration.py'))
    for repetition in range(1, 11):
        fixture['test_real_lfm_summary_compaction_and_exact_unhide_restore']()
        print('INTEGRATION_PASS', repetition, flush=True)
