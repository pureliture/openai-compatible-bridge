"""Opt-in synthetic live regression using the production LFMSummarizer."""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openai_compatible_bridge.context_compaction import load_settings, _required_evidence_indexes
from openai_compatible_bridge.lfm_summary import LFMSummarizer
from openai_compatible_bridge.providers.ollama import OllamaChatClient

async def exercise(output: Path) -> None:
    fixture = runpy.run_path(str(ROOT / 'tests/test_lfm_live_integration.py'))
    cases = {
        'catalog': (fixture['_source'](), ('sample-addon', 'exact', '12', 'pass')),
        'optional_warning': ('violet-module catalog uses exact-name matching.\n' * 60 +
                             'Warning: optional descriptions omitted.', ('violet-module', 'exact', 'optional', 'omitt')),
        'known_exit': (json.dumps({'output': '2 passed units\n' * 60, 'exit_code': 0}), ('2', 'pass', '0')),
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
                    if name == 'unknown_exit':
                        assert 'exit code 0' not in text and 'exit_code 0' not in text
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
