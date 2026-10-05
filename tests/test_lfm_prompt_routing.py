"""Production request regression: structure-based prompts, no case-name routing."""
import asyncio
import json
import pytest
from openai_compatible_bridge.lfm_summary import LFMSummarizer, LFMUnavailable
from openai_compatible_bridge.context_compaction import load_settings

@pytest.mark.parametrize('original,structured', [
    ('arbitrary-module performs exact-name lookup.\nCommand result: 12 checks passed.', False),
    (json.dumps({'output':'2 passed units','exit_code':0}), True),
    ('violet-module catalog uses exact-name matching.\n'*60+'Warning: optional descriptions omitted.', True),
])
def test_request_uses_input_structure_without_invocation_context(original, structured):
    captured = []
    async def generate(**kwargs):
        captured.append(kwargs)
        return {'text':json.dumps({'summary':original.splitlines()[0] if not structured else '2 passed units.'})}
    settings = load_settings({})
    async def exercise():
        try:
            await LFMSummarizer(generate=generate,settings=settings).summarize(
                original,(),lambda:None,invocation={'command':'DO_NOT_SEND'},context={'purpose':'DO_NOT_SEND'})
        except LFMUnavailable:
            pass  # Only request assembly is under test; existing validator still rejects bad output.
    asyncio.run(exercise())
    assert len(captured)==1
    request=captured[0]
    user=request['messages'][1]['content']
    if structured:
        assert user.startswith('Write factual findings as JSON.\n\n')
        assert 'result' in json.loads(user.split('\n\n',1)[1])
    else:
        packet=json.loads(user)
        assert packet['source']==original
        assert 'Preserve the subject name' in request['messages'][0]['content']
    assert 'DO_NOT_SEND' not in json.dumps(request)
    assert request['temperature']==0
    assert request['reasoning']=={'effort':'none'}
