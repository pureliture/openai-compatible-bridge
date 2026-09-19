# Milestones: Palantir Foundry Multi-Provider Structured Tool Calling

- **Selected Contract**: `agentic-execution`
- **Approved Requirements & Design**: [prompt_draft.md](file:///Users/ddalkak/.gemini/antigravity/brain/b8d708df-4b04-4a42-9691-229650282f33/prompt_draft.md)
- **Runtime Policy**: All models must use `Gemini 3.8 Flash (High)` exclusively.
- **Git Worktree**: `/Users/ddalkak/Projects/openai-compatible-bridge/.worktrees/foundry-tool-calling` (`agentic/foundry-tool-calling` branched from `origin/main` @ `9199b57`)

## Slices

- [x] **Slice 1: Bridge Request/Response & Streaming Foundation for Tool Calling**
  - Scope: Update `main.py` models and streaming handler to accept `tools`/`tool_choice`, preserve tool messages in history, emit `tool_calls` in response and SSE deltas.
  - Observable Result: Public OpenAI chat completion contract accepts tool calling payloads and streams tool chunks with `finish_reason: "tool_calls"` without breaking Vertex/Ollama.
  - Evidence: Verified via `test_foundry_alias_routes_tools_and_receives_tool_calls_non_stream` and `test_foundry_alias_streams_tool_call_deltas` in `tests/test_foundry_chat.py`. 283 baseline tests passing.

- [x] **Slice 2: Foundry OpenAI Protocol (GPT Family) Structured Tool Calling**
  - Scope: `providers/foundry.py` OpenAI protocol mapping for `tools`, `tool_choice`, response `tool_calls`, streaming `tool_calls` SSE deltas, and multi-turn tool history continuation.
  - Observable Result: GPT family non-streaming & streaming tool calling tests pass.
  - Evidence: `tests/test_foundry_tool_calling.py` (5/5 passed: `test_foundry_openai_tool_calling_non_stream`, `test_foundry_openai_tool_calling_stream`, `test_foundry_openai_multi_turn_history`, `test_foundry_openai_tool_calling_with_assistant_text_and_empty_arguments`, `test_foundry_openai_stream_fallback_finish_reason`). All 288 tests passed (`uv run pytest -q`), `uv run python -m compileall -q openai_compatible_bridge` 0 errors.

- [x] **Slice 3: Foundry Anthropic Protocol (Claude Family) Structured Tool Calling**
  - Scope: Anthropic `input_schema` mapping, `tool_use`/`tool_result` translation, consecutive `tool` message merging into single `user` turn, streaming `content_block_delta` SSE chunks.
  - Observable Result: Claude family non-streaming & streaming tool calling tests pass.
  - Evidence: `tests/test_foundry_tool_calling.py` (6/6 Anthropic tests passed: `test_foundry_anthropic_tool_calling_non_stream`, `test_foundry_anthropic_tool_calling_stream`, `test_foundry_anthropic_multi_turn_history`, `test_foundry_anthropic_tool_choice_variants_and_turn_alternation`, `test_foundry_anthropic_stream_fallback_finish_reason`, `test_foundry_anthropic_tool_calling_with_text_and_empty_args`). All 315 project tests passed (`uv run pytest -q`), `uv run python -m compileall -q openai_compatible_bridge tests` 0 errors.

- [x] **Slice 4: Foundry xAI Responses Protocol (Grok Family) Structured Tool Calling**
  - Scope: xAI Responses `tools` mapping, `function_call`/`function_call_output` item translation, streaming `response.output_item.*` SSE chunks.
  - Observable Result: Grok family non-streaming & streaming tool calling tests pass.
  - Evidence: `tests/test_foundry_tool_calling.py` (6/6 xAI tests passed: `test_foundry_xai_tool_calling_non_stream`, `test_foundry_xai_tool_calling_stream`, `test_foundry_xai_multi_turn_history`, `test_foundry_xai_stream_fallback_finish_reason`, `test_foundry_xai_tool_calling_with_text_and_empty_args`, `test_foundry_xai_tool_choice_variants`). All 340 project tests passed (`uv run pytest -q`), `uv run python -m compileall -q openai_compatible_bridge tests` 0 errors.

- [x] **Slice 5: End-to-End Regression & Error Handling Verification**
  - Scope: Palantir Foundry error unwrapping (LanguageModelService:LlmHttpClientError, responseBody Optional[...] JSON, errorMessage/errorCode fallback), upstream 502 HTML, upstream 504 Timeout and 502 Connection Error mapping, 4-tier E2E Hermes multi-turn simulation (OpenAI GPT, Anthropic Claude, xAI Grok non-streaming and streaming), zero regression across full test suite, compileall check.
  - Observable Result: Zero regressions across entire suite, 10 new E2E & error handling tests passing (27/27 in `tests/test_foundry_tool_calling.py`, 366/366 total suite passing), clean Python compilation with 0 errors.
  - Evidence:
    - `uv run pytest tests/test_foundry_tool_calling.py -v`: 27 passed in 0.26s (including all 4 error handling E2E tests and 6 Hermes 4-tier multi-turn tests).
    - `uv run pytest -q`: 366 passed, 1 warning in 1.63s (100% pass, zero regression).
    - `uv run python -m compileall -q openai_compatible_bridge tests`: 0 errors.

## Active Slice
- None (All 5 Slices Completed)

