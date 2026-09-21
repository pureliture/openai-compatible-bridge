# Milestones: Streaming Usage Reporting

- **Selected Contract**: `agentic-execution`
- **Goal**: Enable streaming `usage` chunk reporting in `/v1/chat/completions` (supporting OpenAI `stream_options.include_usage`) so that Hermes Agent and other clients receive real-time token counts and trigger session compaction properly.

## Requirements & Design Alignment
- Support `stream_options: {"include_usage": true}` in `OpenAIChatRequest`.
- In streaming response generators (`_chat_completions_stream` and `_chat_completions_stream_buffered_structured_repair`), emit a standard OpenAI-compatible usage chunk (`choices: []`, `usage: {...}`) before `[DONE]`.
- Provide fallback estimation if upstream does not return explicit usage in stream events.
- Zero regression on existing 428 tests.

## Slices

### Slice 1: Stream Options & Usage Chunk Emission [COMPLETED]
- **Status**: COMPLETED
- **Implementation**:
  - Added `StreamOptions` and `stream_options` field to `OpenAIChatRequest` in `main.py`.
  - Added `STREAM_INCLUDE_USAGE_DEFAULT` env var support.
  - Implemented `include_usage` check in `_chat_completions_stream` and `_chat_completions_stream_buffered_structured_repair`.
  - Emitted standard OpenAI-compatible usage chunk (`choices: []`, `usage: {...}`) prior to `data: [DONE]`.
  - Implemented token estimation fallback if upstream events omit usage.
  - Added comprehensive test suite in `tests/test_stream_usage.py` covering default behavior, `stream_options.include_usage=True`, fallback estimation, `STREAM_INCLUDE_USAGE_DEFAULT` env override, and Foundry streaming.
- **Observable Evidence**:
  - 438 pytest unit/integration tests passed (`uv run pytest -q`).
  - Validated that SSE chunk streams output expected OpenAI-spec usage chunk containing `prompt_tokens`, `completion_tokens`, and `total_tokens`.
