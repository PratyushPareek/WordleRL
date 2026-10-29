# Attempt 1 (2026-10-29, A100-SXM4-40GB, commit dc16218): both items failed

TRL loaded the trainer's model copy in float32 (15.4 GiB instead of 7.5 GiB in bf16).
- baseline-qwen3-4b-think: vLLM could not get its 65% memory share and did not start.
- grpo-qwen3-4b-think: vLLM started; the first training update ran out of GPU memory.
Fixed in commit ae71ad5 (model_init_kwargs dtype bfloat16).

`grpo-qwen3-4b-think/games.jsonl` holds the 16 thinking games played before the out-of-memory error
(untrained model, words funky and fluke, 8 games each): 2 solved, 12 hit the 8,192-token limit.
Generating them took about 5 minutes. The session cost 0.44 compute units.
