# Attempt 2 (2026-10-29, A100-SXM4-40GB, commit 77caf22): stopped on purpose

With the bf16 loading fix, vLLM started, but the baseline ran out of memory after playing its 20 games:
TRL's evaluation re-scores the whole eval batch (20 games x ~8k tokens) with the training model at once.
The training item would have hit the same error at its step-25 evaluation (~2.5 h in), so the VM was
stopped ~1 min into training. Fixed in the next commit: evaluation only plays and scores games
(WordleGRPOTrainer.prediction_step). The session cost about 1 compute unit.
