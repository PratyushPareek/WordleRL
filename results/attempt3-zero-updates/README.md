# Attempt 3 (2026-10-29, A100-SXM4-40GB, commit 5a1cb1c): baseline OK, training stopped after step 1

Step 1 ran (5.6 min, no out-of-memory error) but its update was zero: grad_norm 2.8e-13. TRL's default
vLLM correction ("sequence_mask") weights each game by the product of per-token vLLM/trainer probability
ratios; over ~7k thinking tokens the product was 1e-11 to 1e-42 (importance_sampling_ratio max 5e-11).
Fixed in the next commit: vllm_importance_sampling_mode="token_truncate". Session cost 1.59 units.
