# AISP Reproduction Report

## What Was Implemented

- Deterministic greedy decoding for TinyLlama-1.1B-Chat.
- Top-p Best-of-N with a matched sample budget against the strongest enabled AISP-style search method.
- Faithful AISP with Gaussian perturbations in pre-logit space, greedy rollout decoding, adaptive importance-sampling updates, best-response tracking, and final-sample evaluation from `q(V | U_kappa, sigma^2)`.
- Tsallis-AISP as an explicit extension: same pre-logit perturbation rollouts as AISP, but with Tsallis-MPPI-inspired deformed-exponential weighting over iteration-normalized sampled costs.
- HH-RLHF loading, reward scoring, diversity/coherence metrics, artifact saving, and plotting.

## Aggregate Results

| method       |   average_reward |   average_diversity |   average_coherence |   average_runtime_seconds |
|:-------------|-----------------:|--------------------:|--------------------:|--------------------------:|
| best_of_n    |         1.79363  |            0.968316 |            0.603468 |                  45.3236  |
| greedy       |         1.5721   |            0.9061   |            0.59352  |                   9.78937 |
| aisp         |         0.703237 |            0.857308 |            0.406428 |                 179.54    |
| tsallis_aisp |         0.650635 |            0.893617 |            0.42312  |                 204.359   |

## Assumptions

- The course-project requirements in the user message were treated as the accessible proposal spec because macOS blocked direct reads from the Downloads folder.
- The evaluation subset is a fixed shuffled slice of `Anthropic/hh-rlhf` default/test rather than the full HH benchmark due local compute limits.
- Coherence uses SimCSE sentence embeddings from `princeton-nlp/sup-simcse-bert-base-uncased`.

## Deviations From The Paper

- The experiment configs use smaller budgets and shorter generations than the paper defaults so the pipeline can run end-to-end on a local MPS setup.
- The reward model is an HH-RLHF-compatible off-the-shelf Hugging Face sequence classifier instead of the paper's larger reward models.
- Diversity is computed over generated token ids from the base model tokenizer, which is a faithful token-sequence interpretation of the paper's n-gram formula.
- Tsallis-AISP is an explicit extension motivated by the Tsallis-MPPI update rule from Wang et al. (2021), not a method defined in the original AISP paper.

## Outcome

- Best average-reward method on this run: `best_of_n`.
- AISP did not beat BoN under the matched sample budget.
- Tsallis-AISP did not beat BoN under the matched sample budget.
- External judge win-rate harness available: no, left disabled

## Next Step For Tsallis-AISP

- Tune Tsallis-specific hyperparameters `r`, `elite_fraction`, and `sigma_sq` on the same held-out subset used for Gaussian AISP.
- If TinyLlama remains brittle, try shorter perturbation horizons or norm-constrained pre-logit perturbations before adding more algorithmic complexity.