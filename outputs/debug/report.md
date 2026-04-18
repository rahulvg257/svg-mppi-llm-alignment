# AISP Reproduction Report

## What Was Implemented

- Deterministic greedy decoding for TinyLlama-1.1B-Chat.
- Top-p Best-of-N with a matched sample budget `N = n * kappa`.
- Faithful AISP with Gaussian perturbations in pre-logit space, greedy rollout decoding, adaptive importance-sampling updates, best-response tracking, and final-sample evaluation from `q(V | U_kappa, sigma^2)`.
- HH-RLHF loading, reward scoring, diversity/coherence metrics, artifact saving, and plotting.

## Aggregate Results

| method    |   average_reward |   average_diversity |   average_coherence |   average_runtime_seconds |
|:----------|-----------------:|--------------------:|--------------------:|--------------------------:|
| best_of_n |         1.79015  |            0.968316 |            0.653928 |                   10.621  |
| greedy    |         1.5721   |            0.9061   |            0.59352  |                    8.4588 |
| aisp      |         0.535113 |            0.853608 |            0.425721 |                   78.7974 |

## Assumptions

- The course-project requirements in the user message were treated as the accessible proposal spec because macOS blocked direct reads from the Downloads folder.
- The evaluation subset is a fixed shuffled slice of `Anthropic/hh-rlhf` default/test rather than the full HH benchmark due local compute limits.
- Coherence uses SimCSE sentence embeddings from `princeton-nlp/sup-simcse-bert-base-uncased`.

## Deviations From The Paper

- The experiment configs use smaller budgets and shorter generations than the paper defaults so the pipeline can run end-to-end on a local MPS setup.
- The reward model is an HH-RLHF-compatible off-the-shelf Hugging Face sequence classifier instead of the paper's larger reward models.
- Diversity is computed over generated token ids from the base model tokenizer, which is a faithful token-sequence interpretation of the paper's n-gram formula.

## Outcome

- Best average-reward method on this run: `best_of_n`.
- AISP did not beat BoN under the matched sample budget.
- External judge win-rate harness available: no, left disabled

## Next Step For Tsallis-AISP

- Add a separate Tsallis weight/update implementation behind a new method flag rather than folding it into the current Gaussian AISP path.
- Tune Tsallis-specific hyperparameters on the same held-out subset and keep the reward/evaluation interface unchanged for apples-to-apples comparison.