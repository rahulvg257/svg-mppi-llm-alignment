# svg_tmpc

**Test-time LLM alignment via variational-inference sampling-based optimal control.**

`svg_tmpc` is a research package that compares four decoding strategies on
[TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
using the [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
preference dataset:

1. **Baseline** — unmodified greedy / nucleus decoding.
2. **Best-of-N** — sample N continuations, keep the highest-reward one.
3. **AISP** — representation-editing decoding via vanilla MPPI weights, after
   Kanai et al., *Test-Time Alignment of LLMs via Sampling-Based Optimal Control in pre-logit space*.
4. **Tsallis-AISP** — our novel contribution. Replaces the softmax MPPI weights
   with the q-exponential (Tsallis) weights from
   Wang et al., *Variational Inference MPC using Tsallis Divergence* (RSS 2021),
   yielding more robust exploration of the perturbation distribution.

The package is a self-contained pip-installable Python project and ships two
console scripts: `svg-tmpc-run` (generate) and `svg-tmpc-eval` (score).

## Installation

```bash
pip install -e .
# or with dev tools
pip install -e ".[dev]"
```

`svg_tmpc` depends on `torch>=2.0`, `transformers>=4.40`, `datasets`, `nltk`,
`pyyaml`, `pandas`, `numpy`, and `tqdm`. The first run will trigger the usual
HuggingFace model + dataset downloads.

NLTK is used for sentence-BLEU. If you have not previously downloaded NLTK data
the smoothing-only path used by `self_bleu` does *not* require any download, but
if you extend the diversity metric you may need:

```python
import nltk
nltk.download("punkt")
```

## Quick start

```bash
# 1. Generate responses with all four methods
svg-tmpc-run --config configs/default.yaml --output-dir outputs/

# 2. Compute diversity / coherence / reward / win-rate
svg-tmpc-eval --output-dir outputs/

# 3. (optional) sweep hyperparameters
svg-tmpc-sweep --config configs/default.yaml \
               --sweep  configs/sweep_example.yaml \
               --output-dir sweeps/initial/
```

The generation step writes `outputs/responses.json` and the evaluation step
writes `outputs/metrics.csv` plus a formatted table to stdout. The sweep
harness writes `sweeps/initial/sweep_summary.csv` plus per-run subdirectories.

## Hyperparameter sweeps

`svg-tmpc-sweep` loads the backbone + reward model **once** and iterates a
Cartesian grid of overrides on top of a base config. Each grid point produces
its own subdirectory containing `responses.json`, `metrics.csv`, and the
exact `overrides.json` applied. A flat `sweep_summary.csv` collects one row
per (run, method) so you can pivot in pandas:

```python
import pandas as pd
df = pd.read_csv("sweeps/initial/sweep_summary.csv")
df.pivot_table(index="tsallis_mppi.q", columns="method", values="avg_reward")
```

The sweep spec is YAML — see [configs/sweep_example.yaml](configs/sweep_example.yaml).
Keys are dotted paths into the base config (`tmpc.sigma`, `tsallis_mppi.q`,
`best_of_n.N`, etc.) and values are lists. The Cartesian product of all axes
is enumerated; every combination re-runs every enabled method. Sweeping
`backbone.*` or `reward.*` does not reload the model — run separate sweeps
for those.

## Larger backbones (Llama-2-7B, Llama-3-8B)

The backbone wrapper is architecture-agnostic for any Llama-family causal LM —
swapping in a 7B or 8B instruct model is a config change:

```bash
# Llama-2-7B-chat (>=24 GB VRAM, bf16)
svg-tmpc-run --config configs/llama2_7b.yaml --output-dir outputs_llama2_7b/

# Llama-3-8B-Instruct (>=32 GB VRAM, bf16)
svg-tmpc-run --config configs/llama3_8b.yaml --output-dir outputs_llama3_8b/

# Llama-3-8B-Instruct on a single 24 GB card (NF4 + double-quant, ~5 GB weights)
svg-tmpc-run --config configs/llama3_8b_titanrtx.yaml --output-dir outputs_llama3_8b/
```

All three models are gated on HuggingFace — request access and run
`huggingface-cli login` once before invoking the CLI.

### Memory recipes

| Setup                          | dtype    | Quantization        | Weights | Notes                          |
| ------------------------------ | -------- | ------------------- | ------- | ------------------------------ |
| Llama-2-7B, TITAN RTX 24 GB    | bfloat16 | none                | ~13 GB  | `configs/llama2_7b.yaml`       |
| Llama-3-8B, TITAN RTX 24 GB    | bfloat16 | none                | ~16 GB  | `configs/llama3_8b.yaml`       |
| Llama-3-8B, **TITAN RTX 24 GB**| float16  | NF4 + double-quant  | ~5 GB   | `configs/llama3_8b_titanrtx.yaml` |
| Any 7B–8B, multi-GPU           | bfloat16 | none, `device_map: "auto"` | ~13–16 GB sharded | set `backbone.device_map: "auto"` |

### TITAN RTX (24 GB) recipe

The `llama3_8b_titanrtx.yaml` preset enables 4-bit NF4 quantization with double
quantization, which brings Llama-3-8B's weights to ~5 GB. With the OpenAssistant
DeBERTa reward model (~1.5 GB) and K=16 batched TMPC rollouts, peak VRAM stays
comfortably under 24 GB. Requirements:

- `pip install bitsandbytes` (and a recent NVIDIA driver supporting CUDA ≥ 11.6)
- GPU compute capability ≥ 7.5 (TITAN RTX, RTX 30/40-series, A-series). Older
  cards (e.g. GTX 10-series) are unsupported by bitsandbytes 4-bit.
- If you OOM during TMPC rollouts, reduce `tmpc.K` / `tsallis_mppi.K` to 8.

### Hyperparameter notes for larger backbones

The default `sigma=0.1` was tuned for TinyLlama's 2048-dim hidden state.
Llama-2-7B and Llama-3-8B both have hidden_size=4096, so the noise norm doubles
at the same `sigma`. The Llama-3 presets default to `sigma=0.05` to compensate;
for Llama-2-7B you may want to **re-sweep sigma** in `[0.03, 0.05, 0.1]` —
`svg-tmpc-sweep` is the easiest way to do this.

Llama-3-Instruct also uses a different chat template (`<|start_header_id|>...`)
than the `Human:`/`Assistant:` format produced by the HH-RLHF loader. The model
will still produce coherent continuations under the simpler format, but for
absolute-quality numbers you may want to preprocess prompts with
`tokenizer.apply_chat_template` before calling the samplers.

## Algorithm sketch

At every autoregressive decoding step *t*:

1. Run the backbone forward on the current sequence and capture the post-final-
   norm hidden state `h_t` for the last position via a forward hook.
2. Draw `K` Gaussian perturbations `eps_k ~ N(0, sigma^2 * I)` and form
   `h_k = h_t + eps_k`.
3. Apply the LM head directly to each `h_k` to pick a first rollout token, then
   greedily decode an additional `H - 1` tokens to produce the rollout.
4. Score every rollout with a fixed reward model: `r_k = R(prompt + rollout_k)`.
5. Convert rewards into importance weights:
   - **TMPC** uses softmax weights, `w_k ∝ exp(r_k / lambda)`.
   - **Tsallis-MPPI** uses the q-exponential,
     `w_k ∝ max(0, 1 + (q - 1) * (r_k - r_max) / lambda)^{1/(q-1)}`.
     `q → 1` recovers TMPC, `q > 1` sparsifies, and `q < 1` broadens.
6. Form `h_t* = sum_k w_k * h_k` and replace the original hidden state with
   `h_t*` for this step's token selection.

All hyperparameters live in [`configs/default.yaml`](configs/default.yaml) with
inline documentation.

## Package layout

```
svg_tmpc/
├── pyproject.toml
├── README.md
├── configs/
│   └── default.yaml
└── svg_tmpc/
    ├── models/        # backbone + reward model wrappers
    ├── samplers/      # baseline / best-of-N / TMPC / Tsallis-MPPI
    ├── data/          # HH-RLHF prompt loader
    ├── metrics/       # distinct-n, self-BLEU, perplexity, reward, win rate
    ├── experiments/   # ExperimentRunner + svg-tmpc-eval
    └── utils/         # logging + seeding helpers
```

## Programmatic use

```python
from svg_tmpc.models.backbone import Backbone
from svg_tmpc.models.reward import RewardModel
from svg_tmpc.samplers.tsallis_mppi import TsallisMPPISampler

backbone = Backbone()
reward = RewardModel()

sampler = TsallisMPPISampler(
    backbone=backbone,
    reward_model=reward,
    K=16, H=8, sigma=0.1, lambda_=1.0, q=1.5,
)
print(sampler.generate_one("Human: How can I learn Python?\n\nAssistant:", max_new_tokens=64))
```

## Reproducing a run

The default config evaluates 50 HH-RLHF test prompts with
`max_new_tokens=64`, `K=16`, `H=8`. On a single A100 the four-method sweep
takes roughly 30–60 minutes; on CPU the TMPC variants are impractical. Override
the device and dtype with command-line flags or a custom YAML.

## References

- Kanai, S., et al. *Test-Time Alignment of LLMs via Sampling-Based Optimal Control in pre-logit space.*
- Wang, Z., et al. *Variational Inference MPC using Tsallis Divergence.* RSS 2021.
- Williams, G., et al. *Aggressive Driving with Model Predictive Path Integral
  Control.* ICRA 2016.
- Bai, Y., et al. *Training a Helpful and Harmless Assistant with RLHF.* 2022.
- Zhang, P., et al. *TinyLlama: An Open-Source Small Language Model.* 2024.

## License

MIT.
