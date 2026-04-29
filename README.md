# svg_tmpc

**Test-time LLM alignment via variational-inference sampling-based optimal control.**

`svg_tmpc` is a research package that compares four decoding strategies on
[TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
using the [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
preference dataset:

1. **Baseline** — unmodified greedy / nucleus decoding.
2. **Best-of-N** — sample N continuations, keep the highest-reward one.
3. **TMPC** — representation-editing decoding via vanilla MPPI weights, after
   Wang et al., *Test-time Alignment for Large Language Models via Textual
   Model Predictive Control* (ICLR 2026).
4. **Tsallis-MPPI** — our novel contribution. Replaces the softmax MPPI weights
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

## Larger backbones (Llama-2-7B)

The backbone wrapper is architecture-agnostic for any Llama-family causal LM,
so swapping in `meta-llama/Llama-2-7b-chat-hf` is a config change:

```bash
svg-tmpc-run --config configs/llama2_7b.yaml --output-dir outputs_llama2_7b/
```

Llama-2-7B is gated; request access on HuggingFace and run `huggingface-cli
login` first. Memory tips:

- `dtype: bfloat16` → ~13 GB weights, fits on a single 24 GB GPU.
- For tighter VRAM, set `backbone.load_in_8bit: true` (requires the optional
  `bitsandbytes` package) — drops to ~7 GB and forces `device_map="auto"`.
- For multi-GPU sharding without quantization, set `backbone.device_map: "auto"`.

The default sigma=0.1 was tuned for TinyLlama's 2048-dim hidden state. Llama-2-7B
has a 4096-dim hidden state, so you may want to **re-sweep sigma** (try 0.03,
0.05, 0.1) when moving to the larger backbone — `svg-tmpc-sweep` is the easiest
way to do this.

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

- Wang, K.-D., et al. *Test-time Alignment for Large Language Models via Textual
  Model Predictive Control.* ICLR 2026.
- Wang, Z., et al. *Variational Inference MPC using Tsallis Divergence.* RSS 2021.
- Williams, G., et al. *Aggressive Driving with Model Predictive Path Integral
  Control.* ICRA 2016.
- Bai, Y., et al. *Training a Helpful and Harmless Assistant with RLHF.* 2022.
- Zhang, P., et al. *TinyLlama: An Open-Source Small Language Model.* 2024.

## License

MIT.
