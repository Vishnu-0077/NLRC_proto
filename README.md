# Reservoir Computing & Chaos-Inspired ML

A collective research codebase for **undergraduate reservoir-computing experiments**: chaotic time-series forecasting, lightweight nonlinear reservoirs, polynomial readouts (NGRC), classic echo state networks, and GLS-neuron feature extraction for classification.

The work centers on **Generalized Lipschitz System (GLS) / tent-map** nonlinearities and **ridge-regression readouts**, with benchmarks such as **Mackey–Glass**, the **logistic map**, **Lorenz**, and the **Iris** dataset.

---

## Repository layout

```
NLRC proto/
├── README.md                 ← this file (overview of the whole tree)
├── RC/                       ← classic Echo State Networks (ESN)
├── NLRC/                     ← Nonlinear Reservoir Computing variants
│   ├── base_version/         ← baseline NLRC (GLS chain + ridge)
│   ├── leaky_NLRC/           ← NLRC + leaky exponential memory
│   ├── NLRC_feature_extraction/  ← statistical + polynomial features
│   └── RC_tentmap_functionchange/  ← ESN with sigmoid → tent activation
├── NGRC/                     ← Next Generation Reservoir Computing
└── Neurochaos/               ← GLS + TTSS features + SVM (Iris)
```

Each subdirectory has its own `README.md` with hyperparameters, API notes, and design detail.

---

## Project areas at a glance

| Directory | Goal | Core idea | Typical benchmark |
|-----------|------|-----------|-------------------|
| [**RC/**](RC/) | Baseline reservoir computing | Random recurrent `W`, leaky integrator, `tanh`, ridge readout | Mackey–Glass, Lorenz (CSV) |
| [**NLRC/**](NLRC/) | Lightweight “reservoir” without large `W` | Scalar input → chained **GLS/tent map** → small feature vector → ridge | Mackey–Glass, logistic map |
| [**NGRC/**](NGRC/) | Polynomial reservoir computing | Lag window + `PolynomialFeatures` + standardized ridge | Mackey–Glass |
| [**Neurochaos/**](Neurochaos/) | Chaos-inspired **classification** | GLS iteration → **TTSS** features → SVM | Iris |

---

## Shared concepts

### GLS / tent map neuron

Many scripts use the same piecewise map with threshold `b = 0.5`:

- If `x ≥ b`: `(1 − x) / (1 − b)`
- Else: `x / b`

This appears in NLRC (`gls_neuron_gen`), the tent-map ESN (`RC_tentmap_functionchange`), and Neurochaos (`gls_neuron_gen`). Values are often clipped or scaled to `[0, 1]` before iteration.

### Training pattern (forecasting scripts)

1. Build features from training inputs (reservoir state, GLS chain, polynomials, etc.).
2. Fit readout weights in closed form:  
   `w_out = Y · Φᵀ · (ΦΦᵀ + λI)⁻¹` (ridge regression).
3. **Autoregressive test**: seed with one test value; each prediction becomes the next input (ground truth is not fed step-by-step during rollout).

### Benchmarks

| Series / data | Where used |
|---------------|------------|
| Mackey–Glass | `RC/`, `NLRC/base_version`, `NLRC/RC_tentmap_functionchange`, `NLRC_feature_extraction/feature_only.py`, `NGRC/` |
| Logistic map | `NLRC/leaky_NLRC`, `NLRC_feature_extraction/feature_polynomial.py` |
| Lorenz (CSV) | `RC/rc_with_lorenz.py` |
| Iris | `Neurochaos/code.py` |

---

## Architecture comparison

```mermaid
flowchart TB
  subgraph RC_ESN["RC — Echo State Network"]
    u1["u"] --> W["Random W, W_in"]
    W --> tanh["tanh (or tent in variant)"]
    tanh --> x["State x"]
    x --> ridge1["Ridge w_out"]
  end

  subgraph NLRC_flow["NLRC — Nonlinear RC"]
    u2["Scalar u"] --> gls["GLS chain (n steps)"]
    gls --> feat["Small Φ e.g. n+1 dims"]
    feat --> ridge2["Ridge w_out"]
  end

  subgraph NGRC_flow["NGRC"]
    u3["Lag window k"] --> poly["PolynomialFeatures"]
    poly --> ridge3["Ridge w_out"]
  end

  subgraph Neuro["Neurochaos"]
    u4["Iris features"] --> ttss["GLS → TTSS"]
    ttss --> svm["SVM"]
  end
```

| Approach | Feature size | Recurrent matrix | Main nonlinearity |
|----------|--------------|------------------|-------------------|
| **RC (ESN)** | `1 + insize + ressize` (e.g. 301) | Yes (`W`, spectral radius ≈ 0.95) | `tanh` |
| **NLRC base** | `n + 1` (e.g. 11) | No | GLS iterations on scalar |
| **Leaky NLRC** | `n + 1` + temporal EMA | No | GLS + leak factor `a` |
| **NLRC feature stats** | 6 or polynomial of `5(k+1)` | No | GLS + energy, entropy, variance, firing rate |
| **RC tent map** | Same as ESN | Yes | `sigmoid` → `tentmap` |
| **NGRC** | `C(k+deg, deg)` monomials | No | Polynomials of lags |
| **Neurochaos** | `10 targets × 4 features` = 40 | No | GLS symbolic trace → TTSS |

---

## NLRC variants (detail in [NLRC/README.md](NLRC/README.md))

| Folder | Class / style | Distinction |
|--------|---------------|-------------|
| [base_version/](NLRC/base_version/) | `nlrc` | Baseline: `[1, X₀, …, Xₙ₋₁]` from GLS chain |
| [leaky_NLRC/](NLRC/leaky_NLRC/) | `nlrc` | Leaky exponential memory over feature columns |
| [NLRC_feature_extraction/](NLRC/NLRC_feature_extraction/) | `nlfea` | Statistics from GLS chain; optional delay + polynomials (NGRC-like) |
| [RC_tentmap_functionchange/](NLRC/RC_tentmap_functionchange/) | Script ESN | Same Mackey–Glass RC demo as `RC/`, but `tanh` → **sigmoid + tent map** |

---

## Dependencies

Install what you need per experiment:

```bash
# Common forecasting stack
pip install numpy scipy scikit-learn matplotlib

# RC Lorenz script, NGRC Excel demo
pip install pandas openpyxl
```

| Package | Used by |
|---------|---------|
| `numpy`, `scipy` | All forecasting folders |
| `scikit-learn` | Scaling, ridge-style fits, `PolynomialFeatures`, SVM, metrics |
| `matplotlib` | NLRC / NGRC / feature-extraction plots |
| `pandas`, `openpyxl` | `RC/rc_with_lorenz.py`, `NGRC/code.py` (Excel path in demo) |

---

## Quick start

From the repository root:

```bash
# Classic reservoir computing
python RC/rc_with_mackayglass.py
python RC/rc_with_lorenz.py          # update CSV path inside the script

# NLRC family
python NLRC/base_version/code.py
python NLRC/leaky_NLRC/code.py
python NLRC/NLRC_feature_extraction/feature_only.py
python NLRC/NLRC_feature_extraction/feature_polynomial.py
python NLRC/RC_tentmap_functionchange/code.py

# NGRC
python NGRC/code.py                  # set Excel data path in script

# Classification
python Neurochaos/code.py
```

Scripts typically print **MSE** (forecasting) or **accuracy** (Neurochaos) and may open matplotlib figures.

---

## Configuration notes

- **Hardcoded paths**: `RC/rc_with_lorenz.py` and `NGRC/code.py` use absolute or local paths to data files—change them before running on your machine.
- **Reproducibility**: ESN scripts set `np.random.seed(42)` for fixed reservoir weights.
- **Scaling**: Most NLRC demos use `MinMaxScaler` to `[0, 1]`; classic `RC/` Mackey–Glass and tent-map scripts often run on raw generated series.
- **Recursive testing**: Only the first test point (or trained reservoir state) seeds rollout; later steps use model outputs only.

---

## Results (from bundled demos / docs)

| Experiment | Reported metric |
|------------|-----------------|
| NLRC base (Mackey–Glass, 10-step test) | MSE ~`4.15e-10` — see [NLRC/base_version/README.md](NLRC/base_version/README.md) |
| Neurochaos (Iris, 80% test split) | **94%** accuracy — see [Neurochaos/README.md](Neurochaos/README.md) |
| Other scripts | MSE printed to stdout; tune `n`, `k`, `deg`, `reg`, `a`, `ressize` per README in each folder |

---

## Further reading (per-folder)

| Topic | Document |
|-------|----------|
| ESN Lorenz vs Mackey–Glass | [RC/README.md](RC/README.md) |
| NLRC overview & variant table | [NLRC/README.md](NLRC/README.md) |
| Baseline NLRC API & plots | [NLRC/base_version/README.md](NLRC/base_version/README.md) |
| Leaky memory constructor | [NLRC/leaky_NLRC/README.md](NLRC/leaky_NLRC/README.md) |
| Feature-only vs polynomial NLRC | [NLRC/NLRC_feature_extraction/README.md](NLRC/NLRC_feature_extraction/README.md) |
| Tent-map ESN vs `tanh` RC | [NLRC/RC_tentmap_functionchange/README.md](NLRC/RC_tentmap_functionchange/README.md) |
| NGRC class & Mackey–Glass demo | [NGRC/README.md](NGRC/README.md) |
| GLS, TTSS, SVM pipeline | [Neurochaos/README.md](Neurochaos/README.md) |

---

## Remote

This tree is tracked as a git repository (remote: `NLRC_proto` on GitHub). Clone or push from your fork as needed; subdirectory READMEs are the source of truth for parameters and behavior.

---

## Summary

**NLRC proto** collects parallel lines of research on **reservoir-style learning** for dynamics and simple classification: full **echo state networks**, **compact NLRC** models tied to GLS/tent maps, **NGRC** polynomial readouts, and **Neurochaos** symbolic features. Use this README for orientation; use each folder’s README and `code.py` for experiments and reproduction.
