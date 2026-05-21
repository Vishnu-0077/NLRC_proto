# Reservoir Computing with Tent Map Activation (`code.py`)

This folder contains a **single-script Echo State Network (ESN)** that predicts a Mackey–Glass time series. It follows the same overall layout as the classic reservoir-computing demo in [`RC/rc_with_mackayglass.py`](../../RC/rc_with_mackayglass.py), but replaces the reservoir’s **`tanh` activation** with a **sigmoid + tent map** pipeline on the pre-activations.

The experiment name (`RC_tentmap_functionchange`) refers to that activation change only: weights, sizes, training length, ridge regression, and recursive testing are otherwise aligned with the original Mackey–Glass RC script.

---

## What this script does (high level)

1. **Generate** a Mackey–Glass series (or optionally a logistic map via `test_data`).
2. **Build** a random recurrent reservoir (`W`, `W_in`), scale `W` to a target spectral radius.
3. **Train** by running the reservoir over the first `train_length` samples, collecting state snapshots after a washout period, and fitting a linear readout `w_out` with ridge regression.
4. **Test** autoregressively for `test_length` steps: each predicted value becomes the next input `u`.

There is no class wrapper—the logic is linear in one file, matching the style of the original `RC` scripts.

---

## Dependencies

| Package | Use in this script |
|---------|-------------------|
| `numpy` | Arrays, reservoir dynamics, readout |
| `scipy.linalg` | Matrix inverse for ridge regression |
| `pandas` | Imported but **not used** in the current script |
| `sklearn.preprocessing.MinMaxScaler` | Imported but **not used** (unlike NLRC demos that scale to `[0, 1]`) |

Install as usual, e.g. `pip install numpy scipy pandas scikit-learn`.

---

## Data generation

### `mackey_glass(length=2000, tau=17, beta=0.2, gamma=0.1, n=10, dt=1.0)`

Euler integration of the Mackey–Glass delay equation:

- Initial segment `x[:tau+1] = 1.2`
- Update uses delayed sample `x[t - tau]` and current `x[t]`

The main script uses `data = mackey_glass()` and one-step-ahead targets `y_data = data[1:]`.

### `test_data(a=3.95, i=0.5422, length=2000)`

Logistic map: `x[t+1] = a * x[t] * (1 - x[t])`. Present for alternate benchmarks; **not** used in the default `mackey_glass()` path.

---

## Hyperparameters (script-level)

| Symbol | Value | Role |
|--------|-------|------|
| `insize`, `outsize` | `1` | Scalar input / output |
| `ressize` | `300` | Number of reservoir units |
| `init_len` | `50` | Washout: states before this index are not stored for training |
| `train_length` | `1000` | Training time steps |
| `test_length` | `100` | Autoregressive forecast horizon |
| `a` | `0.3` | Leak rate in `x ← (1-a)x + a·r` |
| `target_radius` | `0.95` | Spectral radius of `W` after scaling |
| `reg` | `1e-8` | Ridge term on `x_history @ x_history.T` |
| `np.random.seed(42)` | — | Reproducible `W`, `W_in` |

---

## Reservoir setup

1. **`W_in`**: shape `(ressize, insize+1)`, entries uniform in `[-0.5, 0.5]`. The extra column is the **bias** (constant `1` stacked with input `u`).
2. **`W`**: shape `(ressize, ressize)`, same random range.
3. **Spectral radius**: `W` is multiplied by `target_radius / max|λ(W)|` so dynamics stay in the typical ESN stable regime.

Initial state: `x = zeros(ressize, 1)`.

---

## Tent map (`tentmap`)

```python
def tentmap(x):
    b = 0.5
    return np.where(x < b, x / b, (1 - x) / (1 - b))
```

For each component in `[0, 1]`:

- If `x < b`: map to `[0, 1]` via `x / b`
- Else: map via `(1 - x) / (1 - b)`

This is the same **piecewise linear “tent”** used as the GLS neuron map in the NLRC codebase (`gls_neuron_gen` in `base_version/code.py`), applied here **vectorized** to the full pre-activation vector `r`.

Parameter `b = 0.5` matches the default threshold in those NLRC helpers; this script does not expose `b` as a hyperparameter.

---

## Reservoir update: where this differs from original RC

### Original RC (`RC/rc_with_mackayglass.py`)

One step of reservoir state:

```text
pre = W @ x + W_in @ [1; u]
x   ← (1 - a) * x + a * tanh(pre)
```

`tanh` is applied **directly** to the affine combination of previous state and input. No extra bounding step before the nonlinearity.

### This script (`code.py`)

```text
pre = W @ x + W_in @ [1; u]
r   = sigmoid(pre) = 1 / (1 + exp(-pre))
r   = tentmap(r)
x   ← (1 - a) * x + a * r
```

| Aspect | Original RC | This script |
|--------|-------------|-------------|
| Nonlinearity on `pre` | `tanh(pre)` (range roughly `(-1, 1)`) | `sigmoid(pre)` then `tentmap` (per unit in `[0, 1]` before tent) |
| Role of sigmoid | None | Forces pre-activations into `(0, 1)` so `tentmap` is well-defined |
| Tent map | Not used | Applied to every unit of `r` |
| Leaky integration | `x = (1-a)x + a·tanh(...)` | Same formula with `r` instead of `tanh(...)` |
| Training / test loops | Same structure | Same structure (only inner update changes) |

**Why sigmoid before tent map:** Comment in code: bound activations between 0 and 1 before the tent map, which is defined on that interval. `tanh` already bounds values but not to `[0, 1]`; the tent map expects a unit-interval style domain like the GLS / NLRC feature maps.

**What did not change vs original Mackey–Glass RC:**

- Mackey–Glass generator and default lengths
- `ressize = 300`, `train_length = 1000`, `init_len = 50`, `test_length = 100`, `a = 0.3`
- Random seed, `W` / `W_in` initialization, spectral-radius scaling
- Feature matrix: each column is `[1; u; x]` (bias, input, full reservoir state)
- Ridge readout: `w_out = yt @ x_history.T @ inv(x_history @ x_history.T + reg·I)`
- Testing: start `u = data[train_length]`, then `u ← y` each step

---

## Training phase

For `i = 0 … train_length - 1`:

1. `u = data[i]`
2. Update `x` with the **sigmoid → tent map → leaky** rule above
3. If `i >= init_len`, store `np.vstack((1, u, x))` in `x_history`

Targets for the readout:

```text
yt = y_data[init_len : train_length]   # shape (1, train_length - init_len)
```

`w_out` has shape `(outsize, 1 + insize + ressize)` — one linear combination of bias, input, and all reservoir units.

---

## Testing phase (autoregressive)

Continues from the **trained** reservoir state `x` (not reset).

For each test step:

1. Same reservoir update as in training
2. `y = w_out @ [1; u; x]`
3. **`u = y`** for the next step (closed-loop prediction; ground-truth `data` is not fed in during the rollout)

The script prints shapes of `x_history`, `w_out`, and `Y`, plus a scalar summary over the test window at the end.

---

## Relation to NLRC (same repo family, different paradigm)

Do not confuse this file with **`NLRC/base_version/code.py`**:

| | Classic RC (this folder) | NLRC (`base_version`) |
|--|--------------------------|------------------------|
| Dynamics | Large random `W`, recurrent state `x` | No `W`; `n` chained GLS/tent iterations on scalar `u` |
| Feature size | `1 + insize + ressize` (301 here) | `n + 1` (e.g. 11) |
| Nonlinearity | Sigmoid + tent on reservoir pre-activations | Repeated `gls_neuron_gen` on a scalar chain |
| Scaling | None in this script | `MinMaxScaler` on train/test |

This project is **ESN-style RC with a changed activation**, not the lightweight NLRC feature extractor—though the tent map itself is mathematically akin to the GLS neuron map.

---

## File layout

| File | Purpose |
|------|---------|
| `code.py` | Full pipeline: data, reservoir, train, autoregressive test |
| `README.md` | This document |

---

## Quick reference: original vs modified update (code locations)

**Original** (`rc_with_mackayglass.py`, training loop):

```python
x = x*(1-a) + a*np.tanh(np.dot(W,x) + np.dot(W_in, np.vstack((1,u))))
```

**This repo** (`code.py`, training and test loops):

```python
r = np.dot(W, x) + np.dot(W_in, np.vstack((1, u)))
r = 1 / (1 + np.exp(-r))
r = tentmap(r)
x = x*(1-a) + a*r
```

Everything else in the Mackey–Glass pipeline is intentionally kept parallel so comparisons isolate the **activation function change**.
