# Leaky NLRC — Time Series Prediction

This project implements a **leaky Non-Linear Reservoir Computer (NLRC)** for one-step-ahead forecasting on chaotic time series. The implementation lives in `code.py` and combines:

1. **GLS neuron feature extraction** — nonlinear expansion of each input sample into a fixed-length feature vector.
2. **Memory constructor** — temporal leaky integration over those features (the main distinction from a standard NLRC).
3. **Ridge regression readout** — linear weights `w_out` map integrated features to the target.

The bundled demo fits on a logistic map, predicts a short test horizon, and reports MSE plus a plot of ground truth vs. prediction.

---

## Pipeline overview

```
input u  →  build_features (GLS chain)  →  memory_constructor (leaky EMA over time)
                                                    ↓
                              fit: ridge solve for w_out  |  predict: w_out · [1; X]
```

| Stage | Role |
|--------|------|
| `build_features` | At each time index, run the input through `n` GLS neuron iterations and stack `[1, x₀, …, xₙ₋₁]ᵀ` as one column of the feature matrix. |
| `memory_constructor` | Smooth feature columns across training time with leak factor `a`; store end-of-train state in `final_x`. |
| `memory_constructor_pred` | During inference, blend new instantaneous features with `final_x` and update state each step. |
| `fit` / `predict` | Train `w_out` on integrated features; autoregressively forecast using previous prediction as next input. |

---

## Memory constructor (detailed)

The memory constructor is the **leaky** part of this NLRC. Instead of using only the features at the current time step, it maintains a **running, exponentially weighted memory** of past feature vectors. That gives the readout access to information from earlier in the series without explicitly storing a sliding window of raw inputs.

### Training: `memory_constructor(history_x)`

**Input:** `history_x` with shape `(n + 1, T)` — output of `build_features`, one column per time step (row 0 is the bias `1`, rows `1…n` are GLS features).

**Output:** `x` with the same shape, where each column is a **leaky blend** of the current raw features and the previous integrated state.

Initialization and recurrence:

```python
x[:, 0] = history_x[:, 0]                    # first time: no prior memory

for i in range(1, T):
    x[:, i] = a * x[:, i-1] + (1 - a) * history_x[:, i]
```

Interpretation:

- **`a` (leak / memory strength):** In `[0, 1]`.  
  - **`a → 1`:** Strong memory — integrated features change slowly; past dominates.  
  - **`a → 0`:** Weak memory — `x[:, i] ≈ history_x[:, i]`; behaves like a memoryless NLRC.  
- **Row-wise update:** Every feature dimension (including the bias row in the loop, though readout uses `x[1:, :]`) is smoothed with the **same** scalar `a`, so all GLS channels share one leak timescale.
- **End state:** After the loop, the model stores  
  `self.final_x = x[1:, -1].flatten()`  
  i.e. the last column’s GLS features **without** the bias. This vector is the **memory state at the end of training** and is carried into prediction so train and test dynamics stay consistent.

Geometric view: for each feature index `k`, the sequence `{x[k, i]}` is an exponential moving average of `{history_x[k, i]}`. The constructor turns an instantaneous feature snapshot into a **low-pass filtered trajectory** over training time, which can stabilize learning on noisy or fast-varying chaotic signals.

### Inference: `memory_constructor_pred(X)`

At each test step, `build_features` is not applied across a full batch; instead a single input `u` produces an instantaneous feature vector `X` (length `n`). Prediction must apply the **same leaky rule** as during training, but only for one step at a time, using the state left from the previous step (or from the end of `fit`):

```python
X = (1 - a) * X.flatten() + a * self.final_x
self.final_x = X.flatten()
return X
```

This is the **one-step equivalent** of the training recurrence: new features are mixed with `final_x`, then `final_x` is updated for the next autoregressive step. The readout then uses `np.dot(w_out, [1, X])` in `predict`.

**Important:** `predict` feeds each predicted `y` back as the next `u`, so both the GLS chain and `final_x` evolve recursively on the test horizon. Errors can accumulate; `a` strongly affects how much past context vs. new input drives each step.

### How memory ties into `fit` and `predict`

```text
fit:
  history_x = build_features(data)           # (n+1) × train_len
  history_x = memory_constructor(history_x)  # leaky integration over train_len
  w_out = Y · history_xᵀ · (history_x · history_xᵀ + reg·I)⁻¹

predict (each step i):
  X ← GLS chain from current u
  X ← memory_constructor_pred(X)             # leak + update final_x
  y ← w_out · [1; X]
  u ← y for next i
```

Training integrates memory **over the entire training sequence** before solving for `w_out`. Inference integrates memory **one step at a time** while rolling forward. Consistency depends on using the same `a` and on initializing `final_x` from the last training column (done inside `memory_constructor` during `fit`).

---

## GLS neuron (`gls_neuron_gen`)

Piecewise nonlinear map on `(0, 1)` with threshold `b`:

- If `x ≥ b`: return `(1 - x) / (1 - b)`
- Else: return `x / b`

Inputs are clipped to `(ε, 1 - ε)` for numerical stability. Chaining `n` applications starting from the current sample yields the reservoir state used in features.

---

## Class `nlrc` — parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `a` | `0.3` | Leak factor for `memory_constructor` / `memory_constructor_pred`. |
| `b` | `0.5` | GLS neuron threshold. |
| `reg` | `1e-8` | Ridge regularization for `w_out`. |
| `n` | `10` | Number of GLS iterations (feature dimension is `n`). |
| `test_len` | `10` | Horizon length in `predict`. |

---

## Demo script (bottom of `code.py`)

The example uses a **logistic map** (`test_data`), MinMax scaling to `[0, 1]`, `train_len = 100`, `test_len = 10`, and `a = 0` in the demo model (no leak in that run — adjust `a` to enable memory). It prints MSE, plots original vs. predicted test values, and prints shapes of raw vs. memory-augmented feature matrices.

**Dependencies:** `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

**Run:**

```bash
python code.py
```

---

## Helper generators

- **`mackey_glass(...)`** — Mackey–Glass delay differential equation series (alternative benchmark; not used in the default demo).
- **`test_data(a, i, length)`** — Logistic map `x_{t+1} = a x_t (1 - x_t)`.

---

## Design notes (from inline comments)

1. GLS binary-trace-style values act as the feature extractor for the time series input.
2. There is no explicit spatial neighborhood; depth is controlled by a fixed number of GLS iterations `n`.
3. The training feature matrix (after memory construction) is used to compute `w_out`.
4. `w_out` drives recursive prediction: the previous output becomes the next input.

For reproducibility and experimentation, tune **`a`** first when studying memory effects, then **`n`**, **`b`**, and **`reg`**.
