# Reservoir Computing (Echo State Networks)

This folder contains two Python scripts that implement **Echo State Networks (ESN)** — a form of reservoir computing — for one-step-ahead prediction on chaotic time series. Both scripts share the same core pipeline (random reservoir, leaky integrator dynamics, ridge regression for the readout, then autoregressive testing).

## Requirements

- Python 3
- `numpy`
- `scipy`
- `pandas` (Lorenz script only, for CSV loading)

Install dependencies:

```bash
pip install numpy scipy pandas
```

## Shared approach

Each script:

1. Builds a reservoir of size `ressize` with random input weights `W_in` and recurrent weights `W`.
2. Scales `W` so its spectral radius is **0.95** (stability / echo-state property).
3. Updates the reservoir state with a **leaky integrator** (`a = 0.3`):

   `x ← (1 - a)·x + a·tanh(W·x + W_in·[1, u])`

4. During **training**, collects augmented states `[1, u, x]` after a washout period `init_len`.
5. Fits output weights `w_out` via **ridge regression** (`reg = 1e-8`) to predict the next target from reservoir states.
6. During **testing**, runs `test_length` steps in **closed loop**: the predicted output is fed back as the next input (details differ per script).
7. Reports **mean squared error (MSE)** on the test segment: average of `(true - predicted)²`.

---

## `rc_with_lorenz.py`

Predicts the Lorenz attractor time series from **external CSV data**.

### Data

- Reads `traindata.csv` (path is hardcoded: `/home/vishnu/Downloads/traindata.csv`).
- Uses rows **1000–65000** (drops the first 1000 samples as burn-in).
- Input `u` is a **3-dimensional** vector per timestep (`insize = 3`).
- Target `y` is the **next value of the first column** (`y_data = data shifted by one step on column 0`).

### Hyperparameters

| Parameter        | Value  |
|-----------------|--------|
| Reservoir size  | 100    |
| Input size      | 3      |
| Init washout    | 100    |
| Train length    | 2000   |
| Test length     | 100    |
| Leak rate `a`   | 0.3    |

### Testing (closed loop)

After training, prediction starts at `train_length`. For each test step:

- The network predicts `y` from `[1, u, x]`.
- The next input is built from the **prediction** and **two true components** from the CSV:

  `u = [y_pred, data[t+1, col1], data[t+1, col2]]`

So only the first dimension is predicted autoregressively; the other two channels still come from the dataset during the test window.

### Run

```bash
python rc_with_lorenz.py
```

Update the CSV path in the script if your data is elsewhere. The script prints `x_history` and `w_out` shapes and the test **MSE**.

---

## `rc_with_mackayglass.py`

Predicts the **Mackey–Glass** delay differential equation time series, generated **inside the script** (no external file).

### Data generation

- **`mackey_glass()`** — integrates the Mackey–Glass equation with default parameters (`tau=17`, `beta=0.2`, `gamma=0.1`, `n=10`, `dt=1.0`, length 2000). Initial segment `x[:tau+1] = 1.2`.
- **`test_data()`** — logistic map generator (present in the file but **not used** by the main script).
- Main series: `data = mackey_glass()`, target `y_data = data[1:]` (one-step-ahead on the scalar series).

### Hyperparameters

| Parameter        | Value  |
|-----------------|--------|
| Reservoir size  | 300    |
| Input size      | 1      |
| Init washout    | 50     |
| Train length    | 1000   |
| Test length     | 100    |
| Leak rate `a`   | 0.3    |

### Testing (fully autoregressive)

During testing, the next input is **only** the previous prediction:

`u = y`

The model must sustain the Mackey–Glass dynamics from its own outputs alone.

### Run

```bash
python rc_with_mackayglass.py
```

Prints a small preview of `x_history` (first 10×10 via pandas), shapes, and test **MSE**.

---

## Comparison

| Aspect              | Lorenz script              | Mackey–Glass script        |
|---------------------|----------------------------|----------------------------|
| Data source         | CSV file                   | Synthetic `mackey_glass()` |
| Input dimension     | 3                          | 1                          |
| Reservoir size      | 100                        | 300                        |
| Test feedback       | Mixed (pred + true dims)   | Fully autoregressive       |

Both scripts fix the random seed (`np.random.seed(42)`) for reproducible reservoir weights.

## Notes

- Change `train_length`, `test_length`, `ressize`, `init_len`, or `reg` in either file to experiment with capacity and overfitting.
- The Lorenz script depends on a valid CSV path and column layout (at least 3 numeric columns).
- MSE is printed to stdout; there is no plotting or model saving in either script.
