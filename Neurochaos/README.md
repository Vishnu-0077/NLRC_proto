# Neurochaos

A classification pipeline that uses **Generalized Lipschitz System (GLS) neurons** to turn raw inputs into symbolic, chaos-inspired features, then trains a **Support Vector Machine (SVM)** on those features to classify Iris flowers.

## Overview

The script does not feed raw Iris measurements directly into the classifier. Instead, each scaled feature value is driven through a GLS neuron until it settles near a **reference target** `y`. Along the way, the neuron emits a **binary symbolic sequence** (0s and 1s). From that sequence we compute a **TTSS** (time-to-symbolic-state style) statistic: the fraction of steps that were `1`. This is repeated for several reference targets and all features, producing a high-dimensional representation that the SVM learns from.

```
Raw Iris data → MinMax scale → GLS iteration per (sample, feature, target) → TTSS features → SVM → class prediction
```

## Results

**Accuracy: 94%**

On the Iris test split (80% of data held out for testing), the pipeline reaches **94%** classification accuracy using GLS feature extraction plus SVM.

---

## Data preparation

At the bottom of `code.py`, the Iris dataset is loaded and split:

| Step | What happens |
|------|----------------|
| `load_iris()` | 150 samples, 4 features (sepal/petal length and width), 3 classes |
| `train_test_split(..., test_size=0.8)` | 20% train (30 samples), 80% test (120 samples) |
| `MinMaxScaler` | Each feature scaled to `[0, 1]` so GLS updates stay in a bounded range |
| `y_target = np.linspace(0.1, 0.9, 10)` | Ten fixed reference values used as attractor states during GLS iteration |

The reference targets are evenly spaced between 0.1 and 0.9. Each one defines a separate “view” of how each input value converges toward that reference.

---

## Core functions

### `gls_neuron_gen(X, b, ss)` — one GLS step (skew binary method)

This is the atomic update of a single scalar value `X` (one cell in the feature matrix).

- **`b`**: threshold, fixed at `0.5` in `iterate_gls_neuron`
- **`ss`**: list that accumulates the binary symbolic output over iterations

**Behavior:**

1. If `X >= b`: append `1` to `ss`, return `(1 - X) / (1 - b)` — maps the upper half toward 1
2. If `X < b`: append `0` to `ss`, return `X / b` — maps the lower half toward 0

So each step both **records a symbol** (0 or 1) and **transforms** `X` for the next step. The piecewise maps are the “skew binary” dynamics: values above the threshold are pulled upward in normalized form; values below are pulled toward 0.

---

### `iterate_gls_neuron(x, y, eps)` — TTSS for one reference target

For a full 2D input matrix `x` (samples × features) and a single reference `y`:

1. For every sample `i` and feature `j`, start with `val = x[i][j]` and an empty sequence `ss`.
2. **Iterate** while `val` is **outside** the band `[y - eps, y + eps]` and `N < 1000`:
   - Increment `N`
   - Set `val = gls_neuron_gen(val, b=0.5, ss)`
3. When the loop stops, count how many `1`s appeared in `ss`: `h = ss.count(1)`.
4. Store **TTSS** at `ttss[i][j] = h / N` (or `0` if `N == 0`).

**Interpretation:** TTSS is the **proportion of GLS steps that fired “high” (symbol 1)** before the value entered the ε-neighbourhood of `y`. Values that converge quickly with many 1s get a different signature than values that linger with mostly 0s.

**Parameters:**

- **`eps`**: width of the convergence band around `y` (default `0.1` in `main()`)
- **`max_iter`**: safety cap of 1000 iterations per cell

---

### `ttss_allocations(x, y_set, eps)` — TTSS for all reference targets

`y_set` is the list of reference targets (the ten values from `np.linspace(0.1, 0.9, 10)`).

For each `y` in `y_set`, the code calls `iterate_gls_neuron(x, y, eps)` and stacks the results. The output shape is `(num_targets, num_samples, num_features)` — one TTSS matrix per reference.

---

### `mean_representation(final_ttss, y_lst, m, n)` — optional aggregation

In `main()`, this function is called but its result `final_M` is **not** passed to the SVM; the pipeline instead reshapes the full TTSS tensor.

For each reference index `i` and feature column `j`, it takes the **mean of TTSS across all samples** for that target and feature. That yields a `(num_targets × num_features)` summary matrix. It is useful if you want a compact global descriptor per target rather than per-sample features.

---

### `model_train(X_train, y_train)` — SVM classifier

Wraps `sklearn.svm.SVC` in a `Pipeline` and fits on the engineered TTSS features and Iris class labels (`y_train`).

---

## `main()` — end-to-end training and evaluation

```python
eps = 0.1

# Training features
train_ttss = ttss_allocations(X_train, y_target, eps).transpose(1, 0, 2)
final_M = mean_representation(train_ttss, y_target, len(y_target), b)  # computed, not used for SVM
train_ttss = train_ttss.reshape(len(X_train), len(y_target) * b)

# Test features (same eps and y_target, no separate fit)
test_ttss = ttss_allocations(X_test, y_target, eps).transpose(1, 0, 2).reshape(len(X_test), len(y_target) * b)

model = model_train(train_ttss, y_train)
y_pred = model.predict(test_ttss)
return accuracy_score(y_test, y_pred)
```

**Shape logic:**

- After `ttss_allocations`: `(10 targets, 30 train samples, 4 features)`
- After `.transpose(1, 0, 2)`: `(30, 10, 4)` — samples first
- After `.reshape(..., 10 * 4)`: `(30, 40)` — each sample is a **40-dimensional** vector (10 targets × 4 features)

The test set is transformed the same way into `(120, 40)` features. The SVM sees **40 TTSS-derived features per flower**, not the original 4 measurements.

Running `python code.py` prints the accuracy from `main()`.

---

## `main_2()` — epsilon sweep (optional)

`main_2()` is not called when you run the script as-is; it is there for experiments. It tries several neighbourhood widths:

`epsilon = [0.2, 0.1, 0.05, 0.02, 0.01]`

For each `eps`, it builds train/test TTSS features (with reshape only, no `mean_representation`), trains an SVM, and stores rounded test accuracy in a dictionary. Smaller `eps` means tighter convergence to each reference `y`, which can change TTSS values and thus accuracy.

---

## Requirements

- Python 3
- `numpy`
- `scikit-learn`

```bash
pip install numpy scikit-learn
```

## Usage

```bash
python code.py
```

This executes `print(main())` and reports test accuracy (94% for the reported run).

## Files

| File      | Description |
|-----------|-------------|
| `code.py` | GLS neuron feature extraction, TTSS construction, SVM training and evaluation |
| `README.md` | Project and code documentation |
