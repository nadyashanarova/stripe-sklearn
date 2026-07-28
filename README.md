# Stripe Learning

Stripe Learning is a Python implementation of Yakubovich's Stripe
algorithms for binary classification with a scikit-learn compatible
API.

The project provides both linear and kernelized implementations of the
L-system and C-system classifiers, while preserving the mathematical
behaviour of the original research implementation.

Unlike gradient-based classifiers, Stripe algorithms are
**projection-based**: parameters are corrected only when a constraint
is violated, using a finite-update projection step rather than
iterative gradient descent. This gives training dynamics that don't
depend on a learning rate or a learning-rate schedule.

## What it implements

Yakubovich formulated learning as satisfying a system of linear
inequalities ("hyperslab" constraints) around a target hyperplane, in
two complementary ways:

- **L-system** — constraints derived from a relaxed stationarity
  condition of a regularized quadratic loss.
- **C-system** — pointwise tolerance constraints directly on the
  prediction error, with a three-zone ("multizone") correction rule.

Both formulations are provided in a linear form and a kernelized form
(via an RBF or linear kernel, operating on kernel expansion
coefficients instead of raw features), for four estimators in total:

| Estimator | Formulation | Feature space |
|---|---|---|
| `StripeLClassifier` | L-system | linear |
| `StripeCClassifier` | C-system | linear |
| `StripeLKernelClassifier` | L-system | kernelized (RBF / linear kernel) |
| `StripeCKernelClassifier` | C-system | kernelized (RBF / linear kernel) |

All four are binary classifiers, matching the scope of the original
research (see `references/paper.pdf` for the full mathematical
treatment).

## Installation

The package isn't published on PyPI yet; install it from a local clone:

```bash
git clone <repository-url>
cd stripe-kernel
pip install .
```

For development (editable install, plus test dependencies):

```bash
pip install -e ".[test]"
```

Requirements: Python >= 3.9, NumPy >= 1.23, scikit-learn >= 1.3.

## Quickstart

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from stripe_learning import StripeLClassifier

X, y = make_classification(n_samples=200, n_features=10, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = StripeLClassifier(random_state=0)
clf.fit(X_train, y_train)

clf.predict(X_test[:5])
clf.decision_function(X_test[:5])
clf.score(X_test, y_test)
```

Every estimator also supports incremental learning:

```python
import numpy as np

clf = StripeLClassifier(random_state=0)
classes = np.unique(y_train)
clf.partial_fit(X_train[:50], y_train[:50], classes=classes)
clf.partial_fit(X_train[50:], y_train[50:])
```

and drops into standard scikit-learn tooling without any special
handling:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(StandardScaler(), StripeLClassifier(random_state=0))

search = GridSearchCV(pipe, param_grid={"stripelclassifier__lam": [0.0, 0.05, 0.1]})
search.fit(X_train, y_train)
```

See [`examples/basic_usage.py`](examples/basic_usage.py) for a runnable
walkthrough covering all four estimators, `partial_fit`, and
`Pipeline`/`GridSearchCV`.

> **Note:** Stripe algorithms use fixed absolute tolerances (`eps`,
> `delta_m`, ...), so — like most projection-based or margin-based
> methods — they are sensitive to feature scale. Scaling features
> (e.g. with `StandardScaler`) before fitting is recommended.

## Estimators

All four estimators implement the standard scikit-learn classifier
interface — `fit`, `partial_fit`, `predict`, `decision_function`,
`get_params`/`set_params`, and `sklearn.base.clone` support — and are
each documented in detail in their own docstring (`help(StripeLClassifier)`
etc.). Binary classification only, by design (matching the scope of
the underlying theory).

**`StripeLClassifier`** — linear L-system. Learns a weight vector by
correcting coordinates of a running regularized second-moment system
whenever a residual exceeds `delta_m`. Key parameters: `delta_m`,
`lam`, `mode`, `epochs_count`.

**`StripeCClassifier`** — linear C-system. Learns a weight vector via a
three-zone multizone correction applied per training sample. Key
parameters: `eps`, `delta_ratio`, `beta`, `lam`, `init_mode`.

**`StripeLKernelClassifier`** — kernelized L-system. Same correction
rule as `StripeLClassifier`, operating on a kernel expansion over an
accumulated support-point dictionary. Key parameters: `kernel`,
`gamma`, `delta_m`, `lam`.

**`StripeCKernelClassifier`** — kernelized C-system. Same correction
rule as `StripeCClassifier`, operating on the kernel Gram matrix. Key
parameters: `kernel`, `gamma`, `eps`, `delta_ratio`, `beta_param`.

One behavioral note worth calling out explicitly: on the two kernel
estimators, `partial_fit(X, y, classes=None)` both accumulates the
supplied samples into the kernel dictionary *and* runs one correction
epoch, in a single call — there's no separate "accumulate" step to
remember. Every other scikit-learn-standard usage pattern (a loop of
`partial_fit(X_batch, y_batch)` calls) works as expected.

## Project structure

```text
stripe-kernel/
├── src/stripe_learning/     # the package
│   ├── linear_l.py          # StripeLClassifier
│   ├── linear_c.py          # StripeCClassifier
│   ├── kernel_l.py          # StripeLKernelClassifier
│   └── kernel_c.py          # StripeCKernelClassifier
├── tests/                   # one test file per estimator
├── examples/
│   └── basic_usage.py       # runnable usage walkthrough
├── references/              # original research code + paper (read-only)
├── IMPLEMENTATION_PLAN.md   # extended technical documentation
├── MATH_AUDIT.md            # extended technical documentation
└── pyproject.toml
```

`references/` contains the original, untouched research implementation
and the paper the algorithms are based on. It's kept in the repository
for provenance: every estimator's test suite includes numerical parity
tests asserting that, for equivalent hyperparameters, the new
implementation matches this original code exactly.

## Design principles

- **Mathematical fidelity.** Every estimator's core update rule,
  normalization, regularization order, and sign conventions match its
  reference implementation exactly — verified by automated parity
  tests, not just by inspection.
- **Full scikit-learn compatibility.** Every estimator passes the
  applicable subset of `sklearn.utils.estimator_checks.check_estimator`
  and works with `Pipeline`, `GridSearchCV`, `cross_validate`, and
  `sklearn.base.clone`.
- **Consistent API across all four estimators.** The same constructor
  conventions, threshold handling, `partial_fit` semantics, and
  validation errors apply uniformly, even where the underlying
  reference implementations differed from each other.
- **No silent mathematical changes.** Where an estimator's behavior was
  changed relative to its reference implementation, the change is
  API/engineering-only (constructor shape, input validation, dead-code
  removal) and documented — never a silent change to a formula, sign,
  boundary, or update rule.

## Running the tests

```bash
pip install -e ".[test]"
pytest
```

Each estimator has its own test file (`tests/test_linear_l.py`,
`tests/test_linear_c.py`, `tests/test_kernel_l.py`,
`tests/test_kernel_c.py`) covering scikit-learn compatibility,
`Pipeline`/`GridSearchCV`/`cross_validate` integration, fitted-state
behavior, deterministic training, and numerical parity against the
reference implementation in `references/`.


## License

MIT — see [`LICENSE`](LICENSE).
