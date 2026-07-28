"""Load classes straight out of ``references/*.py`` for parity testing.

The files under ``references/`` are immutable research artifacts (see
``CLAUDE.md``) and must never be edited -- not even to add an
``__init__.py`` or to strip their unrelated experimental imports
(matplotlib, pandas, torch, tqdm, hyperopt, TabZilla, ...). Several of
those imports are optional dependencies this test suite does not
otherwise need, so rather than installing a heavy ML stack just to be
able to `import` two classes, this loader stubs out any of those
optional modules that aren't already installed, only for the duration
of executing the reference file, then removes the stubs from
``sys.modules`` again -- leaving them in place would leak into other
tests (e.g. scikit-learn's own optional-pandas detection elsewhere in
the test session). The real module is used instead of a stub whenever
it is already importable, and is never touched or removed.

Nothing in ``references/`` is modified by this process.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

REFERENCES_DIR = Path(__file__).resolve().parent.parent / "references"


def _ensure_stub(name, attrs=None, newly_stubbed=None):
    """Insert a dummy module into ``sys.modules`` if it isn't importable.

    Records the names it actually inserted (as opposed to names that were
    already real/importable) into ``newly_stubbed`` so the caller can undo
    the insertion afterwards.
    """
    try:
        importlib.import_module(name)
        return
    except ImportError:
        pass

    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        sub_name = ".".join(parts[:i])
        if sub_name in sys.modules:
            continue
        stub = types.ModuleType(sub_name)
        sys.modules[sub_name] = stub
        if newly_stubbed is not None:
            newly_stubbed.append(sub_name)
        if i > 1:
            setattr(sys.modules[".".join(parts[: i - 1])], parts[i - 1], stub)

    if attrs:
        target = sys.modules[name]
        for attr in attrs:
            if not hasattr(target, attr):
                setattr(target, attr, lambda *a, **k: None)


def _stub_optional_experimental_dependencies():
    newly_stubbed = []
    _ensure_stub("matplotlib", newly_stubbed=newly_stubbed)
    _ensure_stub("matplotlib.pyplot", newly_stubbed=newly_stubbed)
    _ensure_stub("pandas", newly_stubbed=newly_stubbed)
    _ensure_stub("torch", newly_stubbed=newly_stubbed)
    _ensure_stub("torch.nn", newly_stubbed=newly_stubbed)
    _ensure_stub("tqdm", attrs=["tqdm"], newly_stubbed=newly_stubbed)
    _ensure_stub(
        "hyperopt",
        attrs=["fmin", "tpe", "hp", "STATUS_OK", "space_eval"],
        newly_stubbed=newly_stubbed,
    )

    # scipy's array-API compatibility layer probes `torch.Tensor` (via
    # `issubclass`) as soon as it notices a `torch` entry in `sys.modules`,
    # which our bare stub above does not have -- give it a real (if inert)
    # class so that probe doesn't crash sklearn's own import chain while
    # the reference file is being executed.
    torch_stub = sys.modules.get("torch")
    if "torch" in newly_stubbed and not hasattr(torch_stub, "Tensor"):
        torch_stub.Tensor = type("Tensor", (), {})

    return newly_stubbed


def _load_module(filename):
    path = REFERENCES_DIR / filename
    module_name = f"_stripe_reference_{path.stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    newly_stubbed = _stub_optional_experimental_dependencies()
    dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # never write a __pycache__/ under references/
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = dont_write_bytecode
        # Stubs only need to exist while the reference file's own
        # `import ...` statements run; remove them again so they don't
        # leak into unrelated code (e.g. scikit-learn's pandas detection)
        # for the rest of the test session.
        for name in newly_stubbed:
            sys.modules.pop(name, None)

    return module


def load_linear_reference():
    """Return the ``references/linear_reference.py`` module, unmodified."""
    return _load_module("linear_reference.py")


def load_kernel_reference():
    """Return the ``references/kernel_reference.py`` module, unmodified."""
    return _load_module("kernel_reference.py")
