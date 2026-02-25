# --- anchor: kernel_loader.load_kernel_object (new spec without :object) ---
"""Utilities for loading external kernels.

Kernel spec formats
-------------------
- File path (extension optional): "/abs/path/bcic_iv_2a_kernel" or "/abs/path/bcic_iv_2a_kernel.py"
- Module path: "my_kernels.bcic_iv_2a_kernel"

The module must expose a single kernel object named ``KERNEL`` which implements:
    apply(udataset, raw, row)
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from types import ModuleType
from typing import Any, Tuple


def _load_module_from_file(file_path: str) -> ModuleType:
    """Load a Python module from a file path."""
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Kernel file not found: {file_path}")

    module_name = f"eegunity_external_kernel_{abs(hash(file_path))}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import kernel file: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_module_from_import_path(module_path: str) -> ModuleType:
    """Load a Python module from an import path."""
    return importlib.import_module(module_path)


def _resolve_file_path(spec: str) -> str | None:
    """Resolve a kernel file path from a spec string.

    The spec may omit the '.py' extension.

    Returns
    -------
    str | None
        Absolute file path if spec points to a file, otherwise None.
    """
    # If user already passed a .py file path.
    if spec.endswith(".py") and os.path.isfile(spec):
        return os.path.abspath(spec)

    # If user omitted extension but points to an existing file.
    if os.path.isfile(spec):
        return os.path.abspath(spec)

    # Try adding ".py" (your requested new style).
    candidate = f"{spec}.py"
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)

    return None


def load_kernel_object(spec: str) -> Tuple[Any, str]:
    """Load a kernel object from a spec string.

    Parameters
    ----------
    spec
        Kernel spec string. Either a file path (extension optional) or a Python
        module import path.

    Returns
    -------
    (kernel, normalized_spec)
        The loaded kernel object and a normalized spec string.
        If spec is a file path, normalized_spec is the resolved absolute file path.
        If spec is a module path, normalized_spec is the original spec.

    Raises
    ------
    AttributeError
        If the module does not define ``KERNEL``.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Kernel spec must be a non-empty string.")

    spec = spec.strip()

    file_path = _resolve_file_path(spec)
    if file_path is not None:
        module = _load_module_from_file(file_path)
        kernel = getattr(module, "KERNEL")
        return kernel, file_path

    # Otherwise treat as module import path.
    module = _load_module_from_import_path(spec)
    kernel = getattr(module, "KERNEL")
    return kernel, spec