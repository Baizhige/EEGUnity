"""Imports for the optional LLM extra. Missing packages raise at call time."""


def require_optional(module_name: str, purpose: str):
    """Import an optional module or explain how to install the llm extra."""
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{purpose} needs the optional dependency {module_name}. "
            "Install it with: pip install 'eegunity[llm]'"
        ) from exc
