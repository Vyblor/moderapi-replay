"""Interactive TUI mode using Textual (3 screens).

Screens:
    1. Summary — overall gate result, per-attribute pass/fail
    2. Disagreements — texts where Perspective and Detoxify disagree most
    3. Threshold Explorer — adjust threshold and see agreement change live

Eng review: Windows fallback to HTML report if Textual unavailable.
"""

from __future__ import annotations


def launch_tui() -> None:
    """Launch the interactive TUI. Falls back to HTML on Windows/no-terminal."""
    try:
        from textual.app import App  # noqa: F401
    except ImportError:
        print("TUI requires the [tui] extra: pip install moderapi-replay[tui]")
        print("Falling back to HTML report...")
        return

    # TODO: Implement Textual TUI
    print("TUI not yet implemented — coming in Phase 1 Week 2")
