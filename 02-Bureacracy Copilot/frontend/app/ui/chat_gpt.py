"""
Feature-flag chat entrypoint.

Maintains the feature flag path while delegating to UI v3 chat renderer.
"""

from app.ui import chat


def render(client=None, show_history_sidebar: bool = False) -> None:
    """Render ChatGPT-like path using the same center chat panel.

    Args:
        client: Backend API client.
        show_history_sidebar: Reserved compatibility flag.
    """
    _ = show_history_sidebar
    chat.render(client=client)