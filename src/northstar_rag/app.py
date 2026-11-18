"""Command-line entry point: `uv run northstar-rag` prints how to start the app."""

UI_PATH = "src/northstar_rag/ui/streamlit_app.py"


def main() -> None:
    """Entry point for the `northstar-rag` command."""
    print("NorthStar Bank Assistant")
    print(f"Start the chat app with:  uv run streamlit run {UI_PATH}")


if __name__ == "__main__":
    main()
