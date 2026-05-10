"""
Central configuration loader.
Call load_env() once at application startup (FastAPI lifespan or CLI entry point).
Never call load_dotenv() in individual modules — it creates import-order dependencies.
"""
from dotenv import load_dotenv
from pathlib import Path


def load_env() -> None:
    """
    Load .env from project root.
    Safe to call multiple times — dotenv is idempotent.
    """
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
    # override=False: real env vars (set in shell/CI) take precedence over .env
    # This means your production environment can override .env values safely