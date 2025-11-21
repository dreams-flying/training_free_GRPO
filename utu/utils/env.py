import importlib.metadata
import os

from dotenv import find_dotenv, load_dotenv

# Load .env file if it exists, but don't raise error if not found
# This makes the .env file optional while still supporting it when present
dotenv_path = find_dotenv(raise_error_if_not_found=False)
if dotenv_path:
    load_dotenv(dotenv_path, verbose=True, override=True)
else:
    # Warn user if .env file is not found
    import warnings
    warnings.warn(
        "No .env file found. Please create a .env file from .env.example template. "
        "See README.md for more details.",
        UserWarning
    )


class EnvUtils:
    @staticmethod
    def get_env(key: str, default: str | None = None) -> str | None:
        if default is None:
            res = os.getenv(key)
            if not res:
                raise ValueError(f"Environment variable {key} is not set")
            return res
        return os.getenv(key, default)

    @staticmethod
    def assert_env(key: str | list[str]) -> None:
        if isinstance(key, list):
            for k in key:
                EnvUtils.assert_env(k)
        else:
            if not os.getenv(key):
                raise ValueError(f"Environment variable {key} is not set")

    @staticmethod
    def ensure_package(package_name: str) -> None:
        try:
            importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            raise ValueError(f"Package {package_name} is required but not installed!") from None
