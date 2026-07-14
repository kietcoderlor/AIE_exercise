from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    model_name: str
    anthropic_api_key: str


def main() -> None:
    settings = Settings()
    print(f"Model Name from Settings: {settings.model_name}")


if __name__ == "__main__":
    main()

