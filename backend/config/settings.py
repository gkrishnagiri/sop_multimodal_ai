from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "SOP Multimodal AI"
    data_dir: str = "data"
    videos_dir: str = "data/videos"

    # Hugging Face
    hf_token: str | None = None

    # OpenAI / LLM
    openai_api_key: str | None = None
    openai_tracing: bool = False
    llm_model: str = "gpt-4o-mini"

    # Browser / future UI-agent settings
    browser_mode: str = "headed"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()