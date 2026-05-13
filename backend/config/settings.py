from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "SOP Multimodal AI"
    data_dir: str = "data"
    videos_dir: str = "data/videos"
    hf_token: str | None = None

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()