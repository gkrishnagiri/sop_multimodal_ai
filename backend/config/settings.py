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

    # OCR settings
    ocr_frame_stride: int = 1
    ocr_continue_on_frame_error: bool = True
    ocr_min_confidence: float = 0.0
    ocr_fail_pipeline_if_all_frames_fail: bool = False

    # Diarization settings
    # 0 or None means full-length diarization.
    diarization_max_duration_seconds: int | None = 0

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()