import os
from pathlib import Path

from dotenv import load_dotenv
from pyannote.audio import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"

# Load backend/.env explicitly
load_dotenv(PROJECT_ROOT / ".env")

hf_token = os.getenv("HF_TOKEN")

print("HF_TOKEN present:", bool(hf_token))

if not hf_token:
    raise RuntimeError(
        "HF_TOKEN is missing. Add HF_TOKEN=your_token_here to backend/.env"
    )

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=hf_token,
)

print("pyannote pipeline loaded ok")