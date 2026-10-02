from pathlib import Path
from typing import Any
import os
import secrets
import torch

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field, ConfigDict

from kev.predictors import LocalPredictor


# ============================================================
# Configuration
# ============================================================

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

BASE_DIR = Path(__file__).resolve().parent

API_KEY_FILE = BASE_DIR / ".api_key"


def resolve_api_key() -> str:
    """
    Read the API key from, in order:

    1. the KEV_API_KEY environment variable,
    2. a Colab secret named KEV_API_KEY,
    3. a key generated on first run and saved to .api_key.

    The key is never stored in git: .api_key is gitignored.
    """

    key = os.environ.get("KEV_API_KEY")

    if not key:
        try:
            from google.colab import userdata

            key = userdata.get("KEV_API_KEY")
        except Exception:
            key = None

    if not key and API_KEY_FILE.exists():
        key = API_KEY_FILE.read_text().strip()

    if not key:
        key = secrets.token_urlsafe(24)
        API_KEY_FILE.write_text(key)
        print(
            f"No API key found. Generated one and saved it to {API_KEY_FILE}:\n\n"
            f"    {key}\n\n"
            "Set it as KEV_API_KEY (env var or Colab secret) to keep it stable.\n"
        )

    return key


API_KEY = resolve_api_key()

KEV_MODEL = BASE_DIR / "models" / os.environ.get("KEV_MODEL", "kev-0.8b")

DEVICE = os.environ.get("KEV_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu"
)

HOST = os.environ.get("KEV_HOST", "127.0.0.1")
PORT = int(os.environ.get("KEV_PORT", "8000"))


# ============================================================
# App
# ============================================================

app = FastAPI(
    title="Kev Decision API",
    description=(
        "Decision API powered by Kev. "
        "Uses CUDA when a GPU is available, otherwise CPU. "
        "Supports noul, choice, and score questions."
    ),
    version="1.0.0",
)


print("=" * 60)
print("Kev Decision API")
print("=" * 60)
print(f"Model:  {KEV_MODEL}")
print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU:    {torch.cuda.get_device_name(0)}")

print()


# ============================================================
# Load model ONCE
# ============================================================

print("Loading Kev-0.8B...")

CONTEXT = {
    "max_state": 4096,
    "max_branch": 8192,
    "max_packed": 12288,
    "truncate": False,
}

predictor = LocalPredictor(
    run=str(KEV_MODEL),
    device=DEVICE,
    context=CONTEXT,
)

print("Kev loaded.")
print()


# ============================================================
# Public API schemas
# ============================================================

class NoulQuestion(BaseModel):
    """
    Binary true/false decision.
    """

    type: str = Field("noul", pattern="^noul$")
    instructions: Any | None = None
    criteria: dict[str, Any] | None = None


class ChoiceQuestion(BaseModel):
    """
    Select one option from named choices.
    """

    type: str = Field("choice", pattern="^choice$")
    instructions: Any | None = None
    criteria: dict[str, Any] = Field(..., min_length=1)


class ScoreQuestion(BaseModel):
    """
    Select one ordered score level.
    """

    type: str = Field("score", pattern="^score$")
    instructions: Any | None = None
    criteria: list[Any] = Field(
        ...,
        min_length=1,
        max_length=255,
    )


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion


class PredictRequest(BaseModel):
    """
    SystemOne-style decision request.
    """

    model_config = ConfigDict(extra="forbid")

    state: Any
    model: str = "kev-latest"
    questions: dict[str, Question] = Field(
        ...,
        min_length=1,
    )


# ============================================================
# Response schemas
# ============================================================

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


# ============================================================
# Helpers
# ============================================================
def verify_api_key(x_api_key: str | None = Header(default=None)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )


def get_question_label(question: Question):
    if question.type == "noul":
        return 0

    if question.type == "choice":
        return next(iter(question.criteria.keys()))

    if question.type == "score":
        return 0

    raise ValueError(
        f"Unsupported question type: {question.type}"
    )


def build_kev_record(request: PredictRequest) -> dict:
    """
    Convert the public API request into the internal structure
    expected by Kev's predictor.
    """

    questions = {}

    for question_id, question in request.questions.items():

        q = {
            "type": question.type,
            "instructions": question.instructions,
            "label": get_question_label(question),
            "src": "api",
        }

        # Kev's to_record()/materialize() expects criteria for
        # choice and score questions.
        if question.criteria is not None:
            q["criteria"] = question.criteria

        questions[question_id] = q

    return {
        "state": request.state,
        "questions": questions,
    }


# ============================================================
# Health
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():

    return HealthResponse(
        status="ok",
        model=str(KEV_MODEL),
        device=DEVICE,
    )


# ============================================================
# Predict
# ============================================================

@app.post("/predict")
def predict(
    request: PredictRequest,
    _: None = Depends(verify_api_key),
):

    # Convert public request into Kev's internal format.
    record = build_kev_record(request)

    # Run Kev.
    result = predictor(record)

    results = {}

    for qid, question in request.questions.items():

        probabilities = result["probabilities"][qid]
        logits = result["logits"][qid]

        # Highest-probability option.
        choice = max(
            probabilities,
            key=probabilities.get,
        )

        output = {
            "type": question.type,
            "choice": choice,
            "probabilities": probabilities,
            "logits": logits,
        }

        # For score questions, also return the actual
        # human-readable criterion.
        if question.type == "score":

            index = int(choice)

            if 0 <= index < len(question.criteria):
                output["choice_label"] = question.criteria[index]

        results[qid] = output
        print(f"\n\n input_tokens: {result['input_tokens']},\n latency_ms: {result['latency_ms']},\n result: {results}\n\n")

    return {
        "model": str(KEV_MODEL),
        "results": results,
        "latency_ms": result["latency_ms"],
        "input_tokens": result["input_tokens"],
        "inference_temperature": result["inference_temperature"],
    }


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
    )
