# Kev Decision API — Colab

A [FastAPI](https://fastapi.tiangolo.com/) service that exposes the
[Kev](https://github.com/jaredpalmer/kev) decision models over HTTP, built to run on a
free Google Colab GPU runtime.

Kev is a family of small decision models built on Qwen3.5 by
[Jared Palmer](https://github.com/jaredpalmer). Give it a piece of text (`state`) and a set
of questions, and it answers them together and returns a probability distribution for every
answer — `noul` (yes/no), `choice` (pick an option) and `score` (pick a level on an ordered
scale).

This repository is a **third-party wrapper**. It is not affiliated with, endorsed by, or
supported by the Kev authors. Kev itself is Apache-2.0 and is cloned from GitHub at setup
time; none of its code is redistributed here.

> **Scope:** this is the **Colab** version. It downloads the checkpoint from the Hugging Face
> Hub and runs on GPU when one is available. The local CPU-only variant, which repoints the
> checkpoint at a hand-downloaded Qwen base model, is out of scope and is not published here.

---

## Quickstart

One cell in a Colab notebook. Pick **Runtime → Change runtime type → GPU** first, then:

```bash
!git clone https://github.com/<owner>/<repo>.git && cd <repo> && bash setup_colab.sh
```

`setup_colab.sh` does everything, in this order:

1. reports the Python, torch and GPU it found
2. clones `jaredpalmer/kev` into `./kev`
3. copies `main.py`, `requirements.txt` and this README into `./kev` (`README_COLLAB.md`)
4. `cd kev` — everything from here runs inside the kev checkout, because `import kev` only
   resolves there
5. installs the requirements and upgrades `huggingface_hub`
6. creates `models/` and downloads `jaredpalmer/kev-0.8b` into `models/kev-0.8b`
7. starts the server on `http://127.0.0.1:8000`

The script is idempotent. Re-running it skips the clone and the model download, so it is safe
to use after a Colab runtime reset:

```bash
cd <repo> && bash setup_colab.sh
```

kev has no root `main.py` and no root `requirements.txt`, so step 3 overwrites nothing
upstream.

### Step by step, if you prefer

```bash
git clone https://github.com/jaredpalmer/kev.git
cd kev
# copy main.py, requirements.txt and setup_colab.sh from this repository into kev/
bash setup_colab.sh
```

The script detects that it is already inside a kev checkout and skips steps 2 and 3.

Under the hood, `setup_colab.sh` is exactly the manual sequence:

```bash
pip install --upgrade "huggingface_hub>=0.34,<2.0"
pip install -r requirements.txt
mkdir -p models
hf download jaredpalmer/kev-0.8b --local-dir models/kev-0.8b   # or kev-4b
python main.py
```

> **Why the `huggingface_hub` upper bound?** `pip install --upgrade huggingface_hub`
> on its own installs 2.x, and `transformers`, `tokenizers` and `datasets` all declare
> `huggingface-hub<2`. The result is a broken environment that only shows up as an import
> error at model load. `<2.0` still provides the `hf` CLI.

`torch` is left open-ended in `requirements.txt` on purpose. kev's `pyproject.toml` caps it
at `<2.9`, but Colab ships a preinstalled CUDA build, and honouring that cap means a
multi-GB reinstall. If a future torch release breaks kev on your runtime, pin it:

```bash
pip install "torch>=2.6,<2.9"
```

The script is idempotent — the model download is skipped if `models/kev-0.8b` already exists,
so re-running it is safe after a Colab runtime reset.

`requirements.txt` is safe to add next to kev's own `pyproject.toml`: kev does not ship a
`requirements.txt`, so nothing upstream is overwritten.

---

## Requirements

- Python 3.12 or 3.13 (Colab's default qualifies)
- A GPU runtime is strongly recommended. Free Colab gives a T4, which runs `kev-0.8b` comfortably
- About 5 GB of disk for the adapter and the base model

## Models

| Model | Base model | Disk | Notes |
|---|---|---|---|
| `kev-0.8b` (default) | Qwen3.5-0.8B-Base | ~5 GB | Fits the free T4. Use this one. |
| `kev-4b` | Qwen3.5-4B-Base | ~12 GB | Slow on a T4 and close to the VRAM limit. Colab Pro (A100) recommended. |

```bash
KEV_MODEL=kev-4b bash setup_colab.sh
```

Accuracy rises with size (Kev-4B is several points above Kev-0.8B on the community Decision
Index). See the [upstream model table](https://github.com/jaredpalmer/kev#models) for the
numbers.

## Configuration

All optional, all environment variables read by `main.py`:

| Variable | Default | Meaning |
|---|---|---|
| `KEV_MODEL` | `kev-0.8b` | Checkpoint directory under `models/` |
| `KEV_DEVICE` | `cuda` if available, else `cpu` | Force a device |
| `KEV_HOST` | `127.0.0.1` | Bind address. Use `0.0.0.0` for direct LAN access |
| `KEV_PORT` | `8000` | Port |
| `KEV_API_KEY` | generated on first run | Value required in the `X-API-Key` header |

### The API key

`main.py` never stores a key in git. It looks in this order:

1. the `KEV_API_KEY` environment variable,
2. a Colab secret named `KEV_API_KEY`,
3. a key it generates on first run and writes to `.api_key` (gitignored).

If none is set, the key is printed once at startup. To pin your own, add it as a Colab secret
(**🔑 Secrets** in the left sidebar, name it `KEV_API_KEY`) or set it before starting:

```python
import os
os.environ["KEV_API_KEY"] = "choose-something-long-and-random"
```

---

## API

Base URL: `http://127.0.0.1:8000`

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | none | Liveness, loaded model, device |
| `GET /docs` | none | Interactive Swagger UI |
| `POST /predict` | `X-API-Key` header | Answer a set of questions about a state |

### Health

```bash
curl http://127.0.0.1:8000/health
```

```json
{ "status": "ok", "model": "/content/kev/models/kev-0.8b", "device": "cuda" }
```

### Predict

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEV_API_KEY" \
  -d '{
    "state": "Shoes arrived two weeks late and in the wrong size. I also see two charges on my card.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "returns": "Exchanges, refunds, wrong or damaged items",
          "shipping": "Delivery status, delays, lost packages",
          "billing": "Charges, invoices, payment problems"
        }
      },
      "escalate": {
        "type": "noul",
        "instructions": "Does this need urgent human attention?"
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"]
      }
    }
  }'
```

```json
{
  "model": "/content/kev/models/kev-0.8b",
  "results": {
    "department": {
      "type": "choice",
      "choice": "returns",
      "probabilities": { "returns": 0.41, "shipping": 0.33, "billing": 0.26 },
      "logits": { "returns": 0.12, "shipping": -0.19, "billing": -0.44 }
    },
    "escalate": {
      "type": "noul",
      "choice": "true",
      "probabilities": { "false": 0.29, "true": 0.71 },
      "logits": { "false": -0.55, "true": 0.68 }
    },
    "frustration": {
      "type": "score",
      "choice": "1",
      "choice_label": "Frustrated",
      "probabilities": { "0": 0.12, "1": 0.55, "2": 0.33 },
      "logits": { "0": -1.02, "1": 0.31, "2": -0.09 }
    }
  },
  "latency_ms": 412,
  "input_tokens": 96,
  "inference_temperature": 2.35
}
```

| Field | Meaning |
|---|---|
| `choice` | Highest-probability option, or the option name |
| `choice_label` | The human-readable criterion, for `score` questions |
| `probabilities` | Softmax distribution over the options |
| `logits` | Raw pointer-head scores before softmax |
| `latency_ms` | Model time for the request |
| `input_tokens` | Tokens in the state and questions |
| `inference_temperature` | The calibration temperature fitted to this checkpoint |

Treat `probabilities` as a ranking and routing signal, not a calibrated real-world
probability. Threshold them on your own data before you automate anything.

### Question types

**`noul`** — binary. `criteria` is optional: `{"true": "...", "false": "..."}`.

```json
{ "type": "noul", "instructions": "Is this ticket about billing?" }
```

**`choice`** — one option out of 1–255. `criteria` is a required object of
`name -> description`.

```json
{ "type": "choice", "instructions": "Which team should handle this?",
  "criteria": { "returns": "...", "shipping": "...", "billing": "..." } }
```

**`score`** — one level out of 1–255, ordered lowest to highest. `criteria` is a required list.

```json
{ "type": "score", "instructions": "How urgent is this ticket?",
  "criteria": ["can wait", "this week", "today"] }
```

Questions are answered independently — they never see each other's answers — so you can ask
all of them in one request.

### From Python

```python
import os
import requests

BASE_URL = "http://127.0.0.1:8000"

response = requests.post(
    f"{BASE_URL}/predict",
    headers={"X-API-Key": os.environ["KEV_API_KEY"]},
    json={
        "state": "We are hiring a Python backend engineer to build FastAPI APIs.",
        "questions": {
            "relevant": {
                "type": "noul",
                "instructions": "Is this relevant to a Python backend consultancy?",
            }
        },
    },
    timeout=120,
)

print(response.json()["results"]["relevant"])
```

---

## Exposing the API with ngrok

`setup_colab.sh` ends by starting the server in the foreground, which blocks the cell. To put
it in the background instead, run it yourself from the kev checkout:

```python
%cd kev
!nohup python main.py > server.log 2>&1 &
```

Wait for the server, then open the tunnel:

```python
!pip install -q pyngrok

from pyngrok import ngrok, conf

# Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
authtoken = ""
conf.get_default().auth_token = authtoken

# Open HTTP tunnel to port 8000
public_url = ngrok.connect(8000).public_url
print(f"\nPublic API Base URL: {public_url}")
```

`ngrok.connect(8000)` works against the `127.0.0.1` bind address — the tunnel terminates at
ngrok's edge and forwards to your loopback interface, so you do not need `KEV_HOST=0.0.0.0`.

Call it from another notebook or project:

```python
import requests

PUBLIC_URL = "https://xxxx-xxxx-xxxx.ngrok-free.app"
KEY = "your KEV_API_KEY"

response = requests.post(
    f"{PUBLIC_URL}/predict",
    headers={"X-API-Key": KEY},
    json={
        "state": "The customer wants a refund for a late order.",
        "questions": {"urgent": {"type": "noul", "instructions": "Escalate this?"}},
    },
    timeout=120,
)
print(response.json()["results"]["urgent"]["probabilities"])
```

Health checks are unauthenticated, so you can also verify the tunnel with:

```bash
curl https://xxxx-xxxx-xxxx.ngrok-free.app/health
```

> **Security.** The ngrok URL is on the public internet. Anyone who has it *and* your API key
> can spend your Colab GPU. Keep `KEV_API_KEY` out of git, out of shared notebooks, and rotate
> it if the URL leaks. Stop the tunnel when you are done:
> `ngrok.kill()`. Colab also kills the runtime when you close the tab long enough, which closes
> the tunnel with it.

---

## Troubleshooting

**"GPU: none detected - this will run on CPU and be very slow."**
Runtime → Change runtime type → GPU, then restart the session. Free Colab T4 is enough for
`kev-0.8b`.

**`torch.cuda.OutOfMemoryError`**
Drop to `KEV_MODEL=kev-0.8b`, shorten the `state`, or use fewer questions per request.

**Disk full in Colab**
Colab gives a limited ephemeral disk. The checkpoint and the Qwen base model land in
`kev/models/` and the Hugging Face cache; `du -sh kev/models ~/.cache/huggingface` shows where
they went, and `rm -rf kev/models/*` clears them (they re-download on the next run).

**`ModuleNotFoundError: No module named 'kev'`**
`main.py` has to run from the root of a kev checkout, where the `kev/` package lives. The
script handles this by copying itself into `kev/` and running from there; if you moved `main.py`
somewhere else, `cd` back or re-run `bash setup_colab.sh` from this repository's root.

**`error: the kev package is not importable from ...`**
The clone in `./kev` is incomplete or the `kev/` package is missing from it. Remove and let the
script redo it: `rm -rf kev && bash setup_colab.sh`.

**Server dies between cells**
Free Colab suspends idle runtimes. Restart it and re-run `cd <repo> && bash setup_colab.sh`;
the clone and the model download are skipped.

**Want kev's own server instead?**
Upstream ships one: `uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009`,
with a `POST /v1/systemone` endpoint that matches the TypeSafe System One API, so the TypeSafe
Python SDK works against it unchanged. This project is a smaller, dependency-light alternative
with a `/predict` endpoint — not a replacement.

---

## License

Apache-2.0. See [LICENSE](LICENSE).

Kev and the model weights are separate works by the Kev authors, also Apache-2.0, obtained
from [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev) and
[huggingface.co/jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b). Neither is
included in this repository.

Thanks to [Jared Palmer](https://github.com/jaredpalmer) for Kev and the model cards.