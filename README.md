[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AliHaSSan-13/kev-server/blob/main/README.md)

# Kev Server

A [FastAPI](https://fastapi.tiangolo.com/) HTTP service for the
[Kev](https://github.com/jaredpalmer/kev) decision models. Runs on a Google Colab GPU runtime
or on any local machine with Python 3.12 / 3.13 — same three steps either way.

Kev is a family of small decision models built on Qwen3.5 by
[Jared Palmer](https://github.com/jaredpalmer). Give it a piece of text (`state`) and a set of
questions, and it answers them together, returning a probability distribution for every answer
— `noul` (yes/no), `choice` (pick an option) and `score` (pick a level on an ordered scale).

> **Going straight to Colab?** Jump to [Running on Google Colab](https://github.com/AliHaSSan-13/kev-server#running-on-google-colab) —
> the same install-and-run flow, plus which runtime to pick and what the free GPU can handle.
>
> The badge at the top opens this page as a notebook, already connected to a Colab runtime, so
> you can add a cell and paste the quickstart instead of copying the clone URL by hand.

This repository is a **third-party wrapper**. It is not affiliated with, endorsed by, or
supported by the Kev authors. Kev itself is Apache-2.0 and is cloned from GitHub at setup time;
none of its code is redistributed here.

---

## Quickstart

```bash
git clone https://github.com/AliHaSSan-13/kev-server-colab.git
cd kev-server-colab
bash setup.sh
```

That is the whole thing: clone kev, drop this project's files in, install, download the model,
start the server on `http://127.0.0.1:8000`.

`setup.sh` does this, in order:

1. reports the Python, torch and GPU it found
2. clones `jaredpalmer/kev` into `./kev`
3. copies `main.py`, `requirements.txt` and this README into `./kev` (as `README_COLLAB.md`)
4. `cd kev` — everything from here runs inside the kev checkout, because `import kev` only
   resolves there
5. installs the requirements and upgrades `huggingface_hub`
6. creates `models/` and downloads `jaredpalmer/kev-0.8b` into `models/kev-0.8b`
7. starts the server on `http://127.0.0.1:8000` and stays in the foreground

kev has no root `main.py` and no root `requirements.txt`, so step 3 overwrites nothing upstream.

The script is idempotent — re-running it skips the clone and the model download:

```bash
cd kev-server-colab && bash setup.sh
```

### Step by step, if you prefer

```bash
git clone https://github.com/jaredpalmer/kev.git
cd kev
# copy main.py, requirements.txt and setup.sh from this repository into kev/
bash setup.sh
```

The script detects that it is already inside a kev checkout and skips steps 2 and 3. Under the
hood it is exactly the manual sequence:

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

> **`torch` is left open-ended on purpose.** kev's `pyproject.toml` caps it at `<2.9`, but
> Colab ships a preinstalled CUDA build and honouring that cap means a multi-GB reinstall. If a
> future torch release breaks kev on your machine, pin it: `pip install "torch>=2.6,<2.9"`.

---

## Requirements

- Python 3.12 or 3.13 (kev declares `>=3.12,<3.14`)
- `git` and `pip`
- About 5 GB of disk for the checkpoint and the base model
- A GPU is optional. Without one the service still runs, on CPU, slowly.

The first run downloads the model from the Hugging Face Hub and caches it, so every run after
that works offline. Set `HF_HUB_OFFLINE=1` to assert that.

## Models

| Model | Base model | Disk | Notes |
|---|---|---|---|
| `kev-0.8b` (default) | Qwen3.5-0.8B-Base | ~5 GB | The one to use unless you have a reason not to. |
| `kev-4b` | Qwen3.5-4B-Base | ~12 GB | Several accuracy points better, needs a real GPU. |

```bash
KEV_MODEL=kev-4b bash setup.sh
```

Accuracy rises with size — see the
[upstream model table](https://github.com/jaredpalmer/kev#models) for the numbers against the
community Decision Index.

## Configuration

All optional, all environment variables read by `main.py`:

| Variable | Default | Meaning |
|---|---|---|
| `KEV_MODEL` | `kev-0.8b` | Checkpoint directory under `models/` |
| `KEV_DEVICE` | `cuda` if available, else `cpu` | Force a device |
| `KEV_HOST` | `127.0.0.1` | Bind address. Use `0.0.0.0` to accept traffic from other machines |
| `KEV_PORT` | `8000` | Port |
| `KEV_API_KEY` | generated on first run | Value required in the `X-API-Key` header |

### The API key

`main.py` never stores a key in git. It looks in this order:

1. the `KEV_API_KEY` environment variable,
2. a Colab secret named `KEV_API_KEY`,
3. a key it generates on first run and writes to `.api_key` (gitignored).

If none is set, the key is printed once at startup. To pin your own:

```bash
export KEV_API_KEY="choose-something-long-and-random"
```

On Colab, add it as a secret instead — the **🔑 Secrets** panel in the left sidebar, named
`KEV_API_KEY`.

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
{ "status": "ok", "model": "/path/to/kev/models/kev-0.8b", "device": "cuda" }
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
  "model": "/path/to/kev/models/kev-0.8b",
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

Treat `probabilities` as a ranking and routing signal, not a calibrated real-world probability.
Threshold them on your own data before you automate anything.

### Question types

**`noul`** — binary. `criteria` is optional: `{"true": "...", "false": "..."}`.

```json
{ "type": "noul", "instructions": "Is this ticket about billing?" }
```

**`choice`** — one option out of 1–255. `criteria` is a required object of `name -> description`.

```json
{ "type": "choice", "instructions": "Which team should handle this?",
  "criteria": { "returns": "...", "shipping": "...", "billing": "..." } }
```

**`score`** — one level out of 1–255, ordered lowest to highest. `criteria` is a required list.

```json
{ "type": "score", "instructions": "How urgent is this ticket?",
  "criteria": ["can wait", "this week", "today"] }
```

Questions are answered independently — they never see each other's answers — so you can ask all
of them in one request.

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

## Running on a local machine

Nothing extra is required — `bash setup.sh` installs into whatever Python is on your `PATH`. Two
things worth doing first:

**Use a virtualenv**, so the install doesn't touch your system packages:

```bash
python3 -m venv .venv && source .venv/bin/activate
bash setup.sh
```

**On a CPU-only Linux box, install torch from the CPU index first**, otherwise pip pulls
several GB of CUDA wheels you cannot use:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
bash setup.sh
```

`requirements.txt` asks for `torch>=2.6`, so the CPU build already installed is kept.

On a CUDA machine the default PyPI wheel is what you want. For Qwen3.5 bases on CUDA, kev's
own docs recommend `pip install flash-linear-attention` for throughput.

Reach the server from another machine on your network with `KEV_HOST=0.0.0.0 bash setup.sh` —
and read the [ngrok section](https://github.com/AliHaSSan-13/kev-server#exposing-the-api-with-ngrok) first if the machine is not yours.

## Running on Google Colab

Same `bash setup.sh`, GPU instead of CPU. Two extra things:

**Pick a GPU runtime.** Runtime → Change runtime type → GPU. Free Colab gives a T4, which runs
`kev-0.8b` comfortably; `kev-4b` is slow on a T4 and close to the VRAM limit, so use Colab Pro
(A100) if you want the 4B model.

**Use the Terminal panel**, at the bottom left. It is the easiest place to run a long-lived
server, and it gives you a second tab for ngrok later:

```bash
git clone https://github.com/AliHaSSan-13/kev-server-colab.git
cd kev-server-colab
bash setup.sh
```

Leave that tab running. Open a second tab (`+` in the Terminal panel) for anything else —
including [ngrok](https://github.com/AliHaSSan-13/kev-server#exposing-the-api-with-ngrok).

Free Colab notes:

- The runtime is ephemeral. When it resets, clone and run `setup.sh` again; the model
  re-downloads unless it is still on disk.
- Idle sessions are suspended, and the GPU is not guaranteed. If the model moves to CPU,
  requests take minutes instead of milliseconds.
- Colab's disk is limited. `du -sh kev/models ~/.cache/huggingface` shows where the ~5 GB went;
  `rm -rf kev/models/*` clears it (it re-downloads).

---

## Troubleshooting

**`GPU: none detected - this will run on CPU and be very slow.`**
Colab: Runtime → Change runtime type → GPU, then restart the session. Locally: check
`python -c "import torch; print(torch.cuda.is_available())"` inside the same environment the
server runs in.

**`torch.cuda.OutOfMemoryError`**
Drop to `KEV_MODEL=kev-0.8b`, shorten the `state`, or ask fewer questions per request.

**`ModuleNotFoundError: No module named 'kev'`**
`main.py` has to run from the root of a kev checkout, where the `kev/` package lives. The script
handles this by copying itself into `kev/` and running from there. If you moved `main.py`
elsewhere, `cd` back or re-run `bash setup.sh` from this repository's root.

**`error: the kev package is not importable from ...`**
The clone in `./kev` is incomplete or the `kev/` package is missing from it. Remove it and let
the script redo it: `rm -rf kev && bash setup.sh`.

**`error: python 3 not found on PATH`**
Install Python 3.12 or 3.13, or point at it: `PY=...` is not configurable, but
`python3.12 -m venv .venv` then `bash setup.sh` from the activated venv works.

**Transformers, tokenizers or datasets fail to import after install**
`huggingface_hub` is 2.x. `pip install "huggingface_hub<2.0"` — `setup.sh` already does this,
but a manual install from the step-by-step section can skip it.

**Disk full**
`du -sh kev/models ~/.cache/huggingface`, then `rm -rf kev/models/*`. Both re-download on the
next run.

**Want kev's own server instead?**
Upstream ships one: `uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009`,
with a `POST /v1/systemone` endpoint matching the TypeSafe System One API, so the TypeSafe
Python SDK works against it unchanged. This project is a smaller, dependency-light alternative
with a `/predict` endpoint — not a replacement.

---

## Exposing the API with ngrok

Works the same locally and on Colab: the service already listens on `127.0.0.1:8000`, and ngrok
forwards a public URL to it. `ngrok http 8000` needs no change to `KEV_HOST` — the tunnel
terminates at ngrok's edge and forwards to your loopback interface.

Leave the server running. Open a **second terminal tab** (Colab: the Terminal panel at the
bottom left, `+` for a new tab) and put ngrok there, so the two are independent processes and
restarting either leaves the other alone.

```bash
pip install -q pyngrok
```

Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken and add it once:

```bash
ngrok config add-authtoken YOUR_AUTHTOKEN
```

Then open the tunnel:

```bash
ngrok http 8000
```

ngrok prints the public URL in its dashboard:

```
Forwarding                    https://a1b2c3d4-5678.ngrok-free.app -> http://localhost:8000
```

Copy that URL into whatever you want to call it from:

```python
import requests

PUBLIC_URL = "https://a1b2c3d4-5678.ngrok-free.app"
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

Or from the terminal. In the kev directory, the key `main.py` generated is in `.api_key`:

```bash
cd kev-server-colab/kev

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $(cat .api_key)" \
  -d '{
    "state": "The customer wants a refund for a late order.",
    "questions": {"urgent": {"type": "noul", "instructions": "Escalate this?"}}
  }'

# through the tunnel; health needs no key
curl -s https://a1b2c3d4-5678.ngrok-free.app/health
```

Press `Ctrl+C` in the ngrok tab to bring the tunnel down; the server keeps running in the other
tab.

> **Security.** The ngrok URL is on the public internet. Anyone who has it *and* your API key
> can spend your GPU. Keep `KEV_API_KEY` out of git, out of shared notebooks, and rotate it if
> the URL leaks. Colab also kills the runtime when you leave the tab idle long enough, which
> closes the tunnel with it.

---

## License

Apache-2.0. See [LICENSE](LICENSE).

Kev and the model weights are separate works by the Kev authors, also Apache-2.0, obtained from
[github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev) and
[huggingface.co/jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b). Neither is
included in this repository.

Thanks to [Jared Palmer](https://github.com/jaredpalmer) for Kev and the model cards.