# Common entrypoints for restorecoins. Install `just` via `brew install just`.

default:
    @just --list

# --- Setup ---------------------------------------------------------------

setup-py:
    python3.12 -m venv .venv
    .venv/bin/pip install -U pip
    .venv/bin/pip install -r training/requirements.txt
    .venv/bin/pip install -r backend/requirements.txt

setup-fe:
    cd frontend && npm install

# --- Tests / quality ----------------------------------------------------

test:
    .venv/bin/python -m pytest training/tests -v

build-fe:
    cd frontend && npm run build

# --- Backend ------------------------------------------------------------

backend-mock:
    RESTORECOINS_MODE=mock .venv/bin/uvicorn backend.app:app --reload --port 7860

backend-real:
    RESTORECOINS_MODE=real .venv/bin/uvicorn backend.app:app --reload --port 7860

# --- Frontend -----------------------------------------------------------

dev-fe:
    cd frontend && VITE_BACKEND_URL=http://127.0.0.1:7860 VITE_BASE=/ npm run dev

# --- Data ---------------------------------------------------------------

scrape limit="50":
    .venv/bin/python -m training.scrape_ocre --limit {{limit}} --out data/raw

filter:
    .venv/bin/python -m training.scrape_ocre --filter --in data/raw --out data/filtered

fixtures n="20":
    .venv/bin/python -m training.fixtures --out data/fixtures --n {{n}}

# --- Training (long; needs a real GPU) ----------------------------------

train-lora data="data/filtered":
    accelerate launch -m training.train_lora --data_root {{data}} --output_dir runs/lora-latest

# MPS-friendly: 100-step smoke test on M-series Mac to verify the loop works.
train-lora-mps-smoke data="data/wikimedia_raw":
    .venv/bin/python -m training.train_lora --device mps --steps 100 --rank 16 \
        --resolution 512 --grad_accum 1 --save_every 100 \
        --data_root {{data}} --output_dir runs/lora-mps-smoke

# Full MPS training run — multi-hour, only after the smoke passes.
train-lora-mps data="data/wikimedia_raw":
    .venv/bin/python -m training.train_lora --device mps --steps 2000 --rank 16 \
        --resolution 768 --grad_accum 4 --save_every 500 \
        --data_root {{data}} --output_dir runs/lora-mps-latest

train-mask data="data/filtered":
    .venv/bin/python -m training.train_mask_proposer --data_root {{data}} --output_dir runs/mask-latest

# --- Eval ---------------------------------------------------------------

eval backend="http://localhost:7860" set="data/filtered":
    .venv/bin/python -m training.evaluate --backend {{backend}} --eval_set {{set}} --output runs/eval-latest

# --- Smoke ---------------------------------------------------------------

smoke:
    just backend-mock &
    sleep 2 && curl -sf http://127.0.0.1:7860/version | jq .
    kill %1
