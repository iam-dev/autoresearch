# MnemeBrain Sidecar Setup

This document covers how to install and run [mnemebrain-lite](https://pypi.org/project/mnemebrain-lite/) for the MnemeBrain experiment conditions (C and D).

## What is mnemebrain-lite?

mnemebrain-lite is a belief-state engine that provides:

- **BeliefMemory** — a structured memory store backed by [Kuzu](https://kuzudb.com/) graph database
- **Belnap four-valued logic** — beliefs can be TRUE, FALSE, BOTH (contradictory evidence), or NEITHER (no evidence)
- **Evidence-weighted confidence** — each piece of evidence carries a polarity (SUPPORTS/ATTACKS), weight, and reliability score
- **Semantic similarity search** — finds beliefs related to a query using sentence-transformer embeddings
- **Contradiction surfacing** — detects when accumulated evidence creates a BOTH state and supports contextual revision
- **FastAPI server** — optional HTTP API for remote access (used by Condition D)

The companion package [mnemebrain](https://pypi.org/project/mnemebrain/) (the SDK) provides a Python HTTP client that wraps the mnemebrain-lite API.

## Installation

### For Condition C (passive belief memory)

Condition C uses mnemebrain-lite as a direct Python library — no server needed.

```bash
# Apple Silicon or Linux (local sentence-transformer embeddings)
pip install "mnemebrain-lite[embeddings]"

# Intel Mac (no local PyTorch wheels available for x86_64 macOS)
pip install "mnemebrain-lite[openai]"
# Then set: export OPENAI_API_KEY=sk-...
```

The `[embeddings]` extra installs `sentence-transformers` for local embedding computation. The `[openai]` extra uses the OpenAI embeddings API instead, which avoids the need for a local PyTorch installation.

### For Condition D (active prediction/recommendation)

Condition D needs both the library and the SDK client:

```bash
# Apple Silicon or Linux
pip install "mnemebrain-lite[embeddings]" mnemebrain

# Intel Mac
pip install "mnemebrain-lite[openai]" mnemebrain
```

### Other install variants

```bash
pip install mnemebrain-lite                # core only (no embeddings — limited functionality)
pip install "mnemebrain-lite[all]"           # everything (API + embeddings + OpenAI)
```

## Running as library (Condition C)

Condition C uses `BeliefMemory` directly in-process. No server is required. The `PassiveHooks` class in `mnemebrain_hooks.py` handles this automatically:

```python
from mnemebrain_core.memory import BeliefMemory
from mnemebrain_core.providers.base import EvidenceInput

# CPU-only to avoid competing with training for GPU VRAM
memory = BeliefMemory(db_path="./results/beliefs", device="cpu")

# Record an observation as a belief
memory.believe(
    claim="matrix_lr=0.04 depth=4 achieved val_bpb=1.872",
    evidence=[EvidenceInput(
        source_ref="run_1",
        content="val_bpb=1.872, steps=1450, diverged=False",
        polarity="supports",
        weight=0.75,
        reliability=0.95,
    )],
)

# Query beliefs about a parameter
explanation = memory.explain(claim="matrix_lr=0.04 improves training")
# Returns: truth_state, supporting_count, attacking_count, evidence chain
```

The belief database is stored at `results/beliefs/` (a Kuzu graph database directory). This is created automatically on first use.

When running Condition C, just set the environment variable and run normally:

```bash
AUTORESEARCH_CONDITION=C uv run train.py
```

## Running as server (Condition D)

Condition D uses the mnemebrain-lite FastAPI server as a sidecar process and queries it via the mnemebrain SDK client.

### Starting the server

```bash
# IMPORTANT: Force CPU-only to reserve GPU for training
CUDA_VISIBLE_DEVICES="" mnemebrain
```

This starts the FastAPI server on port 8000. You can verify it is running:

```bash
curl http://localhost:8000/docs    # OpenAPI documentation
curl http://localhost:8000/health  # Health check
```

### Running training with Condition D

In a separate terminal (or after backgrounding the server):

```bash
CUDA_VISIBLE_DEVICES="" mnemebrain &    # background the server
AUTORESEARCH_CONDITION=D uv run train.py
```

### How ActiveHooks uses the server

The `ActiveHooks` class in `mnemebrain_hooks.py` connects to the server via the mnemebrain SDK:

```python
from mnemebrain import MnemeBrainClient

client = MnemeBrainClient(base_url="http://localhost:8000")

# believe, explain, search, revise — all via HTTP
client.believe(claim="...", evidence=[...])
explanation = client.explain(claim="...")
similar = client.search(query="...", limit=5)
```

Prediction and recommendation logic lives in `ActiveHooks._predict()` and `ActiveHooks._recommend()`, not in the server. ActiveHooks queries the belief store, then applies its own reasoning to generate predictions and recommendations.

### Customizing the server

By default, the server binds to `http://localhost:8000`. If you need a different port or host:

```bash
CUDA_VISIBLE_DEVICES="" mnemebrain --host 0.0.0.0 --port 9000
```

Then set the `MNEMEBRAIN_URL` environment variable or pass the URL directly when constructing `ActiveHooks`.

## CPU-only requirement

All mnemebrain-lite computation (embeddings, graph queries, belief operations) must run on CPU only. The GPU is reserved exclusively for training. This is a fairness requirement for the experiment: if mnemebrain-lite consumed GPU VRAM (even a few hundred MB), Conditions C and D would train with less effective memory than A and B, creating a confound.

Since mnemebrain operations happen BETWEEN runs (not during training), the CPU overhead does not affect training throughput. It adds only a few seconds of pre/post-run latency, which is not measured as part of the 5-minute training budget.

### How CPU-only is enforced

| Condition | Mechanism |
|-----------|-----------|
| C (in-process library) | `BeliefMemory(device="cpu")` prevents sentence-transformers from loading onto GPU/MPS |
| D (separate server) | `CUDA_VISIBLE_DEVICES=""` before the `mnemebrain` command forces all inference to CPU |

## Using the mnemebrain SDK client

The `mnemebrain` PyPI package provides a thin HTTP client for the mnemebrain-lite API:

```python
from mnemebrain import MnemeBrainClient

client = MnemeBrainClient(base_url="http://localhost:8000")

# Store a belief with evidence
client.believe(
    claim="LR 0.06 tends to destabilize training",
    evidence=[{
        "source_ref": "run_12",
        "content": "diverged at step 200",
        "polarity": "supports",
        "weight": 0.88,
        "reliability": 0.95,
    }],
)

# Explain a claim (get truth state + evidence chain)
result = client.explain(claim="LR 0.06 destabilizes training")
print(result.truth_state)       # "TRUE", "FALSE", "BOTH", or "NEITHER"
print(result.supporting_count)  # number of supporting evidence items
print(result.attacking_count)   # number of attacking evidence items

# Semantic search for related beliefs
similar = client.search(query="learning rate stability", limit=5)
for belief in similar:
    print(belief.claim, belief.truth_state)

# Revise beliefs (contextual refinement)
client.revise(
    claim="LR 0.03 improves training",
    context={"warmup": ">100 steps"},
)
```

## Troubleshooting

### TOKENIZERS_PARALLELISM warning

When using mnemebrain-lite in-process (Condition C), you may see:

```
huggingface/tokenizers: The current process just got forked, setting parallelism to false
```

This is harmless. `PassiveHooks` sets `TOKENIZERS_PARALLELISM=false` automatically to suppress it. If you still see warnings, set it manually:

```bash
TOKENIZERS_PARALLELISM=false AUTORESEARCH_CONDITION=C uv run train.py
```

### Intel Mac: no wheels for sentence-transformers

PyTorch no longer provides x86_64 macOS wheels. On Intel Macs, use the OpenAI embeddings variant:

```bash
pip install mnemebrain-lite[openai]
export OPENAI_API_KEY=sk-...
```

This uses the OpenAI embeddings API instead of local sentence-transformers. The latency is slightly higher but functionally equivalent for the experiment.

### Server not reachable (Condition D)

If `ActiveHooks` fails to connect:

1. Check the server is running: `curl http://localhost:8000/health`
2. Check the port is not in use: `lsof -i :8000`
3. Verify CPU-only launch: `CUDA_VISIBLE_DEVICES="" mnemebrain`
4. Check logs for startup errors

### Belief database corruption

The Kuzu database at `results/beliefs/` can occasionally become corrupted after a hard crash. To reset:

```bash
rm -rf results/beliefs/
# The next Condition C run will create a fresh database
```

This only affects Condition C beliefs. Individual run JSONs in `results/runs/` are unaffected.

### High CPU usage during pre/post-run

Embedding computation (sentence-transformers) is CPU-intensive. On machines with limited cores, the few seconds of pre/post-run CPU work may cause a brief spike. This is expected and does not affect training since mnemebrain operations run strictly between training runs.
