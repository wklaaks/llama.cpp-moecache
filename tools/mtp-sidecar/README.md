# MTP Sidecar Builder

Build an **MTP (NextN / draft) sidecar GGUF** from a Hugging Face *safetensors*
model — downloading **only the tensors the sidecar needs**, never the whole
model. Designed to be handed to a friend: they run one script, pass a model URL,
and it does the checks, the partial download, the conversion, and the validation.

```
make_mtp_sidecar.py      ← the tool (single file, stdlib + torch + huggingface_hub)
README.md                ← this file
```

---

## What it produces

A small self-contained `.gguf` (a few GB, e.g. ~4 GB at `q8_0`) that holds just:

| Part | Tensors |
|------|---------|
| Token embeddings | `model.language_model.embed_tokens.weight` |
| Output head | `lm_head.weight` |
| Final MTP layer | `mtp.layers.0.*` (self-attn, MoE experts, hyper-connection mixer) |
| MTP combiner | `mtp.fc_embedding`, `mtp.fc_hidden`, `mtp.pre_fc_norm_*`, … |

That's everything `llama-server` needs as the speculative-draft model. It is
served alongside the full trunk model:

```bash
llama-server -m <base.gguf> -md mtp-<name>.gguf \
             --spec-type draft-mtp --spec-draft-n-max 3
```

---

## Why it only downloads a fraction

The tool wraps the **fork's** converter in `--remote` mode:

```
<fork>/convert_hf_to_gguf.py --mtp --remote --outtype q8_0 <hf-repo-id>
```

In `--remote` mode the converter:

1. downloads **only** `config.json` + tokenizer (not the weights),
2. lists every tensor in the HF repo via the safetensors index,
3. keeps **only** the MTP set (embeddings + `lm_head` + the final MTP layer +
   its `nextn` combiner), and
4. streams exactly those tensors over HTTP **Range reads**.

So for a 360 GB / 131-shard model it touches only the ~30 shards that actually
contain MTP tensors — a few tens of GB at most instead of 360.

> ⚠️ This requires the **qwen4exp fork** of llama.cpp, *not* stock upstream.
> Upstream's `qwen4exp.py` converter hard-codes `no_mtp=True` and cannot emit a
> sidecar. The fork carries the per-block MTP NextN combiner. Point
> `--fork-dir` at your `llama.cpp-qwen4exp` checkout.

---

## Requirements

- **Python ≥ 3.10**
- **`torch`** and **`huggingface_hub`** (both already used by the converter)
- The **llama.cpp-qwen4exp fork** checked out somewhere on disk
- Enough free disk (the tool checks this for you before it touches the network)

Install the Python deps if you don't have them:

```bash
pip install -U torch huggingface_hub
```

---

## Quick start

```bash
# 1) Preview only — no download, no write. Safe to run first.
./make_mtp_sidecar.py Qwen/Qwen3.8-Flash-Next --dry-run

# 2) For real — build into ~/models using your fork.
./make_mtp_sidecar.py Qwen/Qwen3.8-Flash-Next \
    --out-dir ~/models \
    --fork-dir ~/src/llama.cpp-qwen4exp
```

The tool will, in order:

1. **Check the environment** — python version, torch, huggingface_hub, fork
   present, and that the converter supports `--mtp` (it refuses stock
   llama.cpp).
2. **Inspect the repo** — fetch `config.json`, detect architecture, layer
   count, and the NextN/MTP layer count. If the model has no MTP layer it
   aborts early.
3. **Guard disk space** — read the safetensors index `total_size`, require
   ~20% of it + 4 GB headroom, and **abort before any download** if there
   isn't enough.
4. **Run the converter** — the actual partial download + conversion.
5. **Validate the output** — parse the produced GGUF header (built-in, no
   extra deps) and confirm the `nextn` combiner, embeddings, and output head
   are all present.

On success it prints the serve command to use.

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `model` (positional) | — | HF repo id (`owner/name`), a full `https://huggingface.co/…` URL, or a local dir already holding `config.json` + safetensors |
| `--out-dir` | `.` | where to write the `.gguf` |
| `--fork-dir` | `$LLAMA_CPP_QWEN4EXP` or `~/src/llama.cpp-qwen4exp` | path to the qwen4exp fork |
| `--outtype` | `q8_0` | output quantization (`q8_0`, `bf16`, `f16`, `f32`) |
| `--dry-run` | off | preview only; no download, no write |
| `--min-free-gb` | auto | override the minimum-free-space threshold |
| `--token` | `$HF_TOKEN` | HF token for gated repos |
| `--extra-args` | — | extra args passed verbatim to the converter |

Pass a **raw safetensors base model** (the one with the MTP/nextn layer), **not**
a pre-converted GGUF repo — the tool reads `config.json` + the safetensors index
to plan the partial download, and a GGUF repo has neither.

---

## Example: the Qwen3.8 Next model with the MTP/nextn layer

This is the working example the tool was validated against.

**Hugging Face repo (raw safetensors):**

```
https://huggingface.co/Qwen/Qwen3.8-Flash-Next
```

- Architecture: `Qwen4ExpForConditionalGeneration` (`model_type: qwen4_exp`)
- Trunk layers: **48** (`num_hidden_layers`)
- MTP/NextN layers: **1** (`mtp_num_hidden_layers`)
- Total weights: **~360 GB** across **131** safetensors shards
- MTP/nextn tensors: **31** tensors, spread across ~28 of those shards

So passing just the repo id:

```bash
./make_mtp_sidecar.py https://huggingface.co/Qwen/Qwen3.8-Flash-Next \
    --out-dir ~/models --fork-dir ~/src/llama.cpp-qwen4exp
```

…is all your friend needs to type.

### Concrete safetensors URLs

The MTP/nextn tensors do **not** live in one shard — they're interleaved with
trunk tensors across the file range `model-00037-of-00131` →
`model-00124-of-00131`. The two "always-needed" tensors sit at the tail:

```
# token embeddings
https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/model-00130-of-00131.safetensors

# output head (lm_head)
https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/model-00131-of-00131.safetensors
```

Representative MTP-layer shards (each is a normal ~2.7 GB shard; the converter
pulls only the MTP tensors out of them via Range reads):

```
# MTP fc_embedding combiner
https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/model-00120-of-00131.safetensors

# MTP fc_hidden combiner
https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/model-00124-of-00131.safetensors

# MTP self-attn projections
https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/model-00108-of-00131.safetensors

# MTP MoE experts
https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/model-00060-of-00131.safetensors
https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/model-00062-of-00131.safetensors
```

You do **not** need to enumerate these yourself — the tool reads
`model.safetensors.index.json` and the converter's `--mtp` filter selects the
exact tensors. The list above is for reference / manual verification.

> Note: `jamesrogers/Qwen3.8-Flash-Next-MTP-MXFP4-GGUF` is a **pre-converted
> GGUF** repo (tags `gguf`/`mxfp4`). It has no `config.json`/safetensors, so it
> is *not* a valid input to this tool. Use the raw `Qwen/Qwen3.8-Flash-Next`
> safetensors model instead.

---

## Disk space

The tool estimates the requirement from the safetensors index and refuses to
start if the target volume can't hold it:

```
need ≈ 0.20 × (total model bytes) + 4 GB headroom
```

(20% comfortably covers the final MTP layer + its nextn combiner + embeddings +
head at `q8_0`, plus the temp file the converter writes while quantizing.) For
the 360 GB Qwen3.8-Flash-Next that's a ~76 GB requirement. Override with
`--min-free-gb N` if you know better.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `converter … does not appear to support --mtp` | You pointed at stock llama.cpp. Use the **qwen4exp fork** (`--fork-dir`). |
| `could not fetch config.json` | Wrong repo id, network issue, or you passed a GGUF repo. Use a raw safetensors repo. |
| `this model does not declare a NextN/MTP layer` | The model genuinely has no MTP block — nothing to build. |
| `NOT ENOUGH DISK SPACE` | Free up space or point `--out-dir` at a larger volume. |
| Gated repo / 401 | Set `HF_TOKEN` or pass `--token hf_…`. |
| `torch not importable` | `pip install -U torch huggingface_hub`. |

---

## Verified

Validated end-to-end against `Qwen/Qwen3.8-Flash-Next`:

- repo-id parsing (bare id, full URL, `/tree/main` URL, local dir) ✓
- MTP detection → `block_count=48, nextn=1, arch=Qwen4ExpForConditionalGeneration` ✓
- disk math → total 360 GB, needs ~76 GB, passes on a 338 GB volume ✓
- built-in GGUF validator correctly reads a known-good 4.14 GB sidecar
  (v3, `qwen4exp`, 35 tensors; `nextn`/embeddings/output all present) ✓
- `--dry-run` pass-through → passed all pre-checks, downloaded config+tokenizer,
  detected arch, entered remote mode ✓