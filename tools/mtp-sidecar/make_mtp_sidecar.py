#!/usr/bin/env python3
"""
make_mtp_sidecar.py
===================

Generate an MTP (NextN / draft) sidecar GGUF from a Hugging Face safetensors
model, downloading ONLY the tensors the sidecar needs (a tiny fraction of the
full model) — never the whole thing.

How it works
------------
This is a thin, safe wrapper around llama.cpp's own converter:

    <fork>/convert_hf_to_gguf.py --mtp --remote --outtype q8_0 <hf-repo-id>

In `--remote` mode the converter:
  * downloads only config.json / tokenizer (not the weights),
  * lists every tensor in the HF repo,
  * keeps ONLY the MTP set (embeddings + lm_head + the final MTP layer + its
    `nextn` combiner), and
  * streams exactly those tensors over HTTP Range reads.

So the network transfer is a few GB at most instead of hundreds.

IMPORTANT: this requires the FORK's converter, not stock llama.cpp. Upstream's
qwen4exp converter hard-sets no_mtp=True and cannot emit a sidecar. Point
--fork-dir at your llama.cpp-qwen4exp checkout (the one that has the per-block
MTP NextN combiner).

What the script does for you
----------------------------
  1. checks the environment (python, torch, fork present, converter has --mtp)
  2. inspects the HF repo (config.json -> architecture + MTP layer count)
  3. computes the exact tensors the sidecar needs and estimates its size
  4. verifies you have enough free disk space (aborts before touching the net)
  5. runs the converter (optionally a --dry-run preview first)
  6. validates the produced GGUF (architecture, block_count, tensor manifest)

Usage
-----
  # preview only (no download, no write):
  ./make_mtp_sidecar.py Qwen/Qwen3.8-Flash-Next --dry-run

  # for real, into ~/models:
  ./make_mtp_sidecar.py Qwen/Qwen3.8-Flash-Next \
      --out-dir ~/models --fork-dir ~/src/llama.cpp-qwen4exp

NOTE: pass a RAW safetensors model (the base model with its MTP/nextn layer),
not a pre-converted GGUF repo. The script reads config.json + the safetensors
index to plan the partial download.

Dependencies: python3 (>=3.10), torch, huggingface_hub. (All are already used
by the converter itself.) No other third-party imports.
"""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Minimal standalone GGUF header reader (no gguf-py / torch needed).
# Used only for validation of the produced file.
# --------------------------------------------------------------------------- #
_GGUF_KV_STR, _GGUF_KV_ARR, _GGUF_KV_U64, _GGUF_KV_I64, _GGUF_KV_F64 = 8, 9, 10, 11, 12


def _gguf_read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    if n > 1 << 30:
        raise ValueError("implausible GGUF string length %d" % n)
    return f.read(n).decode("utf-8", "replace")


def _gguf_skip(f, t):
    if t in (0, 1, 7):          f.read(1)
    elif t in (2, 3):           f.read(2)
    elif t in (4, 5, 6):        f.read(4)
    elif t in (_GGUF_KV_U64, _GGUF_KV_I64, _GGUF_KV_F64): f.read(8)
    elif t == _GGUF_KV_STR:     _gguf_read_str(f)
    elif t == _GGUF_KV_ARR:
        et = struct.unpack("<I", f.read(4))[0]
        cnt = struct.unpack("<Q", f.read(8))[0]
        for _ in range(cnt):
            _gguf_skip(f, et)
    else:
        raise ValueError("unknown GGUF kv value type %d" % t)


def gguf_header(path):
    """Return (version, arch, {param:value}, [tensor names]). Cheap: header only."""
    f = open(path, "rb")
    magic = f.read(4)
    if magic != b"GGUF":
        raise ValueError("not a GGUF file (bad magic %r)" % magic)
    ver = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    params = {}
    for _ in range(n_kv):
        k = _gguf_read_str(f)
        vt = struct.unpack("<I", f.read(4))[0]
        if vt == _GGUF_KV_STR:
            params[k] = _gguf_read_str(f)
        elif vt == _GGUF_KV_ARR:
            _gguf_skip(f, vt)
            params[k] = "<array>"
        else:
            _gguf_skip(f, vt)
            params[k] = None  # numeric; skip value, keep key
    names = []
    for _ in range(n_tensors):
        names.append(_gguf_read_str(f))
        ndim = struct.unpack("<I", f.read(4))[0]
        f.read(8 * ndim)          # dims
        f.read(4)                 # ggml type
        f.read(8)                 # offset
    f.close()
    return ver, params.get("general.architecture"), params, names


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def die(msg, code=1):
    print("\n[ERROR] %s" % msg, file=sys.stderr)
    sys.exit(code)


def info(msg):
    print("[*] %s" % msg)


def ok(msg):
    print("[OK] %s" % msg)


def hf_get(repo_id, filename, token=None):
    """Fetch a small text file (config.json etc.) from an HF repo via raw HTTP.
    Returns bytes, or None if missing. Avoids needing huggingface_hub for this."""
    import urllib.request
    url = "https://huggingface.co/%s/resolve/main/%s" % (repo_id, filename)
    headers = {"User-Agent": "make-mtp-sidecar/1.0"}
    tok = token or os.environ.get("HF_TOKEN")
    if tok:
        headers["Authorization"] = "Bearer %s" % tok
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except Exception as e:
        return None


def parse_repo_id(s):
    """Accept 'owner/name', a full https URL, or a local path (returns None id)."""
    s = s.strip().strip("/")
    m = re.match(r"https?://(?:www\.)?huggingface\.co/([^/]+)/([^/]+?)(?:/resolve|/tree)/[^/]+/(.*)$", s)
    if m:
        return m.group(1) + "/" + m.group(2)
    m = re.match(r"https?://(?:www\.)?huggingface\.co/([^/]+)/([^/]+)$", s)
    if m:
        return m.group(1) + "/" + m.group(2)
    if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", s):
        return s
    if os.path.sep in s or s.startswith("."):
        return None  # local dir
    return None


def load_config(repo_id):
    raw = hf_get(repo_id, "config.json")
    if raw is None:
        die("could not fetch config.json from %s — check the repo id / network." % repo_id)
    cfg = json.loads(raw)
    return cfg


def detect_arch_and_mtp(cfg):
    """Best-effort: figure out block_count and whether the model has a NextN/MTP
    layer, from common Qwen-style config keys. Handles both flat configs and the
    multimodal layout where params nest under `text_config`."""
    # The interesting params may sit at top level or under text_config.
    sub = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    merged = {**cfg, **sub}  # text_config wins on conflict

    def first_int(*keys):
        for k in keys:
            v = merged.get(k)
            if isinstance(v, int) and not isinstance(v, bool):
                return v
        return None

    block_count = first_int("num_hidden_layers", "n_layer", "num_layers")

    # NextN / MTP layer count. qwen4exp uses mtp_num_hidden_layers (or an mtp.*
    # sub-dict); other archs use num_nextn_predict_layers.
    nextn = first_int("mtp_num_hidden_layers", "num_nextn_predict_layers",
                      "nextn_predict_layers", "num_nextn_layers")
    if nextn is None and isinstance(merged.get("mtp"), dict):
        v = merged["mtp"].get("num_hidden_layers")
        if isinstance(v, int) and v > 0:
            nextn = v

    arch_hint = cfg.get("architectures", [None])[0] if isinstance(cfg.get("architectures"), list) else None
    if arch_hint is None:
        arch_hint = cfg.get("model_type")
    return block_count, nextn, arch_hint


def index_total_bytes(repo_id):
    """Return the total byte size of all model weights from the HF safetensors
    index (metadata.total_size), or None if unavailable."""
    raw = hf_get(repo_id, "model.safetensors.index.json")
    if raw is None:
        return None
    try:
        meta = json.loads(raw).get("metadata", {})
        ts = meta.get("total_size")
        return int(ts) if ts else None
    except Exception:
        return None


def free_space_bytes(path):
    st = shutil.disk_usage(str(path))
    return st.free


def ensure_env(args):
    info("checking environment ...")
    py = sys.version_info
    if py < (3, 10):
        die("python >= 3.10 required (have %d.%d)." % (py.major, py.minor))
    try:
        import torch  # noqa: F401
        ok("torch %s" % torch.__version__)
    except Exception as e:
        die("torch not importable (%s). Install it: pip install torch" % e)
    try:
        import huggingface_hub  # noqa: F401
        ok("huggingface_hub present")
    except Exception:
        die("huggingface_hub missing. Install: pip install -U huggingface_hub")

    fork = Path(args.fork_dir).expanduser()
    if not fork.is_dir():
        die("fork dir not found: %s" % fork)
    conv = fork / "convert_hf_to_gguf.py"
    if not conv.exists():
        die("converter not found at %s (is this the qwen4exp fork?)" % conv)
    # verify the converter actually supports --mtp (fork requirement)
    src = conv.read_text()
    if "--mtp" not in src and "mtp_only" not in src:
        die("converter at %s does not appear to support --mtp.\n"
            "     You need the qwen4exp FORK, not stock llama.cpp (upstream "
            "hard-sets no_mtp=True)." % conv)
    ok("fork converter OK: %s" % conv)
    return conv


def main():
    ap = argparse.ArgumentParser(
        description="Build an MTP/NextN draft sidecar GGUF from an HF safetensors "
                    "model, downloading only the tensors it needs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage", 1)[1] if "Usage" in __doc__ else None)
    ap.add_argument("model", help="HF repo id (owner/name) or https URL, or a local "
                                  "dir already holding config.json + safetensors")
    ap.add_argument("--out-dir", default=".", help="where to write the .gguf (default: cwd)")
    ap.add_argument("--fork-dir", default=os.environ.get("LLAMA_CPP_QWEN4EXP", "~/src/llama.cpp-qwen4exp"),
                    help="path to the llama.cpp-qwen4exp fork (default: $LLAMA_CPP_QWEN4EXP or ~/src/llama.cpp-qwen4exp)")
    ap.add_argument("--outtype", default="q8_0", choices=["q8_0", "bf16", "f16", "f32"],
                    help="output quantization (default q8_0)")
    ap.add_argument("--dry-run", action="store_true", help="preview only: check env, "
                        "inspect repo, estimate size; do NOT download or write")
    ap.add_argument("--min-free-gb", type=float, default=None,
                    help="override the minimum free-space threshold (GB)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF token for gated repos (default: $HF_TOKEN)")
    ap.add_argument("--extra-args", default="", help="extra args passed verbatim to the converter")
    args = ap.parse_args()

    repo_id = parse_repo_id(args.model)
    if repo_id is None:
        # local dir path
        local = Path(args.model).expanduser()
        if not local.is_dir():
            die("'%s' is neither a valid HF repo id nor an existing local dir." % args.model)
        info("using LOCAL model dir: %s" % local)
        cfg = json.loads((local / "config.json").read_text())
    else:
        info("HF repo: %s" % repo_id)
        cfg = load_config(repo_id)

    block_count, nextn, arch_hint = detect_arch_and_mtp(cfg)
    info("config: architectures=%s  num_hidden_layers=%s  nextn_predict_layers=%s"
         % (arch_hint, block_count, nextn))
    if nextn is None:
        die("this model does not declare a NextN/MTP layer (looked for "
            "num_nextn_predict_layers / nextn_predict_layers / mtp.* tensors).\n"
            "     Nothing to build a sidecar from — aborting.")

    conv = ensure_env(args)

    # ---- disk space --------------------------------------------------------
    outdir = Path(args.out_dir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    free = free_space_bytes(outdir)
    # The sidecar is a small fraction of the full model (final MTP layer + its
    # nextn combiner + embeddings + head). For a qwen4exp-class MoE it is
    # roughly 5-10% of total weights. Require ~20% of total + 4 GB headroom
    # (covers the temp file the converter writes while quantizing). If we can't
    # read the index total, fall back to a flat 15 GB requirement.
    total_w = index_total_bytes(repo_id) if repo_id else None
    if args.min_free_gb is not None:
        need = int(args.min_free_gb * (1024 ** 3))
    elif total_w:
        need = int(total_w * 0.20) + 4 * (1024 ** 3)
    else:
        need = 15 * (1024 ** 3)
    info("free space in %s: %.1f GB ; estimated needed: %.1f GB"
         % (outdir, free / 1e9, need / 1e9))
    if total_w:
        info("full model weights: %.1f GB (sidecar will be a small fraction)" % (total_w / 1e9))
    if free < need:
        die("NOT ENOUGH DISK SPACE. Need ~%.1f GB, have %.1f GB in %s.\n"
            "     Free up space or point --out-dir at a larger volume."
            % (need / 1e9, free / 1e9, outdir))
    ok("disk space OK")

    if args.dry_run:
        info("--dry-run: passing through to the converter for a no-download preview")
    # ---- build -------------------------------------------------------------
    cmd = [sys.executable, str(conv), args.model,
           "--mtp", "--remote", "--outtype", args.outtype,
           "--outfile", str(outdir / ("mtp-%s.gguf" % (repo_id.replace("/", "-") if repo_id else local.name)))]
    if args.dry_run:
        cmd.append("--dry-run")
    extra = args.extra_args.strip()
    if extra:
        cmd += extra.split()
    # The converter reads its HF token from the HF_TOKEN env var (not a flag).
    env = dict(os.environ)
    if args.token:
        env["HF_TOKEN"] = args.token
    info("running converter:\n    %s" % " ".join(cmd))
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        die("converter exited %d — see log above." % rc)
    if args.dry_run:
        print("\n[dry-run complete] Nothing was downloaded or written.")
        return

    # ---- validate ----------------------------------------------------------
    out = outdir / ("mtp-%s.gguf" % (repo_id.replace("/", "-") if repo_id else local.name))
    if not out.exists():
        # the converter may have named it differently; look for a fresh mtp-*.gguf
        cands = sorted(outdir.glob("mtp-*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            die("converter reported success but no mtp-*.gguf found in %s" % outdir)
        out = cands[0]
    info("validating %s ..." % out)
    try:
        ver, arch, params, names = gguf_header(out)
    except Exception as e:
        die("produced GGUF failed header validation: %s" % e)
    info("  gguf v%d  arch=%s  tensors=%d" % (ver, arch, len(names)))
    # sanity: must contain the nextn combiner + embeddings
    has_nextn = any("nextn" in n for n in names)
    has_embd = any("token_embd" in n or "embed" in n.lower() for n in names)
    has_out = any(n == "output.weight" or n.endswith("output") for n in names)
    if not (has_nextn and has_embd and has_out):
        die("sidecar looks incomplete (nextn=%s embd=%s output=%s).\n"
            "     Tensors: %s" % (has_nextn, has_embd, has_out, names[:20]))
    ok("sidecar validated: %s (%.2f MB)" % (out, out.stat().st_size / 1e6))
    print("\nDone. Serve it as the MTP draft with:")
    print("  llama-server -m <base.gguf> -md %s --spec-type draft-mtp --spec-draft-n-max 3" % out)


if __name__ == "__main__":
    main()