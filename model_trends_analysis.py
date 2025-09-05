#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os
from datetime import datetime
import re

"""
Model Trends Analysis — standalone (multi-run + multi-file, provider-aware, packaging-aware)

What this script does:
- Ingests **all** raw scrape files under `scraped_data/` (CSV/JSON/JSONL), recursively.
- Handles **multiple incremental runs** safely: dedups to the latest record per `model_id` using
  last_modified → created_at → downloads.
- Separates **provider** (HF owner/parent) from **underlying model family** (derived from the
  model name, tags, and base-model fields), to avoid confusing quantizers/fine-tuners with
  base families.
- Computes **canonical model identities** (collapsing alternate packaging and quantization) and also
  preserves per-package detail to analyze local-AI–relevant formats.
- Writes **CSV tables** for inspection (not just graphs): family, provider, model (canonical),
  packaging and quantization breakdowns, time series, inventory, and canonical representatives.

Run:
  export SCRAPED_DIR=/path/to/raw   # optional (defaults to ./scraped_data)
  python model_trends_analysis.py

Inputs expected (any subset is fine; script coalesces names):
  - model id:     model_id | model | model_name | repo_id | id | name
  - downloads:    downloads | downloadCount | downloads_count
  - likes:        likes | likeCount | likes_count | stars
  - license:      license | license_name
  - tags:         tags | tag_string | categories
  - created_at:   created_at | createdAt | createdDate | created
  - last_modified:last_modified | lastModified | updated_at | modified_at
  - params:       params | parameters | n_params | n_parameters | size_params
  - family:       family | model_family (hint only)
  - base_model:   base_model | original_model | source_model | upstream_model (strong hint)
  - files:        files  (list of filenames from your scraper; used for packaging/quant)

Outputs (CSV):
  - out/ingest_inventory.csv
  - out/records_canonical.csv                     (one row per canonical model representative)
  - out/model_canonical_summary.csv               (popularity by canonical model)
  - out/family_summary.csv                        (popularity by derived family)
  - out/provider_summary.csv                      (popularity by provider/owner)
  - out/packaging_by_family.csv                   (counts/downloads by packaging)
  - out/packaging_by_model.csv                    (per canonical model packaging mix)
  - out/quantization_by_family.csv                (counts/downloads by quant level)
  - out/quantization_by_model.csv                 (per canonical model quant mix)
  - out/family_timeseries_monthly.csv             (monthly new canonical bases & downloads)

Outputs (PNG):
  - out/top_families_canonical_downloads.png
  - out/families_unaligned_share.png
  - out/monthly_new_bases_top_families.png
  - out/params_distribution_canonical.png (if params available)
  - out/popularity_vs_unaligned_share.png

Headline stats:
  - out/key_datapoints.json
"""

RAW_DIR = os.environ.get("SCRAPED_DIR", "scraped_data")
PATTERNS = ["*.csv", "*.json", "*.jsonl"]

# Heuristics
UNALIGNED_MARKERS = [
    "uncensored", "no-safety", "nosafety", "unsafe", "jailbreak",
    "abliterated", "unfiltered", "nsfw", "anythinggoes", "deguard",
    "de-aligned", "dealigned", "unrestrict", "de-align"
]

# family tokens to look for inside model names / tags / base_model
FAMILY_TOKENS = [
    "llama", "mistral", "mixtral", "qwen", "gemma", "phi", "falcon", "deepseek",
    "yi", "xverse", "olmo", "minicpm", "wizardlm", "nemotron", "command",
    "granite", "glm", "baichuan", "internlm", "qwen2", "qwen3", "deepseek-r1",
]

# packaging formats (local-AI relevant)
PACKAGING_TOKENS = [
    "gguf",           # container for llama.cpp
    "safetensors",    # container/format
    "pth", "pt", "onnx", "bin",
    "llama-cpp", "gguf-my-repo"
]

EXTENSIONS = (".gguf", ".safetensors", ".bin", ".pth", ".pt", ".onnx")

# quantization levels and dtypes (simple substring checks)
QUANT_TOKENS = ["q2", "q3", "q4", "q5", "q6", "q8", "int4", "int8", "nf4", "fp16", "bf16", "fp8", "fp32"]

QUANT_REGEXPS = [
    r"\bq[234568](?:_[01])?\b",          # q2, q3, q4, q5, q6, q8, q4_0, q8_0
    r"\bq[234568]_k(?:_[a-z]+)?\b",      # q4_k_m, q5_k_s, etc.
    r"\biq[1-8](?:_[a-z]+)?\b",          # iq2_xs, iq3_m, ...
    r"\b(?:gptq|awq|exl2|imatrix)\b",    # quantization methods
    r"\bint[48]\b",
    r"\b(?:4-?bit|8-?bit)\b",
    r"\bnf4\b",
    r"\bfp(?:8|16|32)\b",
    r"\bbf16\b"
]

ALIASES = {
    "model": ["model_id", "model", "model_name", "repo_id", "id", "name"],
    "downloads": ["downloads", "downloadCount", "downloads_count"],
    "likes": ["likes", "likeCount", "likes_count", "stars"],
    "license": ["license", "license_name"],
    "tags": ["tags", "tag_string", "categories"],
    "created_at": ["created_at", "createdAt", "createdDate", "created"],
    "last_modified": ["last_modified", "lastModified", "updated_at", "modified_at"],
    "params": ["params", "parameters", "n_params", "n_parameters", "size_params"],
    "family": ["family", "model_family"],
    "base_model": ["base_model", "original_model", "source_model", "upstream_model"],
    "files": ["files"],
}

def _coalesce(df, keys):
    for k in keys:
        if k in df.columns:
            return k
    return None

def _read(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".jsonl":
        return pd.read_json(path, lines=True)
    if suf == ".json":
        try:
            return pd.read_json(path, orient="records")
        except Exception:
            return pd.read_json(path)
    raise ValueError(f"Unsupported file: {path}")

def _to_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    parts = s.replace("|", ",").replace(";", ",").split(",")
    return [p.strip() for p in parts if p.strip()]

def _owner_and_name(model_id: str):
    mid = (model_id or "").strip()
    if "/" in mid:
        owner, name = mid.split("/", 1)
    else:
        owner, name = "", mid
    return owner.lower(), name.lower()

def _first_non_null(*vals):
    for v in vals:
        if pd.notna(v) and v != "":
            return v
    return np.nan

def _find_token(text: str, tokens: list):
    t = (text or "").lower()
    for tok in tokens:
        if tok in t:
            return tok
    return ""

def _collect_tokens(text: str, tokens: list):
    t = (text or "").lower()
    hits = []
    for tok in tokens:
        if tok in t:
            hits.append(tok)
    return sorted(list(set(hits)))

def _derive_family(owner: str, model_name: str, tags_text: str, family_hint: str, base_hint: str):
    # priority: explicit family hint → base_model hint → tokens from tags/name → fallback: first token of model_name
    if family_hint:
        return family_hint
    if base_hint:
        bh = base_hint
        # If base_model is owner/base, prefer the base part for token detection
        if "/" in bh:
            bh = bh.split("/")[-1]
        hit = _find_token(bh, FAMILY_TOKENS)
        if hit:
            return hit
        # otherwise use a cleaned base string as a family-ish label
        return bh.split("-")[0].split("_")[0]
    hit = _find_token(tags_text + " " + model_name, FAMILY_TOKENS)
    if hit:
        return hit
    # fallback to first token of name
    for sep in ["-", "_", " "]:
        if sep in model_name:
            return model_name.split(sep, 1)[0]
    return model_name

def _first_base_from_tags_list(tags_list):
    """Prefer base_model:* from tags_raw, skipping :quantized:/:merge:/:adapter:."""
    for t in (tags_list or []):
        ts = str(t).lower()
        if not ts.startswith("base_model:"):
            continue
        if any(x in ts for x in (":quantized:", ":merge:", ":adapter:")):
            continue
        v = ts.split("base_model:", 1)[1].strip()
        if "/" in v:
            v = v.split("/", 1)[1]  # take repo part
        if v and v not in {"none", "model"}:
            return v
    return None


def _clean_key(s):
    """Lowercase, remove packaging/quant/dtype tokens, collapse separators, strip extensions."""
    s = (s or "").lower()
    # normalize separators
    s = s.replace(" ", "-").replace("_", "-").replace("/", "-")
    # strip known extensions at end
    for ext in EXTENSIONS:
        if s.endswith(ext):
            s = s[: -len(ext)]
    # drop packaging tokens on hyphen/dot boundaries
    parts = re.split(r"[-.]", s)
    parts = [p for p in parts if p not in set(PACKAGING_TOKENS)]
    s = "-".join(parts)
    # remove quant/dtype patterns
    for rx in QUANT_REGEXPS:
        s = re.sub(rx, "", s)
    # collapse leftovers
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    # remove trailing isolated zeros created by q8_0 stripping
    s = re.sub(r"-0$", "", s)
    return s


def _canonical_model_key_from_row(row):
    """
    Prefer: base_model tag → base_hint → filename stem → repo name.
    Keep behavior suffixes (e.g., instruct/chat) if present in the chosen string.
    """
    # 1) base_model:* from tags_raw
    from json import loads
    tags_raw = row.get("tags_raw", "")
    if isinstance(tags_raw, str):
        try:
            tags_list = loads(tags_raw) if tags_raw.strip().startswith("[") else []
        except Exception:
            tags_list = []
    else:
        tags_list = list(tags_raw) if tags_raw is not None else []
    base_from_tags = _first_base_from_tags_list(tags_list)
    if base_from_tags:
        return _clean_key(base_from_tags)

    # 2) base_hint
    bh = (row.get("base_hint") or "").strip().lower()
    if bh and bh not in {"none", "model"}:
        return _clean_key(bh.split("/")[-1])

    # 3) filenames (take longest model file stem)
    files_raw = row.get("files_raw", "")
    stems = []
    if isinstance(files_raw, str):
        try:
            files_list = loads(files_raw) if files_raw.strip().startswith("[") else []
        except Exception:
            files_list = []
    else:
        files_list = list(files_raw) if files_raw is not None else []
    for f in files_list:
        fs = str(f).lower()
        for ext in EXTENSIONS:
            if fs.endswith(ext):
                stems.append(fs[: -len(ext)])
                break
    if stems:
        stems.sort(key=len, reverse=True)
        return _clean_key(stems[0])

    # 4) fallback: repo name
    return _clean_key(row.get("model_name_raw") or "")


def _safe_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            return json.loads(v)
        except Exception:
            return []
    return []

def _detect_packaging_quant(row):
    # Build a big lowercase haystack from id/name/tags/files
    parts = [
        str(row.get("model_id") or ""),
        str(row.get("model_name_raw") or ""),
        " ".join(_safe_list(row.get("tags_raw"))),
        " ".join(_safe_list(row.get("files_raw"))),
    ]
    hay = " ".join(parts).lower()

    # Packaging: simple token presence on hyphen/space/underscore/dot boundaries
    packaging = set()
    tokens = re.split(r"[^a-z0-9.+]+", hay)  # keep '.' for extensions like .onnx if present in hay
    token_set = set(tokens)
    for tok in PACKAGING_TOKENS:
        if tok in token_set or f".{tok}" in hay:
            packaging.add(tok)

    # Quantization/dtype: regex matches (normalized to lowercase)
    quant = set()
    for rx in QUANT_REGEXPS:
        for m in re.findall(rx, hay):
            quant.add(m.lower())

    return sorted(packaging), sorted(quant)

def write_repo_catalog(raw_df: pd.DataFrame, out_path: str = "repo_catalog.csv"):
    rows = []
    for _, r in raw_df.iterrows():
        model_id = str(r.get("model_id") or "")
        provider, repo_name = (model_id.split("/", 1) + [""])[:2] if "/" in model_id else ("", model_id)

        packaging_detected, quant_detected = _detect_packaging_quant(r)

        rows.append({
            "model_id": model_id,
            "provider": provider,
            "repo_name": repo_name,
            "canonical_model_key": r.get("canonical_model_key") or r.get("canonical_model_key_fixed") or "",
            "family": r.get("family") or r.get("family_hint") or "",
            "base_hint": r.get("base_hint") or "",
            "unaligned": r.get("unaligned") or "",
            "is_packaged": r.get("is_packaged") or "",
            "packaging_detected": ",".join(packaging_detected),
            "quant_detected": ",".join(quant_detected),
            "created_at": r.get("created_at") or "",
            "last_modified": r.get("last_modified") or "",
            "license": r.get("license") or "",
            "downloads": r.get("downloads") or "",
            "likes": r.get("likes") or "",
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote per-repo catalog: {out_path}")


# ---------------- load all files ----------------
paths = []
for pat in PATTERNS:
    paths.extend(Path(RAW_DIR).rglob(pat))

if not paths:
    raise SystemExit(f"No raw scrape files found under: {RAW_DIR}")

inv_rows = []
frames = []
for p in sorted(paths):
    try:
        df = _read(p)
    except Exception as e:
        print(f"[WARN] Skipping {p}: {e}")
        continue

    before = len(df)
    cols = {key: _coalesce(df, ALIASES[key]) for key in ALIASES}
    mid = cols["model"]
    if not mid:
        print(f"[WARN] {p.name}: no recognizable model id column; cols={list(df.columns)[:8]}")
        continue

    mini = pd.DataFrame()
    mini["model_id"] = df[mid].astype(str)
    mini["downloads"] = pd.to_numeric(df[cols["downloads"]], errors="coerce") if cols["downloads"] else np.nan
    mini["likes"] = pd.to_numeric(df[cols["likes"]], errors="coerce") if cols["likes"] else np.nan
    mini["license"] = df[cols["license"]] if cols["license"] else ""
    mini["tags_raw"] = df[cols["tags"]] if cols["tags"] else ""
    mini["created_at"] = pd.to_datetime(df[cols["created_at"]], errors="coerce") if cols["created_at"] else pd.NaT
    mini["last_modified"] = pd.to_datetime(df[cols["last_modified"]], errors="coerce") if cols["last_modified"] else pd.NaT
    mini["params"] = pd.to_numeric(df[cols["params"]], errors="coerce") if cols["params"] else np.nan
    mini["family_hint"] = df[cols["family"]].astype(str).str.lower().str.strip() if cols["family"] else ""
    mini["base_hint"] = df[cols["base_model"]].astype(str).str.lower().str.strip() if cols["base_model"] else ""
    mini["files_raw"] = df[cols["files"]] if cols["files"] else ""
    mini["source_file"] = p.name

    after = len(mini)
    inv_rows.append({"file": p.name, "rows_in": before, "rows_out": after})
    frames.append(mini)

if not frames:
    raise SystemExit("No usable rows after reading inputs.")

raw = pd.concat(frames, ignore_index=True)

# fill basics
raw["downloads"] = raw["downloads"].fillna(0)
raw["likes"] = raw["likes"].fillna(0)
raw["license"] = raw["license"].fillna("")

# provider and model_name
owners, names = [], []
for mid in raw["model_id"].astype(str).tolist():
    o, n = _owner_and_name(mid)
    owners.append(o)
    names.append(n)
raw["provider"] = owners
raw["model_name_raw"] = names

# tags & files text
raw["tags_text"] = [" ".join(map(str, _to_list(x))).lower() for x in raw["tags_raw"]]
raw["files_text"] = [" ".join(map(str, _to_list(x))).lower()
                     for x in raw.get("files_raw", pd.Series([""] * len(raw)))]

# unaligned heuristic
raw["unaligned"] = raw["tags_text"].apply(lambda t: any(m in t for m in UNALIGNED_MARKERS)) | \
                   raw["model_id"].str.lower().str.contains("uncensored|abliterated", na=False)

# family derivation (owner is NOT used as family)
raw["family"] = [
    _derive_family(raw.loc[i, "provider"], raw.loc[i, "model_name_raw"],
                   raw.loc[i, "tags_text"], raw.loc[i, "family_hint"], raw.loc[i, "base_hint"])
    for i in range(len(raw))
]

# --- LLM vs non-LLM filter (conservative, overinclusive) -----------------------
import re

# Optional: dedupe first so each repo is considered once
if "model_id" in raw.columns:
    raw = raw.drop_duplicates(subset=["model_id"]).reset_index(drop=True)

NON_LLM_PIPELINES = {
    "text-embedding", "feature-extraction", "sentence-similarity",
    "fill-mask", "token-classification", "sequence-classification",
    "image-classification", "object-detection", "image-segmentation",
    "image-to-image", "text-to-image", "image-to-text",
    "depth-estimation", "image-feature-extraction",
    "audio-classification", "automatic-speech-recognition", "text-to-speech",
    "speech-segmentation", "voice-activity-detection",
    "diffusers", "dancediffuserpipeline",
}

# Broad tokens (require >=2 hits unless also in STRONG_NON_LLM_TOKENS)
NON_LLM_NAME_TOKENS = {
    # diffusion / image
    "stable-diffusion", "sdxl", "sd-", "latent-diffusion", "controlnet",
    "unet", "vae", "inpaint", "txt2img", "img2img",
    # vision
    "vit", "resnet", "yolo", "detr", "sam", "segment-anything",
    # audio/speech
    "whisper", "wav2vec", "hubert", "tacotron", "fastspeech", "tts", "musicgen", "dancediffusion", "dancediffuser",
    # embeddings families
    "sentence-transformers", "minilm", "gte", "e5", "bge", "mpnet", "sbert",
    # general embedding keywords
    "embedding", "embeddings", "encoder-only", "feature-extraction",
}

# Tokens that should exclude on a *single hit* (your troublesome cases go here)
STRONG_NON_LLM_TOKENS = {
    "prompt-uie-base", "prompt-uie-medical-base",
    "ernie-ctm-nptag", "ernie-ctm-base", "ernie-1.0",
    "mc-bert",
    "anime-collaborative-filtering",
}

# Optional: exact owner/repo slugs to always exclude (use lowercase)
STRONG_NON_LLM_SLUGS = {
    "freedomking/prompt-uie-base",
    "freedomking/prompt-uie-medical-base",
    "freedomking/ernie-ctm-nptag",
    "freedomking/ernie-ctm-base",
    "freedomking/ernie-1.0",
    "freedomking/mc-bert",
    "nomiwai/anime-collaborative-filtering",
}

LLM_POSITIVE_TAGS = {
    "text-generation", "text2text-generation", "causal-lm",
    "conversational", "chat", "instruct", "rlhf"
}

# --- helpers -------------------------------------------------------------------
_norm_re = re.compile(r"[^a-z0-9]+")
def _norm(s: str) -> str:
    return _norm_re.sub(" ", (s or "").lower()).strip()

def _token_hit(text_norm: str, token: str) -> bool:
    # check raw token, normalized token, and token without separators
    t_raw = token.lower()
    t_norm = _norm(token)
    t_compact = re.sub(r"\s+", "", t_norm)
    return (
        t_raw in text_norm or
        t_norm in text_norm or
        t_compact in text_norm.replace(" ", "")
    )

def _any_token_hit(text_norm: str, tokens: set[str]) -> bool:
    return any(_token_hit(text_norm, t) for t in tokens)

def _count_token_hits(text_norm: str, tokens: set[str]) -> int:
    return sum(1 for t in tokens if _token_hit(text_norm, t))

# --- main predicate ------------------------------------------------------------
def _is_llmish(row) -> bool:
    tags = (row.get("tags_text") or "")
    model_id = (row.get("model_id") or row.get("model_name_raw") or "").lower()
    name = (row.get("model_name_raw") or "").lower()
    fam  = (row.get("family") or "").lower()
    base = (row.get("base_hint") or "").lower()

    hay_tags_norm = _norm(tags)
    hay_all_norm = _norm(" ".join([model_id, name, fam, base, tags]))

    # 0) Hard exclude specific slugs
    if model_id in STRONG_NON_LLM_SLUGS:
        return False

    # 1) Strong LLM signals → keep
    if _any_token_hit(hay_tags_norm, LLM_POSITIVE_TAGS):
        return True
    # If you have an existing FAMILY_TOKENS for LLMs, keep those
    if "FAMILY_TOKENS" in globals() and _any_token_hit(hay_all_norm, FAMILY_TOKENS):
        return True

    # 2) Non-LLM pipeline tags → exclude
    if _any_token_hit(hay_tags_norm, NON_LLM_PIPELINES):
        return False

    # 3) Strong non-LLM tokens → single hit excludes
    if _any_token_hit(hay_all_norm, STRONG_NON_LLM_TOKENS):
        return False

    # 4) Broad non-LLM tokens → require >=2 hits to avoid false positives
    nonllm_hits = _count_token_hits(hay_all_norm, NON_LLM_NAME_TOKENS)
    if nonllm_hits >= 2:
        return False

    # 5) Default keep (overinclusive)
    return True

# Apply the filter
before_n = len(raw)
raw = raw[raw.apply(_is_llmish, axis=1)].reset_index(drop=True)
after_n = len(raw)
print(f"[INFO] LLM-ish filter kept {after_n}/{before_n} repos")
# -------------------------------------------------------------------------------

# canonical model identity (collapse packaging + quantization only)
raw["canonical_model_key"] = raw.apply(_canonical_model_key_from_row, axis=1)

# Output the full Repo catalog
write_repo_catalog(raw, out_path="repo_catalog.csv")

# packaging + quantization signals (use model_id + name + tags + files)
raw["packaging_hits"] = [
    ",".join(_collect_tokens(raw.loc[i, "model_id"] + " " + raw.loc[i, "model_name_raw"] + " " +
                             raw.loc[i, "tags_text"] + " " + raw.loc[i, "files_text"], PACKAGING_TOKENS))
    for i in range(len(raw))
]
raw["quant_hits"] = [
    ",".join(_collect_tokens(raw.loc[i, "model_id"] + " " + raw.loc[i, "model_name_raw"] + " " +
                             raw.loc[i, "tags_text"] + " " + raw.loc[i, "files_text"], QUANT_TOKENS))
    for i in range(len(raw))
]

# multi-run dedup: keep latest per model_id
raw["_k1"] = raw["last_modified"].fillna(pd.Timestamp(0))
raw["_k2"] = raw["created_at"].fillna(pd.Timestamp(0))
raw["_k3"] = raw["downloads"].fillna(0)
raw = raw.sort_values(["_k1", "_k2", "_k3"], ascending=[False, False, False])
raw_dedup = raw.drop_duplicates(subset=["model_id"], keep="first").reset_index(drop=True).copy()

# ---------------- canonical representatives per canonical_model_key ----------------
# choose a representative repo per canonical model (prefer un-packaged, then max downloads)
def _is_packaged(s):
    s = s or ""
    for tok in PACKAGING_TOKENS:
        if tok in s:
            return True
    return False

raw_dedup["is_packaged"] = raw_dedup.apply(
    lambda r: _is_packaged(r.get("model_id", "")) or
              _is_packaged(r.get("model_name_raw", "")) or
              _is_packaged(r.get("tags_text", "")) or
              _is_packaged(r.get("files_text", "")),
    axis=1,
)

raw_dedup["last_t"] = raw_dedup.apply(lambda r: _first_non_null(r.get("last_modified"), r.get("created_at")), axis=1)

reps = []
for key, grp in raw_dedup.groupby("canonical_model_key"):
    g = grp.sort_values(["is_packaged", "downloads", "last_t"], ascending=[True, False, False])
    reps.append(g.iloc[0])
reps_df = pd.DataFrame(reps)

# package totals per canonical model
pkg_totals = raw_dedup.groupby("canonical_model_key").agg(
    package_downloads_total=("downloads", "sum"),
    package_repo_count=("model_id", "count"),
    providers_set=("provider", lambda s: sorted(list(set(s)))),
    families_set=("family", lambda s: sorted(list(set(s)))),
    packagings_set=("packaging_hits", lambda s: sorted(list(set(",".join([x for x in s if x]).split(","))))),
    quant_set=("quant_hits", lambda s: sorted(list(set(",".join([x for x in s if x]).split(","))))),
).reset_index()

canon = reps_df.merge(pkg_totals, on="canonical_model_key", how="left")

# ---------------- summaries ----------------
# family summary
family_summary = canon.groupby("family", as_index=False).agg(
    models_canonical=("canonical_model_key", "nunique"),
    downloads_canonical=("downloads", "sum"),
    downloads_pkg_total=("package_downloads_total", "sum"),
    providers_unique=("providers_set", lambda s: len(sorted(set([p for sub in s for p in sub])))),
    unaligned_share=("unaligned", "mean"),
    params_median=("params", "median"),
)

# provider summary (who is publishing/hosting)
provider_summary = canon.groupby("provider", as_index=False).agg(
    canonical_models_hosted=("canonical_model_key", "nunique"),
    downloads_canonical=("downloads", "sum"),
    downloads_pkg_total=("package_downloads_total", "sum"),
)

# model canonical summary
model_canonical_summary = canon[[
    "canonical_model_key", "family", "provider", "model_name_raw", "downloads", "package_downloads_total",
    "package_repo_count", "providers_set", "families_set", "packagings_set", "quant_set", "unaligned", "params"
]].copy()
model_canonical_summary.rename(columns={"downloads": "downloads_canonical_rep"}, inplace=True)

# packaging by family
packaging_records = []
for _, r in raw_dedup.iterrows():
    fam = r["family"]
    dls = r["downloads"]
    packs = _collect_tokens(r["model_id"] + " " + r["model_name_raw"] + " " + r["tags_text"] + " " + r.get("files_text", ""), PACKAGING_TOKENS)
    qhits = _collect_tokens(r["model_id"] + " " + r["model_name_raw"] + " " + r["tags_text"] + " " + r.get("files_text", ""), QUANT_TOKENS)
    ptype = packs[0] if packs else "none"
    qlvl = qhits[0] if qhits else "none"
    packaging_records.append({"family": fam, "packaging": ptype, "quant": qlvl, "downloads": dls})
packaging_df = pd.DataFrame(packaging_records)
packaging_by_family = packaging_df.groupby(["family", "packaging"], as_index=False).agg(
    repos=("packaging", "count"), downloads=("downloads", "sum")
)
quant_by_family = packaging_df.groupby(["family", "quant"], as_index=False).agg(
    repos=("quant", "count"), downloads=("downloads", "sum")
)

# packaging & quantization by canonical model
pkg_model_rows = []
for key, grp in raw_dedup.groupby("canonical_model_key"):
    for _, r in grp.iterrows():
        packs = _collect_tokens(r["model_id"] + " " + r["model_name_raw"] + " " + r["tags_text"] + " " + r.get("files_text", ""), PACKAGING_TOKENS)
        qs = _collect_tokens(r["model_id"] + " " + r["model_name_raw"] + " " + r["tags_text"] + " " + r.get("files_text", ""), QUANT_TOKENS)
        pkg_model_rows.append({
            "canonical_model_key": key,
            "provider": r["provider"],
            "family": r["family"],
            "packaging": packs[0] if packs else "none",
            "quant": qs[0] if qs else "none",
            "downloads": r["downloads"],
        })
packaging_by_model = pd.DataFrame(pkg_model_rows).groupby(
    ["canonical_model_key", "packaging"], as_index=False
).agg(repos=("packaging", "count"), downloads=("downloads", "sum"))
quant_by_model = pd.DataFrame(pkg_model_rows).groupby(
    ["canonical_model_key", "quant"], as_index=False
).agg(repos=("quant", "count"), downloads=("downloads", "sum"))

# time series (monthly) from canonical reps
canon["ts"] = [ _first_non_null(canon.loc[i, "created_at"], canon.loc[i, "last_modified"]) for i in range(len(canon)) ]
canon["ts_month"] = pd.to_datetime(canon["ts"]).dt.to_period("M").dt.to_timestamp()
family_ts = canon.dropna(subset=["ts_month"]).groupby(["ts_month", "family"], as_index=False).agg(
    new_canonical_models=("canonical_model_key", "nunique"),
    downloads_canonical=("downloads", "sum"),
)

# ---------------- write CSV outputs ----------------
outdir = Path("out")
outdir.mkdir(parents=True, exist_ok=True)

pd.DataFrame(inv_rows).to_csv(outdir / "ingest_inventory.csv", index=False)
reps_df.to_csv(outdir / "records_canonical.csv", index=False)
model_canonical_summary.to_csv(outdir / "model_canonical_summary.csv", index=False)
family_summary.sort_values("downloads_canonical", ascending=False).to_csv(outdir / "family_summary.csv", index=False)
provider_summary.sort_values("downloads_canonical", ascending=False).to_csv(outdir / "provider_summary.csv", index=False)
packaging_by_family.to_csv(outdir / "packaging_by_family.csv", index=False)
packaging_by_model.to_csv(outdir / "packaging_by_model.csv", index=False)
quant_by_family.to_csv(outdir / "quantization_by_family.csv", index=False)
quant_by_model.to_csv(outdir / "quantization_by_model.csv", index=False)
family_ts.to_csv(outdir / "family_timeseries_monthly.csv", index=False)

# ---------------- plots ----------------
plt.figure(figsize=(11,6))
_topF = family_summary.sort_values("downloads_canonical", ascending=False).head(12)
plt.bar(_topF["family"], _topF["downloads_canonical"])
plt.title("Top Families by Canonical Downloads")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(outdir / "top_families_canonical_downloads.png", dpi=200)
plt.close()

plt.figure(figsize=(11,6))
_topU = family_summary.sort_values("unaligned_share", ascending=False).head(12)
plt.bar(_topU["family"], _topU["unaligned_share"])
plt.title("Families by Share of Unaligned Canonical Models")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(outdir / "families_unaligned_share.png", dpi=200)
plt.close()

plt.figure(figsize=(11,6))
_top6 = _topF["family"].head(6).tolist()
sub = family_ts[family_ts["family"].isin(_top6)]
for fam in _top6:
    ss = sub[sub["family"] == fam]
    plt.plot(ss["ts_month"], ss["new_canonical_models"], label=fam)
plt.title("Monthly New Canonical Models (Top Families)")
plt.legend()
plt.tight_layout()
plt.savefig(outdir / "monthly_new_bases_top_families.png", dpi=200)
plt.close()

if canon["params"].notna().any():
    plt.figure(figsize=(11,6))
    vals = canon["params"].dropna().values
    plt.hist(vals, bins=25)
    plt.title("Parameter Count Distribution (Canonical)")
    plt.tight_layout()
    plt.savefig(outdir / "params_distribution_canonical.png", dpi=200)
    plt.close()

plt.figure(figsize=(11,6))
plt.scatter(family_summary["downloads_canonical"], family_summary["unaligned_share"])
plt.title("Popularity vs. Unaligned Share (by Family)")
plt.xlabel("Canonical Downloads")
plt.ylabel("Unaligned Share")
plt.tight_layout()
plt.savefig(outdir / "popularity_vs_unaligned_share.png", dpi=200)
plt.close()

# ---------------- headline datapoints ----------------
key = {}
if not family_summary.empty:
    topfam = family_summary.sort_values("downloads_canonical", ascending=False).iloc[0]
    key["most_downloaded_family"] = str(topfam["family"])
    key["most_downloaded_family_canonical_downloads"] = int(topfam["downloads_canonical"])
    dense = family_summary[family_summary["models_canonical"] >= 5]
    if not dense.empty:
        hu = dense.sort_values("unaligned_share", ascending=False).iloc[0]
        key["highest_unaligned_family"] = str(hu["family"])
        key["highest_unaligned_share"] = float(hu["unaligned_share"])

if not family_ts.empty:
    cutoff = family_ts["ts_month"].max() - pd.Timedelta(days=90)
    rec = family_ts[family_ts["ts_month"] >= cutoff]
    grow = rec.groupby("family", as_index=False)["new_canonical_models"].sum()
    if not grow.empty:
        fg = grow.sort_values("new_canonical_models", ascending=False).iloc[0]
        key["fastest_growth_family_last_90d"] = str(fg["family"])
        key["fastest_growth_new_bases_90d"] = int(fg["new_canonical_models"])

with open(outdir / "key_datapoints.json", "w") as f:
    json.dump(key, f, indent=2)

print(f"[OK] Files ingested: {len(paths)}  |  unique repos after dedup: {raw_dedup.shape[0]}")
print(f"[OK] CSVs written to: {outdir}")
