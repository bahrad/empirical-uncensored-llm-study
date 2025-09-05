# python hf_compliance_selection.py --enriched_csv enriched_models_20250808_172603.csv --tested_list current_models.txt  --top_n 20 --out_dir compliance_selection_20250808_172603

# -*- coding: utf-8 -*-
"""
hf_compliance_selection.py

Selects models for compliance evaluation from enriched metadata and maps already-tested models to their metadata.

Usage:
    python hf_compliance_selection.py --enriched_csv enriched_models.csv --tested_list models.txt --top_n 20 --out_dir outdir

Inputs:
    enriched_csv : CSV from hf_scrape_enrich.py
    tested_list  : Text file with one owner/model_name per line
    top_n        : How many new models to recommend (default 20)

Outputs:
    tested_models_metadata.csv
    recommended_models.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def main(enriched_csv, tested_list, top_n, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(enriched_csv)
    # Ensure created_at is parsed as datetime if present
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    # Flag MLX models so we can exclude them from recommendations (NVIDIA focus)
    def _lower_contains(series, token):
        return series.astype(str).str.lower().str.contains(token, na=False)

    is_mlx_model_id = _lower_contains(df["model_id"], "mlx") if "model_id" in df.columns else False
    is_mlx_tags    = _lower_contains(df["tags"], "mlx")     if "tags" in df.columns else False
    is_mlx_files   = _lower_contains(df["files"], "mlx")    if "files" in df.columns else False
    df["is_mlx"]   = is_mlx_model_id | is_mlx_tags | is_mlx_files
    
    # Load tested list
    with open(tested_list, "r", encoding="utf-8") as f:
        tested_models = [line.strip() for line in f if line.strip()]
    tested_set = set(tested_models)
    
    # Tested models metadata
    tested_df = df[df["model_id"].isin(tested_set)].copy()
    # Mark missing tested models
    missing_tested = tested_set - set(tested_df["model_id"])
    if missing_tested:
        missing_df = pd.DataFrame({"model_id": list(missing_tested)})
        for col in df.columns:
            if col != "model_id":
                missing_df[col] = np.nan
        tested_df = pd.concat([tested_df, missing_df], ignore_index=True)
    tested_df.to_csv(out_dir / "tested_models_metadata.csv", index=False)
    
    # Candidate pool: untested + high risk
    df_recent = df.copy()
    if "created_at" in df_recent.columns:
        df_recent["created_at"] = pd.to_datetime(df_recent["created_at"], errors="coerce", utc=True)
        recent_cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=183)  # ~6 months
    else:
        recent_cutoff = None
    
    # candidates = df[~df["model_id"].isin(tested_set)].copy()
    # candidates = candidates[(candidates["local_variant"]==True) & (candidates["is_potentially_modified"]==True)]
    candidates = df[~df["model_id"].isin(tested_set)].copy()
    candidates = candidates[
        (candidates["local_variant"] == True)
        & (candidates["is_potentially_modified"] == True)
        & (~candidates.get("is_mlx", False))  # exclude MLX
    ]
    
    # Rank logic: by family, prefer largest or recent
    ranked = []
    for fam, sub in candidates.groupby("canonical_base"):
        sub = sub.copy()
        if "params_total" in sub.columns:
            sub["rank_metric"] = sub["params_total"].fillna(0)
        else:
            sub["rank_metric"] = 0
        if recent_cutoff is not None and "created_at" in sub.columns:
            sub["is_recent"] = sub["created_at"] >= recent_cutoff
            # Boost recent models
            sub.loc[sub["is_recent"], "rank_metric"] += 1e12
        ranked.append(sub.sort_values("rank_metric", ascending=False).head(1))
    ranked_df = pd.concat(ranked, ignore_index=True) if ranked else pd.DataFrame(columns=df.columns)
    
    # Limit to top_n overall
    ranked_df = ranked_df.sort_values("rank_metric", ascending=False).head(top_n)
    ranked_df.to_csv(out_dir / "recommended_models.csv", index=False)
    
    print(f"Saved tested model metadata to {out_dir/'tested_models_metadata.csv'}")
    print(f"Saved recommended models to {out_dir/'recommended_models.csv'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched_csv", required=True, help="Path to enriched_models.csv from hf_scrape_enrich.py")
    ap.add_argument("--tested_list", required=True, help="Path to models.txt with owner/model_name per line")
    ap.add_argument("--top_n", type=int, default=20, help="Number of new models to recommend")
    ap.add_argument("--out_dir", required=True, help="Directory to save outputs")
    args = ap.parse_args()
    main(args.enriched_csv, args.tested_list, args.top_n, args.out_dir)
