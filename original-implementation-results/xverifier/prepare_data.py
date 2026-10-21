"""Download pinned inputs, verify original row IDs, and embed text without labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

RAW = "Persius/imr-appeals"
RAW_REV = "c68b70e0faf74cd2b5731ea02ad7ba7d99a11b8b"
CLEAN = "utcompliance1/imr-appeals-cleaned"
CLEAN_REV = "0abb1e862c71e2b0558b560fceedab3d528759f7"
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDER_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    clean = load_dataset(CLEAN, revision=CLEAN_REV)
    raw = load_dataset(RAW, revision=RAW_REV)
    full = pd.concat([raw[k].to_pandas().assign(split=k) for k in ("train", "test")], ignore_index=True)
    full["row_id"] = np.arange(len(full))
    annotated = pd.concat([s.to_pandas() for s in clean.values()]).sort_values("row_id")
    assert annotated.row_id.tolist() == full.row_id.tolist()
    for col in ("text", "decision", "split"):
        assert annotated[col].tolist() == full[col].tolist(), f"Original row ID mismatch: {col}"
    full["near_dup_group_id"] = annotated.near_dup_group_id.to_numpy()
    full["clean"] = full.row_id.isin(set(clean["train"]["row_id"]) | set(clean["test"]["row_id"]))
    # Neither full_text nor any other metadata is passed to the embedding model.
    full = full[["row_id", "text", "decision", "split", "near_dup_group_id", "clean"]]
    full.to_parquet(args.work / "cases.parquet", index=False)
    embedding_path = args.work / "embeddings.npz"
    if not embedding_path.exists():
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(4)
        model = SentenceTransformer(EMBEDDER, revision=EMBEDDER_REV, device=args.device)
        vectors = model.encode(full.text.tolist(), batch_size=64, show_progress_bar=True,
                               normalize_embeddings=True, convert_to_numpy=True)
        np.savez(embedding_path, row_id=full.row_id.to_numpy(), representation=vectors.astype(np.float32))
    metadata = {
        "raw_dataset": RAW, "raw_revision": RAW_REV,
        "clean_dataset": CLEAN, "clean_revision": CLEAN_REV,
        "embedding_model": EMBEDDER, "embedding_revision": EMBEDDER_REV,
        "embedding_input": "text", "embedding_normalized": True,
        "embedding_max_seq_length": 256,
        "row_alignment_verified": True,
        "rows": {k: len(v) for k, v in raw.items()},
        "clean_rows": {k: len(v) for k, v in clean.items()},
        "embedding_sha256": sha(embedding_path), "cases_sha256": sha(args.work / "cases.parquet"),
    }
    (args.work / "data_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
