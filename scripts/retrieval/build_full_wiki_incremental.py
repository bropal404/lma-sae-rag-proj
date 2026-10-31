"""Disk-constrained incremental BM25 indexing of DPR Wikipedia (Pyserini 2.4).

Uses 22 compressed HF shards sequentially. DO NOT use an incomplete index for
benchmark labels. An interrupted indexing operation leaves an IN_PROGRESS
marker, because a partially committed Lucene shard cannot safely be replayed.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from pyserini.index.lucene import LuceneIndexReader, LuceneIndexer

REPO = "castorini/odqa-wiki-corpora"
PREFIX = "wiki-text-100w-karpukhin/"
MIN_FREE_BEFORE_SHARD = 3 * 1024**3
BATCH = 1000


def write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--max-shards", type=int, default=0,
                    help="0 = all remaining; 1 = validate first shard")
    args = ap.parse_args()
    if args.max_shards < 0:
        ap.error("--max-shards must be >= 0")

    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    index = root / "index"
    stage = root / "stage"
    stage.mkdir(exist_ok=True)
    state_file = root / "state.json"
    inflight = root / "IN_PROGRESS"
    ready_file = root / "FULL_INDEX_READY"

    if inflight.exists():
        raise SystemExit(
            f"UNSAFE TO RESUME: {inflight} exists. The previous shard may be "
            "partially committed. Do not blindly append it again."
        )
    if state_file.exists():
        state = json.loads(state_file.read_text())
    else:
        if index.exists() and any(index.iterdir()):
            raise SystemExit("Index exists without checkpoint; refusing to append.")
        state = {"repo": REPO, "revision": None, "completed": [], "docs": 0}

    api = HfApi()
    if not state["revision"]:
        state["revision"] = api.dataset_info(REPO).sha
        write_json_atomic(state_file, state)
    revision = state["revision"]
    shards = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset", revision=revision)
                    if f.startswith(PREFIX) and f.endswith(".jsonl.gz"))
    if len(shards) != 22:
        raise RuntimeError(f"Expected 22 DPR shards; found {len(shards)}")
    if any(f not in shards for f in state["completed"]):
        raise RuntimeError("Checkpoint refers to an unknown shard")
    if state["completed"] != shards[:len(state["completed"])]:
        raise RuntimeError("Checkpoint shards are not a contiguous prefix")

    started = 0
    for shard in shards[len(state["completed"]):]:
        if args.max_shards and started >= args.max_shards:
            break
        free = shutil.disk_usage(root).free
        if free < MIN_FREE_BEFORE_SHARD:
            raise SystemExit(f"STOP: only {free/1024**3:.2f} GiB free; "
                             "do not risk filling home storage")
        print(f"Downloading {shard}; free={free / 1024**3:.2f} GiB", flush=True)
        source = Path(hf_hub_download(
            repo_id=REPO, repo_type="dataset", revision=revision,
            filename=shard, local_dir=stage,
        ))
        print(f"Downloaded: {source} ({source.stat().st_size/1024**2:.1f} MiB)", flush=True)

        # Checkpoint before altering the Lucene index. If indexing is interrupted,
        # this prevents replaying a potentially half-committed shard.
        inflight.write_text(shard + "\n")
        indexer = LuceneIndexer(
            args=["-index", str(index), "-storeRaw"],
            append=bool(state["completed"]), threads=8,
        )
        count = 0
        batch: list[str] = []
        try:
            with gzip.open(source, "rt", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if not row.get("docid") or not row.get("text"):
                        continue
                    raw = json.dumps({
                        "id": str(row["docid"]),
                        "contents": f'{row.get("title", "")}\n{row["text"]}',
                    }, ensure_ascii=False)
                    batch.append(raw)
                    if len(batch) >= BATCH:
                        indexer.add_batch_raw(batch)
                        count += len(batch)
                        batch.clear()
                        if count % 100000 == 0:
                            print(f"  {count:,} docs in {shard}", flush=True)
                if batch:
                    indexer.add_batch_raw(batch)
                    count += len(batch)
            indexer.close()
        except BaseException:
            # Preserve marker: do not automatically re-index a partial shard.
            print(f"FAILED during {shard}. Check {inflight} before resuming.", flush=True)
            raise

        reader = LuceneIndexReader(str(index))
        actual = reader.stats()["documents"]
        del reader
        expected = state["docs"] + count
        if actual != expected:
            raise RuntimeError(f"Index document count mismatch: {actual} != {expected}. "
                               f"Preserved {inflight} for diagnosis")
        state["completed"].append(shard)
        state["docs"] = expected
        write_json_atomic(state_file, state)
        inflight.unlink()
        source.unlink()
        started += 1
        print(f"COMMITTED {len(state['completed'])}/{len(shards)} shards; "
              f"docs={expected:,}; free={shutil.disk_usage(root).free / 1024**3:.2f} GiB",
              flush=True)

    if len(state["completed"]) == len(shards):
        ready_file.write_text(f"{state['docs']} documents; revision={revision}\n")
        print("FULL_INDEX_READY", state["docs"], index, flush=True)
    else:
        print(f"PARTIAL INDEX: {len(state['completed'])}/{len(shards)} shards. "
              "Do not use for final labels.", flush=True)


if __name__ == "__main__":
    main()
