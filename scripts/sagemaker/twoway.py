"""Two-way routers for H3: each three-way probe with ABSTAIN folded into RETRIEVE."""
import argparse

from saerag.config import load_config, path
from saerag.schema import read_jsonl, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--variant", required=True)
ap.add_argument("--layers", nargs="+", type=int, required=True)
ap.add_argument("--files", nargs="+", required=True)
args = ap.parse_args()
cfg = load_config(args.config)

for L in args.layers:
    for kind in ("resid_last", "sae16k_last"):
        pid = f"{cfg['model']['tag']}__{args.variant}__{kind}@L{L}__l2"
        for f in args.files:
            src = path(cfg, "results", "routers", pid, f"probs_{f}.jsonl")
            if not src.exists():
                continue
            rows = [{"qid": r["qid"], "p": {"ANSWER": r["p"].get("ANSWER", 0.0),
                                            "RETRIEVE": r["p"].get("RETRIEVE", 0.0) + r["p"].get("ABSTAIN", 0.0),
                                            "ABSTAIN": 0.0}} for r in read_jsonl(src)]
            write_jsonl(path(cfg, "results", "routers", f"{pid}__2way", f"probs_{f}.jsonl"), rows)
        print("wrote", f"{pid}__2way")
