#!/usr/bin/env python3
"""Print a comparison table from one or more benchmark.py result JSONs.

  python compare.py results/*.json
"""
import json, sys

paths = sys.argv[1:]
if not paths:
    sys.exit("usage: compare.py result1.json result2.json ...")

rows = []
for path in paths:
    with open(path) as f:
        r = json.load(f)
    t, inf = r["train"], r.get("inference", {})
    rows.append({
        "name": r["device_name"][:26],
        "prec": r["precision"],
        "train_ips": t["median_img_per_s"],
        "epoch_s": t["median_epoch_sec"],
        "total_s": t["total_sec"],
        "acc": t["final_test_acc"],
        "to90": t["epochs_to_0.90"],
        "i_bs1": inf.get("bs1", {}).get("lat_ms_p50"),
        "i_bs128": inf.get("bs128", {}).get("img_per_s"),
    })

cols = [("device", "name", 26), ("prec", "prec", 5), ("train img/s", "train_ips", 11),
        ("s/epoch", "epoch_s", 8), ("total s", "total_s", 8), ("acc", "acc", 6),
        ("->90%", "to90", 6), ("infer bs1 ms", "i_bs1", 12), ("infer bs128 i/s", "i_bs128", 15)]

def line(cells):
    print("  ".join(str(c).ljust(w) for c, (_, _, w) in zip(cells, cols)))

line([h for h, _, _ in cols])
line(["-" * w for _, _, w in cols])
for r in rows:
    line([r[k] for _, k, _ in cols])

base = rows[0]["train_ips"] or 1
print()
for r in rows:
    print(f"  {r['name']:<26}  {r['train_ips'] / base:6.2f}x  train throughput vs '{rows[0]['name']}'")
