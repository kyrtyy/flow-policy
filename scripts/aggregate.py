"""Mean ± std over training seeds from eval_sweep's results.csv.  python scripts/aggregate.py results/fast"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
groups = defaultdict(list)
for r in csv.DictReader(open(out / "results.csv")):
    groups[(r["policy"], r["variant"])].append(r)


def ms(rows, k, fmt):
    xs = [float(r[k]) for r in rows]
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{statistics.mean(xs):{fmt}} ± {sd:{fmt}}"


md = ["| policy | variant | seeds | success % | coverage | latency ms |", "|---|---|---|---|---|---|"]
for (p, v), rows in groups.items():
    md.append(f"| {p} | {v} | {len(rows)} | {ms(rows, 'pc_success', '.1f')} | "
              f"{ms(rows, 'avg_max_reward', '.3f')} | {ms(rows, 'latency_ms', '.1f')} |")
(out / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
