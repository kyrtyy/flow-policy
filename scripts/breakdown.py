"""Success, coverage and failure breakdown per (policy, variant), pooled over seeds, from eval_info.json files.

    python scripts/breakdown.py results/full results/guided_full

Failures are split by final coverage: gross (< 0.6, e.g. wrong orientation) vs near miss (>= 0.85).
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

rows = defaultdict(list)  # (dir, policy, variant) -> [(success %, coverage, max_rewards, successes)]
for root in map(Path, sys.argv[1:]):
    for f in sorted(root.glob("*/*/eval_info.json")):
        m = json.loads(f.read_text())["per_task"][0]["metrics"]
        policy = f.parent.parent.name.split("_s")[0]
        rows[(root.name, policy, f.parent.name)].append(m)

print("| results | policy | variant | seeds | success % | coverage | fails | gross <0.6 | near >=0.85 |")
print("|---|---|---|---|---|---|---|---|---|")
for (root, policy, variant), ms in rows.items():
    succ = [100 * sum(m["successes"]) / len(m["successes"]) for m in ms]
    cov = [sum(m["max_rewards"]) / len(m["max_rewards"]) for m in ms]
    fails = [r for m in ms for r, s in zip(m["max_rewards"], m["successes"], strict=True) if not s]
    sd = statistics.stdev(succ) if len(succ) > 1 else 0.0
    print(
        f"| {root} | {policy} | {variant} | {len(ms)} | {statistics.mean(succ):.1f} ± {sd:.1f} | "
        f"{statistics.mean(cov):.3f} | {len(fails)} | {sum(r < 0.6 for r in fails)} | {sum(r >= 0.85 for r in fails)} |"
    )
