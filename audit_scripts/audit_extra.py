"""Extra audit checks (cached data only; KMeans refit as in audit_numbers.py)."""
import json, os
from collections import Counter
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "data", "cache"); R = os.path.join(ROOT, "data", "results")
load = lambda p: json.load(open(p, encoding="utf-8"))

nq = np.array([x["mean_logprob"] for x in load(f"{C}/nq_generations.json")])
tq = np.array([x["mean_logprob"] for x in load(f"{C}/tqa_generations.json")])
grid = np.unique(np.concatenate([nq, tq]))
Fn = np.searchsorted(np.sort(nq), grid, side="right") / len(nq)
Ft = np.searchsorted(np.sort(tq), grid, side="right") / len(tq)
bad = grid[Fn < Ft]
print("Fig6: #thresholds where F_NQ < F_TQA:", len(bad), "range", (bad.min(), bad.max()) if len(bad) else None,
      "max(F_TQA-F_NQ)", (Ft - Fn).max())

for nm in ["nq", "tqa"]:
    s = np.array([x["entail_score"] for x in load(f"{C}/{nm}_entailment.json")])
    print("Fig7", nm, "frac<0.1", np.mean(s < 0.1), "frac>0.9", np.mean(s > 0.9), "frac in [0.1,0.9]", np.mean((s >= 0.1) & (s <= 0.9)))

# synthetic cluster detail
from sklearn.cluster import KMeans
sd = load(f"{C}/synth_qa_data.json"); sl = np.array([x["entail_label"] for x in load(f"{C}/synth_qa_entailment.json")])
topics = np.array([r["topic"] for r in sd])
emb = np.load(f"{C}/synth_qa_embeddings.npy")
lb = KMeans(n_clusters=10, random_state=42, n_init=10).fit(emb).labels_
pair = load(f"{C}/synthetic_a_pair_indices.json")
S = np.zeros(len(sd), bool); S[pair["source_idx"]] = True
T = np.zeros(len(sd), bool); T[pair["target_idx"]] = True
for t, k in [("music and composers", 7), ("literature and authors", 8)]:
    m = topics == t
    print(t, "in own cluster", k, ": n", int((m & (lb == k)).sum()), "acc", sl[m & (lb == k)].mean(),
          "| outside: n", int((m & (lb != k)).sum()), "acc", sl[m & (lb != k)].mean() if (m & (lb != k)).any() else None,
          "| clusters outside", Counter(lb[m & (lb != k)].tolist()))
    print("   S", int((m & S).sum()), "T", int((m & T).sum()), "unused", int((m & ~S & ~T).sum()))
for k in range(10):
    m = lb == k
    top, cnt = Counter(topics[m]).most_common(1)[0]
    print("cluster", k, "n", int(m.sum()), "majority", top, cnt, f"purity={cnt/m.sum():.3f}")
# expected composition under the sampling law: alpha weights on A clusters
A = set(pair["A_topics"])
inA = np.isin(lb, list(A))
print("pool n in A", int(inA.sum()), "in B", int((~inA).sum()))
print("source frac from A", (S & inA).sum() / 800, "target frac from A", (T & inA).sum() / 800)
