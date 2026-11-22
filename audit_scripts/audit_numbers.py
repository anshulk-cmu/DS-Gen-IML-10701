"""Audit: recompute paper numbers from cached JSON/NPY only.

No generation, entailment scoring, classifier training, or 500-split re-runs.
The only fit performed is KMeans(K=10, random_state=42, n_init=10) on the cached
synthetic-pool embeddings, to recover cluster labels; it is checked against the
cluster histograms stored in cache/synthetic_a_pair_indices.json.

Run: python -I audit_scripts/audit_numbers.py   (from the repo root)
"""
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.stats import beta as beta_dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "data", "cache")
R = os.path.join(ROOT, "data", "results")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def hdr(s):
    print("\n" + "=" * 8 + " " + s + " " + "=" * 8)


trapz = getattr(np, "trapezoid", None) or np.trapz

# ---------------------------------------------------------------- caches
nq_d, tqa_d = load(f"{C}/nq_data.json"), load(f"{C}/tqa_data.json")
nq_g, tqa_g = load(f"{C}/nq_generations.json"), load(f"{C}/tqa_generations.json")
nq_e, tqa_e = load(f"{C}/nq_entailment.json"), load(f"{C}/tqa_entailment.json")

hdr("Dataset sizes and accuracy (N002,N003,N006,N007)")
for name, d, g, e in [("NQ", nq_d, nq_g, nq_e), ("TQA", tqa_d, tqa_g, tqa_e)]:
    lab = np.array([x["entail_label"] for x in e])
    print(name, "n_data", len(d), "n_gen", len(g), "n_ent", len(e),
          "correct", int(lab.sum()), "acc", lab.mean())
    assert [x["idx"] for x in d] == [x["idx"] for x in g] == [x["idx"] for x in e]
nq_multi = sum(1 for r in nq_d if len(r["all_answers"]) > 1)
print("NQ questions with >1 gold answer (only answer[0] used as reference):", nq_multi)
tqa_alias = np.mean([len(r["all_answers"]) for r in tqa_d])
print("TQA mean #answers incl aliases (only primary used):", tqa_alias)

hdr("Greedy generation facts (N004,N005,N019)")
for name, g in [("NQ", nq_g), ("TQA", tqa_g)]:
    ns = Counter(len(x["sampled_answers"]) for x in g)
    ntok = np.array([len(x["token_logprobs"]) for x in g])
    print(name, "samples per q:", dict(ns), "max greedy tokens:", ntok.max(),
          "mean greedy tokens:", ntok.mean(),
          "mean_logprob==mean(token_logprobs):",
          all(abs(x["mean_logprob"] - (np.mean(x["token_logprobs"]) if x["token_logprobs"] else 0.0)) < 1e-9 for x in g))

# FER feasibility: e=0 & entail_score >= tau_cp
hdr("FER feasibility: entail_label==0 with high entail_score")
for name, e in [("NQ", nq_e), ("TQA", tqa_e)]:
    s = np.array([x["entail_score"] for x in e]); lab = np.array([x["entail_label"] for x in e])
    print(name, "max score among label 0:", s[lab == 0].max(),
          "min score among label 1:", s[lab == 1].min(),
          "#label0 with score>=0.45:", int(((lab == 0) & (s >= 0.45)).sum()),
          "#label0 with score>=0.5:", int(((lab == 0) & (s >= 0.5)).sum()))

# ---------------------------------------------------------------- baseline
base = load(f"{R}/baseline_results.json")
ps = base["per_split"]
hdr("Split sizes (N027) and grid size H (N029)")
print("cal_size", Counter(p["cal_size"] for p in ps), "zu", Counter(p["zu_size"] for p in ps),
      "ze", Counter(p["ze_size"] for p in ps), "H", Counter(p["grid_size_H"] for p in ps),
      "in-domain n_total", Counter(p["indomain_test"]["n_total"] for p in ps))
print("floor(3610*0.7)=", int(np.floor(3610 * 0.7)), " floor(2527*0.75)=", int(np.floor(2527 * 0.75)))
print("alpha_adj = (0.02-1e-5)/20 =", (0.02 - 1e-5) / 20)
tcp = np.array([p["tau_cp"] for p in ps])
print("tau_cp across splits: min", tcp.min(), "median", np.median(tcp), "max", tcp.max(),
      "#splits tau_cp<0.5:", int((tcp < 0.5).sum()))
# empirical FER count per split on TQA (all items with label 0 and score >= tau_cp)
tq_s = np.array([x["entail_score"] for x in tqa_e]); tq_l = np.array([x["entail_label"] for x in tqa_e])
fer_items = [int(((tq_l == 0) & (tq_s >= t)).sum()) for t in tcp]
print("TQA items with e=0 & s>=tau_cp (pool-wide, per split): max", max(fer_items), "mean", np.mean(fer_items))


def table1_row(res, key):
    s = res[key]
    vals = [p[f"{'indomain_test' if key=='indomain' else 'shifted_test'}"] for p in res["per_split"]]
    fdr = np.array([v["fdr_e"] for v in vals]); eff = np.array([v["efficiency"] for v in vals])
    return dict(validity=s["validity_rate"], mean_fdr=s["mean_fdr_e"], std_fdr_json=s["std_fdr_e"],
                std_fdr_ddof0=fdr.std(), std_fdr_ddof1=fdr.std(ddof=1),
                mean_eff=s["mean_efficiency"], std_eff_json=s["std_efficiency"],
                std_eff_ddof0=eff.std(), std_eff_ddof1=eff.std(ddof=1),
                recomputed_validity=float(np.mean([v["valid"] for v in vals])),
                recomputed_mean_fdr=fdr.mean(), recomputed_mean_eff=eff.mean())


hdr("Table 1 M1 rows")
for k in ["indomain", "shifted"]:
    print("M1", k, table1_row(base, k))

hdr("M1 vacuous analysis (C2)")
tau_none = [p["tau1"] is None for p in ps]
sh_zero = [p["shifted_test"]["n_selected"] == 0 for p in ps]
id_zero = [p["indomain_test"]["n_selected"] == 0 for p in ps]
print("#tau1 None:", sum(tau_none), " #NQ n_sel==0:", sum(sh_zero), " #TQA-test n_sel==0:", sum(id_zero))
print("tau1None == NQ zero:", tau_none == sh_zero, " tau1None == TQA zero:", tau_none == id_zero)
sh_valid = np.array([p["shifted_test"]["valid"] for p in ps])
nv = ~np.array(sh_zero)
print("NQ valid total", int(sh_valid.sum()), " valid among non-vacuous", int(sh_valid[nv].sum()), "of", int(nv.sum()))
id_valid = np.array([p["indomain_test"]["valid"] for p in ps]); idnv = ~np.array(id_zero)
print("TQA in-domain valid among non-vacuous", int(id_valid[idnv].sum()), "of", int(idnv.sum()),
      "=", id_valid[idnv].mean())
sh_fdr = np.array([p["shifted_test"]["fdr_e"] for p in ps])
print("NQ splits with fdr_e==0:", int((sh_fdr == 0).sum()), " min non-vacuous NQ fdr:", sh_fdr[nv].min(),
      " mean non-vac NQ fdr:", sh_fdr[nv].mean())
print("Fig10 legend valid frac TQA:", np.mean(np.array([p['indomain_test']['fdr_e'] for p in ps]) <= 0.25),
      " NQ:", np.mean(sh_fdr <= 0.25))

hdr("Fig 19 tau1 stability")
t1 = np.array([p["tau1"] for p in ps if p["tau1"] is not None])
print("n", len(t1), "mean", t1.mean(), "median", np.median(t1), "std", t1.std(),
      "IQR", np.percentile(t1, 75) - np.percentile(t1, 25))

hdr("Fig 13 abstention analysis (M1 median tau1 on NQ)")
med = np.median(t1)
f1 = np.array([x["mean_logprob"] for x in nq_g]); lab = np.array([x["entail_label"] for x in nq_e])
sel = f1 >= med
print("median tau", med, "selected", int(sel.sum()), sel.mean(), "sel correct", int((sel & (lab == 1)).sum()),
      (sel & (lab == 1)).sum() / sel.sum(), "sel wrong", int((sel & (lab == 0)).sum()),
      "abst", int((~sel).sum()), (~sel).mean(), "abst correct", int((~sel & (lab == 1)).sum()),
      (~sel & (lab == 1)).sum() / (~sel).sum(), "abst wrong", int((~sel & (lab == 0)).sum()),
      "overall", lab.mean())
# compare with M3 median tau1
iw = load(f"{R}/importance_weighted_results.json")
t3 = np.array([p["tau1"] for p in iw["per_split"] if p["tau1"] is not None])
print("M3 median tau1 over committed splits:", np.median(t3), " n", len(t3))

# ---------------------------------------------------------------- M3
hdr("Table 1 M3 rows")
for k in ["indomain", "shifted"]:
    print("M3", k, table1_row(iw, k))
ips = iw["per_split"]
t_none = np.array([p["tau1"] is None for p in ips])
sz = np.array([p["shifted_test"]["n_selected"] == 0 for p in ips])
iz = np.array([p["indomain_test"]["n_selected"] == 0 for p in ips])
sv = np.array([p["shifted_test"]["valid"] for p in ips]); iv = np.array([p["indomain_test"]["valid"] for p in ips])
sf = np.array([p["shifted_test"]["fdr_e"] for p in ips])
print("#tau1 None", int(t_none.sum()), "#NQ zero", int(sz.sum()), "#TQA zero", int(iz.sum()),
      "NQ valid among committed", int(sv[~sz].sum()), "of", int((~sz).sum()),
      "mean non-vac NQ FDR", sf[~sz].mean(), "min non-vac NQ FDR", sf[~sz].min(),
      "TQA valid among non-vac", int(iv[~iz].sum()), "of", int((~iz).sum()))
print("Fig12 outcomes: abstain", int(sz.sum()), "valid answering", int((sv & ~sz).sum()), "invalid", int((~sv).sum()))
ws = iw["diagnostics"]["weight_stats"]
print("weight_stats", ws)
print("mean n_eff across splits", iw["diagnostics"]["mean_n_eff_across_splits"], "/1895 =",
      iw["diagnostics"]["mean_n_eff_across_splits"] / 1895)
print("weight_max/clip_value = n/sum(clipped raw) =", ws["weight_max"] / ws["clip_value"],
      "=> mean clipped raw weight", ws["clip_value"] / ws["weight_max"])
nes = np.array([p["n_eff_selected"] for p in ips])
print("n_eff_selected for committed splits: min", nes[~t_none].min(), "median", np.median(nes[~t_none]))
print("M3 classifier CV", iw["diagnostics"]["classifier_cv_accuracy"])
eff_sh = np.array([p["shifted_test"]["efficiency"] for p in ips]); print("Fig12 eff mean", eff_sh.mean())

# ---------------------------------------------------------------- conservative
hdr("Table 6 / M2")
cons = load(f"{R}/conservative_results.json")
for opt in ["option_a", "option_b", "option_c"]:
    for k, v in cons[opt].items():
        print(opt, k, "sh val", v["shifted"]["validity_rate"], "sh fdr", v["shifted"]["mean_fdr_e"],
              "sh eff", v["shifted"]["mean_efficiency"], "| id val", v["indomain"]["validity_rate"],
              "id eff", v["indomain"]["mean_efficiency"])
print("log(1.2) =", np.log(1.2), " max fM1 TQA", max(x["mean_logprob"] for x in tqa_g),
      " max fM1 NQ", max(x["mean_logprob"] for x in nq_g), " max tau1 M1", t1.max())

hdr("Table 7 / Fig 1 / Fig 9")
sw = load(f"{R}/epsilon_sweep_results.json")
for e in sw["epsilons"]:
    k = str(e)
    print(e, "M1", sw["method1"][k]["shifted_validity"], sw["method1"][k]["shifted_mean_efficiency"],
          "M2", sw["method2_optC"][k]["shifted_validity"], sw["method2_optC"][k]["shifted_mean_efficiency"],
          "M3", sw["method3"][k]["shifted_validity"], sw["method3"][k]["shifted_mean_efficiency"])

hdr("Tables 3 / 11 counts")
fe = load(f"{R}/synthetic_final_eps_sweep.json")
for pair in ["synthetic", "tqa_nq"]:
    for row in fe[pair]:
        for m in ["m1", "m3"]:
            r = row[m]
            nvc = r["non_vacuous_count"]; nvv = r["non_vacuous_validity"]
            nvalid = None if nvv is None else round(nvv * nvc, 6)
            print(pair, row["epsilon"], m, "val", r["validity_rate"], "eff", r["mean_efficiency"],
                  "vac", r["vacuous_frac"], "vac_n", round(r["vacuous_frac"] * 500, 6), "nv_n", nvc,
                  "nv_valid_n", nvalid, "nv_val", nvv, "check (vac+nvvalid)/500",
                  None if nvalid is None else (r["vacuous_frac"] * 500 + nvalid) / 500,
                  "id_val", r["indomain_validity"], "id_eff", r["indomain_efficiency"])

# ---------------------------------------------------------------- screening
hdr("Screening: TQA->NQ recomputable entries (T1,T2a,T2b,T3)")
fs = np.array([x["mean_logprob"] for x in tqa_g]); ys = np.array([x["entail_label"] for x in tqa_e])
ft = f1; yt = lab
k = max(1, len(ft) // 20); top = np.argsort(-ft)[:k]
print("acc_S", ys.mean(), "acc_T", yt.mean(), "top5_k", k, "acc_top5", yt[top].mean(), "gap", ys.mean() - yt.mean())
wq = load(f"{R}/synthetic_final_weight_quartile.json")
print("weight quartile json", wq)
pop = load(f"{R}/screening_popqa_results.json")
hy = np.array([x["entail_label"] for x in load(f"{C}/popqa_head_entailment.json")])
ty = np.array([x["entail_label"] for x in load(f"{C}/popqa_tail_entailment.json")])
tg = np.array([x["mean_logprob"] for x in load(f"{C}/popqa_tail_generations.json")])
kk = max(1, len(tg) // 20); tt = np.argsort(-tg)[:kk]
print("PopQA recomputed acc_S", hy.mean(), "acc_T", ty.mean(), "acc_top5", ty[tt].mean(), "n", len(hy), len(ty))
print("PopQA json", {k2: pop[k2] for k2 in ["acc_S", "acc_T", "acc_top5", "gap", "acc_clf", "ess_ratio",
                                            "quartile_spread", "slope_logw", "slope_pvalue", "pass_5", "soft_5"]})
hd = load(f"{C}/popqa_head_data.json"); td = load(f"{C}/popqa_tail_data.json")
print("PopQA head s_pop min", min(r["s_pop"] for r in hd), "tail s_pop max", max(r["s_pop"] for r in td))
syn = load(f"{R}/synthetic_final_screening.json")
print("synthetic json", {k2: syn[k2] for k2 in ["acc_S", "acc_T", "acc_top5", "gap", "acc_clf", "ess_ratio", "n_eff",
                                               "quartile_spread", "quartile_accs", "slope_logw", "slope_pvalue"]})
print("code strong-T6 rule (slope<-0.2 and p<0.05): synthetic ->",
      syn["slope_logw"] < -0.2 and syn["slope_pvalue"] < 0.05,
      " PopQA ->", pop["slope_logw"] < -0.2 and pop["slope_pvalue"] < 0.05)
for nm, d in [("PopQA", pop), ("synthetic", syn)]:
    n_pass = sum(bool(d[f"pass_{t}"]) for t in ["1", "2a", "2b", "3", "4", "5", "6"])
    n_soft = sum(1 for t in ["2a", "5"] if d[f"soft_{t}"] and not d[f"pass_{t}"])
    print(nm, "n_pass", n_pass, "n_soft(not pass)", n_soft, "verdict rule n_pass+n_soft>=6:", n_pass + n_soft >= 6)

# ---------------------------------------------------------------- feature plots
hdr("Fig 5 means, Fig 14, Fig 15 r, Fig 16 AUC")
for name, g, e in [("NQ", nq_g, nq_e), ("TQA", tqa_g, tqa_e)]:
    fm1 = np.array([x["mean_logprob"] for x in g]); fm2 = np.array([x["fM2"] for x in e])
    lb = np.array([x["entail_label"] for x in e])
    th = np.sort(np.unique(fm1))
    if len(th) > 500:
        th = th[np.linspace(0, len(th) - 1, 500, dtype=int)]
    npos, nneg = lb.sum(), len(lb) - lb.sum()
    tpr = np.array([((fm1 >= t) & (lb == 1)).sum() / npos for t in th])
    fpr = np.array([((fm1 >= t) & (lb == 0)).sum() / nneg for t in th])
    o = np.argsort(fpr)
    auc_plot = trapz(tpr[o], fpr[o])
    # exact AUC (Mann-Whitney) for reference
    from scipy.stats import rankdata
    rk = rankdata(fm1); auc_exact = (rk[lb == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    print(name, "fM1 mean", fm1.mean(), "r(fM1,y)", np.corrcoef(fm1, lb)[0, 1], "r(fM2,y)", np.corrcoef(fm2, lb)[0, 1],
          "acc", lb.mean(), "AUC(plot method)", auc_plot, "AUC exact", auc_exact,
          "fM1 range", fm1.min(), fm1.max())

hdr("Fig 8 deciles and Fig 17 calibration bins")
for name, g, e in [("NQ", nq_g, nq_e), ("TQA", tqa_g, tqa_e)]:
    fm1 = np.array([x["mean_logprob"] for x in g]); lb = np.array([x["entail_label"] for x in e])
    idx = np.argsort(fm1); n = len(fm1); ds = n // 10
    rates = [lb[idx[d * ds:(d * ds + ds if d < 9 else n)]].mean() for d in range(10)]
    print(name, "decile rates", [round(r, 4) for r in rates], "as %:", [f"{r:.0%}" for r in rates],
          "monotone non-decreasing:", all(rates[i] <= rates[i + 1] for i in range(9)),
          "decreases at:", [i + 1 for i in range(9) if rates[i + 1] < rates[i]])
    edges = np.percentile(fm1, np.linspace(0, 100, 11))
    cnt, rr, cen = [], [], []
    for i in range(10):
        m = (fm1 >= edges[i]) & ((fm1 < edges[i + 1]) if i < 9 else (fm1 <= edges[i + 1]))
        cnt.append(int(m.sum())); rr.append(round(lb[m].mean(), 4)); cen.append(round(fm1[m].mean(), 4))
    print(name, "calib counts", cnt, "rates", rr, "centers", cen,
          "monotone:", all(rr[i] <= rr[i + 1] for i in range(9)))

# ---------------------------------------------------------------- synthetic pool
hdr("Table 8 synthetic pool")
sd = load(f"{C}/synth_qa_data.json"); se = load(f"{C}/synth_qa_entailment.json")
sl = np.array([x["entail_label"] for x in se])
print("pool n", len(sd), "acc", sl.mean(), "2400 yield", len(sd) / 2400)
topics = np.array([r["topic"] for r in sd]); tiers = np.array([r["tier"] for r in sd])
for t in sorted(set(topics), key=lambda t: -sl[topics == t].mean()):
    print(" topic", t, int((topics == t).sum()), sl[topics == t].mean())
for t in sorted(set(tiers)):
    print(" tier", t, int((tiers == t).sum()), sl[tiers == t].mean())
cell = Counter(zip(topics, tiers)); print(" cell min/max n", min(cell.values()), max(cell.values()))

hdr("Table 9 composition + partition (C5)")
pair = load(f"{C}/synthetic_a_pair_indices.json")
print({k: pair[k] for k in ["alpha", "seed", "K", "fm1_cutoff", "n_pool", "A_topics", "B_topics",
                            "source_topic_hist", "target_topic_hist", "source_acc", "target_acc"]})
si, ti = pair["source_idx"], pair["target_idx"]
print("n_S", len(si), "n_T", len(ti), "overlap", len(set(si) & set(ti)))
src_c = Counter(topics[si]); tgt_c = Counter(topics[ti])
for t in sorted(set(topics), key=lambda t: -src_c[t]):
    print(f" {t:28s} S={src_c[t]:4d} T={tgt_c[t]:4d}")
print("acc_S", sl[si].mean(), "acc_T", sl[ti].mean())
# cached source/target slices consistency
ss = load(f"{C}/synth_a_source_entailment.json"); tsx = load(f"{C}/synth_a_target_entailment.json")
print("cached slice acc", np.mean([x["entail_label"] for x in ss]), np.mean([x["entail_label"] for x in tsx]))

emb = np.load(f"{C}/synth_qa_embeddings.npy")
print("synth emb shape", emb.shape, emb.dtype)
from sklearn.cluster import KMeans
km = KMeans(n_clusters=10, random_state=42, n_init=10).fit(emb)
lbls = km.labels_
loc = {g: i for i, g in enumerate(range(len(sd)))}
src_hist = np.bincount(lbls[si], minlength=10).tolist(); tgt_hist = np.bincount(lbls[ti], minlength=10).tolist()
print("KMeans reproduces stored hist:", src_hist == pair["source_topic_hist"], tgt_hist == pair["target_topic_hist"])
cl_acc = np.array([sl[lbls == k].mean() for k in range(10)])
order = np.argsort(-cl_acc)
print("A (top5 by acc):", sorted(order[:5].tolist()), " B:", sorted(order[5:].tolist()),
      " stored A", pair["A_topics"], " stored B", pair["B_topics"])
for k in order:
    m = lbls == k
    comp = Counter(topics[m]).most_common(3)
    print(f" cluster {k}: {'A' if k in pair['A_topics'] else 'B'} n={int(m.sum())} acc={cl_acc[k]:.4f} "
          f"S={src_hist[k]} T={tgt_hist[k]} top topics={comp}")
for t in ["literature and authors", "music and composers"]:
    m = topics == t
    print(t, "cluster membership", Counter(lbls[m].tolist()))
print("A mean acc", cl_acc[pair["A_topics"]].mean(), "B mean acc", cl_acc[pair["B_topics"]].mean())

hdr("Plots in repo")
pngs = sorted(os.listdir(os.path.join(ROOT, "plots"))); print(len(pngs), pngs)
