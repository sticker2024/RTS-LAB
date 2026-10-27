"""Project 2.1 driver: utilization checks, RTA, RM/EDF charts, inversion vs PIP, lab tasks."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rts_sim import *

# ---- Steps 2-3: utilization and bounds
set3 = [Task("t1", 1, 4), Task("t2", 2, 6), Task("t3", 2, 12)]
set4 = set3 + [Task("t4", 1, 8)]
for label, ts in (("3-task set (Sec 2.2.2)", set3), ("4-task set (Sec 2.3.2)", set4)):
    U, n = total_utilization(ts), len(ts)
    print(f"{label}: U = {U:.3f}, rm_bound({n}) = {rm_bound(n):.3f}, "
          f"RM sufficient test {'PASS' if U <= rm_bound(n) else 'INCONCLUSIVE'}, "
          f"EDF U<=1 {'PASS' if U <= 1 else 'FAIL'}")

# ---- Step 7: RTA on the 4-task set
print("\nRTA (RM order) on 4-task set:")
for name, (R, it, ok) in rta_all(set4).items():
    D = next(t.D for t in set4 if t.name == name)
    print(f"  {name}: R = {R}, D = {D}, iterations = {it}, {'meets' if ok else 'MISSES'} deadline")

# ---- Steps 4-6: RM vs EDF charts over hyperperiod
H = hyperperiod(set4)
rm_tl, rm_miss, rm_sw = simulate_rm(set4, H)
edf_tl, edf_miss, edf_sw = simulate_edf(set4, H)
print(f"\nHyperperiod = {H}. RM misses: {rm_miss}, switches {rm_sw}; EDF misses: {edf_miss}, switches {edf_sw}")
names = [t.name for t in set4]
fig, axs = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
plot_gantt(axs[0], rm_tl, names, "Rate Monotonic")
plot_gantt(axs[1], edf_tl, names, "Earliest Deadline First")
axs[1].set_xlabel("time (ms)")
fig.tight_layout()
fig.savefig("rm_edf.png", dpi=150)
# individual charts as the deliverable lists four PNGs
for tl, title, fn in ((rm_tl, "Rate Monotonic", "gantt_rm.png"), (edf_tl, "EDF", "gantt_edf.png")):
    f, a = plt.subplots(figsize=(11, 2.8)); plot_gantt(a, tl, names, title); a.set_xlabel("time (ms)")
    f.tight_layout(); f.savefig(fn, dpi=150); plt.close(f)

# ---- Steps 8-10: priority inversion vs PIP
res = {}
for pip in (False, True):
    tl, blocked, fin = simulate_inversion(pip)
    res[pip] = (tl, blocked, fin)
    tl = tl[:max(i for i, x in enumerate(tl) if x is not None) + 1]   # show until all three tasks finish
    f, a = plt.subplots(figsize=(10, 2.6))
    plot_gantt(a, tl, ["L", "M", "H"],
               f"{'With PIP' if pip else 'No protocol (unbounded inversion)'}: H blocked {blocked} ticks, finishes t={fin}")
    a.set_xlabel("time (ticks)"); f.tight_layout()
    f.savefig("gantt_pip.png" if pip else "gantt_inversion.png", dpi=150); plt.close(f)
print("\nPriority inversion (L: 6-tick critical section from t=0, H arrives t=1, M arrives t=2, 4 ticks):")
print(f"  No protocol: H blocked {res[False][1]} ticks, H finishes at t={res[False][2]}")
print(f"  With PIP   : H blocked {res[True][1]} ticks, H finishes at t={res[True][2]}  (bound: one CS of L = 6)")

# ---- Lab task: sweep of the bound, ln 2 limit
import math
ns = list(range(1, 11))
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(ns, [rm_bound(n) for n in ns], "o-", label="RM bound n(2^(1/n)-1)")
ax.axhline(1.0, color="r", ls="--", label="EDF bound = 1")
ax.axhline(math.log(2), color="gray", ls=":", label="ln 2 = 0.693")
ax.set_xlabel("n tasks"); ax.set_ylabel("utilization bound"); ax.legend(); ax.set_ylim(0.6, 1.05)
fig.tight_layout(); fig.savefig("bound_sweep.png", dpi=150); plt.close(fig)

# ---- Lab task: LLF context switches
llf_tl, llf_miss, llf_sw = simulate_llf(set4, H)
print(f"\nContext switches over hyperperiod: RM={rm_sw}, EDF={edf_sw}, LLF={llf_sw} (LLF misses: {llf_miss})")

# ---- Lab task: DM vs RM on a constrained-deadline set
cd = [Task("a", 1, 5, 2), Task("b", 2, 8, 8), Task("c", 1, 10, 3)]
print("\nConstrained deadlines (RM order vs DM order):")
print("  RM order:", [t.name for t in priority_order(cd, "RM")], " DM order:", [t.name for t in priority_order(cd, "DM")])
print("  RM sim misses:", simulate_rm(cd, hyperperiod(cd), "RM")[1])
print("  DM sim misses:", simulate_rm(cd, hyperperiod(cd), "DM")[1])
