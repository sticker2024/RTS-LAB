"""Project 2.3 - Partitioned multiprocessor scheduling simulator (FF, FFD, BFD, WF).

Per-core admission test: sum(u) + u_new <= rm_bound(n_on_core + 1), recomputed per core.
Every heuristic has the same signature:  heuristic(tasks, m) -> (cores, unplaced)
  cores    : list of m lists of Task
  unplaced : list of Task that fit on no core
"""
from __future__ import annotations
import csv
from math import ceil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rts_sim import Task, rm_bound, total_utilization, hyperperiod, simulate_rm, plot_gantt

EPS = 1e-12


def load(core):
    return sum(t.U for t in core)


def admissible(core, task) -> bool:
    """Liu & Layland test with n = tasks already on the core + 1."""
    return load(core) + task.U <= rm_bound(len(core) + 1) + EPS


def remaining(core, task) -> float:
    """Spare RM capacity on the core if `task` were added."""
    return rm_bound(len(core) + 1) - (load(core) + task.U)


def first_fit(tasks, m):
    cores, unplaced = [[] for _ in range(m)], []
    for t in tasks:                                   # input order
        for c in cores:
            if admissible(c, t):
                c.append(t); break
        else:
            unplaced.append(t)
    return cores, unplaced


def first_fit_decreasing(tasks, m):
    return first_fit(sorted(tasks, key=lambda t: -t.U), m)


def best_fit_decreasing(tasks, m):
    cores, unplaced = [[] for _ in range(m)], []
    for t in sorted(tasks, key=lambda t: -t.U):
        cand = [c for c in cores if admissible(c, t)]
        if cand:
            min(cand, key=lambda c: remaining(c, t)).append(t)   # tightest fit
        else:
            unplaced.append(t)
    return cores, unplaced


def worst_fit(tasks, m):
    cores, unplaced = [[] for _ in range(m)], []
    for t in tasks:                                   # input order
        cand = [c for c in cores if admissible(c, t)]
        if cand:
            max(cand, key=lambda c: remaining(c, t)).append(t)   # emptiest core
        else:
            unplaced.append(t)
    return cores, unplaced


HEURISTICS = {"FF": first_fit, "FFD": first_fit_decreasing, "BFD": best_fit_decreasing, "WF": worst_fit}


def summarize(cores, unplaced, m):
    placed_u = sum(load(c) for c in cores)
    return dict(ok=not unplaced,
                cores_used=sum(1 for c in cores if c),
                per_core=[round(load(c), 3) for c in cores],
                placed_u=placed_u, pct_of_m=100 * placed_u / m,
                unplaced=[t.name for t in unplaced])


def min_cores(heuristic, tasks, limit=None):
    """Smallest m for which the heuristic partitions everything (None if never)."""
    for m in range(1, (limit or len(tasks)) + 1):
        if not heuristic(tasks, m)[1]:
            return m
    return None


# ------------------------------------------------------------ test sets
def make_sets():
    s = {}
    s["1 balanced"] = [Task(f"b{i}", 1, 4) for i in range(12)]                     # 12 x 0.25
    s["2 skewed"] = ([Task("big1", 3, 5), Task("big2", 3, 5), Task("big3", 11, 20)]  # 0.6, 0.6, 0.55
                     + [Task(f"s{i}", 1, 10) for i in range(9)])                     # 9 x 0.1
    s["3 tight"] = ([Task(f"h{i}", 9, 10) for i in range(4)]                         # 4 x 0.9
                    + [Task(f"l{i}", 1, 20) for i in range(4)])                      # 4 x 0.05
    s["4 adversarial"] = [Task(f"a{i}", 51, 100) for i in range(5)]                  # 5 x 0.51
    s["5 order-sensitive"] = ([Task(f"sm{i}", 3, 20) for i in range(4)]              # 4 x 0.15 first
                              + [Task(f"lg{i}", 1, 2) for i in range(4)])            # then 4 x 0.5
    # utilizations 0.10, 0.35, 0.45, 0.40, 0.50 (period 100); only meaningful for m = 3
    s["6 WF-hostile (m=3)"] = [Task(f"w{i}", p, 100) for i, p in enumerate([10, 35, 45, 40, 50])]
    return s


def run_all(m=4, csv_path="partition_results.csv"):
    sets = make_sets()
    rows = []
    print(f"m = {m} cores\n")
    print(f"{'set':<20}{'U_total':>8}{'U/m':>7}  {'heur':<5}{'result':<10}{'cores':>6}{'U placed/m':>12}  per-core utilization")
    for name, ts in sets.items():
        U = total_utilization(ts)
        m_set = 3 if "m=3" in name else m
        for h, fn in HEURISTICS.items():
            cores, un = fn(ts, m_set)
            r = summarize(cores, un, m_set)
            res = "OK" if r["ok"] else f"FAIL({len(un)})"
            print(f"{name:<20}{U:>8.3f}{100*U/m_set:>6.0f}%  {h:<5}{res:<10}{r['cores_used']:>6}{r['pct_of_m']:>11.1f}%  {r['per_core']}")
            rows.append([name, round(U, 3), h, res, r["cores_used"], round(r["pct_of_m"], 1), r["per_core"], r["unplaced"]])
        print(f"{'':<20}lower bound ceil(U) = {ceil(U)} cores; min cores needed: "
              + ", ".join(f"{h}={min_cores(fn, ts)}" for h, fn in HEURISTICS.items()))
        print()
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "U_total", "heuristic", "result", "cores_used", "placed_pct_of_m", "per_core_U", "unplaced"])
        w.writerows(rows)
    return sets


def plot_partition(cores, path, title):
    placed = [t for c in cores for t in c]
    names = [t.name for t in placed]
    H = hyperperiod(placed)
    fig, axs = plt.subplots(len(cores), 1, figsize=(11, 1.0 + 1.9 * len(cores)), sharex=True)
    for i, (ax, c) in enumerate(zip(axs, cores)):
        if not c:
            ax.text(0.5, 0.5, "core empty", ha="center", va="center", transform=ax.transAxes); ax.set_yticks([])
            ax.set_ylabel(f"core {i}"); continue
        tl, miss, _ = simulate_rm(c, H)
        plot_gantt(ax, tl, [t.name for t in c], f"core {i}: U = {load(c):.3f}, deadline misses = {len(miss)}")
    axs[-1].set_xlabel("time")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    sets = run_all(4)
    ts = sets["5 order-sensitive"]
    cores, un = first_fit_decreasing(ts, 4)
    assert not un
    plot_partition(cores, "partition_gantt.png", "Set 5 partitioned with FFD (per-core RM)")
    # explicit check that every core meets its deadlines in simulation
    for i, c in enumerate(cores):
        print(f"core {i}: {[t.name for t in c]} misses = {simulate_rm(c, hyperperiod(ts))[1]}")
