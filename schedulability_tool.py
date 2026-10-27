"""Project 2.2 - RM vs EDF schedulability comparison tool.

Usage:
    python schedulability_tool.py tasks.csv [--out chart.png] [--horizon 500]
    or import and call analyze(tasks, chart_path="x.png") with a list of Task objects.

CSV columns: name, C, T, [D]   (D defaults to T)
"""
from __future__ import annotations
import argparse, csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rts_sim import (Task, total_utilization, rm_bound, hyperperiod, priority_order,
                     response_time, simulate_rm, simulate_edf, plot_gantt)


def load_csv(path):
    tasks = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            d = (row.get("D") or "").strip()
            tasks.append(Task(row["name"].strip(), int(row["C"]), int(row["T"]), int(d) if d else None))
    return tasks


def validate(tasks):
    if not tasks:
        raise ValueError("empty task set")
    for t in tasks:
        if t.C <= 0 or t.T <= 0 or t.D <= 0:
            raise ValueError(f"{t.name}: C, T, D must be positive")
        if t.D > t.T:
            raise ValueError(f"{t.name}: D > T (arbitrary deadlines) not supported")


def analyze(tasks, chart_path=None, max_horizon=500, verbose=True):
    """Returns a dict of results; prints a formatted report when verbose."""
    validate(tasks)
    n = len(tasks)
    constrained = any(t.D < t.T for t in tasks)
    policy = "DM" if constrained else "RM"
    U = total_utilization(tasks)
    bound = rm_bound(n)
    res = {"U": U, "bound": bound, "policy": policy, "n": n}
    lines = []
    lines.append(f"Task set: {n} tasks, {'constrained' if constrained else 'implicit'} deadlines, "
                 f"priority order = {policy}")
    lines.append(f"{'task':<8}{'C':>4}{'T':>5}{'D':>5}{'u':>9}")
    for t in tasks:
        lines.append(f"{t.name:<8}{t.C:>4}{t.T:>5}{t.D:>5}{t.U:>9.4f}")
    lines.append(f"Total U = {U:.4f}")

    # --- infeasible: U > 1 can never be scheduled on one processor
    if U > 1:
        res.update(feasible=False, rm_ok=False, edf_ok=False)
        lines.append("INFEASIBLE: U > 1, no uniprocessor schedule can meet all deadlines "
                     "(both RM and EDF fail).")

    # --- RM: sufficient bound (only valid for implicit deadlines), then RTA fallback
    order = priority_order(tasks, policy)
    if not constrained:
        passed = U <= bound
        lines.append(f"RM bound n(2^(1/n)-1) = {bound:.4f} -> {'PASS (schedulable)' if passed else 'FAIL (inconclusive)'}")
    else:
        passed = False
        lines.append(f"RM/DM bound n(2^(1/n)-1) = {bound:.4f} applies to implicit deadlines only -> skipped, using RTA")
    res["bound_pass"] = passed
    rta = {}
    if U <= 1 and not passed:
        lines.append("Response Time Analysis (priority order, highest first):")
        for i, t in enumerate(order):
            R, it, ok = response_time(t, order[:i])
            rta[t.name] = (R, it, ok)
            shown = f"R={R}" if ok else f"R>{t.D} (stopped at {R})"
            lines.append(f"  {t.name:<8}{shown:<22} D={t.D:<4} iters={it}  {'OK' if ok else 'MISS'}")
        rm_ok = all(v[2] for v in rta.values())
        lines.append(f"RM verdict via RTA: {'SCHEDULABLE' if rm_ok else 'NOT schedulable'}")
    else:
        rm_ok = passed
        if U > 1:
            rm_ok = False
    res["rta"] = rta
    res["rm_ok"] = rm_ok

    # --- EDF
    edf_ok = U <= 1
    res["edf_ok"] = edf_ok
    if not constrained:
        lines.append(f"EDF exact test U <= 1 -> {'PASS' if edf_ok else 'FAIL'}")
    else:
        density = sum(t.C / t.D for t in tasks)
        lines.append(f"EDF: U <= 1 -> {'PASS' if edf_ok else 'FAIL'} (necessary only for D<T); "
                     f"density sum C/D = {density:.4f} -> {'sufficient test PASS' if density <= 1 else 'sufficient test inconclusive'}")
        res["density"] = density

    # --- simulation (confirms the verdicts, supplies the chart)
    H = hyperperiod(tasks)
    horizon = min(H, max_horizon)
    rm_tl, rm_miss, _ = simulate_rm(tasks, horizon, policy)
    edf_tl, edf_miss, _ = simulate_edf(tasks, horizon)
    res.update(hyperperiod=H, horizon=horizon, rm_misses=rm_miss, edf_misses=edf_miss)
    lines.append(f"Simulation over {horizon} ticks (hyperperiod {H}): "
                 f"RM misses = {len(rm_miss)}, EDF misses = {len(edf_miss)}")
    if horizon < H:
        lines.append("  (horizon capped; simulation does not cover a full hyperperiod)")

    if chart_path:
        names = [t.name for t in tasks]
        fig, axs = plt.subplots(2, 1, figsize=(max(8, min(18, horizon * 0.35)), 5), sharex=True)
        plot_gantt(axs[0], rm_tl, names, f"{policy} - {len(rm_miss)} deadline miss(es)")
        plot_gantt(axs[1], edf_tl, names, f"EDF - {len(edf_miss)} deadline miss(es)")
        axs[1].set_xlabel("time")
        fig.tight_layout(); fig.savefig(chart_path, dpi=140); plt.close(fig)
        res["chart"] = chart_path
    res["report"] = "\n".join(lines)
    if verbose:
        print(res["report"])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--out", default="comparison.png")
    ap.add_argument("--horizon", type=int, default=500)
    a = ap.parse_args(argv)
    try:
        analyze(load_csv(a.csv), a.out, a.horizon)
    except (ValueError, KeyError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
