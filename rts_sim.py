"""rts_sim.py - Real-time scheduling simulator (RM, DM, EDF, LLF, PIP).

Pure Python 3.10+; matplotlib only needed for plotting.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil, gcd
from functools import reduce


# ---------------------------------------------------------------- tasks
@dataclass
class Task:
    name: str
    C: int
    T: int
    D: int | None = None

    def __post_init__(self):
        if self.D is None:
            self.D = self.T          # implicit deadline (Section 2.1.2)

    @property
    def U(self) -> float:
        return self.C / self.T


def total_utilization(tasks) -> float:
    return sum(t.U for t in tasks)


def rm_bound(n: int) -> float:
    """Liu & Layland bound n(2^(1/n) - 1)  (Section 2.2.1)."""
    return n * (2 ** (1 / n) - 1)


def hyperperiod(tasks) -> int:
    return reduce(lambda a, b: a * b // gcd(a, b), (t.T for t in tasks), 1)


# ------------------------------------------------- priority ordering
def priority_order(tasks, policy="RM"):
    """Highest priority first. RM: by period; DM: by relative deadline."""
    key = (lambda t: (t.D, t.T)) if policy == "DM" else (lambda t: (t.T, t.D))
    return sorted(tasks, key=key)


# ------------------------------------------------------------- RTA
def response_time(task, higher, limit=None):
    """R = C + sum ceil(R/Tj) Cj. Returns (R, iterations, converged_within_D)."""
    limit = task.D if limit is None else limit
    R = task.C
    it = 0
    while True:
        it += 1
        new = task.C + sum(ceil(R / h.T) * h.C for h in higher)
        if new == R:
            return R, it, R <= limit
        if new > limit:
            return new, it, False
        R = new


def rta_all(tasks, policy="RM"):
    order = priority_order(tasks, policy)
    out = {}
    for i, t in enumerate(order):
        R, it, ok = response_time(t, order[:i])
        out[t.name] = (R, it, ok)
    return out


# ------------------------------------------------ preemptive simulators
def _simulate(tasks, sim_time, pick):
    """Generic unit-tick simulator. pick(ready, t) returns the job to run.
    A job is a dict: task, release, deadline, rem, id. Returns (timeline, misses, switches)."""
    jobs, timeline, misses = [], [], []
    switches, last = 0, None
    for t in range(sim_time):
        for task in tasks:
            if t % task.T == 0:
                jobs.append(dict(task=task, release=t, deadline=t + task.D, rem=task.C))
        for j in jobs:
            if j["rem"] > 0 and t >= j["deadline"] and not j.get("missed"):
                j["missed"] = True
                misses.append((j["task"].name, j["release"], j["deadline"]))
        ready = [j for j in jobs if j["rem"] > 0]
        if not ready:
            timeline.append(None)
            last = None
            continue
        j = pick(ready, t)
        j["rem"] -= 1
        timeline.append(j["task"].name)
        cur = id(j)
        if last is not None and cur != last:
            # count only genuine preemptions / switches between unfinished work
            switches += 1
        last = cur
    for j in jobs:
        if j["rem"] > 0 and not j.get("missed") and j["deadline"] <= sim_time:
            misses.append((j["task"].name, j["release"], j["deadline"]))
    return timeline, misses, switches


def simulate_rm(tasks, sim_time, policy="RM"):
    rank = {t.name: i for i, t in enumerate(priority_order(tasks, policy))}
    return _simulate(tasks, sim_time,
                     lambda ready, t: min(ready, key=lambda j: (rank[j["task"].name], j["release"])))


def simulate_edf(tasks, sim_time):
    return _simulate(tasks, sim_time,
                     lambda ready, t: min(ready, key=lambda j: (j["deadline"], j["release"])))


def simulate_llf(tasks, sim_time):
    return _simulate(tasks, sim_time,
                     lambda ready, t: min(ready, key=lambda j: (j["deadline"] - t - j["rem"], j["release"])))


# --------------------------------------- priority inversion / PIP
def simulate_inversion(pip: bool, horizon=30):
    """Section 2.6.1 scenario.
    L (prio 1) holds the resource from t=0, needs 6 ticks of CPU inside the critical section.
    H (prio 3) arrives t=1: 1 tick before, 1 tick after... modelled as: needs the resource at once,
    then 2 ticks of work inside the section (the section is a single mutex region for H too).
    M (prio 2) arrives t=2, needs 4 ticks, no resource.
    Returns (timeline, H_blocking_ticks, H_finish)."""
    base = {"L": 1, "M": 2, "H": 3}
    arrive = {"L": 0, "H": 1, "M": 2}
    rem = {"L": 6, "H": 2, "M": 4}      # H: 2 ticks needing the resource
    owner = "L"                         # L holds the mutex from t=0
    L_done_cs = False
    timeline, blocked = [], 0
    for t in range(horizon):
        ready = [n for n in rem if arrive[n] <= t and rem[n] > 0]
        if not ready:
            timeline.append(None)
            continue
        eff = dict(base)
        h_blocked = "H" in ready and owner == "L"
        if h_blocked and pip:
            eff["L"] = max(eff["L"], base["H"])        # priority inheritance
        runnable = [n for n in ready if not (n == "H" and owner == "L")]
        run = max(runnable, key=lambda n: eff[n])
        if h_blocked:
            blocked += 1
        rem[run] -= 1
        timeline.append(run)
        if run == "L" and rem["L"] == 0:
            owner = None                               # L releases; priority restored
        elif run == "H" and rem["H"] == 0:
            owner = None
        if owner is None and "H" in rem and rem["H"] > 0 and owner is None:
            owner = "H"                                # H acquires on release
    H_finish = max(i for i, n in enumerate(timeline) if n == "H") + 1
    return timeline, blocked, H_finish


# ---------------------------------------------------------- plotting
def plot_gantt(ax, timeline, names, title, colors=None):
    cmap = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2", "#9D755D", "#BAB0AC"]
    colors = colors or {n: cmap[i % len(cmap)] for i, n in enumerate(names)}
    for row, n in enumerate(names):
        spans = [(i, 1) for i, x in enumerate(timeline) if x == n]
        ax.broken_barh(spans, (row - 0.4, 0.8), facecolors=colors[n])
    idle = [(i, 1) for i, x in enumerate(timeline) if x is None]
    ax.broken_barh(idle, (len(names) - 0.4, 0.8), facecolors="#DDDDDD", hatch="//")
    ax.set_yticks(range(len(names) + 1))
    ax.set_yticklabels(list(names) + ["idle"])
    ax.set_xlim(0, len(timeline))
    ax.set_xticks(range(0, len(timeline) + 1, max(1, len(timeline) // 24)))
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
