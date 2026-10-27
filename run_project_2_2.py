"""Validation sets for Project 2.2."""
from rts_sim import Task
from schedulability_tool import analyze

sets = {
    "A_sec222_three_task": [Task("t1", 1, 4), Task("t2", 2, 6), Task("t3", 2, 12)],
    "B_sec232_four_task": [Task("t1", 1, 4), Task("t2", 2, 6), Task("t3", 2, 12), Task("t4", 1, 8)],
    # RM bound fails but RTA proves RM-schedulable (U=0.9167 > bound(2)=0.8284... classic harmonic-ish)
    "C_bound_fails_rta_ok": [Task("a", 2, 5), Task("b", 4, 10), Task("c", 1, 20)],
    "D_overload": [Task("x", 3, 4), Task("y", 2, 5)],
    "E_constrained": [Task("a", 1, 5, 2), Task("b", 2, 8, 8), Task("c", 1, 10, 3)],
    "F_eight_tasks": [Task(f"t{i}", c, t) for i, (c, t) in enumerate(
        [(1, 10), (1, 12), (3, 20), (1, 15), (3, 30), (4, 40), (3, 25), (1, 50)], 1)],
}
for name, ts in sets.items():
    print("=" * 70, f"\n{name}")
    r = analyze(ts, f"cmp_{name}.png")
    print(f"-> RM ok={r['rm_ok']}, EDF ok={r['edf_ok']}\n")
