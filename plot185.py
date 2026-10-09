"""Plot #185 before/after walk scaling from scaling_{before,after}.json."""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = sys.argv[1]
B, A = (json.load(open(f"{d}/scaling_{t}.json")) for t in ("before", "after"))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"KDTree(k=50)": "#2a78d6", "BruteForce": "#1baf7a"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.titleweight": "bold", "axes.titlesize": 11})
LABELS = {}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURF, layout="constrained")

def series(ax, data, key, name, ls, label):
    xs = sorted(int(n) for n in data[key][name]); ys = [data[key][name][str(n)] for n in xs]
    ax.plot(xs, ys, ls=ls, lw=2, marker="o", ms=5, color=COL[name], mec=SURF, mew=1.5, solid_capstyle="round")
    if label: LABELS.setdefault(ax, []).append((label, xs[-1], ys[-1]))

def place_labels(ax, min_gap_pt=12):
    """Direct labels at line ends, nudged apart vertically so none overlap."""
    items = LABELS.get(ax, [])
    fig.canvas.draw()
    to_disp, to_data = ax.transData.transform, ax.transData.inverted().transform
    pts = sorted(((to_disp((x, y))[1], lbl, x, y) for lbl, x, y in items))
    gap = min_gap_pt * fig.dpi / 72
    placed = []
    for yd, lbl, x, y in pts:
        if placed and yd - placed[-1][0] < gap: yd = placed[-1][0] + gap
        placed.append((yd, lbl, x, y))
    for yd, lbl, x, y in placed:
        dy = (yd - to_disp((x, y))[1]) * 72 / fig.dpi
        ax.annotate(lbl, (x, y), xytext=(6, dy), textcoords="offset points", va="center",
                    fontsize=9, color=INK2)

def style(ax, title, ylabel):
    ax.set(xscale="log", yscale="log", title=title, xlabel="number of points N", ylabel=ylabel, facecolor=SURF)
    ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.margins(x=0.28)

ax = axes[0]
for name in COL:
    bf = name == "BruteForce"
    series(ax, B, "per_step_us", name, "--", None if bf else "KDTree, before")
    series(ax, A, "per_step_us", name, "-", "BruteForce (unchanged)" if bf else "KDTree, after")
style(ax, "Cost per walk step", "µs per step (setup excluded)")

ax = axes[1]
series(ax, B, "setup_s", "KDTree(k=50)", "--", "before: tree only")
series(ax, A, "setup_s", "KDTree(k=50)", "-", "after: tree + kNN table")
style(ax, "KDTree one-off setup", "seconds")

ax = axes[2]
for name in COL:
    bf = name == "BruteForce"
    series(ax, B, "full_s", name, "--", None if bf else "KDTree, before")
    series(ax, A, "full_s", name, "-", "BruteForce (unchanged)" if bf else "KDTree, after")
style(ax, "Full walk (all N steps, incl. setup)", "seconds")

fig.legend(handles=[plt.Line2D([], [], color=COL["KDTree(k=50)"], lw=2, label="KDTree(k=50)"),
                    plt.Line2D([], [], color=COL["BruteForce"], lw=2, label="BruteForce"),
                    plt.Line2D([], [], color=INK2, lw=2, ls="--", label="before #185"),
                    plt.Line2D([], [], color=INK2, lw=2, ls="-", label="after #185")],
           loc="outside upper center", ncols=4, frameon=False, fontsize=10)
for ax in axes: place_labels(ax)
fig.savefig(f"{d}/pr185_scaling.png", dpi=150, facecolor=SURF)
print(f"{d}/pr185_scaling.png")
