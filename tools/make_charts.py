#!/usr/bin/env python3
"""Draw the two README charts from the CSV files in serving/results/.

Standard library only. Run from the repository root:

    python3 tools/make_charts.py

Writes docs/img/score_by_run_{light,dark}.svg and docs/img/throughput_by_load_{light,dark}.svg.
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "img")
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", ours="#2a78d6", stock="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", ours="#3987e5", stock="#d95926"),
}

# Short row labels for the eight scored runs, in order.
RUN_LABELS = {
    1: "NVFP4 expert kernel (replaces Marlin)",
    2: "+ memory fix, split prefill",
    3: "+ KV write-back by span, prefix reuse",
    4: "+ KV pool ratio 0.3 (flag only)",
    5: "+ fused collectives in split prefill",
    6: "+ column-split pooler/unpooler",
    7: "+ per-rank draft context projection",
    8: "same code, second sample",
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def read_csv(name):
    with open(os.path.join(ROOT, "serving", "results", name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def text(x, y, s, fill, size=12, anchor="start", weight="400"):
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" font-family="{FONT}">{s}</text>')


def hbar(x0, x1, y, h, fill, r=4):
    """Horizontal bar: square at the baseline (x0), rounded at the data end (x1)."""
    r = min(r, (x1 - x0) / 2, h / 2)
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z" fill="{fill}"/>')


def vbar(x, w, y_top, y_base, fill, r=4):
    """Column: square at the baseline, rounded on the cap."""
    r = min(r, w / 2, (y_base - y_top) / 2)
    return (f'<path d="M{x:.1f},{y_base:.1f} V{y_top + r:.1f} Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f} '
            f'H{x + w - r:.1f} Q{x + w:.1f},{y_top:.1f} {x + w:.1f},{y_top + r:.1f} V{y_base:.1f} Z" fill="{fill}"/>')


def score_by_run(theme, c):
    runs = read_csv("scored_runs.csv")
    width, left, right, top, pitch, bar_h = 860, 318, 790, 78, 32, 18
    height = top + pitch * len(runs) + 46
    xmax = 2.5
    sx = lambda v: left + (right - left) * v / xmax
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
         f'role="img" aria-label="Official serving score after each of the eight scored runs">',
         f'<rect width="{width}" height="{height}" rx="8" fill="{c["surface"]}"/>',
         text(24, 32, "Official score after each scored run", c["ink"], 15, weight="600"),
         text(24, 52, "Throughput at stock SGLang's interactivity, geometric mean over 1, 4, 16 and 32 users. Stock = 1.00.",
              c["ink2"], 12)]
    base_y = top + pitch * len(runs)
    for t in (0.5, 1.0, 1.5, 2.0, 2.5):
        o.append(f'<line x1="{sx(t):.1f}" y1="{top - 6}" x2="{sx(t):.1f}" y2="{base_y}" stroke="{c["grid"]}" stroke-width="1"/>')
    for t in (0, 0.5, 1.0, 1.5, 2.0, 2.5):
        o.append(text(sx(t), base_y + 18, f"{t:.1f}×" if t else "0", c["muted"], 11, "middle"))
    o.append(f'<line x1="{sx(0):.1f}" y1="{top - 6}" x2="{sx(0):.1f}" y2="{base_y}" stroke="{c["axis"]}" stroke-width="1"/>')
    o.append(text(sx(1.0), base_y + 34, "stock SGLang", c["ink2"], 11, "middle"))
    for i, r in enumerate(runs):
        y = top + pitch * i + (pitch - bar_h) / 2
        step, score = int(r["step"]), float(r["score"])
        month, day = int(r["submitted_utc"][5:7]), int(r["submitted_utc"][8:10])
        o.append(text(24, y + bar_h - 5, f"{MONTHS[month - 1]} {day}", c["muted"], 12))
        o.append(text(78, y + bar_h - 5, RUN_LABELS[step], c["ink2"], 12))
        o.append(hbar(sx(0), sx(score), y, bar_h, c["ours"]))
        o.append(text(sx(score) + 8, y + bar_h - 5, f"{score:.3f}×", c["ink"], 12, weight="600"))
    o.append("</svg>")
    return "\n".join(o)


def throughput_by_load(theme, c):
    rows = [r for r in read_csv("scored_runs_per_load.csv") if r["step"] == "7"]
    width, height, left, right, top, base_y = 640, 372, 64, 612, 96, 318
    ymax = 35000
    sy = lambda v: base_y - (base_y - top) * v / ymax
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
         f'role="img" aria-label="Tokens per second per GPU at stock interactivity, stock SGLang against the best patch, at 1, 4, 16 and 32 users">',
         f'<rect width="{width}" height="{height}" rx="8" fill="{c["surface"]}"/>',
         text(24, 32, "Throughput per GPU at each load, best run (2.437×)", c["ink"], 15, weight="600"),
         text(24, 52, "Tokens/s per GPU at the per-user speed stock reaches at that load (the judge's metric).", c["ink2"], 12),
         f'<rect x="24" y="66" width="12" height="12" rx="2" fill="{c["stock"]}"/>', text(42, 76.5, "stock SGLang", c["ink2"], 12),
         f'<rect x="140" y="66" width="12" height="12" rx="2" fill="{c["ours"]}"/>', text(158, 76.5, "patched", c["ink2"], 12)]
    for t in (10000, 20000, 30000):
        o.append(f'<line x1="{left}" y1="{sy(t):.1f}" x2="{right}" y2="{sy(t):.1f}" stroke="{c["grid"]}" stroke-width="1"/>')
        o.append(text(left - 8, sy(t) + 4, f"{t // 1000}K", c["muted"], 11, "end"))
    o.append(text(left - 8, base_y + 4, "0", c["muted"], 11, "end"))
    o.append(f'<line x1="{left}" y1="{base_y}" x2="{right}" y2="{base_y}" stroke="{c["axis"]}" stroke-width="1"/>')
    group_w, bar_w, gap = (right - left) / len(rows), 24, 2
    for i, r in enumerate(rows):
        cx = left + group_w * (i + 0.5)
        stock, ours = float(r["stock_tok_s_per_gpu"]), float(r["ours_tok_s_per_gpu_at_stock_speed"])
        x_stock, x_ours = cx - bar_w - gap / 2, cx + gap / 2
        o.append(vbar(x_stock, bar_w, sy(stock), base_y, c["stock"]))
        o.append(vbar(x_ours, bar_w, sy(ours), base_y, c["ours"]))
        # Value labels sit on the caps, each on its own side of the pair so they never cross the neighbour.
        o.append(text(x_stock + bar_w - 2, sy(stock) - 6, f"{stock:,.0f}", c["ink"], 11, "end"))
        o.append(text(x_ours + 2, sy(ours) - 6, f"{ours:,.0f}", c["ink"], 11, "start", "600"))
        users = int(r["concurrent_users"])
        o.append(text(cx, base_y + 18, f"{users} user" + ("" if users == 1 else "s"), c["ink2"], 12, "middle"))
        o.append(text(cx, base_y + 34, f"{float(r['ratio_vs_stock']):.2f}×", c["muted"], 11, "middle"))
    o.append("</svg>")
    return "\n".join(o)


def main():
    os.makedirs(OUT, exist_ok=True)
    for theme, colors in THEMES.items():
        for name, draw in (("score_by_run", score_by_run), ("throughput_by_load", throughput_by_load)):
            path = os.path.join(OUT, f"{name}_{theme}.svg")
            with open(path, "w", encoding="utf-8") as f:
                f.write(draw(theme, colors) + "\n")
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
