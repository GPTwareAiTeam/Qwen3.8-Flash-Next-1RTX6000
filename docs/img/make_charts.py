"""
Regenerates docs/img/throughput.svg and docs/img/prompt_resume.svg from docs/img/chart_data.json (the measured values
also listed in docs/measurements.md, section "README run"). Plain SVG, no dependencies:

    python docs/img/make_charts.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"
C_V2, C_V3, AXIS, GRID, TXT = "#9aa5b1", "#2f6fdb", "#444", "#e3e6ea", "#222"


def bars(groups, series, colors, title, ylabel, path, fmt = "{:.0f}"):
    W, H, L, R, T, B = 640, 360, 64, 16, 44, 56
    pw, ph = W - L - R, H - T - B
    vmax = max(v for s in series.values() for v in s)
    top = 10 ** len(str(int(vmax))) / 10
    top = next(t for t in (top * k for k in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10)) if t >= vmax * 1.08)
    o = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='white'/>",
         f"<text x='{W / 2}' y='22' text-anchor='middle' font-size='15' font-weight='600' fill='{TXT}'>{title}</text>"]
    for i in range(6):
        v = top * i / 5
        y = T + ph - ph * i / 5
        o.append(f"<line x1='{L}' x2='{W - R}' y1='{y:.1f}' y2='{y:.1f}' stroke='{GRID}'/>")
        o.append(f"<text x='{L - 6}' y='{y + 4:.1f}' text-anchor='end' font-size='11' fill='{AXIS}'>{v:g}</text>")
    o.append(f"<text x='14' y='{T + ph / 2}' transform='rotate(-90 14 {T + ph / 2})' text-anchor='middle' "
             f"font-size='12' fill='{AXIS}'>{ylabel}</text>")
    n, k = len(groups), len(series)
    gw = pw / n
    bw = gw * 0.7 / k
    for gi, g in enumerate(groups):
        x0 = L + gi * gw + gw * 0.15
        for si, (name, vals) in enumerate(series.items()):
            v = vals[gi]
            h = ph * v / top
            x = x0 + si * bw
            o.append(f"<rect x='{x:.1f}' y='{T + ph - h:.1f}' width='{bw - 2:.1f}' height='{h:.1f}' fill='{colors[name]}'/>")
            o.append(f"<text x='{x + bw / 2 - 1:.1f}' y='{T + ph - h - 4:.1f}' text-anchor='middle' font-size='10' "
                     f"fill='{TXT}'>{fmt.format(v)}</text>")
        o.append(f"<text x='{L + gi * gw + gw / 2:.1f}' y='{T + ph + 16}' text-anchor='middle' font-size='12' "
                 f"fill='{AXIS}'>{g}</text>")
    lx = L + 8
    for name in series:
        o.append(f"<rect x='{lx}' y='{H - 18}' width='12' height='12' fill='{colors[name]}'/>")
        o.append(f"<text x='{lx + 16}' y='{H - 8}' font-size='12' fill='{TXT}'>{name}</text>")
        lx += 16 + 8 * len(name) + 24
    o.append(f"<line x1='{L}' x2='{W - R}' y1='{T + ph}' y2='{T + ph}' stroke='{AXIS}'/>")
    o.append("</svg>")
    open(path, "w").write("\n".join(o) + "\n")


def main():
    d = json.load(open(os.path.join(HERE, "chart_data.json")))
    t = d["throughput"]
    bars([f"{u} user{'s' if u > 1 else ''}" for u in t["users"]], {"V2": t["V2"], "V3": t["V3"]},
         {"V2": C_V2, "V3": C_V3}, t["title"], "total tokens/s", os.path.join(HERE, "throughput.svg"))
    p = d["prompt_resume"]
    names = list(p["series"])
    bars(p["cases"], p["series"], {names[0]: C_V2, names[1]: C_V3}, p["title"],
         "time to first token, s", os.path.join(HERE, "prompt_resume.svg"), fmt = "{:.2f}")


if __name__ == "__main__":
    main()
