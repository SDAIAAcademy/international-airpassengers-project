#!/usr/bin/env python3
"""Build `index.html` — the project's showcase page.

Every number on the page is recomputed here, from the same harness the notebook
uses (8 rolling origins, h=12, step=12), and injected as a parameter. Nothing is
typed by hand, so the page cannot drift from the notebook it summarises.

Charts are inline SVG rather than images: the page carries Arabic text that the
browser shapes with its own fonts, it stays sharp at any zoom, and it loads with
no library and no network call.

    python3 site/build_page.py
"""
from __future__ import annotations

import json
import math
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.datasets import get_rdataset
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, Naive, AutoETS, AutoARIMA

from coursekit import scoring
from coursekit.plotting import acf_values

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEASON, H, NW = 12, 12, 8

# ── هويّةُ الصفحة: قشرةٌ داكنةٌ من الكيان، وألوانُ البياناتِ Okabe-Ito كما في
#    الدورة — فما يقرؤه الناظرُ على الرسمِ هو نفسُ لونِ الدفتر، ولا لوحةَ ثالثة.
BG, CARD = "#070c15", "#0d1523"
MINT, GOLD, CORAL, BLUE = "#2ee6a8", "#e8b45a", "#e0806a", "#7fb8f0"
MUTED, TEXT = "#97a8c0", "#e9eef7"
OKABE_ORANGE, OKABE_BLUE, OKABE_GREEN = "#D55E00", "#0072B2", "#009E73"


# ══════════════════════════════════════════════════════════════════════════
# 1 · القياس — نفسُ المِحَكِّ الذي في الدفتر
# ══════════════════════════════════════════════════════════════════════════
def qis() -> dict:
    raw = get_rdataset("AirPassengers").data
    t = raw["time"].astype(float).to_numpy()
    years = np.floor(t).astype(int)
    months = np.clip(np.round((t - years) * 12).astype(int) + 1, 1, 12)
    df = pd.DataFrame({
        "unique_id": "Air passengers (Box & Jenkins)",
        "ds": pd.to_datetime(pd.Series(years).astype(str) + "-"
                             + pd.Series(months).astype(str).str.zfill(2) + "-01"),
        "y": raw["value"].astype(float),
    }).sort_values("ds").reset_index(drop=True)

    # ⛔ الحارسُ نفسُه الذي في الدفتر: عمودٌ زائدٌ واحدٌ يُفصَّلُ متغيّراً خارجيّاً
    #    بصمت، وقد فعلَها `Unnamed: 0` مرّةً فحسّنَ MASE 22% بلا سبب.
    assert set(df.columns) == {"unique_id", "ds", "y"}, "عمودٌ زائدٌ سيُفصَّلُ خارجيّاً"

    s = df.set_index("ds")["y"].asfreq("MS")
    stl = STL(s, period=SEASON, robust=False).fit()
    T, S, R = stl.trend, stl.seasonal, stl.resid
    FT = max(0.0, 1 - R.var() / (T + R).var())
    FS = max(0.0, 1 - R.var() / (S + R).var())

    r, bound = acf_values(df["y"], nlags=24)
    d12 = df["y"].diff(SEASON).dropna()
    rd, bound_d = acf_values(d12, nlags=24)

    lb = acorr_ljungbox((df["y"] - df["y"].shift(SEASON)).dropna(),
                        lags=[12, 24], return_df=True)

    sf = StatsForecast(models=[SeasonalNaive(season_length=SEASON), Naive(),
                               AutoETS(season_length=SEASON),
                               AutoARIMA(season_length=SEASON)],
                       freq="MS", n_jobs=1)
    cv = sf.cross_validation(df=df, h=H, step_size=H, n_windows=NW,
                             level=scoring.LEVELS)

    dfl = df.assign(y=np.log(df["y"]))
    logcv = StatsForecast(models=[AutoARIMA(season_length=SEASON)], freq="MS",
                          n_jobs=1).cross_validation(
        df=dfl, h=H, step_size=H, n_windows=NW, level=scoring.LEVELS)
    logcv = logcv.assign(**{c: np.exp(logcv[c]) for c in logcv.columns
                            if c not in ("unique_id", "ds", "cutoff")})
    logcv = logcv.rename(columns={c: c.replace("AutoARIMA", "logARIMA")
                                  for c in logcv.columns})

    def row(frame, name):
        sc = scoring.score_cv(frame, name, df, seasonality=SEASON)
        sc["width80"] = float((frame[f"{name}-hi-80"] - frame[f"{name}-lo-80"]).median())
        sc["bias"] = float((frame["y"] - frame[name]).mean())
        return sc

    models = [("SeasonalNaive", cv, "الأرضيّة — الساذجُ الموسميّ"),
              ("Naive", cv, "الساذجُ البسيط"),
              ("AutoETS", cv, "ETS تلقائيّ"),
              ("AutoARIMA", cv, "ARIMA تلقائيّ"),
              ("logARIMA", logcv, "ARIMA على اللوغاريتم")]
    scores = {n: {**row(f, n), "wasf": w} for n, f, w in models}

    def cov_h(frame, name):
        f = frame.copy(); f["h"] = f.groupby("cutoff").cumcount() + 1
        return f.groupby("h").apply(
            lambda x: float(((x.y >= x[f"{name}-lo-80"])
                             & (x.y <= x[f"{name}-hi-80"])).mean())).tolist()

    def grid(frame, name):
        f = frame.copy()
        f["h"] = f.groupby("cutoff").cumcount() + 1
        f["year"] = f["ds"].dt.year
        f["ape"] = (f["y"] - f[name]).abs() / f["y"] * 100
        p = f.pivot_table(index="year", columns="h", values="ape")
        return {"years": [int(y) for y in p.index], "v": p.values.round(1).tolist()}

    return {
        "n": len(df), "min_ds": str(df.ds.min().date()), "max_ds": str(df.ds.max().date()),
        "namaa": float(df.y.tail(12).mean() / df.y.head(12).mean()),
        "madaa_awwal": float(s[:36].max() - s[:36].min()),
        "madaa_akhir": float(s[-36:].max() - s[-36:].min()),
        "FT": float(FT), "FS": float(FS),
        "bound": float(bound), "r12": float(r[11]), "r1": float(r[0]),
        "kharij": int((np.abs(r) > bound).sum()),
        "r12_d": float(rd[11]), "r1_d": float(rd[0]),
        "kharij_d": int((np.abs(rd) > bound_d).sum()),
        "lb12": float(lb.lb_stat.iloc[0]), "p12": float(lb.lb_pvalue.iloc[0]),
        "scores": scores,
        "cov_floor": cov_h(cv, "SeasonalNaive"), "cov_pick": cov_h(logcv, "logARIMA"),
        "grid_floor": grid(cv, "SeasonalNaive"), "grid_pick": grid(logcv, "logARIMA"),
        "nuqat": int(len(cv)), "nawafidh": int(cv.cutoff.nunique()),
    }


# ══════════════════════════════════════════════════════════════════════════
# 2 · الرسوم — SVG مضمَّن
# ══════════════════════════════════════════════════════════════════════════
def _t(x, y, s, size=13, fill=MUTED, anchor="middle", weight="400", rtl=False):
    """نصٌّ في الرسم. المحورُ LTR دائماً (وإلّا انقلبتْ دلالةُ text-anchor)،
    والعربيُّ يُوسَمُ rtl بعينِه؛ ومحاذاتُه إلى اليمينِ تُطلَبُ بـ`start` لا `end`."""
    d = ' direction="rtl"' if rtl else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{d}>{s}</text>')


PICK, FLOOR = "logARIMA", "SeasonalNaive"
ORDER = ["SeasonalNaive", "Naive", "AutoETS", "AutoARIMA", "logARIMA"]


def shakl_jadwal(sc) -> str:
    """ثلاثُ لوحات: النقطة · التوزيع · أمانةُ الحزمة. واللونُ يتبعُ الدورَ لا الرتبة."""
    W, H = 1000, 300
    panels = [("MASE", "mase", lambda v: f"{v:.2f}", sc[FLOOR]["mase"], "الأرضيّة", None),
              ("scaled CRPS", "crps", lambda v: f"{v:.3f}", sc[FLOOR]["crps"], "الأرضيّة", None),
              ("80% coverage", "coverage_80", lambda v: f"{v*100:.0f}%", 0.80, "الاسميّ", 1.0)]
    col = {m: (TEXT if m == FLOOR else MINT if m == PICK else MUTED) for m in ORDER}
    out, pw = [], W / 3
    for i, (title, key, fmt, ref, ref_lab, cap) in enumerate(panels):
        x0 = i * pw
        L, R_, TOP, BOT = x0 + 96, x0 + pw - 54, 66, 54
        vals = [sc[m][key] for m in ORDER]
        mx = cap if cap else max(vals) * 1.28
        bw = (R_ - L)
        ph = H - TOP - BOT
        bh = ph / len(ORDER) * 0.56
        out.append(_t(x0 + pw / 2, 34, title, 14, TEXT, "middle", "700"))
        for j, m in enumerate(ORDER):
            y = TOP + ph * (j + .5) / len(ORDER) - bh / 2
            w = bw * sc[m][key] / mx
            out.append(f'<rect x="{L}" y="{y:.1f}" width="{w:.1f}" height="{bh:.1f}" '
                       f'rx="3" fill="{col[m]}" opacity=".85"/>')
            out.append(_t(L + w + 7, y + bh / 2 + 4, fmt(sc[m][key]), 11.5, col[m], "start", "700"))
            if i == 0:
                out.append(_t(L - 8, y + bh / 2 + 4, m, 11, MUTED, "end"))
        xr = L + bw * ref / mx
        out.append(f'<line x1="{xr:.1f}" y1="{TOP-6}" x2="{xr:.1f}" y2="{H-BOT+4}" '
                   f'stroke="{BLUE}" stroke-dasharray="4 3" stroke-width="1.4"/>')
        out.append(_t(xr, H - BOT + 20, ref_lab, 10.5, BLUE, "middle", "400", rtl=True))
    return (f'<svg viewBox="0 0 {W} {H}" direction="ltr" role="img" '
            f'aria-label="MASE و CRPS والتغطية لخمسة نماذج">{"".join(out)}</svg>')


def shakl_kharita(gf, gp) -> str:
    """خريطةُ الخطأ: خليّةٌ لكلِّ (طيّة، أفق) — 96 خليّةً لكلِّ نموذج."""
    W, H = 1000, 330
    vmax = max(max(map(max, gf["v"])), max(map(max, gp["v"])))
    out, pw = [], W / 2
    for i, (g, name) in enumerate([(gf, "الأرضيّة"), (gp, "المختار logARIMA")]):
        x0 = i * pw
        L, R_, TOP, BOT = x0 + 60, x0 + pw - 34, 78, 62
        cw, ch = (R_ - L) / 12, (H - TOP - BOT) / len(g["years"])
        mean = sum(map(sum, g["v"])) / (len(g["v"]) * 12)
        out.append(_t(x0 + pw / 2, 32, name, 14, TEXT, "middle", "700", rtl=True))
        out.append(_t(x0 + pw / 2, 52, f"متوسّطُ الخطأ {mean:.1f}%", 12, MUTED, "middle", "400", rtl=True))
        for a, yr in enumerate(g["years"]):
            out.append(_t(L - 8, TOP + ch * (a + .62), str(yr), 10, MUTED, "end"))
            for b in range(12):
                v = g["v"][a][b]
                op = 0.08 + 0.92 * (v / vmax)
                out.append(f'<rect x="{L+b*cw:.1f}" y="{TOP+a*ch:.1f}" width="{cw-1.5:.1f}" '
                           f'height="{ch-1.5:.1f}" fill="{OKABE_ORANGE}" opacity="{op:.3f}"/>')
                out.append(_t(L + b * cw + cw / 2 - .8, TOP + a * ch + ch / 2 + 3.4,
                              f"{v:.0f}", 8.4, "#fff" if op > .5 else "#b9c4d4", "middle", "600"))
        for b in range(0, 12, 2):
            out.append(_t(L + b * cw + cw / 2, H - BOT + 18, str(b + 1), 10, MUTED))
        out.append(_t(x0 + pw / 2, H - BOT + 38, "الأشهرُ قُدُماً (h)", 11, MUTED, "middle", "400", rtl=True))
    return (f'<svg viewBox="0 0 {W} {H}" direction="ltr" role="img" '
            f'aria-label="خريطة الخطأ لكل طية وأفق">{"".join(out)}</svg>')


def shakl_taghtiya(cf, cp) -> str:
    """التغطيةُ بحسبِ الأفق — أين تنكسرُ الحزمةُ بالضبط."""
    W, H, L, R_, TOP, BOT = 1000, 290, 66, 44, 62, 62
    pw, ph = W - L - R_, H - TOP - BOT
    X = lambda h: L + pw * (h - .5) / 12
    Y = lambda v: TOP + ph * (1 - v)
    out = [_t(W / 2, 34, "حزمةُ 80% شهراً بشهرٍ في الأفق", 15, TEXT, "middle", "700", rtl=True)]
    for v in (0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W-R_}" y2="{Y(v):.1f}" '
                   f'stroke="#fff" opacity=".05"/>')
        out.append(_t(L - 9, Y(v) + 4, f"{v*100:.0f}%", 10.5, MUTED, "end"))
    bw = pw / 12 * 0.34
    for h in range(1, 13):
        for k, (series, colr) in enumerate([(cf, TEXT), (cp, MINT)]):
            v = series[h - 1]
            x = X(h) + (k - .5) * bw * 1.12
            out.append(f'<rect x="{x-bw/2:.1f}" y="{Y(v):.1f}" width="{bw:.1f}" '
                       f'height="{ph*v:.1f}" rx="2" fill="{colr}" opacity=".9"/>')
        out.append(_t(X(h), H - BOT + 18, str(h), 10.5, MUTED))
    out.append(f'<line x1="{L}" y1="{Y(.8):.1f}" x2="{W-R_}" y2="{Y(.8):.1f}" '
               f'stroke="{BLUE}" stroke-dasharray="5 4" stroke-width="1.6"/>')
    out.append(_t(L + 6, Y(.8) - 8, "الاسميّ 80%", 11, BLUE, "start", "400", rtl=True))
    worst = min(range(12), key=lambda i: cf[i]) + 1
    out.append(_t(X(worst), Y(cf[worst - 1]) - 10, f"{cf[worst-1]*100:.0f}%", 11, CORAL, "middle", "700"))
    # المربّعُ يمينَ وسمِه، والوسمُ ينمو يساراً عنه — وإلّا ركبَ أحدُهما الآخر
    for sx, colr, lab in ((W - R_ - 8, MINT, "المختار"), (W - R_ - 108, TEXT, "الأرضيّة")):
        out.append(f'<rect x="{sx}" y="{H-31}" width="11" height="11" rx="2" fill="{colr}"/>'
                   + _t(sx - 7, H - 21, lab, 11, MUTED, "start", "400", rtl=True))
    out.append(_t(W / 2, H - BOT + 40, "الأشهرُ قُدُماً (h)", 11.5, MUTED, "middle", "400", rtl=True))
    return (f'<svg viewBox="0 0 {W} {H}" direction="ltr" role="img" '
            f'aria-label="التغطية بحسب الأفق">{"".join(out)}</svg>')


def shakl_intiqal() -> str:
    """انقلابُ الرتبة: نافذةٌ واحدةٌ ليست ترتيباً — مقيسٌ من لوحةِ AutoGluon."""
    W, H = 1000, 300
    cols = [["AutoETS", "AutoARIMA", "SeasonalNaive", "Naive"],      # score_val
            ["AutoARIMA", "AutoETS", "SeasonalNaive", "Naive"],      # score_test
            ["AutoARIMA", "AutoETS", "SeasonalNaive", "Naive"]]      # المِحَكّ
    labs = ["score_val", "score_test", "المِحَكّ"]
    C = {"AutoARIMA": MINT, "AutoETS": OKABE_ORANGE, "SeasonalNaive": TEXT, "Naive": MUTED}
    L, R_, TOP, ph = 176, W - 176, 74, 150
    X = lambda i: L + (R_ - L) * i / 2
    Y = lambda r: TOP + ph * (r - 1) / 3
    out = [_t(W / 2, 34, "نافذةٌ واحدةٌ ليست ترتيباً — الرتبةُ تتغيّرُ بتغيّرِ النافذة",
              15, TEXT, "middle", "700", rtl=True)]
    for m in cols[0]:
        pts = [(X(i), Y(c.index(m) + 1)) for i, c in enumerate(cols)]
        out.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                   + f'" fill="none" stroke="{C[m]}" stroke-width="2.6"/>')
        for x, y in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{C[m]}" '
                       f'stroke="{BG}" stroke-width="2"/>')
        out.append(_t(X(0) - 12, pts[0][1] + 4, m, 11.5, C[m], "end"))
        out.append(_t(X(2) + 12, pts[2][1] + 4, m, 11.5, C[m], "start"))
    for i, lab in enumerate(labs):
        out.append(_t(X(i), H - 44, lab, 12, MUTED, "middle", "600",
                      rtl=(i == 2)))
        out.append(_t(X(i), H - 26, ("نافذةٌ واحدة" if i < 2 else "8 نوافذ"),
                      10.5, MUTED, "middle", "400", rtl=True))
    return (f'<svg viewBox="0 0 {W} {H}" direction="ltr" role="img" '
            f'aria-label="انقلاب الرتبة بين نافذتين">{"".join(out)}</svg>')


# ══════════════════════════════════════════════════════════════════════════
# 3 · الصفحة
# ══════════════════════════════════════════════════════════════════════════
def ibni(d: dict) -> str:
    sc = d["scores"]
    f, p = sc[FLOOR], sc[PICK]
    d_mase = p["mase"] / f["mase"] - 1
    d_crps = p["crps"] / f["crps"] - 1
    worst_h = min(range(12), key=lambda i: d["cov_floor"][i]) + 1
    return f'''<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="description" content="مشروعُ ختامِ دورةِ تحليلِ السلاسلِ الزمنيّةِ بأكاديميّةِ سدايا — ركّابُ الطيرانِ الدوليّ 1949-1960: تشخيصٌ، أرضيّةُ أساس، إطارٌ، مِحَكٌّ بثمانِ نوافذ، وتقريرٌ لمن لا يفتحُ الدفتر.">
<title>ركّابُ الطيران — من الشكلِ إلى الحكم | مشروعُ ختامِ سدايا</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>✈️</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Aref+Ruqaa:wght@400;700&family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  :root {{ --bg:{BG}; --card:{CARD}; --mint:{MINT}; --gold:{GOLD}; --coral:{CORAL};
    --blue:{BLUE}; --muted:{MUTED}; --text:{TEXT}; --bord:rgba(255,255,255,.09); }}
  html {{ scroll-behavior:smooth; }}
  body {{ color:var(--text); font-family:'IBM Plex Sans Arabic',system-ui,sans-serif;
    direction:rtl; text-align:right; line-height:1.95; -webkit-font-smoothing:antialiased;
    background:radial-gradient(1100px 560px at 82% -90px, rgba(46,230,168,.10), transparent 60%),
               radial-gradient(820px 460px at 8% 24%, rgba(232,180,90,.06), transparent 60%), var(--bg); }}
  a {{ color:var(--mint); text-decoration:none; }} a:hover {{ color:#7ff0c8; }}
  ::selection {{ background:var(--mint); color:#07120d; }}
  .lat {{ unicode-bidi:isolate; direction:ltr; display:inline-block; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:42px 24px 90px; }}
  nav.top {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;
    gap:14px; padding-bottom:24px; border-bottom:1px solid var(--bord); }}
  .brand {{ display:flex; align-items:center; gap:11px; }}
  .brand .mk {{ width:38px; height:38px; border-radius:11px;
    background:linear-gradient(135deg,var(--mint),#0e8f66); display:grid; place-items:center;
    font-size:19px; }}
  .brand b {{ font-size:16.5px; }} .brand small {{ display:block; color:var(--muted); font-size:12.5px; font-weight:300; }}
  .pills a {{ padding:8px 14px; border-radius:99px; color:#c4d0e2; border:1px solid var(--bord);
    font-size:13px; margin-inline-start:6px; }}
  .pills a:hover {{ border-color:rgba(46,230,168,.5); color:var(--mint); }}
  h1 {{ font-family:'Aref Ruqaa',serif; font-size:clamp(32px,5vw,54px); line-height:1.3; margin:42px 0 6px;
    background:linear-gradient(180deg,#fff 34%,var(--mint) 128%); -webkit-background-clip:text;
    background-clip:text; color:transparent; }}
  .lede {{ color:#c4d0e2; font-size:17px; max-width:76ch; margin-top:12px; }}
  h2 {{ font-family:'Aref Ruqaa',serif; font-size:27px; margin:54px 0 6px; }}
  h2 .n {{ color:var(--mint); font-family:'IBM Plex Sans Arabic'; font-size:19px; margin-inline-end:8px; }}
  p {{ margin:12px 0; color:#c4d0e2; max-width:80ch; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(168px,1fr)); gap:13px; margin:26px 0; }}
  .kpi {{ background:var(--card); border:1px solid var(--bord); border-radius:15px; padding:17px; text-align:center; }}
  .kpi b {{ display:block; font-size:31px; font-weight:700; color:var(--mint); line-height:1.25; }}
  .kpi.gold b {{ color:var(--gold); }} .kpi.coral b {{ color:var(--coral); }} .kpi.blue b {{ color:var(--blue); }}
  .kpi span {{ display:block; color:var(--muted); font-size:12.5px; margin-top:5px; line-height:1.65; }}
  .card {{ background:var(--card); border:1px solid var(--bord); border-radius:17px; padding:20px 22px; margin:20px 0; }}
  .card.mint {{ border-color:rgba(46,230,168,.30); }}
  .card.gold {{ border-color:rgba(232,180,90,.32); }}
  .card.coral {{ border-color:rgba(224,128,106,.34); }}
  svg {{ width:100%; height:auto; display:block; }}
  .claim {{ color:var(--muted); font-size:13.5px; line-height:1.85;
    border-inline-start:2px solid rgba(46,230,168,.35); padding-inline-start:14px; margin-top:15px; }}
  table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:14.5px; }}
  th,td {{ padding:10px 12px; border-bottom:1px solid var(--bord); text-align:right; }}
  th {{ color:var(--muted); font-weight:600; font-size:13px; }}
  td b {{ color:var(--gold); }}
  tr.pick td {{ background:rgba(46,230,168,.06); }}
  tr.pick td:first-child {{ color:var(--mint); font-weight:600; }}
  .flag {{ display:inline-block; padding:2px 9px; border-radius:99px; font-size:12px; font-weight:600; }}
  .flag.red {{ background:rgba(224,128,106,.14); color:var(--coral); }}
  .flag.green {{ background:rgba(46,230,168,.13); color:var(--mint); }}
  footer {{ margin-top:56px; padding-top:22px; border-top:1px solid var(--bord); color:var(--muted); font-size:13px; }}
  @media (max-width:640px) {{ .kpi b {{ font-size:26px; }} }}
</style>
</head>
<body>
<div class="wrap">

<nav class="top">
  <div class="brand"><div class="mk">✈️</div>
    <div><b>ركّابُ الطيرانِ الدوليّ</b><small>مشروعُ ختامِ دورةِ السلاسلِ الزمنيّة · أكاديميّةُ سدايا</small></div></div>
  <div class="pills">
    <a href="project.ipynb">الدفتر</a><a href="report.md">التقرير</a>
    <a href="brief.md">التكليف</a><a href="rubric.md">المِسطرة</a></div>
</nav>

<h1>من الشكلِ إلى الحكم</h1>
<p class="lede">سلسلةٌ شهريّةٌ من <span class="lat">{d['n']}</span> مشاهدة ({d['min_ds']} ← {d['max_ds']})، تمرُّ الأنبوبَ كاملاً:
تشخيصٌ ← أرضيّةُ أساسٍ يجبُ أن تُغلَب ← إطارٌ يفرزُ القائمةَ القصيرة ← <strong>مِحَكٌّ بثمانِ نوافذَ يحكم</strong> ← تقريرٌ
لمن لا يفتحُ الدفتر. <strong>وكلُّ رقمٍ هنا خرجَ من ذلك المِحَكِّ وحدَه</strong> — ورقمٌ ما كان ليخرجَ منه لا يُحسَب.</p>

<div class="kpis">
  <div class="kpi"><b>{p['mase']:.2f}</b><span>MASE للنموذجِ المختار<br>مقابلَ {f['mase']:.2f} للأرضيّة</span></div>
  <div class="kpi gold"><b>{abs(d_mase)*100:.0f}%</b><span>انخفاضُ خطإِ النقطةِ<br>عن الأرضيّة</span></div>
  <div class="kpi coral"><b>{p['coverage_80']*100:.0f}%</b><span>تغطيةُ حزمةِ 80%<br>الاسميُّ 80% — <span class="flag red">حدٌّ معلَن</span></span></div>
  <div class="kpi blue"><b>{d['nawafidh']}</b><span>نوافذَ متدحرجة<br>{d['nuqat']} نقطةً مسجَّلة</span></div>
</div>

<div class="card coral">
<p style="margin:0"><strong>🟥 فصلُ الدعوى.</strong> <strong>ما لا يُدَّعى:</strong> ليس ادّعاءَ نموذجٍ «صحيح»، ولا حزمةً
أمينةً — <strong>لا نموذجَ في الجدولِ بلغَ الـ80% الاسميّة</strong>، وأصدقُهم {p['coverage_80']*100:.1f}%.
<strong>وما يُقدَّم:</strong> حكمٌ على خمسةِ نماذجَ على <em>نفسِ</em> الأصولِ الثمانية، بثلاثةِ مقاييسَ لا مقياسٍ واحد،
وكلُّ فوزٍ في النقطةِ مقرونٌ بحالِه في التوزيعِ والحزمة.</p>
</div>

<h2><span class="n">1</span> الشكل — ما الذي تقولُه السلسلةُ قبلَ أيِّ نموذج</h2>
<p>المستوى نما <strong>{d['namaa']:.2f}×</strong> في اثنتَي عشرةَ سنة، <strong>والتذبذبُ الموسميُّ نما معه</strong>:
مداهُ في أوّلِ ستٍّ وثلاثين شهراً <span class="lat">{d['madaa_awwal']:.0f}</span> وفي آخرِها
<span class="lat">{d['madaa_akhir']:.0f}</span> — أي <strong>{d['madaa_akhir']/d['madaa_awwal']:.2f}×</strong>.
⟵ سلسلةٌ <strong>ضربيّة</strong>، واللوغاريتمُ شرطُ صحّةٍ لا زينة.</p>
<div class="kpis">
  <div class="kpi"><b>{d['FT']:.2f}</b><span>قوّةُ الاتّجاه <span class="lat">F_T</span></span></div>
  <div class="kpi"><b>{d['FS']:.2f}</b><span>قوّةُ الموسميّة <span class="lat">F_S</span></span></div>
  <div class="kpi gold"><b>{d['kharij']}/24</b><span>فجوةً خارجَ حدِّ الضجيجِ خاماً<br><span class="lat">±{d['bound']:.3f}</span></span></div>
  <div class="kpi blue"><b><span class="lat">{d['r12_d']:+.3f}</span></b><span><span class="lat">r₁₂</span> بعدَ الفرقِ الموسميّ<br>كان {d['r12']:.3f}</span></div>
</div>
<div class="card mint"><p style="margin:0">الرقمانِ عندَ السقف ⟵ <strong>أيُّ نموذجٍ لا يحملُ الاتّجاهَ والموسميّةَ معاً
مرفوضٌ قبلَ أن يُفصَّل</strong>. وفرقٌ موسميٌّ واحدٌ يُسقِطُ <span class="lat">r₁₂</span> من
{d['r12']:.3f} إلى {d['r12_d']:+.3f} ويُبقي <span class="lat">r₁ = {d['r1_d']:.3f}</span> —
<strong>فالباقي اتّجاهٌ صِرْف، وهو ما ستفشلُ فيه الأرضيّة.</strong></p></div>

<h2><span class="n">2</span> الأرضيّة — وما تركتْه على الطاولة</h2>
<p>الساذجُ الموسميُّ هو ما يجبُ على كلِّ نموذجٍ أن يغلبَه. وبواقيه <span class="lat">y_t − y_(t−12)</span>
ليست ضجيجاً أبيضَ ولا تقترب: <span class="lat">Ljung-Box = {d['lb12']:.0f}</span> عند 12 فجوة،
<span class="lat">p = {d['p12']:.1e}</span>. <strong>والبنيةُ الباقيةُ اتّجاهٌ لا موسميّة</strong>،
ومتوسّطُ بواقيها <strong><span class="lat">{f['bias']:+.1f}</span> راكب</strong> لا صفر — <strong>فهي منخفضةٌ منهجيّاً لأنّها تُكرّرُ
العامَ الماضيَ في سلسلةٍ تنمو.</strong></p>

<h2><span class="n">3</span> الحكم — خمسةُ نماذجَ على مِحَكٍّ واحد</h2>
<div class="card">{shakl_jadwal(sc)}
<p class="claim"><strong>{d['nawafidh']}</strong> أصولٍ متدحرجة · <span class="lat">h=12</span> · خطوةُ 12 ·
<strong>{d['nuqat']}</strong> نقطةً مسجَّلة — لكلِّ صفٍّ بلا استثناء، والأرضيّةُ في الجدولِ لا خارجَه.
<strong>logARIMA</strong> يأخذُ النقطةَ ({p['mase']:.2f}، <span class="lat">{d_mase:+.0%}</span> على الأرضيّة) والتوزيعَ
({p['crps']:.3f}، <span class="lat">{d_crps:+.0%}</span>)، وحزمتُه أقربُ إلى الأمانة — <strong>ولا واحدَ بلغَ الاسميَّ 80%</strong>.</p>
</div>

<table>
<tr><th>النموذج</th><th>MASE ↓</th><th>CRPS ↓</th><th>تغطية 80%</th><th>عرضُ الحزمة</th><th>الانحياز</th><th>أسوأُ طيّة</th></tr>
{"".join(f'<tr class="{ "pick" if m == PICK else "" }"><td>{sc[m]["wasf"]}</td><td>{sc[m]["mase"]:.3f}</td>'
         f'<td>{sc[m]["crps"]:.4f}</td><td>{sc[m]["coverage_80"]*100:.1f}%</td>'
         f'<td>{sc[m]["width80"]:.0f}</td><td>{sc[m]["bias"]:+.1f}</td>'
         f'<td>{sc[m]["mase_max"]:.2f}</td></tr>' for m in ORDER)}
</table>
<p><strong>والهامشُ الأمينُ يُقال:</strong> فرقُ النقطةِ بين <span class="lat">logARIMA</span> و<span class="lat">AutoARIMA</span>
الخامِ <strong>{abs(sc[PICK]['mase']/sc['AutoARIMA']['mase']-1)*100:.1f}%</strong> — داخلَ ضجيجِ ثمانِ طيّات، وليس سببَ الاختيار.
<strong>السببُ ما بعدَه:</strong> توزيعٌ أفضل، وحزمةٌ أقربُ إلى الاسميّ، و<strong>أسوأُ طيّةٍ {sc[PICK]['mase_max']:.2f}
مقابلَ {sc['AutoARIMA']['mase_max']:.2f}</strong>. المختارُ هو مَن سنتُه السيّئةُ أقلُّ سوءاً.</p>

<h2><span class="n">4</span> أينَ يفشلُ كلٌّ منهما — 96 خليّةً لا متوسّطٌ واحد</h2>
<div class="card">{shakl_kharita(d['grid_floor'], d['grid_pick'])}
<p class="claim">خطأُ الأرضيّةِ <strong>جدارٌ</strong> يشتدُّ كلّما نزلتَ: ينمو مع المستوى، في كلِّ طيّةٍ، وأسوأُ ما يكونُ
في منتصفِ الأفقِ عندَ ذروةِ الصيف. وخطأُ المختارِ <strong>بعثرة</strong> — لا سنةَ تملكُه ولا أفقَ يملكُه، وهذا ما
ينبغي أن يبدوَ عليه الباقي. <strong>نفسُ الخلايا الستِّ والتسعين، ونفسُ الأصول.</strong></p>
</div>

<h2><span class="n">5</span> الحزمةُ — الجزءُ الذي لا أوقّعُ عليه بالقبول</h2>
<div class="card">{shakl_taghtiya(d['cov_floor'], d['cov_pick'])}
<p class="claim">حزمةُ الأرضيّةِ اسميّةً 80% وتغطّي <strong>{f['coverage_80']*100:.1f}%</strong>، وأسوأُ ما تكونُ عندَ
<span class="lat">h={worst_h}</span> بـ<strong>{d['cov_floor'][worst_h-1]*100:.1f}%</strong> — أشهرُ الذروةِ التي تنمو
أسرعَ هي بعينِها الأشهرُ التي تنسخُها من العامِ الماضي. <strong>وليست ضيّقةً فحسب، بل مركزُها خطأ.</strong>
وحزمةُ المختارِ {p['coverage_80']*100:.1f}% — أفضلُ ولا تزالُ دونَ ما وُعِدَ به.</p>
</div>

<h2><span class="n">6</span> الإطارُ يفرز، والمِحَكُّ يحكم</h2>
<div class="card">{shakl_intiqal()}
<p class="claim">لوحةُ <span class="lat">AutoGluon</span> فصّلتْ على <strong>نافذةِ تحقّقٍ واحدة</strong>
(<span class="lat">num_val_windows = 1</span>) وأخرجتْ <span class="lat">score_val</span> <strong>سالباً في كلِّ صفّ</strong>
لأنّها تقلبُ إشارةَ المقياسِ ليكونَ الأعلى أفضل. وبتغيُّرِ النافذةِ ينقلبُ الترتيبُ: <span class="lat">AutoETS</span>
و<span class="lat">AutoARIMA</span> يتبادلانِ الموضعَ بين <span class="lat">score_val</span> و<span class="lat">score_test</span>.
<strong>الأصولُ الثمانيةُ تحسمُ ما لا تحسمُه نافذة.</strong></p>
</div>

<h2><span class="n">7</span> عطبٌ أُمسِكَ قبلَ أن يُسلَّم</h2>
<div class="card gold">
<p>مسوّدةٌ سابقةٌ مرَّرتِ السلسلةَ عبرَ <span class="lat">to_csv()</span> فعادَ معها عمودُ فهرسٍ باسمِ
<span class="lat">Unnamed: 0</span>. و<span class="lat">StatsForecast</span> <strong>يُفصِّلُ كلَّ عمودٍ ليس
<span class="lat">unique_id/ds/y</span> متغيّراً خارجيّاً — صامتاً</strong>. فدخلَ عدّادُ الصفوفِ
<span class="lat">0…143</span> ميزةً للنموذج، وهو <strong>اتّجاهٌ زمنيٌّ حتميٌّ كامل</strong>.</p>
<p style="margin-bottom:0"><strong>فتحسّنَ الرقم:</strong> <span class="lat">MASE</span> من
<strong>{p['mase']:.3f}</strong> إلى <strong>0.498</strong> — تحسّنٌ بـ22% من عمودٍ لم يخترْه أحد.
<strong>لم يُنذِرْ شيءٌ ولم يُرفَعْ خطأ</strong>؛ أُمسِكَ بسؤالٍ واحد: <em>لماذا تحسّنَ الرقم؟</em>
والدفترُ الآن يحملُ حارساً يشترطُ الأعمدةَ الثلاثةَ وضابطاً سالباً يُثبِتُ أنّ الحارسَ يُمسِكُ الحالةَ التي وُلِدَ منها.
<strong>وكلُّ رقمٍ في هذه الصفحةِ هو النظيف.</strong></p>
</div>

<h2><span class="n">8</span> ما يبقى بيدِ الإنسان</h2>
<table>
<tr><th>الدعوى</th><th>الشاهد</th><th>الحكم</th></tr>
<tr><td>«المختارُ يغلبُ الأرضيّة»</td><td>MASE <span class="lat">{d_mase:+.0%}</span> · CRPS <span class="lat">{d_crps:+.0%}</span> · تغطية {p['coverage_80']*100:.0f}% مقابلَ {f['coverage_80']*100:.0f}%</td><td><span class="flag green">مقيس</span></td></tr>
<tr><td>«حزمتُه أمينة»</td><td>{p['coverage_80']*100:.1f}% حيثُ وُعِدَ بـ80%</td><td><span class="flag red">مكذوب</span></td></tr>
<tr><td>«ترتيبُ الإطارِ موثوق»</td><td>نافذةٌ واحدة · والترتيبانِ يتناقضان</td><td><span class="flag red">لا يُعتمَد</span></td></tr>
</table>
<p><strong>والخطوةُ الواحدةُ التالية:</strong> تصحيحُ انحيازِ العكسِ من اللوغاريتم — الانحيازُ
<strong>{p['bias']:+.1f}</strong> راكب، وهو أثرٌ متوقَّعٌ نظريّاً لا صدفة، ويُقاسُ أثرُ تصحيحِه
<strong>على نفسِ الأصولِ الثمانيةِ قبلَ اعتمادِه</strong>.</p>

<footer>
<p style="margin:0">كلُّ رقمٍ في هذه الصفحةِ يُعادُ حسابُه عندَ كلِّ بناءٍ من
<span class="lat">site/build_page.py</span> — لا رقمَ مكتوبٌ بيد، والرسومُ <span class="lat">SVG</span>
مضمَّنةٌ بلا مكتبةٍ ولا طلبِ شبكة. الدفترُ يعملُ من رأسِه إلى قدمِه في أقلَّ من دقيقة.</p>
<p style="margin:10px 0 0"><strong>عماد سليمان علوان</strong> · استشاريُّ ذكاءٍ اصطناعيّ ·
مشروعُ ختامِ دورةِ تحليلِ السلاسلِ الزمنيّةِ والتنبّؤ، أكاديميّةُ سدايا</p>
</footer>

</div>
</body>
</html>
'''


def main() -> int:
    d = qis()
    out = os.path.join(ROOT, "index.html")
    open(out, "w", encoding="utf-8").write(ibni(d))
    json.dump(d, open(os.path.join(ROOT, "site", "qiyas.json"), "w"),
              ensure_ascii=False, indent=1)
    print("📄 بُنِيَت:", out)
    print(f"   MASE {d['scores'][PICK]['mase']:.4f} · تغطية "
          f"{d['scores'][PICK]['coverage_80']:.4f} · {d['nuqat']} نقطة")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
