"""
forecast_report.py  -  charts and a CFO-ready HTML report from the saved forecast results (Step 5).
Reads data/forecast/* only, so it can be re-run without retraining.

    python forecast_engine/forecast_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

NAVY, STEEL, GREY, RED, GREEN, AMBER = "#1f3a5f", "#5b7fa6", "#9aa5b1", "#b23a48", "#2e7d5b", "#c98a1b"
LAYOUT = dict(template="plotly_white", font=dict(family="Arial", size=12), margin=dict(l=60, r=30, t=70, b=60))


class ForecastReport:
    def __init__(self, config_path="forecast.yaml"):
        cfgp = Path(config_path).resolve()
        cfg = yaml.safe_load(open(cfgp, encoding="utf-8"))
        self.cfg = cfg
        self.dir = cfgp.parent / cfg["paths"]["out_dir"]
        x = pd.ExcelFile(self.dir / "forecast_results_2026.xlsx")
        self.acc = pd.read_excel(x, "backtest_accuracy", index_col=0)
        self.bm = pd.read_excel(x, "backtest_monthly", index_col=0)
        self.gm = pd.read_excel(x, "backtest_gross_margin", index_col=0)
        self.fy = pd.read_excel(x, "group_pnl_by_year", index_col=0)
        self.bridge = pd.read_excel(x, "bridge_FY26_FY27", index_col=0)["EUR m"]
        self.imp = pd.read_excel(x, "driver_importance", index_col=0).iloc[:, 0]
        self.sens = pd.read_excel(x, "sensitivities")
        self.g = pd.read_parquet(self.dir / "group_pnl_monthly.parquet")
        self.meta = json.load(open(self.dir / "forecast_meta.json"))
        self.sections = []

    def _add(self, title, fig, text):
        if fig is not None:
            fig.update_layout(**LAYOUT)
        self.sections.append((title, fig, text))

    # ----------------------------------------------------------------------------------------- charts
    def backtest(self):
        bm = self.bm / 1e6
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bm.index, y=bm["Actual"], name="Actual", line=dict(color="black", width=3)))
        style = {"AI forecast (LightGBM + drivers)": (NAVY, "solid"),
                 "AI forecast with perfect hindsight on drivers": (NAVY, "dot"),
                 "Holt-Winters (brand x country)": (STEEL, "dash"),
                 "Department budget (bottom-up submissions)": (AMBER, "dash"),
                 "Last year same month": (GREY, "dot")}
        for col, (c, d) in style.items():
            if col in bm:
                fig.add_trace(go.Scatter(x=bm.index, y=bm[col], name=col, line=dict(color=c, dash=d, width=2)))
        fig.update_layout(title="Backtest: Net Sales Jan-Sep 2026 forecast in Dec 2025 vs actual (EUR m, actual FX)",
                          yaxis_title="EUR m", height=460, legend=dict(orientation="h", y=-0.18))
        a = self.acc
        ai, pf = a.loc["AI forecast (LightGBM + drivers)"], a.loc["AI forecast with perfect hindsight on drivers"]
        hw, bud = a.loc["Holt-Winters (brand x country)"], a.loc["Department budget (bottom-up submissions)"]
        ly = a.loc["Last year same month"]
        text = (f"With the consensus available in Dec 2025, the AI forecast missed group Net Sales by {ai['WAPE group monthly']:.1%} "
                f"a month on average (bias {ai['Bias (total)']:+.1%}), against {hw['WAPE group monthly']:.1%} for Holt-Winters and "
                f"{bud['WAPE group monthly']:.1%} for the department budget. At SKU x country level, where commercial decisions are made, "
                f"it is the most accurate method ({ai['WAPE SKU x country x month']:.1%} vs {hw['WAPE SKU x country x month']:.1%} "
                f"Holt-Winters and {ly['WAPE SKU x country x month']:.1%} last year). With the drivers that actually happened, the same "
                f"model misses by only {pf['WAPE group monthly']:.1%} at group level and {pf['WAPE SKU x country x month']:.1%} at SKU level.")
        self._add("1. How accurate is it? Backtest against Jan-Sep 2026", fig, text)

    def accuracy_and_split(self):
        a = self.acc[["WAPE group monthly", "WAPE country x month", "WAPE SKU x country x month"]] * 100
        fig = go.Figure()
        colors = [NAVY, STEEL, GREY]
        for col, c in zip(a.columns, colors):
            fig.add_trace(go.Bar(y=a.index, x=a[col], name=col.replace("WAPE ", ""), orientation="h", marker_color=c,
                                 text=[f"{v:.1f}%" if pd.notna(v) else "n/a" for v in a[col]], textposition="outside"))
        fig.update_layout(barmode="group", title="Forecast error (WAPE, lower is better) by level of detail", height=520,
                          xaxis_title="% of actual Net Sales", yaxis=dict(autorange="reversed", automargin=True),
                          legend=dict(orientation="h", y=-0.12))
        em, ea = self.meta["err_model_eur_m"], self.meta["err_assumptions_eur_m"]
        text = (f"Splitting the Jan-Sep 2026 error: the model itself accounts for EUR {em:+.0f}m, the driver assumptions "
                f"(consensus outlook of Dec 2025) for EUR {ea:+.0f}m. The consensus missed the April 2026 US tariff, the "
                f"spring drop in consumer confidence and the inflation slowdown in Argentina and Turkey. The lesson for the CFO: "
                f"the forecast is only as good as its assumptions, so the value is in re-running it the day assumptions change "
                f"(Step 7 scenario app), not in locking a number in October.")
        self._add("2. Where does the error come from: model or assumptions?", fig, text)

    def margin(self):
        g = self.gm * 100
        fig = go.Figure()
        for col, c, d in [("actual", "black", "solid"), ("AI forecast", NAVY, "solid"), ("AI with perfect hindsight", NAVY, "dot"),
                          ("last year", GREY, "dot")]:
            fig.add_trace(go.Scatter(x=g.index, y=g[col], name=col, line=dict(color=c, dash=d, width=2.5 if col == "actual" else 2)))
        fig.update_layout(title="Gross margin % Jan-Sep 2026: forecast in Dec 2025 vs actual", yaxis_title="%", height=420,
                          legend=dict(orientation="h", y=-0.18))
        mae_pf = (g["AI with perfect hindsight"] - g["actual"]).abs().mean()
        mae_ai = (g["AI forecast"] - g["actual"]).abs().mean()
        text = (f"With the true commodity prices the COGS engine tracks the 2026 margin squeeze within {mae_pf:.2f} points a month; "
                f"with the Dec 2025 consensus (flat commodities) the error is {mae_ai:.2f} points, because the consensus did not see "
                f"the oil and fragrance shock. Two fixes were made after testing: (1) no extrapolation of the 2023-25 margin "
                f"recovery trend, which had made the first version forecast a rising margin; (2) commodity moves pass through "
                f"over 3 months (hedges and contracts), chosen because it fits BOTH the 2022 and the 2026 shock; a 6-month "
                f"setting fitted 2026 best but failed on 2022.")
        self._add("3. Does the cost engine get the margin right?", fig, text)

    def rolling(self):
        g = self.g[self.g.index >= "2024-01-01"]
        fig = go.Figure()
        act, fc = g[g.Type == "Actual"], g[g.Type == "Forecast"]
        fig.add_trace(go.Scatter(x=act.index, y=act.ns / 1e6, name="Actual", line=dict(color="black", width=2.5)))
        fig.add_trace(go.Scatter(x=list(fc.index) + list(fc.index[::-1]), y=list(fc.ns_high / 1e6) + list(fc.ns_low[::-1] / 1e6),
                                 fill="toself", fillcolor="rgba(31,58,95,0.15)", line=dict(width=0), name="~80% range"))
        fig.add_trace(go.Scatter(x=[act.index[-1]] + list(fc.index), y=[act.ns.iloc[-1] / 1e6] + list(fc.ns / 1e6),
                                 name="Forecast", line=dict(color=NAVY, width=2.5, dash="dash")))
        fig.update_layout(title="Rolling forecast: group Net Sales Oct 2026 - Dec 2027 (EUR m)", yaxis_title="EUR m", height=440,
                          legend=dict(orientation="h", y=-0.18))
        fy = self.fy
        t = self.cfg["reporting"]
        text = (f"FY2026 (9 months actual + 3 forecast): Net Sales EUR {fy.loc[2026, 'net_sales'] / 1e6:,.0f}m, gross margin "
                f"{fy.loc[2026, 'gross_margin_pct']:.1%}, contribution after country overheads {fy.loc[2026, 'contribution_margin_pct']:.1%}. "
                f"FY2027: EUR {fy.loc[2027, 'net_sales'] / 1e6:,.0f}m ({fy.loc[2027, 'ns_growth_reported']:+.1%} reported), gross margin "
                f"{fy.loc[2027, 'gross_margin_pct']:.1%}, contribution {fy.loc[2027, 'contribution_margin_pct']:.1%} vs the "
                f"{t['target_ebit_margin']:.0%} EBIT target (note: contribution is before group costs, so the true EBIT gap is larger). "
                f"The shaded range is +/- {t['interval_z']} x the {self.meta['rmspe']:.1%} typical monthly backtest error, widening with horizon.")
        self._add("4. The rolling forecast", fig, text)

    def bridge_chart(self):
        b = self.bridge
        labels = list(b.index)
        fig = go.Figure(go.Waterfall(
            x=labels, measure=["absolute"] + ["relative"] * (len(b) - 2) + ["total"], y=list(b.values[:-1]) + [0],
            text=[f"{b.iloc[0]:,.0f}"] + [f"{v:+,.0f}" for v in b.values[1:-1]] + [f"{b.iloc[-1]:,.0f}"], textposition="outside",
            connector=dict(line=dict(color=GREY)), increasing=dict(marker=dict(color=GREEN)),
            decreasing=dict(marker=dict(color=RED)), totals=dict(marker=dict(color=NAVY))))
        lo = min(b.iloc[0], b.iloc[-1])
        fig.update_layout(title="Net Sales bridge FY2026 -> FY2027 forecast (EUR m)", height=440, showlegend=False,
                          yaxis=dict(range=[lo * 0.95, max(b.iloc[0], b.iloc[-1]) * 1.03]))
        start = b.iloc[0]
        org_ex = (b["Volume"] + b["Mix"] + b["Price (other markets)"] + b["Delisted / ending SKUs"]) / start
        org = org_ex + b["Price (hyperinflation markets)"] / start
        text = (f"Organic growth 2027 is {org:.1%}, but only {org_ex:.1%} excluding hyperinflation pricing, below the "
                f"{self.cfg['reporting']['target_organic_growth']:.0%} target. Growth is almost entirely price (EUR "
                f"{b['Price (other markets)']:+.0f}m outside Argentina/Turkey); volume adds only EUR {b['Volume']:+.0f}m, and FX "
                f"translation costs EUR {b['FX translation']:+.0f}m. This is the gap the CFO agent must address.")
        self._add("5. What drives 2027?", fig, text)

    def drivers(self):
        imp = self.imp.sort_values()
        fig = go.Figure(go.Bar(y=imp.index, x=imp.values * 100, orientation="h", marker_color=NAVY,
                               text=[f"{v:.0%}" for v in imp.values], textposition="outside"))
        fig.update_layout(title="What the volume model relies on (share of explanation, SHAP)", height=380,
                          xaxis=dict(title="%", range=[0, imp.max() * 125]), yaxis=dict(automargin=True))
        s = self.sens.copy()
        rows = "".join(f"<tr><td>{r['driver change (all months after Sep 2026)']}</td><td>{r['net sales EUR m']:+,.0f}</td>"
                       f"<td>{r['gross margin pts']:+.2f}</td><td>{r['contribution EUR m']:+,.0f}</td><td>{r['how']}</td></tr>"
                       for _, r in s.iterrows())
        table = ("<table><tr><th>Driver change (all months after Sep 2026)</th><th>Net Sales FY2027, EUR m</th>"
                 "<th>Gross margin, pts</th><th>Contribution, EUR m</th><th>Mechanism</th></tr>" + rows + "</table>")
        text = (f"The volume model leans most on market size and share ({self.imp.get('Market size & share', 0):.0%} of its explanation): "
                "the outside-in logic is what it actually uses. Every driver is constrained to move volume in the economically "
                "sensible direction (e.g. lower consumer confidence or more private label can never raise sales); the first "
                "version without these constraints produced a wrong-sign confidence effect, which is why they were added. "
                "Sensitivities for FY2027, used by the scenario app:" + table)
        self._add("6. What drives the model, and how sensitive is the forecast?", fig, text)

    def tradeoff(self):
        rows = [
            ("Accuracy, SKU level", "Weak on short or intermittent series", "Best tested (10% with true drivers, 11.5% with consensus)"),
            ("Accuracy, group level", "Good when the year is calm (1.0% in 2026)", "0.4% with true drivers; 3.6% with the Dec 2025 consensus"),
            ("Uses outside-in drivers", "No - trend and seasonality only", "Yes: market, share, private label, CPI, confidence, commodities, FX"),
            ("Can answer 'what if oil +20%?'", "No", "Yes, in seconds (scenario app)"),
            ("Explainability", "High, but only 'trend + season'", "Driver bridge, SHAP shares, sensitivities; price/COGS/FX are explicit formulas"),
            ("Economic sense guaranteed", "n/a", "Monotone constraints on every driver"),
            ("Main weakness", "Blind to shocks and drivers", "Only as good as its assumptions; volume layer is a black box without SHAP"),
        ]
        tbl = "<table><tr><th></th><th>Holt-Winters / last year</th><th>Chosen: LightGBM volume + driver P&L</th></tr>" + \
              "".join(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows) + "</table>"
        text = ("<b>Why this model for a CFO audience.</b> A pure black-box model (e.g. a neural network forecasting Net Sales "
                "directly) might score slightly better but could not show why the number moved. A pure statistical model "
                "(Holt-Winters) is transparent but cannot use market data or answer what-if questions, and in 2026 it was "
                "accurate only because nothing structural changed at group level. The hybrid keeps machine learning where the "
                "pattern is complex (volume across 4,749 series) and uses formulas a controller can audit for price, costs and "
                "FX. Accuracy is reported honestly: with consensus inputs it does not beat Holt-Winters at group level, and we "
                "say so. Recommended use: run both, and treat a large gap between them as a signal to challenge the assumptions."
                + tbl +
                "<b>Limitations.</b> Training uses actual drivers (ex-post) because historical consensus vintages exist only for "
                "Dec 2025 and Sep 2026. The volume model does not know the promotion calendar or new launches after the forecast "
                "date (Marketing's overlay adds them in Step 6). 'Contribution after country overheads' is not EBIT: group costs, "
                "R&D and D&A are missing. The uncertainty range is based on only two backtest folds.")
        self._add("7. Model choice: accuracy vs interpretability", None, text)

    # ----------------------------------------------------------------------------------------- export
    def build(self):
        for f in [self.backtest, self.accuracy_and_split, self.margin, self.rolling, self.bridge_chart, self.drivers, self.tradeoff]:
            f()
        css = ("body{font-family:Arial,sans-serif;max-width:1150px;margin:30px auto;color:#1f2933;padding:0 16px}"
               "h1,h2{color:#1f3a5f}h2{border-top:1px solid #d9dee4;padding-top:24px}"
               ".box{background:#f4f6f8;border-left:4px solid #1f3a5f;padding:10px 14px;margin:8px 0;line-height:1.5}"
               "table{border-collapse:collapse;margin:10px 0;font-size:13px;width:100%}"
               "th,td{border:1px solid #d9dee4;padding:6px 8px;text-align:left;vertical-align:top}th{background:#eef1f4}")
        parts = [f"<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
                 f"<title>Hygea Group - forecast report</title><style>{css}</style></head><body>",
                 "<h1>Hygea Group (fictional) - outside-in P&amp;L forecast</h1>",
                 f"<p>Rolling forecast from the {self.meta['origin'][:7]} actuals. EUR m. Volume: global LightGBM model; "
                 "price, costs, trade, A&amp;P, overheads and FX: driver formulas. Backtest: forecast made with Dec 2025 information only.</p>"]
        first = True
        for title, fig, text in self.sections:
            parts.append(f"<h2>{title}</h2>")
            if fig is not None:
                parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False))
                first = False
            parts.append(f"<div class='box'>{text}</div>")
        parts.append("</body></html>")
        (self.dir / "forecast_report.html").write_text("\n".join(parts), encoding="utf-8")
        return self


def main():
    import sys
    r = ForecastReport(sys.argv[1] if len(sys.argv) > 1 else "forecast.yaml").build()
    for title, _, text in r.sections:
        print(f"\n{title}\n  {text[:400]}")
    print(f"\nReport: {r.dir / 'forecast_report.html'}")


if __name__ == "__main__":
    main()
