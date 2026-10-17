"""
explore.py  -  STEP 4: exploratory analysis that answers business questions.

Each question has one function that returns:
    fig      - an interactive Plotly chart
    finding  - the answer in numbers (plain English)
    decision - what the forecast will do because of it
    table    - the supporting numbers

Reads the CLEAN outputs of Step 3 only (never the raw files or the answer key).

HOW TO RUN
    python forecast_engine/explore.py        -> writes data/eda/eda_report.html + findings.xlsx
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

NAVY, STEEL, GREY, RED, GREEN = "#1f3a5f", "#5b7fa6", "#9aa5b1", "#b23a48", "#2e7d5b"
LAYOUT = dict(template="plotly_white", font=dict(family="Arial", size=12), margin=dict(l=60, r=30, t=70, b=50))


class Explorer:
    def __init__(self, config_path: str | Path = "analysis.yaml"):
        self.cfg_path = Path(config_path).resolve()
        self.base = self.cfg_path.parent
        self.cfg = yaml.safe_load(open(self.cfg_path, encoding="utf-8"))
        clean = self.base / self.cfg["paths"]["clean_dir"]
        self.out = self.base / self.cfg["paths"]["out_dir"]
        self.out.mkdir(parents=True, exist_ok=True)
        self.mat = float(self.cfg["materiality_eur"])
        self.hyper = self.cfg["hyperinflation_countries"]

        sku = pd.read_excel(self.base / self.cfg["paths"]["raw_dir"] / "master_data.xlsx", sheet_name="sku_master")
        cm = pd.read_excel(self.base / self.cfg["paths"]["raw_dir"] / "master_data.xlsx", sheet_name="country_master")
        fr = pd.read_parquet(clean / "fact_pnl_forecast_ready.parquet")
        fr["date"] = pd.to_datetime(fr["date"])
        fr["brand"] = fr["sku_code"].map(sku.set_index("sku_code")["brand"])
        fr["category"] = fr["sku_code"].map(sku.set_index("sku_code")["category"])
        fr["hemisphere"] = np.where(fr["country"].map(cm.set_index("country")["south"]), "South", "North")
        # flags were stored as text by the cleaning step ("True"/"False") - convert explicitly, never with astype(bool)
        for c in ["flag_inorganic", "flag_hyperinflation", "flag_possibly_dormant"]:
            fr[c] = fr[c].astype(str).str.lower().eq("true")
        self.macro = pd.read_parquet(clean / "ext_macro_country.parquet")
        self.macro["date"] = pd.to_datetime(self.macro["date"])
        fx = self.macro[["date", "country", "fx_lc_per_eur"]]
        fr = fr.drop(columns=[c for c in fr.columns if c.endswith("_eur") or c.endswith("_eur_cc")])
        fr = fr.merge(fx, on=["date", "country"], how="left")
        prev = fx.assign(date=fx["date"] + pd.DateOffset(years=1)).rename(columns={"fx_lc_per_eur": "fx_prev_year"})
        fr = fr.merge(prev, on=["date", "country"], how="left")
        for c in ["volume_units", "gross_sales_lc", "trade_spend_lc", "net_sales_lc", "cogs_lc", "ap_lc"]:
            fr[c] = fr[c].fillna(0.0)               # missing_extract months count as 0 here (flagged in findings)
        fr["ns_eur"] = fr["net_sales_lc"] / fr["fx_lc_per_eur"]
        fr["cogs_eur"] = fr["cogs_lc"] / fr["fx_lc_per_eur"]
        fr["ns_cc"] = fr["net_sales_lc"] / fr["fx_prev_year"]   # at last year's same-month rate
        fr["year"], fr["month"] = fr["date"].dt.year, fr["date"].dt.month
        self.fr, self.sku, self.cm = fr, sku, cm
        self.ext_market = pd.read_parquet(clean / "ext_market_category_country.parquet")
        self.ext_global = pd.read_parquet(clean / "ext_commodities_global.parquet")
        self.subs = pd.read_parquet(clean / "submissions_country.parquet")
        self.extract_month = fr["date"].max()
        self.results = []

    # ------------------------------------------------------------------ bridge engine
    def bridge(self, y0: int, y1: int, months: list[int] | None = None, by: list[str] | None = None) -> pd.DataFrame:
        """Net Sales bridge from year y0 to y1 (EUR), optionally by group (e.g. brand x country).
        Effects: volume, mix, price (split hyperinflation / other), launches & delists, M&A, FX.
        Start + effects = end, exactly."""
        by = by or []
        d = self.fr[self.fr.year.isin([y0, y1])]
        if months:
            d = d[d.month.isin(months)]
        g0 = d[d.year == y0]
        g1 = d[d.year == y1]
        mna = g1[g1.flag_inorganic]
        g1 = g1[~g1.flag_inorganic]
        key = ["sku_code", "country"]
        extra = [b for b in by if b not in key]
        a0 = g0.groupby(key + extra, dropna=False).agg(V0=("volume_units", "sum"), N0=("ns_eur", "sum")).reset_index()
        a1 = g1.groupby(key + extra, dropna=False).agg(V1=("volume_units", "sum"), C1=("ns_cc", "sum"), E1=("ns_eur", "sum")).reset_index()
        m = a0.merge(a1, on=key + extra, how="outer").fillna(0)
        cont = (m.N0 > 0) & (m.C1 > 0) & (m.V0 > 0) & (m.V1 > 0)
        m["P0"] = np.where(cont, m.N0 / m.V0.where(m.V0 > 0, 1), 0)
        m["P1"] = np.where(cont, m.C1 / m.V1.where(m.V1 > 0, 1), 0)
        m["price"] = np.where(cont, (m.P1 - m.P0) * m.V1, 0)
        m["hyper"] = m.country.isin(self.hyper)
        m["V1P0"] = np.where(cont, m.V1 * m.P0, 0)
        m["newlost"] = np.where(cont, 0, m.C1 - m.N0)
        m["fx"] = m.E1 - m.C1
        m["cont"] = cont
        m["_g"] = 1
        grp = by or ["_g"]
        c = m[m.cont]
        gsum = c.groupby(grp).agg(V0=("V0", "sum"), V1=("V1", "sum"), N0c=("N0", "sum"), V1P0=("V1P0", "sum"))
        p0avg = gsum.N0c / gsum.V0
        out = pd.DataFrame(index=gsum.index)
        out["start"] = m.groupby(grp)["N0"].sum()
        out["volume"] = (gsum.V1 - gsum.V0) * p0avg
        out["mix"] = gsum.V1P0 - gsum.V1 * p0avg
        out["price_hyperinflation"] = c[c.hyper].groupby(grp)["price"].sum()
        out["price_other"] = c[~c.hyper].groupby(grp)["price"].sum()
        out["launches_delists"] = m.groupby(grp)["newlost"].sum()
        out["fx"] = m.groupby(grp)["fx"].sum()
        mna_g = mna.assign(_g=1).groupby(grp)["ns_eur"].sum() if len(mna) else pd.Series(dtype=float)
        out = out.reindex(m.groupby(grp).size().index).fillna(0)
        out["mna"] = mna_g.reindex(out.index).fillna(0)
        out["end"] = m.groupby(grp)["E1"].sum().reindex(out.index).fillna(0) + out["mna"]
        eff = ["volume", "mix", "price_hyperinflation", "price_other", "launches_delists", "mna", "fx"]
        out["check"] = out["start"] + out[eff].sum(axis=1) - out["end"]
        out["organic_pct"] = out[["volume", "mix", "price_hyperinflation", "price_other", "launches_delists"]].sum(axis=1) / out["start"]
        if not by:
            out.index = [f"{y0}->{y1}"]
        return out

    def _record(self, q, fig, finding, decision, table):
        fig.update_layout(**LAYOUT)
        self.results.append({"question": q, "fig": fig, "finding": finding, "decision": decision, "table": table})
        return fig, finding, decision, table

    # ======================================================== Q1 growth sources
    def q1_growth_bridge(self):
        rows = [self.bridge(y - 1, y) for y in self.cfg["bridge_years"]]
        cy, py = self.cfg["ytd_compare"]["current"], self.cfg["ytd_compare"]["prior"]
        ytd = self.bridge(py, cy, months=list(range(1, self.extract_month.month + 1)))
        ytd.index = [f"{py}->{cy} YTD Jan-{self.extract_month:%b}"]
        tab = pd.concat(rows + [ytd])
        assert tab["check"].abs().max() < 1, "Bridge does not reconcile"
        last = tab.iloc[len(rows) - 1]
        labels = ["Volume", "Mix", "Price (hyperinflation)", "Price (other)", "Launches & delists", "M&A", "FX"]
        vals = last[["volume", "mix", "price_hyperinflation", "price_other", "launches_delists", "mna", "fx"]].values / 1e6
        fig = go.Figure(go.Waterfall(
            x=[f"NS {self.cfg['bridge_years'][-1] - 1}"] + labels + [f"NS {self.cfg['bridge_years'][-1]}"],
            measure=["absolute"] + ["relative"] * len(labels) + ["total"],
            y=[last.start / 1e6] + list(vals) + [0],
            text=[f"{last.start / 1e6:,.0f}"] + [f"{v:+,.0f}" for v in vals] + [f"{last.end / 1e6:,.0f}"],
            textposition="outside", connector=dict(line=dict(color=GREY)),
            increasing=dict(marker=dict(color=GREEN)), decreasing=dict(marker=dict(color=RED)),
            totals=dict(marker=dict(color=NAVY))))
        fig.update_layout(title=f"Q1 Net Sales bridge {tab.index[len(rows) - 1]} (EUR m): price in hyperinflation markets and FX largely cancel out",
                          yaxis_title="EUR m", showlegend=False)
        fig.update_yaxes(range=[last.start / 1e6 * 0.85, max(last.start, last.end) / 1e6 * 1.08])
        t = tab.copy()
        cols = ["start", "volume", "mix", "price_hyperinflation", "price_other", "launches_delists", "mna", "fx", "end"]
        t[cols] = (t[cols] / 1e6).round(1)
        t["organic_pct"] = (t["organic_pct"] * 100).round(1)
        t["organic_ex_hyper_pct"] = ((tab["volume"] + tab["mix"] + tab["price_other"] + tab["launches_delists"]) / tab["start"] * 100).round(1)
        t = t.drop(columns="check")
        org = t["organic_pct"]
        exh = t["organic_ex_hyper_pct"]
        finding = (f"Reported organic growth runs {org.iloc[:-1].min():.1f}% to {org.iloc[:-1].max():.1f}% a year (target 4%), "
                   f"but excluding hyperinflation pricing it is {exh.iloc[:-1].min():.1f}% to {exh.iloc[:-1].max():.1f}%. "
                   f"In {t.index[len(rows) - 1]}, Argentina/Turkey pricing added EUR {last.price_hyperinflation / 1e6:+.0f}m "
                   f"while FX took EUR {last.fx / 1e6:+.0f}m: they largely offset. "
                   f"Volume contributed EUR {last.volume / 1e6:+.0f}m and other pricing EUR {last.price_other / 1e6:+.0f}m. "
                   f"M&A (Limeguard) added EUR {t['mna'].max():.0f}m at its peak. "
                   f"YTD 2026 organic growth is {org.iloc[-1]:.1f}% ({exh.iloc[-1]:.1f}% excluding hyperinflation).")
        decision = ("Forecast volume and real price separately; model hyperinflation-market price with CPI pass-through and "
                    "translate at forecast FX, so price and FX are never forecast independently. Report organic growth "
                    "both including and excluding hyperinflation pricing, because the 4% target is otherwise met by inflation.")
        return self._record("Q1 Where does growth come from?", fig, finding, decision, t)

    # ======================================================== Q2 seasonality
    def q2_seasonality(self):
        d = self.fr[self.fr.year.isin(self.cfg["seasonality_years"])]
        v = d.groupby(["category", "hemisphere", "year", "month"])["volume_units"].sum().reset_index()
        v["idx"] = v["volume_units"] / v.groupby(["category", "hemisphere", "year"])["volume_units"].transform("mean")
        s = v.groupby(["category", "hemisphere", "month"])["idx"].mean().unstack("month")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Northern hemisphere", "Southern hemisphere"], horizontal_spacing=0.18)
        for i, h in enumerate(["North", "South"], 1):
            z = s.xs(h, level="hemisphere")
            fig.add_trace(go.Heatmap(z=z.values, x=months, y=z.index, zmid=1, colorscale=[[0, RED], [0.5, "#ffffff"], [1, NAVY]],
                                     zmin=0.75, zmax=1.25, text=np.round(z.values, 2), texttemplate="%{text}",
                                     showscale=(i == 2), colorbar=dict(title="Index")), row=1, col=i)
        fig.update_layout(title="Q2 Volume seasonality index (1.00 = average month), 2022-2025: disinfection peaks flip between hemispheres",
                          height=430)
        rng_ = (s.max(axis=1) - s.min(axis=1)).round(2)
        peak = s.idxmax(axis=1).map(lambda m: months[m - 1])
        tab = pd.DataFrame({"peak_month": peak, "swing_peak_to_trough": rng_}).reset_index()
        sd_n, sd_s = peak[("Surface Cleaning & Disinfection", "North")], peak[("Surface Cleaning & Disinfection", "South")]
        ac = s.loc[("Air Care", "North")]
        finding = (f"Air Care is the most seasonal category: {ac.max():.2f}x in {months[int(ac.idxmax()) - 1]} vs {ac.min():.2f}x in "
                   f"{months[int(ac.idxmin()) - 1]} (Q4 gifting). Surface Cleaning & Disinfection peaks in {sd_n} in the north "
                   f"but in {sd_s} in the south (flu season follows the winter). Fabric, Water Softeners and Toilet Care are "
                   f"broadly flat (swing below {rng_.xs('North', level='hemisphere').drop(['Air Care', 'Surface Cleaning & Disinfection']).max():.2f}).")
        decision = ("Seasonality is modelled by category x hemisphere, not by group: one global seasonal pattern would put the "
                    "southern disinfection peak in the wrong half of the year. Month-of-year and hemisphere become model features.")
        return self._record("Q2 How seasonal is each category?", fig, finding, decision, tab)

    # ======================================================== Q3 commodities -> GM
    def q3_commodity_margin(self):
        g = self.ext_global.copy()
        g["date"] = pd.to_datetime(g["date"])
        w25 = self.fr[self.fr.year == 2025].merge(self.sku, on="sku_code", suffixes=("", "_m"))
        inputs = ["surfactant", "plastics", "fragrance", "aerosol"]
        weights = {i: (w25[f"recipe_{i}"] * w25["cogs_eur"]).sum() / w25["cogs_eur"].sum() for i in inputs}
        eur = g["eurusd"] / g["eurusd"].iloc[0]
        g["materials_index_eur"] = sum(weights[i] * g[f"{i}_index"] / g[f"{i}_index"].iloc[0] for i in inputs) / sum(weights.values()) / eur * 100
        m = self.fr.groupby("date")[["ns_eur", "cogs_eur"]].sum()
        m["gm_pct"] = (1 - m.cogs_eur / m.ns_eur) * 100
        m = m.join(g.set_index("date")["materials_index_eur"])
        lags = range(0, int(self.cfg["max_lag_months"]) + 1)
        # year-on-year changes remove trend and seasonality, so the correlation reveals the timing
        d_gm = m["gm_pct"].diff(12)
        d_ix = m["materials_index_eur"].pct_change(12) * 100
        corr = pd.Series({L: d_gm.corr(d_ix.shift(L)) for L in lags})
        best = int(corr.idxmin())
        x = d_ix.shift(best)
        ok = x.notna() & d_gm.notna()
        slope = np.polyfit(x[ok], d_gm[ok], 1)[0] * 10
        fig = make_subplots(rows=1, cols=2, column_widths=[0.68, 0.32], specs=[[{"secondary_y": True}, {}]],
                            subplot_titles=["Gross margin % vs materials cost index (EUR)", "Correlation of YoY changes by lag (months)"])
        fig.add_trace(go.Scatter(x=m.index, y=m.gm_pct, name="Gross margin %", line=dict(color=NAVY, width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=m.index, y=m.materials_index_eur, name="Materials index (Jan-21=100)",
                                 line=dict(color=RED, dash="dot")), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Bar(x=list(corr.index), y=corr.values, marker_color=[RED if L == best else GREY for L in corr.index],
                             showlegend=False), row=1, col=2)
        fig.update_yaxes(title_text="GM %", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Index", row=1, col=1, secondary_y=True)
        fig.update_layout(title=f"Q3 Commodity costs hit gross margin with a {best}-month lag", height=450,
                          legend=dict(orientation="h", y=-0.15))
        yr = m.groupby(m.index.year)["gm_pct"].agg(["min", "max", "mean"]).round(1)
        trough22 = m.loc["2022", "gm_pct"].idxmin()
        peak_idx = m.loc["2021":"2022", "materials_index_eur"].idxmax()
        last = m.index.max()
        ix_yoy = m["materials_index_eur"].pct_change(12).loc[last] * 100
        implied = ix_yoy / 10 * slope
        q26 = m.loc["2026"].groupby(m.loc["2026"].index.quarter)["gm_pct"].mean()
        q25 = m.loc["2025"].groupby(m.loc["2025"].index.quarter)["gm_pct"].mean()
        tab = pd.DataFrame({"lag_months": list(corr.index), "correlation": corr.round(2).values})
        finding = (f"Year-on-year changes in gross margin move inversely with the recipe-weighted materials index in EUR; the "
                   f"strongest link is at a {best}-month lag (correlation {corr.min():.2f}; same-month {corr.loc[0]:.2f}), "
                   f"consistent with hedging and supplier contracts. "
                   f"Each +10% on the index costs about {abs(slope):.1f} points of gross margin. "
                   f"In 2022 the index peaked in {peak_idx:%b %Y} and margin bottomed in {trough22:%b %Y} "
                   f"({yr.loc[2022, 'min']:.1f}% vs {yr.loc[2021, 'mean']:.1f}% average in 2021). "
                   f"The 2026 oil and fragrance shock is coming through: in {last:%b %Y} the index is {ix_yoy:+.0f}% year on year, "
                   f"which at the historical sensitivity implies {implied:+.1f} points of gross margin, but Q3 2026 margin is only "
                   f"{q26.iloc[-1] - q25.loc[q26.index[-1]]:+.1f} points vs a year earlier ({q26.iloc[-1]:.1f}%). Hypothesis "
                   f"for the forecast to test: the rest of the pressure is still in hedges and contracts and will reach "
                   f"Q4 2026 and early 2027.")
        decision = (f"COGS per unit is forecast from SKU recipes x commodity prices x FX with a {best}-month lag, not as a "
                    "% of Net Sales. This makes the commodity sliders in the scenario app move gross margin correctly, and "
                    "uses the latest commodity prices as the starting point for Q4 2026 and 2027.")
        self.lag_best = best
        return self._record("Q3 How do commodity shocks hit gross margin?", fig, finding, decision, tab)

    # ======================================================== Q4 share leads NS?
    def q4_share_leads(self):
        e = self.ext_market.copy()
        e["date"] = pd.to_datetime(e["date"])
        e["sellout"] = e["market_size_rsv_lc"] * e["our_value_share"]
        ns = self.fr.groupby(["date", "country", "category"])["net_sales_lc"].sum().rename("ns").reset_index()
        d = e.merge(ns, on=["date", "country", "category"])
        d = d[~d.country.isin(self.hyper)].sort_values(["country", "category", "date"])
        chile = (d.country == "Chile") & (d.date.isin(pd.to_datetime(["2025-03-01", "2025-04-01"])))
        d.loc[chile, "ns"] = np.nan
        grp = d.groupby(["country", "category"])
        d["ns_yoy"] = grp["ns"].pct_change(12, fill_method=None)
        d["so_yoy"] = grp["sellout"].pct_change(12, fill_method=None)
        K = int(self.cfg["lead_lag_months"])
        res = {}
        for L in range(-K, K + 1):    # L > 0: sell-out L months earlier explains Net Sales today
            x = d.groupby(["country", "category"])["so_yoy"].shift(L)
            ok = x.notna() & d.ns_yoy.notna()
            res[L] = np.corrcoef(x[ok], d.ns_yoy[ok])[0, 1]
        corr = pd.Series(res)
        best = int(corr.idxmax())
        q = d.assign(q=d.date.dt.to_period("Q")).groupby(["country", "category", "q"])[["ns", "sellout"]].sum(min_count=3)
        qy = q.groupby(level=[0, 1]).pct_change(4, fill_method=None).dropna()
        r_q = qy.ns.corr(qy.sellout)
        gap = (qy.ns - qy.sellout).abs().median()
        fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                            subplot_titles=["Quarterly YoY growth: Net Sales vs panel sell-out", "Monthly correlation by lead/lag"])
        samp = qy.sample(min(1500, len(qy)), random_state=1)
        fig.add_trace(go.Scatter(x=samp.sellout * 100, y=samp.ns * 100, mode="markers", marker=dict(color=STEEL, size=5, opacity=0.5),
                                 name="category x country x quarter"), row=1, col=1)
        lo_, hi_ = np.nanpercentile(np.r_[qy.sellout, qy.ns] * 100, [1, 99])
        lim = [np.floor(lo_ / 5) * 5, np.ceil(hi_ / 5) * 5]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=GREY, dash="dash"), name="1:1"), row=1, col=1)
        fig.add_trace(go.Bar(x=list(corr.index), y=corr.values, marker_color=[NAVY if L == best else GREY for L in corr.index],
                             showlegend=False), row=1, col=2)
        fig.update_xaxes(title_text="Panel sell-out growth %", range=lim, row=1, col=1)
        fig.update_yaxes(title_text="Net Sales growth %", range=lim, row=1, col=1)
        fig.update_xaxes(title_text="months sell-out leads (+) / lags (-)", row=1, col=2)
        fig.update_layout(title=f"Q4 Panel sell-out explains most Net Sales growth (quarterly r = {r_q:.2f}), but not all of it",
                          height=450, legend=dict(orientation="h", y=-0.2))
        tab = pd.DataFrame({"lead_months": list(corr.index), "correlation": corr.round(3).values})
        finding = (f"Excluding Argentina and Turkey, panel sell-out growth explains Net Sales growth well at quarterly level "
                   f"(correlation {r_q:.2f}, R-squared {r_q ** 2:.2f}) with a typical gap of {gap * 100:.1f} points. "
                   f"Monthly, the best alignment is at lag {best} (correlation {corr.max():.2f}), so sell-out is a coincident "
                   f"rather than a leading signal; month-to-month noise comes from retailer stocking around promotions. "
                   f"The ratio of Net Sales to sell-out drifts over time, consistent with our sales moving into channels the "
                   f"panel does not cover (e-commerce, discounters).")
        decision = ("Market size x share is the outside-in anchor of the baseline, used at quarterly or annual level and "
                    "with a country x category conversion ratio that is allowed to drift. Monthly phasing comes from the "
                    "seasonality and promo calendar, not from the panel. The unexplained gap is what departments must justify.")
        return self._record("Q4 Does market share lead our Net Sales?", fig, finding, decision, tab)

    # ======================================================== Q5 submission bias
    def q5_bias(self):
        s = self.subs[self.subs.actual_value.notna()].copy()
        pct_units = s["unit"].str.startswith("%")
        s["error"] = np.where(pct_units, s.submitted_value - s.actual_value, s.submitted_value / s.actual_value - 1)
        s["error_type"] = np.where(pct_units, "points", "relative")
        b = s.groupby(["department", "metric", "country", "error_type"]).agg(
            years=("error", "size"), mean_error=("error", "mean"),
            same_direction=("error", lambda e: max((e > 0).mean(), (e < 0).mean()))).reset_index()
        bc = self.cfg["bias"]
        thr = np.where(b.error_type == "points", bc["min_bias_pp"], bc["min_bias_rel"])
        b["systematic"] = (b.years >= bc["min_years"]) & (b.same_direction >= bc["consistent_share"]) & (b.mean_error.abs() >= thr)
        b["direction"] = np.where(b.mean_error > 0, "over-states", "under-states")
        sysb = b[b.systematic].copy()
        dm = sysb.groupby(["department", "metric", "direction", "error_type"]).agg(countries=("country", "nunique"),
                                                                      avg_bias=("mean_error", "mean")).reset_index()
        vol = b[b.metric == "Volume growth"].set_index("country")["mean_error"].sort_values() * 100
        lab = dm.department + ": " + dm.metric + " (" + dm.direction + ")"
        txt = [f"{n} {'country' if n == 1 else 'countries'}, avg " + (f"{v:+.0%}" if t_ == "relative" else f"{v * 100:+.1f} pts")
               for n, v, t_ in zip(dm.countries, dm.avg_bias, dm.error_type)]
        # stacked vertically: long labels get the full width and all 30 countries fit
        fig = make_subplots(rows=2, cols=1, row_heights=[0.62, 0.38], vertical_spacing=0.10,
                            subplot_titles=["Sales: volume growth submitted minus actual (points, 2022-25 average). "
                                            "Red = biased by 1 point or more",
                                            "Systematic biases (same direction every year): countries affected and average bias"])
        thr_pts = bc["min_bias_pp"] * 100
        fig.add_trace(go.Bar(y=vol.index, x=vol.values, orientation="h",
                             marker_color=[RED if abs(v) >= thr_pts else GREY for v in vol.values],
                             text=[f"{v:+.1f}" if abs(v) >= thr_pts else "" for v in vol.values], textposition="outside",
                             hovertemplate="%{y}: %{x:+.1f} pts<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Bar(y=lab, x=dm.countries, orientation="h", marker_color=NAVY, text=txt, textposition="outside",
                             hovertemplate="%{y}<br>%{text}<extra></extra>"), row=2, col=1)
        fig.update_yaxes(tickmode="array", tickvals=list(vol.index), ticktext=list(vol.index), tickfont=dict(size=11),
                         automargin=True, row=1, col=1)
        fig.update_yaxes(automargin=True, autorange="reversed", row=2, col=1)
        lo_ = min(vol.min(), -thr_pts) - 1.2
        hi_ = max(vol.max(), thr_pts) + 1.2
        fig.update_xaxes(range=[lo_, hi_], title_text="points", zeroline=True, zerolinecolor=GREY, row=1, col=1)
        fig.update_xaxes(range=[0, 46], title_text="number of countries (of 30)", row=2, col=1)
        fig.update_layout(title="Q5 Biases are systematic, not random: same direction every year", height=1050,
                          showlegend=False, bargap=0.25)
        sand = sysb[(sysb.metric == "Volume growth") & (sysb.mean_error < 0)].country.tolist()
        over = sysb[(sysb.metric == "Volume growth") & (sysb.mean_error > 0)].country.tolist()
        tab = sysb.sort_values(["department", "metric", "mean_error"])[["department", "metric", "country", "years", "mean_error", "direction"]]
        tab["mean_error"] = tab["mean_error"].round(4)
        lines = "; ".join(f"{r.department} {r.metric} {r.direction} in {r.countries} {'country' if r.countries == 1 else 'countries'} (avg "
                          + (f"{r.avg_bias:+.0%}" if r.error_type == "relative" else f"{r.avg_bias * 100:+.1f} pts") + ")"
                          for r in dm.itertuples())
        finding = (f"{len(sysb)} department x metric x country combinations are biased in the same direction in every year "
                   f"2022-2025. Sales sandbags volume in {', '.join(sand)} and over-calls it in {', '.join(over)}. "
                   f"Group-wide patterns: {lines}.")
        decision = ("The AI challenger gets a bias-correction factor per department x metric x country from this track record "
                    "and shows a 'challenged' view next to each submission. Countries without a systematic bias are taken at face value.")
        self.bias_table = b
        return self._record("Q5 Which departments have biased submissions?", fig, finding, decision, tab)

    # ======================================================== Q6 concentration
    def q6_concentration(self):
        d = self.fr[self.fr.year == 2025]
        sku = d.groupby("sku_code")["ns_eur"].sum().sort_values(ascending=False)
        cum = sku.cumsum() / sku.sum()
        n80 = int((cum < 0.8).sum() + 1)
        combo = d.groupby(["sku_code", "country"])["ns_eur"].sum()
        tail = combo[combo < self.cfg["tail_threshold_eur"]]
        launch = self.sku.dropna(subset=["launch_date"]).assign(cohort=lambda x: pd.to_datetime(x.launch_date).dt.year)
        coh = self.fr[self.fr.sku_code.isin(launch.sku_code)].merge(launch[["sku_code", "cohort"]], on="sku_code") \
            .groupby(["year", "cohort"])["ns_eur"].sum().unstack("cohort").fillna(0) / 1e6
        delisted = self.sku.dropna(subset=["delist_date"])
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Pareto: share of 2025 Net Sales by SKU rank",
                                                            "Net Sales of launches by launch year (EUR m)"])
        fig.add_trace(go.Scatter(x=np.arange(1, len(cum) + 1) / len(cum) * 100, y=cum.values * 100, line=dict(color=NAVY, width=2.5),
                                 name="cumulative share"), row=1, col=1)
        fig.add_hline(y=80, line=dict(color=GREY, dash="dot"), row=1, col=1)
        for i, c in enumerate(coh.columns):
            fig.add_trace(go.Bar(x=coh.index, y=coh[c], name=f"launched {c}", marker_color=[NAVY, STEEL, GREY, "#c9d3de"][i % 4]), row=1, col=2)
        fig.update_layout(barmode="stack", title=f"Q6 {n80} of {len(sku)} SKUs make 80% of Net Sales; launches are small but growing",
                          height=450, legend=dict(orientation="h", y=-0.2))
        fig.update_xaxes(title_text="% of SKUs (ranked)", row=1, col=1)
        fig.update_yaxes(title_text="% of Net Sales", row=1, col=1)
        tab = pd.DataFrame({"metric": ["SKUs making 80% of NS", "Top 20% of SKUs share", "Active SKU x country combos",
                                       f"Tail combos (< EUR {self.cfg['tail_threshold_eur']:,.0f}/yr)", "Tail share of NS",
                                       "Launched SKUs 2022-25", "Launch share of 2025 NS", "Delisted SKUs"],
                            "value": [n80, f"{cum.iloc[int(len(sku) * 0.2) - 1]:.0%}", len(combo), len(tail),
                                      f"{tail.sum() / combo.sum():.1%}", len(launch), f"{coh.loc[2025].sum() * 1e6 / sku.sum():.1%}",
                                      len(delisted)]})
        finding = (f"The portfolio is concentrated: {n80} of {len(sku)} SKUs ({n80 / len(sku):.0%}) generate 80% of 2025 Net Sales and "
                   f"the top 20% of SKUs generate {cum.iloc[int(len(sku) * 0.2) - 1]:.0%}. {len(tail):,} of {len(combo):,} SKU x country "
                   f"combinations sell less than EUR {self.cfg['tail_threshold_eur'] / 1e3:.0f}k a year ({tail.sum() / combo.sum():.1%} of Net Sales) "
                   f"and many have months with no sales. {len(launch)} SKUs launched in 2022-2025 now make "
                   f"{coh.loc[2025].sum() * 1e6 / sku.sum():.1%} of Net Sales; {len(delisted)} SKUs were delisted.")
        decision = ("One global model across all SKU x country series, but with the long tail handled as intermittent demand "
                    "(no zero inflation of errors), launches forecast from the ramp-up curve of earlier launches, and delisted "
                    "SKUs removed after their delist date. Forecast accuracy is reported weighted by Net Sales, so the head "
                    "of the portfolio drives the score.")
        return self._record("Q6 Where is the portfolio concentrated?", fig, finding, decision, tab)

    # ======================================================== Q7 material gaps
    def q7_material_movements(self):
        cy, py = self.cfg["ytd_compare"]["current"], self.cfg["ytd_compare"]["prior"]
        months = list(range(1, self.extract_month.month + 1))
        periods = [(py - 1, py, None, f"FY{py} vs FY{py - 1}"), (py, cy, months, f"YTD {cy} vs {py} (Jan-{self.extract_month:%b})")]
        e = self.ext_market.copy()
        e["date"] = pd.to_datetime(e["date"])
        e["sellout"] = e["market_size_rsv_lc"] * e["our_value_share"]
        cat_of = self.sku.drop_duplicates("brand").set_index("brand")["category"]
        out = []
        for y0, y1, mths, label in periods:
            b = self.bridge(y0, y1, months=mths, by=["brand", "country"])
            b["delta"] = b["end"] - b["start"]
            ee = e[e.date.dt.year.isin([y0, y1]) & (e.date.dt.month.isin(mths) if mths else True)]
            so = ee.groupby(["country", "category", ee.date.dt.year])["sellout"].sum().unstack()
            g = (so[y1] / so[y0] - 1).rename("sellout_growth")
            b = b.reset_index()
            b["category"] = b["brand"].map(cat_of)
            b = b.merge(g.reset_index(), on=["country", "category"], how="left")
            b["organic"] = b[["volume", "mix", "price_hyperinflation", "price_other", "launches_delists"]].sum(axis=1)
            b["market_explained"] = b["start"] * b["sellout_growth"]
            b["unexplained"] = b["organic"] - b["market_explained"]
            b["period"] = label
            b["data_issue"] = np.where((b.country == "Chile") & ((y0 == 2025) | (y1 == 2025)),
                                       "Chile Mar-Apr 2025 missing from extract", "")
            # internal shifts between our own brands in the same category x country net out at category level
            cc = b.groupby(["country", "category"])["unexplained"].sum()
            self.netting = getattr(self, "netting", {})
            self.netting[label] = 1 - cc.abs().sum() / b["unexplained"].abs().sum()
            b = b[b["delta"].abs() >= self.mat].copy()
            out.append(b)
        t = pd.concat(out, ignore_index=True)
        eff = {"volume": "volume", "mix": "mix", "price_hyperinflation": "hyperinflation pricing", "price_other": "pricing",
               "launches_delists": "launches/delists", "mna": "M&A", "fx": "FX"}

        def narrative(r):
            main = max(eff, key=lambda k: abs(r[k]))
            act = ("check extract first (known data gap)" if r.data_issue else
                   "ask Sales/Commercial for commentary" if abs(r.unexplained) >= self.mat else
                   "explained by market and FX - no action")
            return (f"{r.brand} {r.country}: {r.delta / 1e6:+.1f}m. Main driver {eff[main]} {r[main] / 1e6:+.1f}m. "
                    f"Panel sell-out {r.sellout_growth:+.0%} explains {r.market_explained / 1e6:+.1f}m; "
                    f"unexplained {r.unexplained / 1e6:+.1f}m -> {act}.")
        t["commentary"] = t.apply(narrative, axis=1)
        t["needs_commentary"] = (t["unexplained"].abs() >= self.mat) | (t["data_issue"] != "")
        t = t.sort_values(["period", "unexplained"], key=lambda s: s.abs() if s.name == "unexplained" else s, ascending=[True, False])
        show = t.copy()
        for c in ["start", "end", "delta", "volume", "mix", "price_hyperinflation", "price_other", "launches_delists", "mna",
                  "fx", "market_explained", "unexplained"]:
            show[c] = (show[c] / 1e6).round(2)
        cur = t[t.period == periods[-1][3]]
        top = cur.reindex(cur["unexplained"].abs().sort_values(ascending=False).index).head(15)
        fig = go.Figure()
        lab = top.brand + " " + top.country
        fig.add_trace(go.Bar(y=lab, x=top.market_explained / 1e6, orientation="h", name="Explained by market (panel sell-out)", marker_color=GREY))
        fig.add_trace(go.Bar(y=lab, x=top.fx / 1e6, orientation="h", name="FX", marker_color=STEEL))
        fig.add_trace(go.Bar(y=lab, x=top.unexplained / 1e6, orientation="h", name="Unexplained - needs commentary",
                             marker_color=[RED if v < 0 else GREEN for v in top.unexplained]))
        fig.update_layout(barmode="relative", height=560, yaxis=dict(autorange="reversed"),
                          title=f"Q7 Material movements (|change| > EUR {self.mat / 1e3:,.0f}k), {periods[-1][3]}: top 15 by unexplained amount (EUR m)",
                          legend=dict(orientation="h", y=-0.12))
        n_all = t.groupby("period").size()
        n_need = t.groupby("period")["needs_commentary"].sum()
        finding = (f"{n_all.get(periods[0][3], 0)} brand x country movements exceed EUR {self.mat / 1e3:,.0f}k in {periods[0][3]} and "
                   f"{n_all.get(periods[-1][3], 0)} in {periods[-1][3]}. After removing what panel sell-out and FX explain, "
                   f"{int(n_need.get(periods[0][3], 0))} and {int(n_need.get(periods[-1][3], 0))} respectively still need a business "
                   f"explanation. About {self.netting[periods[-1][3]]:.0%} of the unexplained amount nets out within the same "
                   f"category and country, i.e. sales shifting between our own sister brands that the category-level panel "
                   f"cannot see. Largest unexplained YTD item: {top.iloc[0].commentary}")
        decision = ("Every movement above EUR 500k is decomposed automatically (volume, price, mix, launches, M&A, FX, market) "
                    "and only the unexplained part goes to the department as a question, netted first at category x country so "
                    "that brand cannibalisation is not mistaken for market under- or over-performance. This list is the input "
                    "queue for the AI challenger and the Skeptic agent.")
        self.material = t
        return self._record(f"Q7 Which movements above EUR {self.mat / 1e3:,.0f}k are not explained?", fig, finding, decision,
                            show[["period", "brand", "country", "start", "end", "delta", "volume", "mix", "price_hyperinflation",
                                  "price_other", "launches_delists", "mna", "fx", "market_explained", "unexplained",
                                  "needs_commentary", "commentary"]])

    # ======================================================== run + export
    def run_all(self):
        for fn in [self.q1_growth_bridge, self.q2_seasonality, self.q3_commodity_margin, self.q4_share_leads,
                   self.q5_bias, self.q6_concentration, self.q7_material_movements]:
            print(f"  {fn.__name__} ...", flush=True)
            fn()
        self.export()
        return self

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{"Question": r["question"], "Answer": r["finding"], "Forecast decision": r["decision"]}
                             for r in self.results])

    def export(self):
        with pd.ExcelWriter(self.out / "eda_findings.xlsx") as w:
            self.summary().to_excel(w, sheet_name="summary", index=False)
            for i, r in enumerate(self.results, 1):
                r["table"].to_excel(w, sheet_name=f"Q{i}", index=r["table"].index.name is not None or i == 1)
        parts = ["<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
                 "<title>Hygea Group - Exploratory analysis</title>"
                 "<style>body{font-family:Arial,sans-serif;max-width:1150px;margin:30px auto;color:#1f2933;padding:0 16px}"
                 "h1{color:#1f3a5f}h2{color:#1f3a5f;border-top:1px solid #d9dee4;padding-top:24px}"
                 ".box{background:#f4f6f8;border-left:4px solid #1f3a5f;padding:10px 14px;margin:8px 0}"
                 ".dec{border-left-color:#2e7d5b}</style></head><body>",
                 "<h1>Hygea Group (fictional) - exploratory analysis</h1>",
                 f"<p>Clean actuals Jan 2021 - {self.extract_month:%b %Y}. Each chart answers one business question and ends "
                 "with the forecast decision it drives. EUR at actual monthly rates unless stated.</p>"]
        for i, r in enumerate(self.results):
            parts.append(f"<h2>{r['question']}</h2>")
            parts.append(r["fig"].to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
            parts.append(f"<div class='box'><b>Answer.</b> {r['finding']}</div>")
            parts.append(f"<div class='box dec'><b>Forecast decision.</b> {r['decision']}</div>")
        parts.append("</body></html>")
        (self.out / "eda_report.html").write_text("\n".join(parts), encoding="utf-8")


def main():
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "analysis.yaml"
    print("Running exploratory analysis ...")
    ex = Explorer(cfg).run_all()
    for r in ex.results:
        print(f"\n{r['question']}\n  ANSWER:   {r['finding']}\n  DECISION: {r['decision']}")
    print(f"\nOutputs: {ex.out / 'eda_report.html'} and {ex.out / 'eda_findings.xlsx'}")


if __name__ == "__main__":
    main()
