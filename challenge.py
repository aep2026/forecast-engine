"""
challenge.py  -  STEP 6: the AI challenger (deterministic part).

Takes every department submission for the budget year and tests it four ways:
  1. GAP vs the outside-in baseline (Step 5), priced in EUR of 2027 contribution
  2. TRACK RECORD: is this department x metric x country systematically biased? (Step 4, Q5)
  3. CONSISTENCY between departments (price vs COGS inflation, volume vs A&P, promo vs trade spend ...)
  4. NARRATIVE: vague or self-contradicting justifications (rule-based here; the agents go deeper in Step 8)

Output: three views of the 2027 P&L (baseline / department / challenged), one row per flag with a
question for the department, and a compact challenge pack for the agent swarm.

    python forecast_engine/challenge.py           (uses challenge.yaml)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

NAVY, STEEL, GREY, RED, GREEN, AMBER = "#1f3a5f", "#5b7fa6", "#9aa5b1", "#b23a48", "#2e7d5b", "#c98a1b"
LAYOUT = dict(template="plotly_white", font=dict(family="Arial", size=12), margin=dict(l=60, r=30, t=70, b=60))

# submission metric -> P&L input it drives
METRIC_MAP = {
    ("Sales", "Volume growth"): "vg",
    ("Sales", "Net price/mix increase"): "pr",
    ("Sales", "Trade spend rate"): "tr",
    ("Marketing", "A&P budget"): "ap",
    ("Supply Chain", "COGS per unit inflation"): "ci",
    ("Finance/HR", "Overhead growth"): "og",
}
INPUT_LABEL = {"vg": "Volume growth", "pr": "Net price/mix increase", "tr": "Trade spend rate", "ap": "A&P budget",
               "ci": "COGS per unit inflation", "og": "Overhead growth"}


def country_pnl(base: dict, x: dict) -> dict:
    """2027 P&L of one country (local currency) from the 2026 base and six inputs."""
    vol = base["vol"] * (1 + x["vg"])
    ns = base["ns"] * (1 + x["vg"]) * (1 + x["pr"])
    gs = ns / (1 - x["tr"])
    cogs = base["cogs"] * (1 + x["vg"]) * (1 + x["ci"])
    ap = x["ap"] * 1e6
    oh = base["oh"] * (1 + x["og"])
    return {"volume": vol, "gross_sales": gs, "trade_spend": gs - ns, "net_sales": ns, "cogs": cogs, "ap": ap,
            "overheads": oh, "contribution": ns - cogs - ap - oh}


class Challenger:
    def __init__(self, config_path="challenge.yaml"):
        p = Path(config_path).resolve()
        self.base = p.parent
        self.cfg = yaml.safe_load(open(p, encoding="utf-8"))
        P = self.cfg["paths"]
        self.clean, self.fc, self.raw = (self.base / P[k] for k in ["clean_dir", "forecast_dir", "raw_dir"])
        self.out = self.base / P["out_dir"]
        self.out.mkdir(parents=True, exist_ok=True)
        self.by, self.y0 = self.cfg["budget_year"], self.cfg["base_year"]
        self.mat = float(self.cfg["materiality_eur"])

    # ----------------------------------------------------------------- inputs
    def load(self):
        cm = pd.read_parquet(self.fc / "country_pnl_monthly.parquet")
        cm["year"] = cm["date"].dt.year
        cols = ["volume", "gross_sales_lc", "trade_spend_lc", "net_sales_lc", "cogs_lc", "ap_lc", "overheads_lc"]
        y = cm[cm.year.isin([self.y0, self.by])].groupby(["country", "year"])[cols].sum()
        eur = cm.assign(ns_eur=cm.net_sales_lc / cm.fx_lc_per_eur).groupby(["country", "year"])["ns_eur"].sum()
        y["fx_eff"] = y["net_sales_lc"] / eur          # Net-Sales-weighted average rate of the year
        self.y = y
        s = pd.read_parquet(self.clean / "submissions_country.parquet")
        self.subs_all = s
        self.subs = s[s.budget_year == self.by].copy()
        self.subs_group = pd.read_parquet(self.clean / "submissions_group.parquet")
        self.targets = pd.read_excel(self.raw / "master_data.xlsx", sheet_name="strategic_targets")
        ol = pd.read_parquet(self.clean / "ext_outlooks.parquet")
        mac = pd.read_parquet(self.clean / "ext_macro_country.parquet")
        mac["date"] = pd.to_datetime(mac["date"])
        o = ol[(ol.vintage == self.cfg["consensus_vintage"]) & (ol.table == "macro")][["date", "country", "cpi_index"]]
        o["date"] = pd.to_datetime(o["date"])
        cpi = pd.concat([mac[["date", "country", "cpi_index"]], o]).drop_duplicates(["date", "country"], keep="first")
        cy = cpi.groupby(["country", cpi.date.dt.year])["cpi_index"].mean().unstack()
        self.cpi_growth = cy[self.by] / cy[self.y0] - 1
        mk = pd.read_parquet(self.clean / "ext_market_category_country.parquet")
        mk["date"] = pd.to_datetime(mk["date"])
        last = mk.date.max()
        a = mk[mk.date > last - pd.DateOffset(months=12)].groupby("country")["our_value_share"].mean()
        b = mk[(mk.date <= last - pd.DateOffset(months=12)) & (mk.date > last - pd.DateOffset(months=24))].groupby("country")["our_value_share"].mean()
        self.share_trend = a - b                       # share points, last 12 months vs the 12 before
        return self

    # ----------------------------------------------------------------- track record
    def track_record(self):
        s = self.subs_all[self.subs_all.actual_value.notna()].copy()
        pct = s["unit"].str.startswith("%")
        s["error"] = np.where(pct, s.submitted_value - s.actual_value, s.submitted_value / s.actual_value - 1)
        s["error_type"] = np.where(pct, "points", "relative")
        b = s.groupby(["department", "metric", "country", "error_type"]).agg(
            years=("error", "size"), mean_error=("error", "mean"),
            same_direction=("error", lambda e: max((e > 0).mean(), (e < 0).mean())),
            history=("error", lambda e: ", ".join(f"{v * 100:+.1f}" for v in e))).reset_index()
        bc = self.cfg["bias"]
        thr = np.where(b.error_type == "points", bc["min_bias_points"], bc["min_bias_relative"])
        b["systematic"] = (b.years >= bc["min_years"]) & (b.same_direction >= bc["consistent_share"]) & (b.mean_error.abs() >= thr)
        self.bias = b
        delivered = s[s.metric.isin(["Promo volume uplift", "COGS savings delivered"])].groupby(["metric", "country"])["actual_value"].mean()
        self.delivered = delivered
        return b

    def correct(self, dept, metric, country, value):
        r = self.bias[(self.bias.department == dept) & (self.bias.metric == metric) & (self.bias.country == country)]
        if r.empty or not r.iloc[0].systematic:
            return value, None
        r = r.iloc[0]
        new = value - r.mean_error if r.error_type == "points" else value / (1 + r.mean_error)
        return new, r

    # ----------------------------------------------------------------- three views
    def views(self):
        rows, inputs = [], {}
        for country in self.y.index.get_level_values(0).unique():
            b0, b1 = self.y.loc[(country, self.y0)], self.y.loc[(country, self.by)]
            base = {"vol": b0.volume, "ns": b0.net_sales_lc, "cogs": b0.cogs_lc, "oh": b0.overheads_lc}
            xb = {"vg": b1.volume / b0.volume - 1,
                  "pr": (b1.net_sales_lc / b1.volume) / (b0.net_sales_lc / b0.volume) - 1,
                  "tr": b1.trade_spend_lc / b1.gross_sales_lc, "ap": b1.ap_lc / 1e6,
                  "ci": (b1.cogs_lc / b1.volume) / (b0.cogs_lc / b0.volume) - 1,
                  "og": b1.overheads_lc / b0.overheads_lc - 1}
            sub = self.subs[self.subs.country == country]
            xd, xc, notes = dict(xb), dict(xb), {}
            for (dept, metric), key in METRIC_MAP.items():
                r = sub[(sub.department == dept) & (sub.metric == metric)]
                if len(r):
                    v = float(r.submitted_value.iloc[0])
                    xd[key] = v
                    xc[key], notes[key] = self.correct(dept, metric, country, v)
            inputs[country] = {"base": base, "baseline": xb, "department": xd, "challenged": xc, "bias": notes,
                               "fx": b1.fx_eff}
            for view, x in [("Baseline (outside-in)", xb), ("Department submissions", xd), ("Challenged", xc)]:
                p = country_pnl(base, x)
                rows.append({"country": country, "view": view, **{k: v / b1.fx_eff for k, v in p.items()}, **{f"in_{k}": v for k, v in x.items()}})
        self.inputs = inputs
        v = pd.DataFrame(rows)
        self.country_views = v
        g = v.groupby("view")[["net_sales", "gross_sales", "cogs", "ap", "overheads", "contribution"]].sum()
        ns26 = (self.y.xs(self.y0, level=1).net_sales_lc / self.y.xs(self.y0, level=1).fx_eff).sum()
        g["gross_margin_pct"] = 1 - g.cogs / g.net_sales
        g["contribution_margin_pct"] = g.contribution / g.net_sales
        g["ns_growth_vs_2026"] = g.net_sales / ns26 - 1
        tgt = self.targets.set_index("target")["value"]
        g["gap_to_ebit_target_eur"] = g.contribution - tgt.get("EBIT margin", 0.22) * g.net_sales
        self.group_views = g.loc[["Baseline (outside-in)", "Department submissions", "Challenged"]]
        return self.group_views

    # ----------------------------------------------------------------- flags
    def flags(self):
        C, out = self.cfg, []
        cs = C["consistency"]
        sev_hi, sev_md = C["severity"]["high_eur"], C["severity"]["medium_eur"]

        def impact(country, key, value, ref="baseline"):
            I = self.inputs[country]
            x = dict(I[ref])
            x[key] = value
            return (country_pnl(I["base"], x)["contribution"] - country_pnl(I["base"], I[ref])["contribution"]) / I["fx"]

        def add(**kw):
            e = abs(kw.get("eur_impact", 0) or 0)
            kw["severity"] = "High" if e >= sev_hi else "Medium" if e >= sev_md or kw.get("force_medium") else "Low"
            kw.pop("force_medium", None)
            out.append(kw)

        for country, I in self.inputs.items():
            sub = self.subs[self.subs.country == country].set_index("metric")
            just = lambda m: sub.loc[m, "justification"] if m in sub.index else ""
            # 1. gap vs baseline + 2. track record
            for (dept, metric), key in METRIC_MAP.items():
                if metric not in sub.index:
                    continue
                d, b, c = I["department"][key], I["baseline"][key], I["challenged"][key]
                e = impact(country, key, d)
                gap = d - b
                big = abs(gap) >= (C["min_gap_relative"] * abs(b) if key == "ap" else C["min_gap_points"])
                if big and abs(e) >= self.mat:
                    add(check="Gap vs outside-in baseline", department=dept, country=country, metric=metric,
                        submitted=d, baseline=b, challenged=c, gap=gap, eur_impact=e, justification=just(metric),
                        finding=f"{metric} {self._fmt(key, d)} vs baseline {self._fmt(key, b)}: "
                                f"{'adds' if e > 0 else 'removes'} EUR {abs(e) / 1e6:.1f}m of 2027 contribution vs the outside-in view.",
                        question=self._question(key, d, b, country))
                br = I["bias"].get(key)
                if br is not None:
                    e2 = impact(country, key, c, ref="department") * -1   # what the bias is worth
                    add(check="Systematic bias (track record)", department=dept, country=country, metric=metric,
                        submitted=d, baseline=b, challenged=c, gap=d - c, eur_impact=e2, justification=just(metric),
                        finding=f"{dept} has {'over' if br.mean_error > 0 else 'under'}-stated {metric.lower()} in "
                                f"{country} in {br.years} of {br.years} years (errors {br.history} "
                                f"{'pts' if br.error_type == 'points' else '%'}). Corrected value {self._fmt(key, c)}.",
                        question=f"Your last {br.years} submissions for {metric.lower()} were {'above' if br.mean_error > 0 else 'below'} "
                                 f"actual every year. What is different this time?", force_medium=True)
            # 3. consistency
            x = I["department"]
            cpi = self.cpi_growth.get(country, np.nan)
            if cpi > cs["hyperinflation_cpi"] and x["pr"] < cpi - cs["hyper_price_shortfall"]:
                e = impact(country, "pr", cpi * 0.9, ref="department") * -1
                add(check="Consistency", department="Sales", country=country, metric="Net price/mix increase",
                    submitted=x["pr"], baseline=cpi, challenged=I["challenged"]["pr"], gap=x["pr"] - cpi, eur_impact=e,
                    justification=just("Net price/mix increase"),
                    finding=f"Price {x['pr']:+.1%} vs consensus inflation {cpi:+.1%}: in a hyperinflation market a price below CPI "
                            f"means a real price cut, or an under-call of revenue.",
                    question="Is a real-terms price cut intended? If not, what price increase is agreed with key accounts?")
            if x["pr"] < x["ci"] - cs["price_vs_cogs_gap"]:
                add(check="Consistency", department="Sales vs Supply Chain", country=country, metric="Price vs COGS inflation",
                    submitted=x["pr"], baseline=x["ci"], challenged=np.nan, gap=x["pr"] - x["ci"],
                    eur_impact=impact(country, "pr", x["ci"], ref="department") * -1,
                    justification=f"Sales: {just('Net price/mix increase')} | Supply Chain: {just('COGS per unit inflation')}",
                    finding=f"Sales plans price {x['pr']:+.1%} while Supply Chain expects unit cost inflation of {x['ci']:+.1%}: "
                            "gross margin erodes and nobody owns the gap.",
                    question="Which department closes the gap between cost inflation and price - and how?")
            vg_gap = x["vg"] - I["baseline"]["vg"]
            ap_chg = x["ap"] / I["baseline"]["ap"] - 1
            if vg_gap > cs["volume_ambition_points"] and ap_chg <= cs["ap_cut"]:
                add(check="Consistency", department="Sales vs Marketing", country=country, metric="Volume vs A&P",
                    submitted=x["vg"], baseline=I["baseline"]["vg"], challenged=np.nan, gap=vg_gap,
                    eur_impact=impact(country, "vg", x["vg"]), justification=just("Volume growth"),
                    finding=f"Sales expects volume {vg_gap * 100:+.1f} points above the market-based view while Marketing "
                            f"cuts A&P {ap_chg:+.0%}: the volume ambition is not funded.",
                    question="What will drive the extra volume if media and promotion support go down?")
            if ap_chg > cs["ap_increase_no_return"] and vg_gap <= 0:
                add(check="Consistency", department="Marketing vs Sales", country=country, metric="A&P vs volume",
                    submitted=x["ap"], baseline=I["baseline"]["ap"], challenged=np.nan, gap=ap_chg,
                    eur_impact=impact(country, "ap", x["ap"]), justification=just("A&P budget"),
                    finding=f"Marketing raises A&P {ap_chg:+.0%} but Sales plans no volume above the baseline: no visible return.",
                    question="What volume or share return is expected from the extra A&P, and is it in the Sales plan?")
            if "Promo volume uplift" in sub.index:
                pu = sub.loc["Promo volume uplift", "submitted_value"]
                hist = self.delivered.get(("Promo volume uplift", country), np.nan)
                if pu > hist + cs["promo_over_history_points"] and x["tr"] <= I["baseline"]["tr"]:
                    add(check="Consistency", department="Marketing vs Sales", country=country, metric="Promo uplift vs trade spend",
                        submitted=pu, baseline=hist, challenged=np.nan, gap=pu - hist, eur_impact=0,
                        justification=just("Promo volume uplift"),
                        finding=f"Marketing claims promo uplift of {pu:.1%} vs {hist:.1%} delivered on average, while Sales keeps "
                                "trade spend at or below the baseline rate: more promotion effect without more promotion money.",
                        question="Which promotions deliver the extra uplift, and where is their cost in the trade spend plan?",
                        force_medium=True)
            if "COGS savings delivered" in sub.index:
                sv = sub.loc["COGS savings delivered", "submitted_value"]
                hist = self.delivered.get(("COGS savings delivered", country), np.nan)
                if sv > cs["savings_vs_history_multiple"] * hist:
                    e = (sv - hist) * self.inputs[country]["base"]["cogs"] / self.inputs[country]["fx"]
                    add(check="Consistency", department="Supply Chain", country=country, metric="COGS savings claimed vs delivered",
                        submitted=sv, baseline=hist, challenged=hist, gap=sv - hist, eur_impact=e,
                        justification=just("COGS savings delivered"),
                        finding=f"Savings claimed {sv:.1%} of COGS vs {hist:.1%} delivered on average since 2022 "
                                f"(EUR {abs(e) / 1e6:.1f}m at risk if history repeats).",
                        question="Which savings projects are new versus already in the run-rate, and are any counted twice?")
            if x["og"] < cpi - cs["overhead_below_cpi_points"]:
                add(check="Consistency", department="Finance/HR", country=country, metric="Overhead growth vs inflation",
                    submitted=x["og"], baseline=cpi, challenged=I["challenged"]["og"], gap=x["og"] - cpi,
                    eur_impact=impact(country, "og", cpi, ref="department") * -1, justification=just("Overhead growth"),
                    finding=f"Overhead growth {x['og']:+.1%} vs consensus inflation {cpi:+.1%}: salary inflation is not covered "
                            "unless headcount falls.",
                    question="Which savings offset salary inflation? Are open roles assumed unfilled?")
            # 4. narrative
            nc = self.cfg["narrative"]
            for metric, r in sub.iterrows():
                t = str(r.justification).lower()
                key = next((k for (d_, m_), k in METRIC_MAP.items() if m_ == metric), None)
                vague = any(p in t for p in nc["vague_phrases"]) and not re.search(r"\d", t)
                contra = (metric == "Volume growth" and r.submitted_value < 0 and any(w in t for w in nc["growth_claims"]))
                share_contra = ("share gain" in t and self.share_trend.get(country, 0) < 0)
                if contra or share_contra:
                    add(check="Narrative", department=r.department, country=country, metric=metric, submitted=r.submitted_value,
                        baseline=I["baseline"].get(key, np.nan) if key else np.nan, challenged=np.nan, gap=np.nan, eur_impact=0,
                        justification=r.justification,
                        finding=("Justification contradicts the number: claims growth/share gains while submitting "
                                 f"{r.submitted_value:+.1%}." if contra else
                                 f"Claims share gains, but panel share fell {self.share_trend.get(country, 0) * 100:+.2f} pts in the last 12 months."),
                        question="Please reconcile the narrative with the number and the market data.", force_medium=True)
                elif vague:
                    add(check="Narrative", department=r.department, country=country, metric=metric, submitted=r.submitted_value,
                        baseline=I["baseline"].get(key, np.nan) if key else np.nan, challenged=np.nan, gap=np.nan, eur_impact=0,
                        justification=r.justification, finding="Vague justification: no quantified driver, account, date or source.",
                        question="What specific, quantified actions support this number?")
        f = pd.DataFrame(out)
        # one sign convention for every check: + = the submission is MORE OPTIMISTIC than the reference (EUR, 2027 contribution)
        f["owner"] = f["department"].str.split(" vs ").str[0]
        f["abs_eur"] = f["eur_impact"].abs()
        order = {"High": 0, "Medium": 1, "Low": 2}
        f = f.sort_values(["severity", "abs_eur"], key=lambda s: s.map(order) if s.name == "severity" else -s).reset_index(drop=True)
        f.insert(0, "flag_id", [f"F{i + 1:03d}" for i in range(len(f))])
        self.flag_table = f.drop(columns="abs_eur")
        return self.flag_table

    @staticmethod
    def _fmt(key, v):
        return f"{v:,.1f}m LC" if key == "ap" else f"{v:+.1%}" if key != "tr" else f"{v:.1%}"

    def _question(self, key, d, b, country):
        q = {"vg": "What specific distribution, listing or share actions explain volume {dir} the market-based view?",
             "pr": "Which price increases are agreed, by account and date, to justify price {dir} the outside-in view?",
             "tr": "Which trade terms change to move the trade-spend rate {dir} the run-rate?",
             "ap": "What is the media and promotion plan behind an A&P budget {dir} the run-rate, and what return is expected?",
             "ci": "Which contracts, hedges or savings make unit cost inflation {dir} the commodity-based view?",
             "og": "Which headcount or cost decisions make overhead growth {dir} the run-rate?"}[key]
        return q.format(dir="above" if d > b else "below")

    # ----------------------------------------------------------------- actuals questions (Step 4 Q7)
    def actuals_questions(self):
        try:
            from forecast_engine.explore import Explorer
            ex = Explorer(self.base / "analysis.yaml")
            ex.q7_material_movements()
            t = ex.material
            t = t[t.needs_commentary & t.period.str.startswith("YTD")]
            t = t.reindex(t.unexplained.abs().sort_values(ascending=False).index)
            self.actq = t[["brand", "country", "delta", "unexplained", "commentary"]].assign(
                delta=lambda d: (d.delta / 1e6).round(2), unexplained=lambda d: (d.unexplained / 1e6).round(2))
        except Exception as e:                                  # analysis is optional for the pack
            self.actq = pd.DataFrame({"note": [f"Q7 analysis not available: {e}"]})
        return self.actq

    # ----------------------------------------------------------------- outputs
    def summary_by_department(self):
        """EUR effect of each department's own inputs on 2027 contribution (no double counting):
        department view and challenged view, each versus the outside-in baseline."""
        keys = {"Sales": ["vg", "pr", "tr"], "Marketing": ["ap"], "Supply Chain": ["ci"], "Finance/HR": ["og"]}
        rows = []
        for dept, ks in keys.items():
            dv = cv = 0.0
            for country, I in self.inputs.items():
                b = country_pnl(I["base"], I["baseline"])["contribution"]
                xd, xc = dict(I["baseline"]), dict(I["baseline"])
                for k in ks:
                    xd[k], xc[k] = I["department"][k], I["challenged"][k]
                dv += (country_pnl(I["base"], xd)["contribution"] - b) / I["fx"]
                cv += (country_pnl(I["base"], xc)["contribution"] - b) / I["fx"]
            f = self.flag_table[self.flag_table.owner == dept]
            rows.append({"department": dept, "submission_vs_baseline_eur_m": dv / 1e6, "challenged_vs_baseline_eur_m": cv / 1e6,
                         "n_flags": len(f), "high": int((f.severity == "High").sum()), "medium": int((f.severity == "Medium").sum())})
        return pd.DataFrame(rows).set_index("department").sort_values("submission_vs_baseline_eur_m", ascending=False)

    def patterns(self, min_countries=8):
        """Issues that repeat across many countries are one pattern, not 30 flags."""
        f = self.flag_table
        g = f.groupby(["check", "department", "metric"]).agg(countries=("country", "nunique"),
                                                            eur_m=("eur_impact", lambda x: x.sum() / 1e6),
                                                            example=("finding", "first")).reset_index()
        return g[g.countries >= min_countries].sort_values("eur_m", key=abs, ascending=False)

    def pack_markdown(self, top=25):
        g, f = self.group_views, self.flag_table
        tgt = self.targets
        lines = [f"# Challenge pack - Budget {self.by} (fictional company Hygea Group)",
                 "", "Data is synthetic. Departments are identified by role and country only (no personal data).", "",
                 "## Strategic targets", tgt[["target", "value", "unit", "definition"]].to_markdown(index=False), "",
                 f"## Three views of the {self.by} P&L (EUR m)",
                 "Baseline = outside-in forecast (market, share, CPI, commodities, FX). Department = submissions as received. "
                 "Challenged = submissions corrected only where a department has a systematic bias in its track record.", ""]
        gv = g.copy()
        for c in ["net_sales", "cogs", "ap", "overheads", "contribution", "gap_to_ebit_target_eur"]:
            gv[c] = (gv[c] / 1e6).round(0)
        for c in ["gross_margin_pct", "contribution_margin_pct", "ns_growth_vs_2026"]:
            gv[c] = (gv[c] * 100).round(1)
        lines += [gv[["net_sales", "ns_growth_vs_2026", "gross_margin_pct", "ap", "overheads", "contribution",
                      "contribution_margin_pct", "gap_to_ebit_target_eur"]].to_markdown(), "",
                  "Note: contribution is after country overheads but before group costs, R&D and D&A - it is not EBIT.", "",
                  "## Effect of each department's inputs on contribution (EUR m, vs outside-in baseline)",
                  "+ = more optimistic than the baseline. Challenged = after removing systematic biases.",
                  self.summary_by_department().round(1).to_markdown(), "",
                  "## Group-wide patterns (same issue in many countries - treat as ONE decision)",
                  self.patterns().round(1).to_markdown(index=False), "",
                  f"## Top {top} individual flags (High/Medium, sorted by EUR; + = submission more optimistic than reference)"]
        t = f[f.severity.isin(["High", "Medium"]) & (f.eur_impact.abs() > 0)].head(top).copy()
        t["eur_impact_m"] = (t.eur_impact / 1e6).round(1)
        for c in ["submitted", "baseline", "challenged"]:
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
        lines.append(t[["flag_id", "severity", "check", "department", "country", "metric", "submitted", "baseline", "challenged",
                        "eur_impact_m", "finding", "justification", "question"]].to_markdown(index=False))
        nar = f[(f.check == "Narrative") & (f.finding.str.startswith("Justification contradicts") | f.finding.str.startswith("Claims share"))]
        lines += ["", "## Narrative contradictions (number and words disagree)",
                  nar[["flag_id", "department", "country", "metric", "submitted", "justification", "finding"]].to_markdown(index=False),
                  "", f"Full list: {len(f)} flags in challenge_pack.xlsx.", "",
                  "## Unexplained actual movements this year (> EUR 500k, after market and FX)"]
        lines.append(self.actq.head(10).to_markdown(index=False))
        lines += ["", "## Known data caveats",
                  "- Chile Mar-Apr 2025 missing from the extract (re-extract requested).",
                  "- Consensus outlooks are naive (trend / flat); the Dec 2025 consensus missed the 2026 US tariff and oil shock.",
                  "- Baseline accuracy (backtest Jan-Sep 2026): 0.4% group error with true drivers, 3.6% with consensus drivers."]
        md = "\n".join(lines)
        (self.out / "challenge_pack.md").write_text(md, encoding="utf-8")
        return md

    def charts(self):
        g = self.group_views
        figs = []
        fig = go.Figure()
        fig.add_trace(go.Bar(x=g.index, y=g.contribution_margin_pct * 100, marker_color=[NAVY, AMBER, STEEL],
                             text=[f"{v:.1%}" for v in g.contribution_margin_pct], textposition="outside", name="Contribution margin"))
        tgt = float(self.targets.set_index("target")["value"].get("EBIT margin", 0.22))
        fig.add_hline(y=tgt * 100, line=dict(color=RED, dash="dash"), annotation_text=f"EBIT target {tgt:.0%}")
        fig.update_layout(title=f"{self.by} contribution margin: three views vs target", yaxis=dict(title="%", range=[0, max(tgt, g.contribution_margin_pct.max()) * 115]),
                          height=400, showlegend=False, **LAYOUT)
        figs.append(fig)
        d = self.summary_by_department()
        fig = go.Figure()
        fig.add_trace(go.Bar(y=d.index, x=d.submission_vs_baseline_eur_m, orientation="h", name="Submission vs baseline", marker_color=AMBER,
                             text=[f"{v:+.0f}m" for v in d.submission_vs_baseline_eur_m], textposition="outside"))
        fig.add_trace(go.Bar(y=d.index, x=d.challenged_vs_baseline_eur_m, orientation="h", name="Challenged vs baseline", marker_color=STEEL,
                             text=[f"{v:+.0f}m" for v in d.challenged_vs_baseline_eur_m], textposition="outside"))
        fig.update_layout(title=f"How much each department's inputs move {self.by} contribution vs the outside-in baseline (EUR m; + = more optimistic)",
                          barmode="group", height=420, yaxis=dict(automargin=True), legend=dict(orientation="h", y=-0.15), **LAYOUT)
        figs.append(fig)
        f = self.flag_table[self.flag_table.eur_impact.abs() > 0].head(15)
        fig = go.Figure(go.Bar(y=f.flag_id + " " + f.department + " - " + f.country + " - " + f.metric, x=f.eur_impact / 1e6,
                               orientation="h", marker_color=[RED if v < 0 else GREEN for v in f.eur_impact]))
        fig.update_layout(title="Top 15 flags (EUR m on 2027 contribution; + = submission more optimistic than reference)", height=560,
                          yaxis=dict(autorange="reversed", automargin=True), xaxis_title="EUR m", **LAYOUT)
        figs.append(fig)
        self.figs = figs
        return figs

    def export(self):
        with pd.ExcelWriter(self.out / "challenge_pack.xlsx") as w:
            pd.DataFrame({"read me": [
                f"Challenge pack for Budget {self.by}. Synthetic data; departments identified by role and country only.",
                "group_views: the P&L under three views. flags: one row per issue with EUR impact and a question for the department.",
                "Severity: High >= EUR 5m impact on contribution; Medium >= EUR 1m, any systematic bias, or a narrative contradiction.",
                "The agent swarm (Step 8) reads challenge_pack.md and argues accept / haircut / reject for each material flag.",
                "No number is changed automatically: every haircut needs a human approver."]}).to_excel(w, sheet_name="read_me", index=False)
            self.group_views.to_excel(w, sheet_name="group_views")
            self.flag_table.to_excel(w, sheet_name="flags", index=False)
            self.summary_by_department().to_excel(w, sheet_name="by_department")
            self.patterns().to_excel(w, sheet_name="patterns", index=False)
            self.country_views.to_excel(w, sheet_name="country_views", index=False)
            self.bias[self.bias.systematic].to_excel(w, sheet_name="bias_track_record", index=False)
            self.actq.to_excel(w, sheet_name="actuals_questions", index=False)
        self.flag_table.to_parquet(self.out / "flags.parquet", index=False)
        self.group_views.to_parquet(self.out / "group_views.parquet")
        self.country_views.to_parquet(self.out / "country_views.parquet", index=False)
        parts = ["<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
                 "<title>Challenge pack</title><style>body{font-family:Arial,sans-serif;max-width:1150px;margin:30px auto;"
                 "padding:0 16px;color:#1f2933}h1,h2{color:#1f3a5f}table{border-collapse:collapse;font-size:12px}"
                 "td,th{border:1px solid #d9dee4;padding:4px 6px;vertical-align:top}</style></head><body>",
                 f"<h1>Challenge pack - Budget {self.by}</h1>"]
        for i, fig in enumerate(self.figs):
            parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
        t = self.flag_table.head(40).copy()
        t["eur_impact"] = (t.eur_impact / 1e6).round(1)
        parts.append("<h2>Top 40 flags</h2>" + t[["flag_id", "severity", "check", "department", "country", "metric",
                                                  "eur_impact", "finding", "question"]].to_html(index=False))
        parts.append("</body></html>")
        (self.out / "challenge_report.html").write_text("\n".join(parts), encoding="utf-8")

    def run(self):
        self.load()
        self.track_record()
        self.views()
        self.flags()
        self.actuals_questions()
        self.pack_markdown()
        self.charts()
        self.export()
        return self


def main():
    import sys
    c = Challenger(sys.argv[1] if len(sys.argv) > 1 else "challenge.yaml").run()
    pd.set_option("display.width", 220)
    g = c.group_views.copy()
    for col in ["net_sales", "contribution", "gap_to_ebit_target_eur"]:
        g[col] = (g[col] / 1e6).round(0)
    for col in ["gross_margin_pct", "contribution_margin_pct", "ns_growth_vs_2026"]:
        g[col] = (g[col] * 100).round(1)
    print(f"\nTHREE VIEWS OF {c.by} (EUR m)\n" + g[["net_sales", "ns_growth_vs_2026", "gross_margin_pct", "contribution",
                                                 "contribution_margin_pct", "gap_to_ebit_target_eur"]].to_string())
    f = c.flag_table
    print(f"\nFLAGS: {len(f)} | by severity {f.severity.value_counts().to_dict()} | by check {f.check.value_counts().to_dict()}")
    print("\nBY DEPARTMENT\n" + c.summary_by_department().round(1).to_string())
    print("\nTOP 12 FLAGS")
    for r in f.head(12).itertuples():
        print(f"  {r.flag_id} [{r.severity}] {r.department} | {r.country} | {r.metric} | EUR {r.eur_impact / 1e6:+.1f}m\n      {r.finding}")
    print(f"\nOutputs in {c.out}")


if __name__ == "__main__":
    main()
