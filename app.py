"""
app.py  -  STEP 7: Hygea Group 2027 scenario room (Streamlit).

Decision this app supports:
    "Is the 2027 plan credible, and which levers close the gap to the 22% margin target?"

Every slider change re-runs the SAME Step 5 forecast model (volume) and driver P&L (price, COGS, FX).
Data: app_data/ (built by forecast_engine/app_bundle.py). Synthetic company - no real or personal data.

Run locally:   streamlit run app.py
"""
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecast_engine.scenario import PRESETS, Levers, ScenarioKit

DATA = Path(__file__).parent / "app_data"
NAVY, STEEL, GREY, RED, GREEN, AMBER, INK = "#1f3a5f", "#6b8cb3", "#a7b1bc", "#b23a48", "#2e7d5b", "#c98a1b", "#1f2933"
PLOT = dict(template="plotly_white", font=dict(family="IBM Plex Sans, Arial, sans-serif", size=13, color=INK),
            margin=dict(l=40, r=20, t=60, b=40))

st.set_page_config(page_title="Hygea 2027 scenario room", page_icon="🧭", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, .stMetric {font-family: 'IBM Plex Sans', Arial, sans-serif;}
h1 {font-family: 'IBM Plex Serif', Georgia, serif; font-weight: 500; color: #1f3a5f; letter-spacing: -0.01em;}
h2, h3 {color: #1f3a5f; font-weight: 600;}
.headline {border-left: 5px solid #1f3a5f; background: #f3f5f8; padding: 14px 18px; font-size: 1.12rem; line-height: 1.5; margin: 6px 0 18px 0;}
.headline.bad {border-left-color: #b23a48;} .headline.good {border-left-color: #2e7d5b;}
.small {color: #5b6672; font-size: 0.88rem;}
div[data-testid="stMetric"] {border: 1px solid #dfe4ea; padding: 10px 14px; border-radius: 4px;}
div[data-testid="stMetric"].big {border-color: #1f3a5f;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------ data
@st.cache_resource(show_spinner="Loading the forecast model ...")
def load_kit():
    return ScenarioKit(DATA / "scenario_kit")


@st.cache_data
def load_tables():
    t = {n: pd.read_parquet(DATA / f"{n}.parquet") for n in
         ["group_pnl_monthly", "backtest_accuracy", "backtest_monthly", "backtest_gross_margin", "group_pnl_by_year",
          "group_views", "flags", "by_department", "patterns", "targets"]}
    t["build"] = pd.read_json(DATA / "build_info.json", typ="series")
    return t


@st.cache_data(max_entries=64, show_spinner="Re-running the forecast ...")
def run(_kit, L: Levers):
    p = _kit.pnl(L)
    return {"fy27": _kit.summary(p), "q4_26": _kit.summary(p, mask=_kit.months[_kit.fmonths].year == 2026),
            "monthly": _kit.monthly(p), "country": _kit.by(p, "country"), "category": _kit.by(p, "category"),
            "region": _kit.by(p, "region"), "bridge": _kit.bridge(L)}


@st.cache_data(max_entries=32, show_spinner="Solving for the target ...")
def gap_table(_kit, L: Levers, target: float):
    d = _kit.close_gap(L, target)
    return d, d.attrs["margin_now"], d.attrs["gap_eur"]


kit, T = load_kit(), load_tables()
target_margin = float(T["targets"].set_index("target")["value"].get("EBIT margin", 0.22))
act = T["group_pnl_monthly"]
act = act[act.Type == "Actual"]
fy26_actual = act[act.index.year == 2026]

# ------------------------------------------------------------------------------------------ sidebar
FIELDS = {f: v for f, v in asdict(Levers()).items()}
for f, v in FIELDS.items():
    st.session_state.setdefault(f, v)


def apply_preset():
    for f, v in asdict(PRESETS[st.session_state["preset"]]).items():
        st.session_state[f] = v


with st.sidebar:
    st.markdown("### Scenario")
    st.selectbox("Start from a preset", list(PRESETS), key="preset", on_change=apply_preset)
    st.caption("Then adjust any lever. Changes apply to all months after Sep 2026.")
    with st.expander("Commodities", expanded=True):
        st.slider("Oil-linked inputs (surfactants, plastics), %", -30, 50, step=5, key="oil_linked_pct")
        st.slider("Fragrance oils, %", -30, 50, step=5, key="fragrance_pct")
        st.slider("Aerosol cans, %", -30, 50, step=5, key="aerosol_pct")
    with st.expander("Inflation (extra points per year)"):
        st.slider("EMEA", -3.0, 10.0, step=0.5, key="infl_emea")
        st.slider("North America", -3.0, 10.0, step=0.5, key="infl_north_am")
        st.slider("Latin America", -3.0, 20.0, step=0.5, key="infl_latam")
        st.slider("APAC", -3.0, 10.0, step=0.5, key="infl_apac")
    with st.expander("Currencies (+ = against us)"):
        st.slider("EUR stronger vs USD and USD-pegged, %", -15, 20, step=1, key="eur_vs_usd_pct")
        st.slider("Turkish lira and Argentine peso weaker, %", -10, 60, step=5, key="hyper_fx_pct")
        st.slider("Other emerging-market currencies weaker, %", -10, 30, step=1, key="other_em_fx_pct")
    with st.expander("Markets and consumers"):
        st.slider("Fabric Care market, % by Dec 2027", -10, 10, step=1, key="mkt_fabric")
        st.slider("Water Softeners & Dishwasher market, %", -10, 10, step=1, key="mkt_water")
        st.slider("Air Care market, %", -10, 10, step=1, key="mkt_air")
        st.slider("Surface Cleaning & Disinfection market, %", -10, 10, step=1, key="mkt_surface")
        st.slider("Toilet Care market, %", -10, 10, step=1, key="mkt_toilet")
        st.slider("Private label share, points", -3.0, 5.0, step=0.5, key="private_label_pts")
        st.slider("Consumer confidence, index points", -15, 10, step=1, key="confidence_pts")
    st.markdown("### Management actions (from Jan 2027)")
    st.slider("Extra price increase, %", 0.0, 10.0, step=0.5, key="price_pct")
    st.slider("Price elasticity of volume", -2.0, 0.0, step=0.1, key="price_elasticity",
              help="Assumption: -0.8 means +1% price loses 0.8% volume. Typical household-care range -0.5 to -1.5.")
    st.slider("Trade spend rate, points", -5.0, 3.0, step=0.5, key="trade_rate_pts")
    st.slider("A&P, %", -40, 20, step=5, key="ap_pct", help="Cost only: no sales response is modelled (conservative).")
    st.slider("Overhead programme, %", -20, 5, step=1, key="overhead_pct")
    st.markdown("### Department view on top")
    view = st.radio("Overlay", ["Outside-in only", "Department submissions", "Challenged submissions"], key="view",
                    help="Adds each view's difference to the outside-in baseline (Step 6) on top of the scenario.")

L = Levers(**{f: float(st.session_state[f]) for f in FIELDS})
R, B = run(kit, L), run(kit, Levers())

gv = T["group_views"]
over = {"Outside-in only": None, "Department submissions": "Department submissions", "Challenged submissions": "Challenged"}[view]
d_ns = d_ct = 0.0
if over:
    d_ns = gv.loc[over, "net_sales"] - gv.loc["Baseline (outside-in)", "net_sales"]
    d_ct = gv.loc[over, "contribution"] - gv.loc["Baseline (outside-in)", "contribution"]

f27, b27 = R["fy27"], B["fy27"]
ns, ct = f27["net_sales"] + d_ns, f27["contribution"] + d_ct
margin, base_margin = ct / ns, b27["contribution"] / b27["net_sales"]
gm = f27["gross_margin"] / f27["net_sales"]
ns26 = fy26_actual["ns"].sum() + R["q4_26"]["net_sales"]
gap_eur = ct - target_margin * ns

# ------------------------------------------------------------------------------------------ header
st.title("2027 scenario room")
st.markdown(f"<div class='small'>Hygea Group (fictional) · decision: <b>is the 2027 plan credible, and which levers close the gap "
            f"to the {target_margin:.0%} margin target?</b> · actuals to {pd.Timestamp(T['build']['last_actual_month']):%b %Y} · EUR</div>",
            unsafe_allow_html=True)

br = R["bridge"].drop(["Base forecast", "Scenario"])
biggest = br.abs().idxmax() if br.abs().max() > 1e6 else None
scen_name = st.session_state["preset"] if L == PRESETS[st.session_state["preset"]] else "this custom scenario"
state = "good" if margin >= target_margin else "bad"
head = (f"Under <b>{scen_name}</b>{' with ' + view.lower() if over else ''}, 2027 contribution is <b>EUR {ct / 1e6:,.0f}m "
        f"({margin:.1%})</b>, {'above' if gap_eur >= 0 else 'below'} the {target_margin:.0%} target by EUR {abs(gap_eur) / 1e6:,.0f}m")
if biggest and L != Levers():
    head += f". Versus the base forecast, the biggest mover is {biggest.split(' (')[0].lower()} (EUR {br[biggest] / 1e6:+,.0f}m)"
head += "."
st.markdown(f"<div class='headline {state}'>{head}</div>", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns([1.5, 1, 1, 1])
k1.metric("2027 contribution margin (after country overheads)", f"{margin:.1%}",
          f"{(margin - target_margin) * 100:+.1f} pts vs {target_margin:.0%} target", delta_color="normal")
k2.metric("2027 Net Sales", f"EUR {ns / 1e6:,.0f}m", f"{ns / ns26 - 1:+.1%} vs 2026")
k3.metric("2027 gross margin", f"{gm:.1%}", f"{(gm - b27['gross_margin'] / b27['net_sales']) * 100:+.1f} pts vs base")
k4.metric("2027 contribution", f"EUR {ct / 1e6:,.0f}m", f"EUR {(ct - b27['contribution']) / 1e6:+,.0f}m vs base")

m = R["monthly"]
mb = B["monthly"]
a = act[act.index >= "2024-01-01"]
fig = go.Figure()
fig.add_trace(go.Scatter(x=a.index, y=(a.ns - a.cogs - a.ap - a.overheads) / a.ns * 100, name="Actual", line=dict(color=INK, width=2.5)))
for df, name, color, dash in [(mb, "Base forecast", GREY, "dot"), (m, "Scenario", NAVY, "solid")]:
    y = (df.net_sales - df.cogs - df.ap - df.overheads) / df.net_sales * 100
    fig.add_trace(go.Scatter(x=[a.index[-1]] + list(df.index), y=[(a.ns - a.cogs - a.ap - a.overheads).iloc[-1] / a.ns.iloc[-1] * 100] + list(y),
                             name=name, line=dict(color=color, width=2.5, dash=dash)))
fig.add_hline(y=target_margin * 100, line=dict(color=RED, dash="dash", width=1), annotation_text=f"{target_margin:.0%} target")
fig.update_layout(title="Monthly contribution margin: December bonus and Q4 media make the shape; the level is the decision",
                  yaxis_title="% of Net Sales", height=360, legend=dict(orientation="h", y=-0.2), **PLOT)
st.plotly_chart(fig, width="stretch")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Why it moved", "Where", "Departments", "Close the gap", "Can I trust it?"])

# ------------------------------------------------------------------------------------------ why
with tab1:
    b = R["bridge"] / 1e6
    fig = go.Figure(go.Waterfall(x=list(b.index), measure=["absolute"] + ["relative"] * (len(b) - 2) + ["total"],
                                 y=list(b.values[:-1]) + [0], text=[f"{v:,.0f}" if i in (0, len(b) - 1) else f"{v:+,.0f}" for i, v in enumerate(b.values)],
                                 textposition="outside", connector=dict(line=dict(color=GREY)),
                                 increasing=dict(marker=dict(color=GREEN)), decreasing=dict(marker=dict(color=RED)),
                                 totals=dict(marker=dict(color=NAVY))))
    lo = min(b.min(), b.iloc[0], b.iloc[-1])
    fig.update_layout(title="2027 contribution: base forecast to scenario, one block of drivers at a time (EUR m)", height=430,
                      yaxis=dict(range=[min(b.iloc[0], b.iloc[-1]) * 0.8, max(b.iloc[0], b.iloc[-1]) * 1.08]), showlegend=False, **PLOT)
    st.plotly_chart(fig, width="stretch")
    st.caption("Blocks are applied in order (volume, inflation, commodities, FX, management), so each bar is the extra effect of that block.")
    rows = ["net_sales", "trade_spend", "cogs", "gross_margin", "ap", "overheads", "contribution"]
    tbl = pd.DataFrame({"Base forecast": [b27[r] for r in rows], "Scenario": [f27[r] for r in rows]},
                       index=["Net Sales", "Trade spend", "COGS", "Gross margin", "A&P", "Overheads", "Contribution"]) / 1e6
    tbl["Difference"] = tbl.Scenario - tbl["Base forecast"]
    st.markdown("**2027 P&L, EUR m** (costs shown as positive amounts; a higher cost is unfavourable)")
    st.dataframe(tbl.style.format("{:,.1f}"), width="stretch")

# ------------------------------------------------------------------------------------------ where
with tab2:
    lvl = st.radio("Show by", ["region", "country", "category"], horizontal=True)
    s, bs = R[lvl], B[lvl]
    metric = "contribution" if "contribution" in s else "net_sales"
    d = pd.DataFrame({"base": bs[metric], "scenario": s[metric]}) / 1e6
    d["change"] = d.scenario - d.base
    d = d.sort_values("change")
    fig = go.Figure(go.Bar(y=d.index, x=d.change, orientation="h", marker_color=[RED if v < 0 else GREEN for v in d.change],
                           text=[f"{v:+,.1f}" for v in d.change], textposition="outside"))
    fig.update_layout(title=f"Change in 2027 {metric.replace('_', ' ')} vs base forecast, by {lvl} (EUR m)",
                      height=max(320, 26 * len(d) + 120), yaxis=dict(automargin=True), **PLOT)
    st.plotly_chart(fig, width="stretch")
    if lvl == "category":
        st.caption("Overheads are held by country, not category, so categories show Net Sales.")
    show = s.copy()
    st.dataframe((show / 1e6).drop(columns=["gross_margin_pct"]).join(show[["gross_margin_pct"]]).style.format(
        {c: "{:,.1f}" for c in show.columns if c != "gross_margin_pct"} | {"gross_margin_pct": "{:.1%}"}), width="stretch")

# ------------------------------------------------------------------------------------------ departments
with tab3:
    st.markdown("**Three views of 2027 at base assumptions** (Step 6). Departments' submissions add up to the target; "
                "the outside-in view does not.")
    g = gv[["net_sales", "contribution", "contribution_margin_pct", "gap_to_ebit_target_eur"]].copy()
    g[["net_sales", "contribution", "gap_to_ebit_target_eur"]] /= 1e6
    st.dataframe(g.style.format({"net_sales": "{:,.0f}", "contribution": "{:,.0f}", "contribution_margin_pct": "{:.1%}",
                                 "gap_to_ebit_target_eur": "{:+,.0f}"}), width="stretch")
    st.markdown("**How much each department's own inputs move 2027 contribution vs the outside-in baseline** "
                "(EUR m; + = more optimistic than the market-based view)")
    st.dataframe(T["by_department"].style.format({"submission_vs_baseline_eur_m": "{:+,.1f}", "challenged_vs_baseline_eur_m": "{:+,.1f}"}),
                 width="stretch")
    st.markdown("**Group-wide patterns** (one decision each, not one per country)")
    st.dataframe(T["patterns"].style.format({"eur_m": "{:+,.1f}"}), width="stretch", hide_index=True)
    f = T["flags"]
    c1, c2, c3 = st.columns(3)
    sev = c1.multiselect("Severity", ["High", "Medium", "Low"], ["High", "Medium"])
    dep = c2.multiselect("Department", sorted(f.owner.unique()), [])
    cty = c3.multiselect("Country", sorted(f.country.unique()), [])
    q = f[f.severity.isin(sev)]
    if dep:
        q = q[q.owner.isin(dep)]
    if cty:
        q = q[q.country.isin(cty)]
    st.caption(f"{len(q)} of {len(f)} flags. Severity rule: High = EUR 5m+ on 2027 contribution; Medium = EUR 1m+, any systematic "
               "bias, or a narrative contradiction. Sign: + = submission more optimistic than the reference.")
    st.dataframe(q[["flag_id", "severity", "check", "department", "country", "metric", "eur_impact", "finding", "question"]]
                 .assign(eur_impact=lambda x: x.eur_impact / 1e6), width="stretch", hide_index=True,
                 column_config={"eur_impact": st.column_config.NumberColumn("EUR m", format="%+.1f"),
                                "finding": st.column_config.TextColumn(width="large"),
                                "question": st.column_config.TextColumn(width="large")})

# ------------------------------------------------------------------------------------------ close the gap
with tab4:
    g, now, gap = gap_table(kit, L, target_margin)
    if now >= target_margin:
        st.success(f"This scenario already reaches {now:.1%}. No further action needed to hit {target_margin:.0%}.")
    else:
        st.markdown(f"Contribution margin in this scenario is **{now:.1%}** (outside-in, before department overlays): "
                    f"EUR **{gap / 1e6:,.0f}m** short of {target_margin:.0%}. Each line shows how far **one** lever would need "
                    "to move on its own, on top of the current settings:")
        g2 = g.copy()
        g2["change needed"] = g2.apply(lambda r: "not reachable alone" if pd.isna(r["change needed"])
                                       else f"{r['change needed']:+.1f} {r['unit']}", axis=1)
        st.dataframe(g2[["lever", "change needed"]], width="stretch", hide_index=True)
        st.caption("Price uses the elasticity set in the sidebar. A&P cuts are shown as cost only; in reality they risk volume.")
    st.markdown("### Recommended actions")
    dep_t = T["by_department"]
    fl = T["flags"]
    usa = fl[(fl.country == "USA") & (fl.metric == "Volume growth")].eur_impact.max() / 1e6
    hyp = fl[(fl.country.isin(["Turkey", "Argentina"])) & (fl.metric == "Net price/mix increase")
             & (fl.check == "Systematic bias (track record)")].eur_impact.sum() / 1e6
    price_need = g.loc[g.lever.str.startswith("Extra price"), "change needed"].iloc[0]
    acts = pd.DataFrame([
        ["Supply Chain plans 2027 unit costs as if the 2026 commodity shock had not happened (27 countries)",
         dep_t.loc["Supply Chain", "submission_vs_baseline_eur_m"], "Re-base COGS on the outside-in commodity view; show hedge cover by month",
         "Supply Chain Director", "15 Nov 2026", "Accept outside-in COGS for the budget?"],
        ["Sales USA: volume +3.2% vs -0.8% baseline, over-stated every year since 2022", usa,
         "Haircut to the challenged value unless new listings are evidenced", "Sales Director USA", "8 Nov 2026", "Approve haircut"],
        ["Pricing needed to close the margin gap", gap / 1e6 if now < target_margin else 0,
         f"Prepare a {price_need:.1f}% price plan by category and account" if pd.notna(price_need) and now < target_margin else "Not needed in this scenario",
         "Revenue Management", "30 Nov 2026", "Mandate the price round?"],
        ["Overhead growth below inflation in 28 countries", dep_t.loc["Finance/HR", "submission_vs_baseline_eur_m"],
         "Either fund salary inflation or name the savings behind it", "Group HR & Finance", "15 Nov 2026", "-"],
        ["Turkey / Argentina price under-called vs inflation every year", hyp,
         "Set price at CPI pass-through; report organic growth with and without hyperinflation", "Sales Directors TR / AR", "8 Nov 2026", "-"],
    ], columns=["Issue", "EUR m exposure", "Recommended action", "Owner", "Due", "Decision required from CFO"])
    st.dataframe(acts.style.format({"EUR m exposure": "{:+,.0f}"}), width="stretch", hide_index=True)
    st.caption("Exposures come from the Step 6 challenge engine (+ = submission more optimistic than the market-based view). "
               "Owners are roles, not people. The agent swarm (Step 8) debates these before anything is changed.")

# ------------------------------------------------------------------------------------------ trust
with tab5:
    st.markdown("### How accurate was this model last time?")
    bm = T["backtest_monthly"] / 1e6
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bm.index, y=bm["Actual"], name="Actual", line=dict(color=INK, width=3)))
    for col, color, dash in [("AI forecast (LightGBM + drivers)", NAVY, "solid"), ("AI forecast with perfect hindsight on drivers", NAVY, "dot"),
                             ("Holt-Winters (brand x country)", STEEL, "dash"), ("Department budget (bottom-up submissions)", AMBER, "dash")]:
        fig.add_trace(go.Scatter(x=bm.index, y=bm[col], name=col, line=dict(color=color, dash=dash)))
    fig.update_layout(title="Jan-Sep 2026 forecast made in Dec 2025: wrong assumptions, not the model, explain most of the miss",
                      height=400, legend=dict(orientation="h", y=-0.25), yaxis_title="EUR m", **PLOT)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(T["backtest_accuracy"].style.format("{:.1%}", na_rep="-"), width="stretch")
    st.markdown("### Reconciliation")
    fy = T["group_pnl_by_year"]
    rec = pd.DataFrame({"Step 5 forecast file": [fy.loc[2027, "net_sales"] / 1e6, fy.loc[2027, "contribution"] / 1e6],
                        "This app, all levers at zero": [b27["net_sales"] / 1e6, b27["contribution"] / 1e6]},
                       index=["FY2027 Net Sales", "FY2027 contribution"])
    rec["Difference"] = rec.iloc[:, 1] - rec.iloc[:, 0]
    st.dataframe(rec.style.format("{:,.2f}"), width="stretch")
    st.markdown("### Definitions, sources and limitations")
    st.markdown(f"""
- **Contribution** = Net Sales - COGS - A&P - country overheads. It is **not EBIT**: group costs, R&D and D&A are not in the data, so the real gap to the {target_margin:.0%} EBIT target is larger.
- **Sign convention:** costs are positive amounts; a higher cost lowers contribution. In the department tables, + means a submission is more optimistic than the outside-in view.
- **Engine:** volume from one LightGBM model across 4,749 SKU x country series (re-run on every change); price follows CPI pass-through by country, COGS follows SKU recipes x commodities x FX with hedging over 3 months, A&P and trade spend follow run-rates, overheads follow inflation.
- **Not modelled:** a sales response to A&P; competitor reactions; new launches after Sep 2026; tariffs other than through the inflation and commodity levers.
- **Data:** synthetic. Actuals to {pd.Timestamp(T['build']['last_actual_month']):%b %Y}; Chile Mar-Apr 2025 missing from the extract. Bundle built {T['build']['built_utc']} UTC from: {', '.join(T['build']['sources'])}.
""")
