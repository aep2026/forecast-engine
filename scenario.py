"""
scenario.py  -  STEP 7: live scenario engine behind the Streamlit app.

Loads the 'scenario kit' saved by Step 5 (model + arrays + estimated elasticities, a few MB) and re-runs
the SAME forecast with changed assumptions. With every lever at zero it reproduces the Step 5 forecast.

Levers
  Outside world : commodities, inflation by region, FX, market size by category, private label, consumer confidence
  Management    : extra price increase (with a price elasticity), trade spend rate, A&P, overhead programme
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecast_engine.forecast import Features, PnLEngine

REGIONS = ["EMEA", "North_Am", "Latam", "APAC"]
HYPER = ["Argentina", "Turkey"]
USD_LINKED = ["USA", "Saudi Arabia", "UAE"]


@dataclass(frozen=True)
class Levers:
    # outside world
    oil_linked_pct: float = 0.0          # surfactants and plastics
    fragrance_pct: float = 0.0
    aerosol_pct: float = 0.0
    infl_emea: float = 0.0               # extra inflation, points per year
    infl_north_am: float = 0.0
    infl_latam: float = 0.0
    infl_apac: float = 0.0
    eur_vs_usd_pct: float = 0.0          # + = EUR stronger vs USD (and USD-pegged)
    hyper_fx_pct: float = 0.0            # + = Turkish lira / Argentine peso weaker
    other_em_fx_pct: float = 0.0         # + = other emerging-market currencies weaker
    mkt_fabric: float = 0.0              # market size vs consensus by Dec 2027, %
    mkt_water: float = 0.0
    mkt_air: float = 0.0
    mkt_surface: float = 0.0
    mkt_toilet: float = 0.0
    private_label_pts: float = 0.0
    confidence_pts: float = 0.0
    # management actions (from Jan 2027)
    price_pct: float = 0.0
    price_elasticity: float = -0.8
    trade_rate_pts: float = 0.0
    ap_pct: float = 0.0
    overhead_pct: float = 0.0

    def volume_key(self):
        """Only these levers change the volume model's inputs (cache key)."""
        return (self.infl_emea, self.infl_north_am, self.infl_latam, self.infl_apac, self.mkt_fabric, self.mkt_water,
                self.mkt_air, self.mkt_surface, self.mkt_toilet, self.private_label_pts, self.confidence_pts)


PRESETS = {
    "Base (Sep 2026 consensus)": Levers(),
    "Oil shock": Levers(oil_linked_pct=25, fragrance_pct=10, aerosol_pct=10, infl_emea=1.5, infl_north_am=1.5, infl_latam=1.5,
                        infl_apac=1.5, confidence_pts=-3),
    "Consumer recession": Levers(mkt_fabric=-3, mkt_water=-3, mkt_air=-6, mkt_surface=-3, mkt_toilet=-2, private_label_pts=2,
                                 confidence_pts=-8, infl_emea=-0.5, infl_north_am=-0.5),
    "Trade war": Levers(oil_linked_pct=10, aerosol_pct=10, infl_north_am=2, eur_vs_usd_pct=5, mkt_fabric=-2, mkt_water=-2,
                        mkt_air=-2, mkt_surface=-2, mkt_toilet=-2, confidence_pts=-4),
    "FX crisis in emerging markets": Levers(hyper_fx_pct=30, other_em_fx_pct=10, infl_latam=5, infl_emea=1, confidence_pts=-2),
}
CAT_LEVER = {"Fabric Care": "mkt_fabric", "Water Softeners & Dishwasher": "mkt_water", "Air Care": "mkt_air",
             "Surface Cleaning & Disinfection": "mkt_surface", "Toilet Care": "mkt_toilet"}


class ScenarioKit:
    def __init__(self, kit_dir: str | Path):
        kit = Path(kit_dir)
        self.meta = json.load(open(kit / "meta.json"))
        cfg = self.meta["cfg"]
        series = pd.read_parquet(kit / "series.parquet")
        A = np.load(kit / "arrays.npz")
        self.months = pd.to_datetime(self.meta["months"])
        o = self.meta["origin"]
        T = len(self.months)
        vol = np.full((len(series), T), np.nan)
        ns = np.full((len(series), T), np.nan)
        vol[:, : o + 1], ns[:, : o + 1] = A["vol"], A["ns"]
        self.P = SimpleNamespace(series=series, vol=vol, ns=ns, act={"fx": A["act_fx"]}, T=T, S=len(series),
                                 C=len(self.meta["countries"]), K=len(self.meta["cats"]), months=self.months, oh=A["oh"],
                                 cfg=cfg, countries=self.meta["countries"], cats=self.meta["cats"])
        self.D0 = {k[2:]: A[k] for k in A.files if k.startswith("D_")}
        self.engine = PnLEngine(self.P, cfg)
        for k in ["beta", "price_drift", "cost_coef", "gp3", "uc3", "cpi3", "cidx3", "tr12", "ap12", "ap_phase"]:
            setattr(self.engine, k, A[f"E_{k}"])
        self.engine.o = o
        self.o = o
        self.F = Features(self.P)
        self.booster = lgb.Booster(model_file=str(kit / "volume_model.txt"))
        self.fmonths = np.arange(o + 1, T)
        self.c = series["c"].values
        self.k = series["k"].values
        self.region = np.array([self.meta["region"][c] for c in self.meta["countries"]])
        self.em = np.array([bool(self.meta["em"][c]) for c in self.meta["countries"]])
        self._vol_cache = {}
        self.y27 = self.months[self.fmonths].year == 2027

    # --------------------------------------------------------------- drivers
    def drivers(self, L: Levers, parts=("volume", "inflation", "commodities", "fx")) -> dict:
        D = {k: v.copy() for k, v in self.D0.items()}
        o, T = self.o, self.P.T
        h = np.arange(T) - o
        ramp = np.clip(h / 12, 0, 1)                         # market / private label changes build up over 12 months
        fut = slice(o + 1, T)
        if "volume" in parts:
            for ki, cat in enumerate(self.P.cats):
                D["mkt"][ki, :, fut] *= (1 + getattr(L, CAT_LEVER[cat]) / 100 * ramp[fut])
            D["pl"][..., fut] += L.private_label_pts / 100 * ramp[fut]
            D["conf"][:, fut] += L.confidence_pts
        if "inflation" in parts:
            pts = {"EMEA": L.infl_emea, "North_Am": L.infl_north_am, "Latam": L.infl_latam, "APAC": L.infl_apac}
            for ci, r in enumerate(self.region):
                D["cpi"][ci, fut] *= (1 + pts[r] / 100) ** (h[fut] / 12)
        if "commodities" in parts:
            D["surfactant_index"][fut] *= 1 + L.oil_linked_pct / 100
            D["plastics_index"][fut] *= 1 + L.oil_linked_pct / 100
            D["fragrance_index"][fut] *= 1 + L.fragrance_pct / 100
            D["aerosol_index"][fut] *= 1 + L.aerosol_pct / 100
        if "fx" in parts:
            for ci, name in enumerate(self.P.countries):
                if name in USD_LINKED:
                    D["fx"][ci, fut] *= 1 + L.eur_vs_usd_pct / 100
                elif name in HYPER:
                    D["fx"][ci, fut] *= 1 + L.hyper_fx_pct / 100
                elif self.em[ci]:
                    D["fx"][ci, fut] *= 1 + L.other_em_fx_pct / 100
            D["eurusd"][fut] *= 1 + L.eur_vs_usd_pct / 100
        return D

    def volume(self, L: Levers) -> np.ndarray:
        key = L.volume_key()
        if key not in self._vol_cache:
            D = self.drivers(L, parts=("volume", "inflation"))   # the volume model also sees inflation
            X = self.F.build(self.o, D, len(self.fmonths))
            vol = np.zeros((self.P.S, len(self.fmonths)))
            vol[X["i"].values, X["t"].values - self.o - 1] = np.clip(self.booster.predict(X[_features()]), 0, None) * X["level"].values
            if len(self._vol_cache) > 64:
                self._vol_cache.clear()
            self._vol_cache[key] = vol
        return self._vol_cache[key]

    # --------------------------------------------------------------- P&L
    def pnl(self, L: Levers, vol: np.ndarray | None = None, parts=("volume", "inflation", "commodities", "fx"),
            management=True) -> dict:
        vol = self.volume(L) if vol is None else vol
        D = self.drivers(L, parts)
        e, m = self.engine, self.fmonths
        is27 = (self.months[m] >= pd.Timestamp("2027-01-01"))[None, :]
        pf = np.where(is27, 1 + L.price_pct / 100, 1.0) if management else 1.0
        v = vol * (pf ** L.price_elasticity if management else 1.0)
        tr0 = e.tr12
        if management:
            e.tr12 = tr0 + L.trade_rate_pts / 100
        try:
            p = e.pnl(v, D, m)
        finally:
            e.tr12 = tr0
        out = {"volume": v}
        for key in ["gross_sales_lc", "trade_spend_lc", "net_sales_lc"]:
            out[key] = np.nan_to_num(p[key]) * pf
        out["cogs_lc"] = np.nan_to_num(p["cogs_lc"])
        out["ap_lc"] = np.nan_to_num(p["ap_lc"]) * pf * ((1 + L.ap_pct / 100) if management else 1.0)
        fx = D["fx"][self.c][:, m]
        for key in ["gross_sales", "trade_spend", "net_sales", "cogs", "ap"]:
            out[f"{key}_eur"] = out[f"{key}_lc"] / fx
        oh_lc = e.overheads(self.o, self.D0, m) * self.D0["fx"][:, None, m]       # base overheads, local currency
        cpi_ratio = D["cpi"][:, m] / self.D0["cpi"][:, m]
        oh_lc = oh_lc * cpi_ratio[:, None, :] ** 0.8                               # salaries follow inflation
        if management:
            oh_lc = oh_lc * np.where(is27[0], 1 + L.overhead_pct / 100, 1.0)[None, None, :]
        out["overheads_eur"] = oh_lc / D["fx"][:, None, m]
        return out

    # --------------------------------------------------------------- summaries
    def summary(self, p: dict, mask=None) -> dict:
        m = self.y27 if mask is None else mask
        ns, gs = p["net_sales_eur"][:, m].sum(), p["gross_sales_eur"][:, m].sum()
        cogs, ap = p["cogs_eur"][:, m].sum(), p["ap_eur"][:, m].sum()
        oh = p["overheads_eur"][..., m].sum()
        return {"net_sales": ns, "gross_sales": gs, "trade_spend": gs - ns, "cogs": cogs, "gross_margin": ns - cogs, "ap": ap,
                "overheads": oh, "contribution": ns - cogs - ap - oh, "volume": p["volume"][:, m].sum()}

    def by(self, p: dict, level: str) -> pd.DataFrame:
        s = self.P.series
        key = {"country": s["country"], "category": s["category"], "region": s["country"].map(self.meta["region"]),
               "brand": s["brand"]}[level].values
        m = self.y27
        d = pd.DataFrame({level: key, "net_sales": p["net_sales_eur"][:, m].sum(1), "cogs": p["cogs_eur"][:, m].sum(1),
                          "ap": p["ap_eur"][:, m].sum(1)}).groupby(level).sum()
        if level in ("country", "region"):
            oh = pd.Series(p["overheads_eur"][..., m].sum((1, 2)), index=self.P.countries)
            if level == "region":
                oh = oh.groupby(oh.index.map(self.meta["region"])).sum()
            d["overheads"] = oh.reindex(d.index).fillna(0)
            d["contribution"] = d.net_sales - d.cogs - d.ap - d.overheads
        d["gross_margin_pct"] = 1 - d.cogs / d.net_sales
        return d

    def monthly(self, p: dict) -> pd.DataFrame:
        return pd.DataFrame({"net_sales": p["net_sales_eur"].sum(0), "cogs": p["cogs_eur"].sum(0), "ap": p["ap_eur"].sum(0),
                             "overheads": p["overheads_eur"].sum((0, 1))}, index=self.months[self.fmonths])

    def bridge(self, L: Levers) -> pd.Series:
        """FY2027 contribution from the base forecast to the scenario, one block of drivers at a time."""
        base = self.summary(self.pnl(Levers()))["contribution"]
        vol = self.volume(L)
        steps = [("Market, private label, confidence (volume)", ("volume",), False),
                 ("Inflation (price pass-through, costs, salaries)", ("volume", "inflation"), False),
                 ("Commodities (COGS)", ("volume", "inflation", "commodities"), False),
                 ("FX translation and USD inputs", ("volume", "inflation", "commodities", "fx"), False),
                 ("Management actions (price, trade, A&P, overheads)", ("volume", "inflation", "commodities", "fx"), True)]
        out, prev = {"Base forecast": base}, base
        for label, parts, mgmt in steps:
            v = self.summary(self.pnl(L, vol=vol, parts=parts, management=mgmt))["contribution"]
            out[label] = v - prev
            prev = v
        out["Scenario"] = prev
        return pd.Series(out)

    def close_gap(self, L: Levers, target_margin: float) -> pd.DataFrame:
        """For each management lever alone: the change needed (on top of the scenario) to reach the target margin."""
        def margin(**kw):
            s = self.summary(self.pnl(Levers(**{**asdict(L), **kw}), vol=self.volume(L)))
            return s["contribution"] / s["net_sales"], s
        now, s0 = margin()
        rows = []
        for lever, label, lo, hi, unit in [("price_pct", "Extra price increase from Jan 2027", 0, 20, "%"),
                                           ("overhead_pct", "Overhead programme", -30, 0, "%"),
                                           ("ap_pct", "A&P reduction", -60, 0, "%"),
                                           ("trade_rate_pts", "Trade spend rate", -6, 0, "pts")]:
            base_v = getattr(L, lever)
            f = lambda x: margin(**{lever: base_v + x})[0] - target_margin
            if now >= target_margin:
                need = 0.0
            else:
                # the levers are close to linear: secant search, 3-4 evaluations instead of ~30
                x0, x1 = 0.0, (hi if hi != 0 else lo)
                f0, f1 = now - target_margin, f(x1)
                need = np.nan
                if f0 * f1 <= 0:
                    for _ in range(4):
                        x2 = x1 - f1 * (x1 - x0) / (f1 - f0) if f1 != f0 else x1
                        x0, f0, x1, f1 = x1, f1, x2, f(x2)
                        if abs(f1) < 1e-5:
                            break
                    need = x1
            rows.append({"lever": label, "change needed": need, "unit": unit})
        d = pd.DataFrame(rows)
        d.attrs["margin_now"] = now
        d.attrs["gap_eur"] = target_margin * s0["net_sales"] - s0["contribution"]
        return d


def _features():
    from forecast_engine.forecast import FEATURES
    return FEATURES
