"""
app.py  -  STEP 7: Hygea Group 2027 scenario room (Streamlit).  SELF-CONTAINED VERSION.

Decision this app supports:
    "Is the 2027 plan credible, and which levers close the gap to the 22% margin target?"

Every slider change re-runs the SAME Step 5 forecast model (volume) and driver P&L (price, COGS, FX).
This single file contains the model code it needs (copied from forecast_engine/forecast.py and scenario.py),
so the app runs with only this file + the app_data files on GitHub. Data files are found by name anywhere
in the repository, so the folder structure of the upload does not matter.
Synthetic company - no real or personal data.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def find(name: str) -> Path:
    """Locate a data file by name anywhere under the app folder (robust to how files were uploaded)."""
    hits = sorted(ROOT.rglob(name), key=lambda p: len(p.parts))
    if not hits:
        raise FileNotFoundError(f"'{name}' not found anywhere in the repository - upload the app_data files")
    return hits[0]


# ================================================================ model code (from forecast_engine/forecast.py)
FEATURES = ['h', 'moy', 'south', 'category', 'country', 'base_price_eur', 'sidx', 'cat_sidx', 'ly_ratio', 'trend_yoy', 'mom3', 'months_since_launch', 'hist_len', 'zero_share', 'mkt_rel', 'share_rel', 'pl_delta', 'conf_delta', 'cpi_infl']

CATEGORICAL = ['category', 'country']

class Features:
    def __init__(self, P: Panel):
        self.P = P
        s = P.series
        self.cat_i, self.cty_i = s["k"].values, s["c"].values
        self.group = (s["k"].values * 2 + s["south"].values)

    def build(self, o: int, D: dict, H: int, cutoff: int | None = None) -> pd.DataFrame:
        """One row per active series x month ahead (t = o + 1 .. o + H)."""
        P, vol = self.P, self.P.vol
        lvl = np.nanmean(vol[:, o - 11:o + 1], axis=1)
        prev = np.nanmean(vol[:, o - 23:o - 11], axis=1)
        active = np.nan_to_num(lvl) > 0
        trend = np.clip(np.where(prev > 0, lvl / prev, np.nan), 0, 5)
        mom3 = np.nanmean(vol[:, o - 2:o + 1], axis=1) / lvl
        hist = np.clip(np.sum(~np.isnan(vol[:, :o + 1]), axis=1), 0, 36)
        zero = np.mean(vol[:, o - 11:o + 1] == 0, axis=1)
        moy_hist = np.arange(o - 23, o + 1) % 12
        win = vol[:, o - 23:o + 1] / lvl[:, None]
        sidx = np.stack([np.nanmean(win[:, moy_hist == m], axis=1) for m in range(12)], axis=1)
        w0 = max(0, o - 35)                       # up to 3 years of history for the category seasonal shape
        gw = vol[:, w0:o + 1]
        gm = np.arange(w0, o + 1) % 12
        grp_sum = np.zeros((self.group.max() + 1, o + 1 - w0))
        np.add.at(grp_sum, self.group, np.nan_to_num(gw))
        cat_sidx = np.stack([grp_sum[:, gm == m].mean(1) for m in range(12)], 1) / grp_sum.mean(1, keepdims=True)
        # drivers relative to the 12 months before the origin
        k, c = self.cat_i, self.cty_i
        mreal = D["mkt"] / D["cpi"][None]
        base_m = np.nanmean(mreal[..., o - 11:o + 1], -1)
        base_s = np.nanmean(D["share"][..., o - 11:o + 1], -1)
        base_pl = np.nanmean(D["pl"][..., o - 11:o + 1], -1)
        base_cf = np.nanmean(D["conf"][..., o - 11:o + 1], -1)
        price0 = np.nansum(P.ns[:, o - 11:o + 1], 1) / np.nansum(P.vol[:, o - 11:o + 1], 1) / P.act["fx"][c, o]
        launch = P.series["launch_date_ix"].values
        delist = P.series["delist_date_ix"].values
        out = []
        for h in range(1, H + 1):
            t = o + h
            if t >= P.T:
                break
            alive = active & ~(delist <= t)
            idx = np.where(alive)[0]
            ly_t = t - 12 if t - 12 <= o else t - 24
            f = {
                "i": idx, "t": np.full(len(idx), t), "level": lvl[idx], "h": np.full(len(idx), h), "moy": np.full(len(idx), t % 12),
                "south": P.series["south"].values[idx], "category": k[idx], "country": c[idx],
                "base_price_eur": P.series["base_price_eur"].values[idx],
                "sidx": sidx[idx, t % 12], "cat_sidx": cat_sidx[self.group[idx], t % 12],
                "ly_ratio": vol[idx, ly_t] / lvl[idx], "trend_yoy": trend[idx], "mom3": mom3[idx],
                "months_since_launch": np.where(np.isnan(launch[idx]), 36, np.clip(t - launch[idx], 0, 36)),
                "hist_len": hist[idx], "zero_share": zero[idx],
                "mkt_rel": mreal[k[idx], c[idx], t] / base_m[k[idx], c[idx]],
                "share_rel": D["share"][k[idx], c[idx], t] / base_s[k[idx], c[idx]],
                "pl_delta": D["pl"][k[idx], c[idx], t] - base_pl[k[idx], c[idx]],
                "conf_delta": D["conf"][c[idx], t] - base_cf[c[idx]],
                "cpi_infl": D["cpi"][c[idx], t] / D["cpi"][c[idx], o] - 1,
                "w_value": lvl[idx] * np.nan_to_num(price0[idx]),
            }
            if cutoff is not None and t <= cutoff:
                y = vol[idx, t] / lvl[idx]
                f["y"] = np.clip(y, 0, self.P.cfg["model"]["ratio_clip"])
                f["w"] = f["w_value"] * P.weight[idx, t]
            out.append(pd.DataFrame(f))
        return pd.concat(out, ignore_index=True)

    def training_set(self, cutoff: int) -> pd.DataFrame:
        P, tl = self.P, self.P.cfg["timeline"]
        first = P.tix[pd.Timestamp(tl["first_training_origin"])]
        D = P.drivers(cutoff, "expost")
        parts = []
        for o in range(first, cutoff, int(tl["origin_step_months"])):
            H = min(int(tl["max_horizon"]), cutoff - o)
            f = self.build(o, D, H, cutoff=cutoff)
            parts.append(f.dropna(subset=["y"]))
        df = pd.concat(parts, ignore_index=True)
        return df[df["w"] > 0]


class PnLEngine:
    """Turns forecast volume into the P&L with explicit, estimated finance relationships."""

    def __init__(self, P: Panel, cfg: dict):
        self.P, self.cfg = P, cfg

    def cost_index(self, D: dict) -> np.ndarray:
        """Recipe-weighted commodity cost index per series in local currency (lagged)."""
        P = self.P
        L = int(self.cfg["drivers"]["commodity_lag_months"])
        lcusd = D["fx"] / D["eurusd"][None]
        idx = np.zeros((P.S, P.T))
        for j in self.cfg["drivers"]["recipe_inputs"]:
            x = D[f"{j}_index"]
            sm = int(self.cfg["drivers"].get("hedge_smoothing_months", 1))
            if sm > 1:   # hedges and supplier contracts spread price moves over several months
                x = pd.Series(x).rolling(sm, min_periods=1).mean().values
            x = np.concatenate([np.repeat(x[:1], L), x[:-L]]) if L else x
            idx += P.series[f"recipe_{j}"].values[:, None] * (x / x[0])[None]
        w = P.series[[f"recipe_{j}" for j in self.cfg["drivers"]["recipe_inputs"]]].sum(1).values[:, None]
        return idx / np.where(w == 0, 1, w) * (lcusd[P.series["c"].values] / lcusd[P.series["c"].values, :1])

    def estimate(self, o: int):
        """Elasticities from actual history up to the origin."""
        P = self.P
        A = P.drivers(o, "expost")
        c, k = P.series["c"].values, P.series["k"].values
        gp = P.gs / P.vol
        uc = P.cogs / P.vol
        cpi = A["cpi"][c]
        cidx = self.cost_index(A)
        sl = slice(12, o + 1)
        dgp = np.log(gp[:, sl] / gp[:, 0:o + 1 - 12])
        dcpi = np.log(cpi[:, sl] / cpi[:, 0:o + 1 - 12])
        duc = np.log(uc[:, sl] / uc[:, 0:o + 1 - 12])
        dci = np.log(cidx[:, sl] / cidx[:, 0:o + 1 - 12])
        lo, hi = self.cfg["drivers"]["price_beta_clip"]
        self.beta = np.zeros(P.C)
        self.price_drift = np.zeros(P.C)
        for ci in range(P.C):
            m = (c == ci)
            y, x = dgp[m].ravel(), dcpi[m].ravel()
            ok = np.isfinite(y) & np.isfinite(x) & (np.abs(y) < 1.5)
            X = np.c_[np.ones(ok.sum()), x[ok]]
            a, b = np.linalg.lstsq(X, y[ok], rcond=None)[0]
            self.beta[ci], self.price_drift[ci] = np.clip(b, lo, hi), np.clip(a, -0.05, 0.05)
        if not self.cfg["drivers"].get("use_trend_drift", False):
            self.price_drift[:] = 0.0      # no extrapolation of past real price trends
        self.cost_coef = np.zeros((P.K, 3))
        for ki in range(P.K):
            m = (k == ki)
            y, x1, x2 = duc[m].ravel(), dci[m].ravel(), dcpi[m].ravel()
            ok = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2) & (np.abs(y) < 1.5)
            X = np.c_[np.ones(ok.sum()), x1[ok], x2[ok]]
            self.cost_coef[ki] = np.linalg.lstsq(X, y[ok], rcond=None)[0]
        if not self.cfg["drivers"].get("use_trend_drift", False):
            self.cost_coef[:, 0] = 0.0     # no extrapolation of past cost trends (2023-25 was a margin recovery)
        # run-rates at the origin
        s3, s12 = slice(o - 2, o + 1), slice(o - 11, o + 1)
        self.o = o
        self.gp3 = np.nansum(P.gs[:, s3], 1) / np.nansum(P.vol[:, s3], 1)
        self.uc3 = np.nansum(P.cogs[:, s3], 1) / np.nansum(P.vol[:, s3], 1)
        self.cpi3 = np.nanmean(cpi[:, s3], 1)
        self.cidx3 = np.nanmean(cidx[:, s3], 1)
        self.tr12 = np.nansum(P.ts[:, s12], 1) / np.nansum(P.gs[:, s12], 1)
        self.ap12 = np.nansum(P.ap[:, s12], 1) / np.nansum(P.ns[:, s12], 1)
        # A&P phasing by category x month (last 24 months)
        w0 = max(0, o - 23)                      # up to 24 months of history
        s24 = slice(w0, o + 1)
        ph = np.ones((P.K, 12))
        for ki in range(P.K):
            m = k == ki
            ap_m, ns_m = np.nansum(P.ap[m, s24], 0), np.nansum(P.ns[m, s24], 0)
            r = ap_m / ns_m
            moy = np.arange(w0, o + 1) % 12
            ph[ki] = np.array([r[moy == mm].mean() for mm in range(12)]) / r.mean()
        self.ap_phase = ph
        return self

    def pnl(self, vol: np.ndarray, D: dict, months: np.ndarray) -> dict:
        """vol: series x months (forecast volume). Returns P&L arrays (local currency and EUR)."""
        P, o = self.P, self.o
        c, k = P.series["c"].values, P.series["k"].values
        cpi = D["cpi"][c][:, months]
        yrs = (months - (o - 1)) / 12
        gp = self.gp3[:, None] * (cpi / self.cpi3[:, None]) ** self.beta[c][:, None] * np.exp(self.price_drift[c][:, None] * yrs)
        gs = vol * gp
        ts = gs * self.tr12[:, None]
        ns = gs - ts
        cidx = self.cost_index(D)[:, months]
        a0, b1, b2 = self.cost_coef[k].T
        uc = self.uc3[:, None] * np.exp(b1[:, None] * np.log(cidx / self.cidx3[:, None])
                                        + b2[:, None] * np.log(cpi / self.cpi3[:, None]) + a0[:, None] * yrs)
        cogs = vol * uc
        ap = ns * self.ap12[:, None] * self.ap_phase[k][:, months % 12]
        fx = D["fx"][c][:, months]
        out = {"volume": vol, "gross_sales_lc": gs, "trade_spend_lc": ts, "net_sales_lc": ns, "cogs_lc": cogs, "ap_lc": ap}
        for key in list(out):
            if key.endswith("_lc"):
                out[key.replace("_lc", "_eur")] = np.nan_to_num(out[key]) / fx
        return out

    def overheads(self, o: int, D: dict, months: np.ndarray) -> np.ndarray:
        """Country x function x months: same month last year x trailing growth."""
        oh = self.P.oh
        g = np.nansum(oh[..., o - 11:o + 1], -1) / np.nansum(oh[..., o - 23:o - 11], -1)
        out = np.zeros(oh.shape[:2] + (len(months),))
        for j, t in enumerate(months):
            kk = int(np.ceil((t - o) / 12))
            out[..., j] = oh[..., t - 12 * kk] * g ** kk
        return out / D["fx"][:, None, months]


# ================================================================ scenario engine (from forecast_engine/scenario.py)
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
        self.meta = json.load(open(find("meta.json")))
        cfg = self.meta["cfg"]
        series = pd.read_parquet(find("series.parquet"))
        A = np.load(find("arrays.npz"))
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
        self.booster = lgb.Booster(model_file=str(find("volume_model.txt")))
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
    return FEATURES


# ================================================================ app
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
    return ScenarioKit(None)


@st.cache_data
def load_tables():
    t = {n: pd.read_parquet(find(f"{n}.parquet")) for n in
         ["group_pnl_monthly", "backtest_accuracy", "backtest_monthly", "backtest_gross_margin", "group_pnl_by_year",
          "group_views", "flags", "by_department", "patterns", "targets"]}
    t["build"] = pd.read_json(find("build_info.json"), typ="series")
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
