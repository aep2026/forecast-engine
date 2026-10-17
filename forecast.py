"""
forecast.py  -  STEP 5: outside-in forecast of the P&L.

DESIGN (one sentence for the CFO)
    Machine learning where the pattern is complex (VOLUME, one global LightGBM model across all
    SKU x country series); explicit finance formulas where the mechanism is known (PRICE follows
    CPI, COGS follows SKU recipe x commodities x FX, TRADE and A&P follow their run-rates, FX translates).

WHAT IT PRODUCES
    1. Backtest: forecast Jan-Sep 2026 using only data to Dec 2025 and the Dec 2025 consensus,
       compared with the actuals and with three benchmarks (last year, Holt-Winters, department budget).
    2. Rolling forecast: Oct 2026 - Dec 2027 from the Sep 2026 actuals and Sep 2026 consensus.
    3. Explanations: driver importance, a Net Sales bridge, and sensitivities for the scenario app.
    4. forecast_results_2026.xlsx (+ parquet files and an HTML report).

HOW TO RUN
    python forecast_engine/forecast.py         (uses forecast.yaml)
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

FEATURES = ["h", "moy", "south", "category", "country", "base_price_eur", "sidx", "cat_sidx", "ly_ratio",
            "trend_yoy", "mom3", "months_since_launch", "hist_len", "zero_share",
            "mkt_rel", "share_rel", "pl_delta", "conf_delta", "cpi_infl"]
CATEGORICAL = ["category", "country"]
FEATURE_GROUPS = {
    "Seasonality": ["moy", "sidx", "cat_sidx", "ly_ratio", "south"],
    "Momentum & trend": ["trend_yoy", "mom3"],
    "Market size & share": ["mkt_rel", "share_rel"],
    "Macro (CPI, confidence, private label)": ["cpi_infl", "conf_delta", "pl_delta"],
    "Lifecycle (launch, tail)": ["months_since_launch", "hist_len", "zero_share"],
    "Structure (country, category, price tier, horizon)": ["country", "category", "base_price_eur", "h"],
}


# =============================================================================== data panel
class Panel:
    """All data as arrays: series (SKU x country) x months, countries x months, category x country x months."""

    def __init__(self, base: Path, cfg: dict):
        self.cfg = cfg
        clean, raw = base / cfg["paths"]["clean_dir"], base / cfg["paths"]["raw_dir"]
        tl = cfg["timeline"]
        self.months = pd.date_range(tl["first_month"], tl["last_month"], freq="MS")
        self.T = len(self.months)
        self.tix = {d: i for i, d in enumerate(self.months)}

        sku = pd.read_excel(raw / "master_data.xlsx", sheet_name="sku_master")
        cm = pd.read_excel(raw / "master_data.xlsx", sheet_name="country_master")
        self.countries = list(cm["country"])
        self.cats = sorted(sku["category"].unique())
        self.C, self.K = len(self.countries), len(self.cats)
        cidx = {c: i for i, c in enumerate(self.countries)}
        kidx = {k: i for i, k in enumerate(self.cats)}

        fr = pd.read_parquet(clean / "fact_pnl_forecast_ready.parquet",
                             columns=["sku_code", "country", "date", "volume_units", "gross_sales_lc", "trade_spend_lc",
                                      "net_sales_lc", "cogs_lc", "ap_lc", "grid_status", "train_weight"])
        fr["date"] = pd.to_datetime(fr["date"])
        ser = fr[["sku_code", "country"]].drop_duplicates().sort_values(["sku_code", "country"]).reset_index(drop=True)
        ser = ser.merge(sku[["sku_code", "brand", "category", "base_price_eur", "launch_date", "delist_date"]
                            + [f"recipe_{j}" for j in cfg["drivers"]["recipe_inputs"]]], on="sku_code", how="left")
        ser["c"] = ser["country"].map(cidx)
        ser["k"] = ser["category"].map(kidx)
        ser["south"] = ser["country"].map(cm.set_index("country")["south"]).astype(int)
        for col in ["launch_date", "delist_date"]:
            d = pd.to_datetime(ser[col])
            ser[col + "_ix"] = ((d.dt.year - self.months[0].year) * 12 + d.dt.month - 1).astype(float)
        self.series = ser
        self.S = len(ser)
        sidx = {(s, c): i for i, (s, c) in enumerate(zip(ser.sku_code, ser.country))}
        i = np.array([sidx[(s, c)] for s, c in zip(fr.sku_code, fr.country)])
        t = fr["date"].map(self.tix).values
        self.extract = int(t.max())
        ok = fr["grid_status"].isin(["reported", "zero_sale"]).values
        for name, col in [("vol", "volume_units"), ("gs", "gross_sales_lc"), ("ts", "trade_spend_lc"),
                          ("ns", "net_sales_lc"), ("cogs", "cogs_lc"), ("ap", "ap_lc")]:
            a = np.full((self.S, self.T), np.nan)
            v = fr[col].values.astype(float)
            a[i[ok], t[ok]] = v[ok]
            setattr(self, name, a)
        self.weight = np.zeros((self.S, self.T))
        self.weight[i, t] = pd.to_numeric(fr["train_weight"], errors="coerce").fillna(1).values
        self.weight[np.isnan(self.vol)] = 0
        # all-NaN columns of a series where nothing was reported at all
        self.valid = ~np.isnan(self.vol) & ~np.isnan(self.ns)

        # external drivers (actuals)
        mac = pd.read_parquet(clean / "ext_macro_country.parquet")
        mac["date"] = pd.to_datetime(mac["date"])
        self.act = {}
        for name, col in [("fx", "fx_lc_per_eur"), ("cpi", "cpi_index"), ("conf", "consumer_confidence")]:
            self.act[name] = self._pivot_c(mac, col, cidx)
        mk = pd.read_parquet(clean / "ext_market_category_country.parquet")
        mk["date"] = pd.to_datetime(mk["date"])
        for name, col in [("mkt", "market_size_rsv_lc"), ("share", "our_value_share"), ("pl", "private_label_share")]:
            self.act[name] = self._pivot_kc(mk, col, cidx, kidx)
        gl = pd.read_parquet(clean / "ext_commodities_global.parquet")
        gl["date"] = pd.to_datetime(gl["date"])
        for col in ["eurusd"] + [f"{j}_index" for j in cfg["drivers"]["recipe_inputs"]]:
            a = np.full(self.T, np.nan)
            a[gl["date"].map(self.tix).values] = gl[col].values
            self.act[col] = a
        self.outlooks = pd.read_parquet(clean / "ext_outlooks.parquet")
        self.outlooks["date"] = pd.to_datetime(self.outlooks["date"])
        self.cidx, self.kidx = cidx, kidx
        oh = pd.read_parquet(clean / "fact_overheads.parquet")
        oh["date"] = pd.to_datetime(oh["date"])
        self.functions = sorted(oh["function"].unique())
        self.oh = np.full((self.C, len(self.functions), self.T), np.nan)
        fi = {f: j for j, f in enumerate(self.functions)}
        self.oh[oh["country"].map(cidx).values, oh["function"].map(fi).values, oh["date"].map(self.tix).values] = oh["overheads_lc"].values

    def _pivot_c(self, df, col, cidx):
        a = np.full((self.C, self.T), np.nan)
        a[df["country"].map(cidx).values, df["date"].map(self.tix).values] = df[col].values
        return a

    def _pivot_kc(self, df, col, cidx, kidx):
        a = np.full((self.K, self.C, self.T), np.nan)
        a[df["category"].map(kidx).values, df["country"].map(cidx).values, df["date"].map(self.tix).values] = df[col].values
        return a

    # ------------------------------------------------------------------ driver views
    def drivers(self, o: int, mode: str = "expost", vintage: str | None = None) -> dict:
        """Driver arrays as seen at origin o.
        expost  = actuals (training, or 'perfect foresight' test)
        vintage = actuals to o, then the published consensus outlook of that vintage
        naive   = actuals to o, then a naive consensus (trend / flat) built the same way"""
        D = {k: v.copy() for k, v in self.act.items()}
        if mode == "expost":
            return D
        fut = np.arange(o + 1, self.T)
        for k in D:
            D[k][..., o + 1:] = np.nan
        if mode == "vintage":
            ol = self.outlooks[self.outlooks.vintage == vintage]
            m = ol[ol.table == "market"]
            for name, col in [("mkt", "market_size_rsv_lc"), ("share", "our_value_share"), ("pl", "private_label_share")]:
                v = self._pivot_kc(m.dropna(subset=[col]), col, self.cidx, self.kidx)
                D[name][..., o + 1:] = v[..., o + 1:]
            m = ol[ol.table == "macro"]
            for name, col in [("fx", "fx_lc_per_eur"), ("cpi", "cpi_index"), ("conf", "consumer_confidence")]:
                v = self._pivot_c(m.dropna(subset=[col]), col, self.cidx)
                D[name][..., o + 1:] = v[..., o + 1:]
            m = ol[ol.table == "global"]
            for col in ["eurusd"] + [f"{j}_index" for j in self.cfg["drivers"]["recipe_inputs"]]:
                a = np.full(self.T, np.nan)
                a[m["date"].map(self.tix).values] = m[col].values
                D[col][o + 1:] = a[o + 1:]
        else:  # naive
            trend = {"mkt", "cpi", "fx"}
            for name in D:
                a = D[name]
                for t in fut:
                    k = int(np.ceil((t - o) / 12))
                    if name in trend:
                        g = np.nansum(a[..., o - 11:o + 1], -1) / np.nansum(a[..., o - 23:o - 11], -1)
                        a[..., t] = a[..., t - 12 * k] * g ** k
                    else:
                        a[..., t] = np.nanmean(a[..., o - 2:o + 1], -1)
        return D


# =============================================================================== features
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


# =============================================================================== volume model
class VolumeModel:
    def __init__(self, cfg):
        self.cfg = cfg["model"]

    def fit(self, df: pd.DataFrame):
        p = {"objective": "regression", "learning_rate": self.cfg["learning_rate"], "num_leaves": self.cfg["num_leaves"],
             "min_data_in_leaf": self.cfg["min_data_in_leaf"], "feature_fraction": self.cfg["feature_fraction"],
             "bagging_fraction": self.cfg["bagging_fraction"], "bagging_freq": self.cfg["bagging_freq"],
             "lambda_l2": self.cfg["lambda_l2"], "seed": self.cfg["seed"], "verbose": -1, "num_threads": 0}
        mono = self.cfg.get("monotone", {})
        if mono:   # economic sense by construction: e.g. a bigger market can never lower volume
            p["monotone_constraints"] = [int(mono.get(f, 0)) for f in FEATURES]
            p["monotone_constraints_method"] = "advanced"
        ds = lgb.Dataset(df[FEATURES], df["y"], weight=df["w"], categorical_feature=CATEGORICAL, free_raw_data=False)
        self.booster = lgb.train(p, ds, num_boost_round=self.cfg["num_boost_round"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.booster.predict(df[FEATURES]), 0, None) * df["level"].values

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.booster.predict(df[FEATURES], pred_contrib=True)
        return pd.DataFrame(c[:, :-1], columns=FEATURES)


# =============================================================================== P&L engine
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


# =============================================================================== benchmarks
class Benchmarks:
    def __init__(self, P: Panel):
        self.P = P

    def seasonal_naive(self, o: int, months: np.ndarray) -> np.ndarray:
        """Same month last year (two years back beyond 12 months)."""
        ns = self.P.ns
        out = np.zeros((self.P.S, len(months)))
        for j, t in enumerate(months):
            out[:, j] = ns[:, t - 12 if t - 12 <= o else t - 24]
        return np.nan_to_num(out)

    def holt_winters(self, o: int, months: np.ndarray) -> np.ndarray:
        """Damped-trend multiplicative Holt-Winters per brand x country, split to SKUs by last-12-month mix."""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        P = self.P
        ns = np.nan_to_num(P.ns[:, : o + 1])
        ser = P.series.assign(g=P.series["brand"] + "|" + P.series["country"])
        out = np.zeros((P.S, len(months)))
        H = int(months.max() - o)
        for g, idx in ser.groupby("g").groups.items():
            idx = np.array(list(idx))
            y = ns[idx].sum(0)
            start = np.argmax(y > 0)
            y = y[start:]
            gap = y <= 0
            if gap.any() and len(y) > 12:                  # fill gaps (e.g. Chile) with last year
                y = y.copy()
                for t in np.where(gap)[0]:
                    y[t] = y[t - 12] if t >= 12 else y[y > 0].mean()
            if len(y) >= 24 and (y > 0).all():
                try:
                    f = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="mul", seasonal_periods=12,
                                             initialization_method="estimated").fit(optimized=True).forecast(H)
                except Exception:
                    f = np.array([y[-12 + (h - 1) % 12] for h in range(1, H + 1)])
            elif len(y) >= 12:
                f = np.array([y[-12 + (h - 1) % 12] for h in range(1, H + 1)])
            else:
                f = np.repeat(y[-3:].mean() if len(y) else 0, H)
            f = np.clip(np.asarray(f), 0, None)
            mix = ns[idx, o - 11:o + 1].sum(1)
            mix = mix / mix.sum() if mix.sum() > 0 else np.repeat(1 / len(idx), len(idx))
            out[idx] = mix[:, None] * f[(months - o - 1)][None]
        return out

    def department_budget(self, subs: pd.DataFrame, year: int, months: np.ndarray) -> pd.DataFrame:
        """Traditional bottom-up budget: last year's Net Sales x (1 + submitted volume growth) x (1 + submitted price)."""
        P = self.P
        s = subs[(subs.budget_year == year) & subs.metric.isin(["Volume growth", "Net price/mix increase"])]
        g = s.pivot_table(index="country", columns="metric", values="submitted_value")
        c = P.series["c"].values
        ly = np.array([t - 12 for t in months])
        ns_ly = np.zeros((P.C, len(months)))
        np.add.at(ns_ly, c, np.nan_to_num(P.ns[:, ly]))
        fac = np.array([(1 + g.loc[cn, "Volume growth"]) * (1 + g.loc[cn, "Net price/mix increase"])
                        if cn in g.index else 1.0 for cn in P.countries])
        return ns_ly * fac[:, None]


# =============================================================================== orchestration
class ForecastRunner:
    def __init__(self, config_path="forecast.yaml"):
        self.cfg_path = Path(config_path).resolve()
        self.base = self.cfg_path.parent
        self.cfg = yaml.safe_load(open(self.cfg_path, encoding="utf-8"))
        self.out = self.base / self.cfg["paths"]["out_dir"]
        self.out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        self.P = Panel(self.base, self.cfg)
        self.F = Features(self.P)
        self.B = Benchmarks(self.P)
        print(f"  panel: {self.P.S:,} SKU x country series, {self.P.T} months ({time.time() - t0:.0f}s)")
        self.log = {}

    def _ix(self, d):
        return self.P.tix[pd.Timestamp(d)]

    # ------------------------------------------------------------------ one forecast
    def forecast(self, o: int, mode: str, vintage: str | None, H: int, model: VolumeModel | None = None):
        P = self.P
        if model is None:
            t0 = time.time()
            tr = self.F.training_set(cutoff=o)
            model = VolumeModel(self.cfg).fit(tr)
            print(f"    trained on {len(tr):,} rows ({time.time() - t0:.0f}s)")
        D = P.drivers(o, mode, vintage)
        X = self.F.build(o, D, H)
        vol_pred = model.predict(X)
        months = np.arange(o + 1, min(o + H, P.T - 1) + 1)
        vol = np.zeros((P.S, len(months)))
        vol[X["i"].values, X["t"].values - o - 1] = vol_pred
        eng = PnLEngine(P, self.cfg).estimate(o)
        pnl = eng.pnl(vol, D, months)
        pnl["overheads_eur"] = eng.overheads(o, D, months)
        return {"o": o, "months": months, "pnl": pnl, "X": X, "model": model, "engine": eng, "D": D}

    # ------------------------------------------------------------------ backtest
    def backtest(self):
        P, tl = self.P, self.cfg["timeline"]
        o = self._ix(tl["backtest_origin"])
        H = P.extract - o
        print(f"  BACKTEST: origin {P.months[o]:%b %Y}, forecasting {H} months with the {tl['backtest_vintage']} consensus")
        fc = self.forecast(o, "vintage", tl["backtest_vintage"], H)
        perfect = self.forecast(o, "expost", None, H, model=fc["model"])
        months = fc["months"]
        fx_act = P.act["fx"][P.series["c"].values][:, months]
        act = np.where(P.valid[:, months], P.ns[:, months], np.nan) / fx_act
        methods = {
            "AI forecast (LightGBM + drivers)": fc["pnl"]["net_sales_lc"] / fx_act,
            "AI forecast with perfect hindsight on drivers": perfect["pnl"]["net_sales_lc"] / fx_act,
            "Last year same month": self.B.seasonal_naive(o, months) / fx_act,
        }
        print("    fitting Holt-Winters benchmark ...")
        methods["Holt-Winters (brand x country)"] = self.B.holt_winters(o, months) / fx_act
        subs = pd.read_parquet(self.base / self.cfg["paths"]["clean_dir"] / "submissions_country.parquet")
        bud_c = self.B.department_budget(subs, P.months[o + 1].year, months) / P.act["fx"][:, months]
        # accuracy tables
        c = P.series["c"].values
        rows = []
        act0 = np.nan_to_num(act)
        a_cty = np.zeros((P.C, len(months)))
        np.add.at(a_cty, c, act0)
        for name, f in methods.items():
            f0 = np.nan_to_num(f)
            sku_w = np.nansum(np.abs(f0 - act0)) / act0.sum()
            f_cty = np.zeros_like(a_cty)
            np.add.at(f_cty, c, f0)
            rows.append({"method": name, "WAPE group monthly": np.abs(f0.sum(0) - act0.sum(0)).sum() / act0.sum(),
                         "WAPE country x month": np.abs(f_cty - a_cty).sum() / a_cty.sum(),
                         "WAPE SKU x country x month": sku_w, "Bias (total)": f0.sum() / act0.sum() - 1})
        rows.append({"method": "Department budget (bottom-up submissions)",
                     "WAPE group monthly": np.abs(bud_c.sum(0) - a_cty.sum(0)).sum() / a_cty.sum(),
                     "WAPE country x month": np.abs(bud_c - a_cty).sum() / a_cty.sum(),
                     "WAPE SKU x country x month": np.nan, "Bias (total)": bud_c.sum() / a_cty.sum() - 1})
        acc = pd.DataFrame(rows).set_index("method")
        # end-to-end: Net Sales translated at the FX the forecast assumed (price and FX move together in
        # hyperinflation markets, so this is how the forecast would have been reported)
        fx_fc = fc["D"]["fx"][c][:, months]
        e2e = np.nan_to_num(fc["pnl"]["net_sales_lc"] / fx_fc).sum(0)
        acc["WAPE group monthly, at forecast FX"] = np.nan
        acc.loc["AI forecast (LightGBM + drivers)", "WAPE group monthly, at forecast FX"] = np.abs(e2e - act0.sum(0)).sum() / act0.sum()
        # gross margin accuracy
        cogs_act = np.nansum(np.where(P.valid[:, months], P.cogs[:, months], np.nan) / fx_act, 0)
        gm_act = 1 - cogs_act / act0.sum(0)
        gm_fc = 1 - (np.nan_to_num(fc["pnl"]["cogs_lc"]) / fx_act).sum(0) / np.nan_to_num(fc["pnl"]["net_sales_lc"] / fx_act).sum(0)
        gm_pf = 1 - (np.nan_to_num(perfect["pnl"]["cogs_lc"]) / fx_act).sum(0) / np.nan_to_num(perfect["pnl"]["net_sales_lc"] / fx_act).sum(0)
        ly = np.array([t - 12 for t in months])
        gm_ly = 1 - np.nansum(P.cogs[:, ly] / P.act["fx"][c][:, ly], 0) / np.nansum(P.ns[:, ly] / P.act["fx"][c][:, ly], 0)
        gm = pd.DataFrame({"actual": gm_act, "AI forecast": gm_fc, "AI with perfect hindsight": gm_pf, "last year": gm_ly},
                          index=P.months[months])
        monthly = pd.DataFrame({name: np.nan_to_num(f).sum(0) for name, f in methods.items()}, index=P.months[months])
        monthly["Department budget (bottom-up submissions)"] = bud_c.sum(0)
        monthly.insert(0, "Actual", act0.sum(0))
        # error decomposition: model vs driver assumptions
        tot_a = act0.sum()
        e_fc = np.nan_to_num(methods["AI forecast (LightGBM + drivers)"]).sum() - tot_a
        e_pf = np.nan_to_num(methods["AI forecast with perfect hindsight on drivers"]).sum() - tot_a
        self.bt = {"fc": fc, "acc": acc, "monthly": monthly, "gm": gm, "err_model": e_pf, "err_assumptions": e_fc - e_pf,
                   "origin": o}
        return self.bt

    def cv_fold(self):
        """Second backtest origin, used only to size the uncertainty range."""
        P, tl = self.P, self.cfg["timeline"]
        o = self._ix(tl["cv_origin"])
        H = 12
        print(f"  CV FOLD: origin {P.months[o]:%b %Y}, naive consensus, {H} months")
        fc = self.forecast(o, "naive", None, H)
        months = fc["months"]
        fx_act = P.act["fx"][P.series["c"].values][:, months]
        a = np.nansum(np.where(P.valid[:, months], P.ns[:, months], np.nan) / fx_act, 0)
        f = np.nansum(fc["pnl"]["net_sales_lc"] / fx_act, 0)
        return pd.DataFrame({"h": np.arange(1, len(months) + 1), "pct_error": f / a - 1, "fold": f"{P.months[o]:%Y-%m}"})

    # ------------------------------------------------------------------ rolling forecast
    def rolling(self):
        P, tl = self.P, self.cfg["timeline"]
        o = P.extract
        H = P.T - 1 - o
        print(f"  ROLLING FORECAST: origin {P.months[o]:%b %Y}, {H} months with the {tl['rolling_vintage']} consensus")
        self.rf = self.forecast(o, "vintage", tl["rolling_vintage"], H)
        return self.rf

    # ------------------------------------------------------------------ explanations
    def explain(self):
        rf, P = self.rf, self.P
        X = rf["X"].sample(min(8000, len(rf["X"])), random_state=1)   # stable ranking, fast
        contrib = rf["model"].contributions(X).abs().mean()
        grp = {g: contrib[f].sum() for g, f in FEATURE_GROUPS.items()}
        imp = pd.Series(grp).sort_values(ascending=False)
        imp = (imp / imp.sum()).rename("share_of_explanation")
        # sensitivities on FY2027 (re-run the P&L with one driver changed)
        o, months = rf["o"], rf["months"]
        y27 = np.isin(P.months[months].year, [2027])
        base = self._fy(rf["pnl"], y27)
        sens = []

        def rerun(mutate, label, note):
            D = {k: v.copy() for k, v in rf["D"].items()}
            mutate(D)
            X2 = self.F.build(o, D, len(months))
            vol = np.zeros((P.S, len(months)))
            vol[X2["i"].values, X2["t"].values - o - 1] = rf["model"].predict(X2)
            p = rf["engine"].pnl(vol, D, months)
            p["overheads_eur"] = rf["engine"].overheads(o, D, months)
            new = self._fy(p, y27)
            sens.append({"driver change (all months after Sep 2026)": label,
                         "volume %": new["volume"] / base["volume"] - 1,
                         "net sales EUR m": (new["net_sales_eur"] - base["net_sales_eur"]) / 1e6,
                         "gross margin pts": (new["gm"] - base["gm"]) * 100,
                         "contribution EUR m": (new["contribution_eur"] - base["contribution_eur"]) / 1e6, "how": note})

        def mk(D):
            D["mkt"][..., o + 1:] *= 1.01
        rerun(mk, "Market size +1%", "volume model (market feature)")
        def shr(D):
            D["share"][..., o + 1:] *= 1 + 0.01 / 0.2
        rerun(shr, "Our value share +1 point (approx.)", "volume model (share feature)")
        def com(D):
            for j in self.cfg["drivers"]["recipe_inputs"]:
                D[f"{j}_index"][o + 1:] *= 1.10
        rerun(com, "Commodity prices +10%", "COGS engine (recipes x commodities)")
        def cpi(D):
            yrs = (np.arange(P.T) - o) / 12
            D["cpi"][:, o + 1:] *= (1.01 ** yrs[o + 1:])[None]
        rerun(cpi, "Inflation +1 point a year", "price pass-through + volume model + COGS engine")
        def cf(D):
            D["conf"][:, o + 1:] -= 5
        rerun(cf, "Consumer confidence -5", "volume model (confidence feature)")
        def pl(D):
            D["pl"][..., o + 1:] += 0.01
        rerun(pl, "Private label share +1 point", "volume model (private label feature)")
        def fx(D):
            D["fx"][:, o + 1:] *= 1.10
            D["eurusd"][o + 1:] *= 1.10
        rerun(fx, "EUR 10% stronger vs all currencies", "FX translation (+ cheaper USD inputs)")
        self.sens = pd.DataFrame(sens)
        self.importance = imp
        self.elasticities = pd.DataFrame({"country": P.countries, "price_pass_through_beta": rf["engine"].beta,
                                          "real_price_drift_pa": rf["engine"].price_drift})
        self.cost_coef = pd.DataFrame(rf["engine"].cost_coef, index=P.cats,
                                      columns=["annual_drift", "commodity_elasticity", "local_inflation_elasticity"])
        return imp, self.sens

    def _fy(self, pnl, mask):
        ns = pnl["net_sales_eur"][:, mask].sum()
        cogs = pnl["cogs_eur"][:, mask].sum()
        ap = pnl["ap_eur"][:, mask].sum()
        oh = pnl["overheads_eur"][..., mask].sum()
        return {"volume": pnl["volume"][:, mask].sum(), "net_sales_eur": ns, "gm": 1 - cogs / ns,
                "contribution_eur": ns - cogs - ap - oh}

    # ------------------------------------------------------------------ results tables
    def actuals_monthly(self) -> pd.DataFrame:
        P = self.P
        c = P.series["c"].values
        fx = P.act["fx"][c]
        m = slice(0, P.extract + 1)
        d = pd.DataFrame(index=P.months[m])
        for k in ["gs", "ts", "ns", "cogs", "ap"]:
            d[k] = np.nansum(getattr(P, k)[:, m] / fx[:, m], 0)
        d["volume"] = np.nansum(P.vol[:, m], 0)
        d["overheads"] = np.nansum(P.oh[..., m] / P.act["fx"][:, None, m], (0, 1))
        return d

    def forecast_monthly(self, f) -> pd.DataFrame:
        p, months = f["pnl"], f["months"]
        d = pd.DataFrame(index=self.P.months[months])
        for k, src in [("gs", "gross_sales_eur"), ("ts", "trade_spend_eur"), ("ns", "net_sales_eur"), ("cogs", "cogs_eur"),
                       ("ap", "ap_eur")]:
            d[k] = p[src].sum(0)
        d["volume"] = p["volume"].sum(0)
        d["overheads"] = p["overheads_eur"].sum((0, 1))
        return d

    def results(self):
        P = self.P
        act = self.actuals_monthly().assign(Type="Actual")
        fc = self.forecast_monthly(self.rf).assign(Type="Forecast")
        g = pd.concat([act, fc])
        g["gross_margin"] = g["ns"] - g["cogs"]
        g["contribution"] = g["gross_margin"] - g["ap"] - g["overheads"]
        g.index.name = "Date"
        # uncertainty range from backtest folds (group monthly % error by horizon)
        errs = pd.concat([self.cv, pd.DataFrame({"h": np.arange(1, len(self.bt["monthly"]) + 1),
                                                 "pct_error": (self.bt["monthly"].iloc[:, 1] / self.bt["monthly"]["Actual"] - 1).values,
                                                 "fold": "2025-12"})])
        rmspe = np.sqrt((errs["pct_error"] ** 2).mean())
        z = self.cfg["reporting"]["interval_z"]
        h = np.arange(1, len(fc) + 1)
        band = z * rmspe * np.sqrt(np.maximum(h, 1) / max(errs["h"].mean(), 1))
        g["ns_low"] = np.where(g.Type == "Forecast", g["ns"] * (1 - np.r_[np.zeros(len(act)), band]), np.nan)
        g["ns_high"] = np.where(g.Type == "Forecast", g["ns"] * (1 + np.r_[np.zeros(len(act)), band]), np.nan)
        self.errs, self.rmspe = errs, rmspe
        self.group_monthly = g
        y = g.groupby(g.index.year)
        fy = pd.DataFrame({"net_sales": y.ns.sum(), "gross_margin_pct": 1 - y.cogs.sum() / y.ns.sum(),
                           "ap": y.ap.sum(), "overheads": y.overheads.sum(), "contribution": y.contribution.sum()})
        fy["contribution_margin_pct"] = fy.contribution / fy.net_sales
        fy["ns_growth_reported"] = fy.net_sales.pct_change()
        fy["months_actual"] = g[g.Type == "Actual"].groupby(g[g.Type == "Actual"].index.year).size().reindex(fy.index).fillna(0).astype(int)
        self.fy = fy
        # FY2027 vs FY2026 bridge (volume, mix, price, FX) at SKU x country level
        self.bridge = self._bridge_26_27()
        return g, fy

    def country_monthly(self) -> pd.DataFrame:
        """Country x month P&L in local currency and EUR: actuals to the origin, then the rolling forecast."""
        P, rf = self.P, self.rf
        c = P.series["c"].values
        rows = []
        for t in range(P.T):
            if t <= rf["o"]:
                vals = {k: np.bincount(c, np.nan_to_num(getattr(P, a)[:, t]), P.C) for k, a in
                        [("volume", "vol"), ("gross_sales_lc", "gs"), ("trade_spend_lc", "ts"), ("net_sales_lc", "ns"),
                         ("cogs_lc", "cogs"), ("ap_lc", "ap")]}
                vals["overheads_lc"] = np.nansum(P.oh[:, :, t], 1)
                fx, typ = P.act["fx"][:, t], "Actual"
            else:
                j = t - rf["o"] - 1
                p = rf["pnl"]
                vals = {k: np.bincount(c, np.nan_to_num(p[k][:, j]), P.C) for k in
                        ["volume", "gross_sales_lc", "trade_spend_lc", "net_sales_lc", "cogs_lc", "ap_lc"]}
                fx, typ = rf["D"]["fx"][:, t], "Forecast"
                vals["overheads_lc"] = p["overheads_eur"][:, :, j].sum(1) * fx
            d = pd.DataFrame({"date": P.months[t], "country": P.countries, "type": typ, "fx_lc_per_eur": fx, **vals})
            rows.append(d)
        return pd.concat(rows, ignore_index=True)

    def _bridge_26_27(self):
        P, rf = self.P, self.rf
        c = P.series["c"].values
        o, months = rf["o"], rf["months"]
        full_lc = {k: np.full((P.S, P.T), np.nan) for k in ["vol", "ns"]}
        full_lc["vol"][:, : o + 1] = P.vol[:, : o + 1]
        full_lc["ns"][:, : o + 1] = P.ns[:, : o + 1]
        full_lc["vol"][:, months] = rf["pnl"]["volume"]
        full_lc["ns"][:, months] = rf["pnl"]["net_sales_lc"]
        fx = rf["D"]["fx"][c]
        y = P.months.year
        m26, m27 = np.where(y == 2026)[0], np.where(y == 2027)[0]
        V0 = np.nansum(full_lc["vol"][:, m26], 1)
        N0 = np.nansum(full_lc["ns"][:, m26] / fx[:, m26], 1)
        V1 = np.nansum(full_lc["vol"][:, m27], 1)
        C1 = np.nansum(full_lc["ns"][:, m27] / fx[:, m26], 1)       # 2027 at 2026 same-month rates
        E1 = np.nansum(full_lc["ns"][:, m27] / fx[:, m27], 1)
        cont = (V0 > 0) & (V1 > 0)
        P0 = np.where(cont, N0 / np.where(V0 > 0, V0, 1), 0)
        P1 = np.where(cont, C1 / np.where(V1 > 0, V1, 1), 0)
        p0avg = N0[cont].sum() / V0[cont].sum()
        hyper = np.isin(P.series["country"].values, self.cfg["drivers"]["hyperinflation_countries"])
        price = np.where(cont, (P1 - P0) * V1, 0)
        b = {"FY2026 (9 months actual + 3 forecast)": N0.sum(),
             "Volume": (V1[cont].sum() - V0[cont].sum()) * p0avg,
             "Mix": (V1[cont] * P0[cont]).sum() - V1[cont].sum() * p0avg,
             "Price (hyperinflation markets)": price[hyper].sum(),
             "Price (other markets)": price[~hyper].sum(),
             "Delisted / ending SKUs": (C1[~cont] - N0[~cont]).sum(),
             "FX translation": (E1 - C1).sum()}
        b["FY2027 forecast"] = E1.sum()
        s = pd.Series(b) / 1e6
        assert abs(s.iloc[0] + s.iloc[1:-1].sum() - s.iloc[-1]) < 0.01
        return s

    # ------------------------------------------------------------------ export
    def export(self):
        P, g = self.P, self.group_monthly
        brief = g[["Type", "ns"]].rename(columns={"ns": "Amount"}).reset_index()
        brief["Amount"] = brief["Amount"].round(0)
        rf = self.rf
        months = rf["months"]
        sku = []
        for j, t in enumerate(months):
            d = pd.DataFrame({"date": P.months[t], "sku_code": P.series.sku_code, "country": P.series.country,
                              "category": P.series.category, "brand": P.series.brand,
                              **{k: rf["pnl"][k][:, j] for k in ["volume", "gross_sales_eur", "trade_spend_eur",
                                                                  "net_sales_eur", "cogs_eur", "ap_eur"]}})
            sku.append(d[d.volume > 0])
        sku = pd.concat(sku, ignore_index=True)
        sku.to_parquet(self.out / "forecast_sku_country_month.parquet", index=False)
        cty = sku.assign(year=sku.date.dt.year).groupby(["country", "year"])[["net_sales_eur", "cogs_eur", "ap_eur"]].sum()
        cat = sku.assign(year=sku.date.dt.year).groupby(["category", "year"])[["net_sales_eur", "cogs_eur", "ap_eur"]].sum()
        card = pd.DataFrame([
            ["Volume model", "One global LightGBM model across all SKU x country series; target = volume / trailing 12-month level"],
            ["Features", ", ".join(FEATURES)],
            ["Training", f"Quarterly forecast origins from {self.cfg['timeline']['first_training_origin'][:7]}, horizons 1-{self.cfg['timeline']['max_horizon']} months, weighted by Net Sales value"],
            ["Drivers in training", "Actual values (ex-post). In the backtest and the rolling forecast: the consensus outlook published at the forecast date"],
            ["Price", "Gross price per unit = last 3 months x (CPI path / CPI)^beta per country, beta estimated from history"],
            ["Trade spend, A&P", "Trailing 12-month rates per series; A&P phased by category x month"],
            ["COGS", f"Unit cost = last 3 months x recipe-weighted commodity index (lag {self.cfg['drivers']['commodity_lag_months']} month) x FX, elasticities by category"],
            ["Overheads", "Same month last year x trailing 12-month growth, by country x function"],
            ["Bottom line", "Contribution after country overheads (no group costs, R&D, D&A yet) - not EBIT"],
            ["Uncertainty range", f"+/- {self.cfg['reporting']['interval_z']} x RMSPE of group monthly backtest errors ({self.rmspe:.1%}), widening with horizon"],
        ], columns=["item", "description"])
        with pd.ExcelWriter(self.out / "forecast_results_2026.xlsx") as w:
            brief.to_excel(w, sheet_name="forecast_results", index=False)
            g.round(0).to_excel(w, sheet_name="group_pnl_monthly")
            self.fy.round(4).to_excel(w, sheet_name="group_pnl_by_year")
            self.bt["acc"].round(4).to_excel(w, sheet_name="backtest_accuracy")
            self.bt["monthly"].round(0).to_excel(w, sheet_name="backtest_monthly")
            self.bt["gm"].round(4).to_excel(w, sheet_name="backtest_gross_margin")
            self.bridge.round(1).rename("EUR m").to_frame().to_excel(w, sheet_name="bridge_FY26_FY27")
            self.importance.round(3).to_frame().to_excel(w, sheet_name="driver_importance")
            self.sens.round(3).to_excel(w, sheet_name="sensitivities", index=False)
            self.elasticities.round(3).to_excel(w, sheet_name="price_elasticities", index=False)
            self.cost_coef.round(3).to_excel(w, sheet_name="cost_elasticities")
            (cty / 1e6).round(2).to_excel(w, sheet_name="by_country_year")
            (cat / 1e6).round(2).to_excel(w, sheet_name="by_category_year")
            card.to_excel(w, sheet_name="model_card", index=False)
        g.to_parquet(self.out / "group_pnl_monthly.parquet")
        self.country_monthly().to_parquet(self.out / "country_pnl_monthly.parquet", index=False)
        self.sens.to_parquet(self.out / "sensitivities.parquet", index=False)
        json.dump({"rmspe": self.rmspe, "origin": str(P.months[rf["o"]].date()),
                   "err_model_eur_m": self.bt["err_model"] / 1e6, "err_assumptions_eur_m": self.bt["err_assumptions"] / 1e6},
                  open(self.out / "forecast_meta.json", "w"), indent=1)
        rf["model"].booster.save_model(str(self.out / "volume_model.txt"))
        self.save_scenario_kit()

    def save_scenario_kit(self):
        """Everything the scenario app needs to re-run the SAME model live, in a few MB (no raw data)."""
        P, rf = self.P, self.rf
        e = rf["engine"]
        kit = self.out / "scenario_kit"
        kit.mkdir(exist_ok=True)
        cols = ["sku_code", "country", "brand", "category", "c", "k", "south", "base_price_eur", "launch_date_ix",
                "delist_date_ix"] + [f"recipe_{j}" for j in self.cfg["drivers"]["recipe_inputs"]]
        P.series[cols].to_parquet(kit / "series.parquet", index=False)
        arrays = {"vol": P.vol[:, : rf["o"] + 1].astype("float32"), "ns": P.ns[:, : rf["o"] + 1].astype("float32"),
                  "act_fx": P.act["fx"], "oh": P.oh}
        arrays.update({f"D_{k}": v for k, v in rf["D"].items()})
        arrays.update({f"E_{k}": getattr(e, k) for k in ["beta", "price_drift", "cost_coef", "gp3", "uc3", "cpi3", "cidx3",
                                                          "tr12", "ap12", "ap_phase"]})
        np.savez_compressed(kit / "arrays.npz", **arrays)
        cm = pd.read_excel(self.base / self.cfg["paths"]["raw_dir"] / "master_data.xlsx", sheet_name="country_master")
        meta = {"origin": int(rf["o"]), "months": [str(m.date()) for m in P.months], "countries": P.countries, "cats": P.cats,
                "functions": P.functions, "region": dict(zip(cm.country, cm.region)), "em": dict(zip(cm.country, cm.em.astype(bool))),
                "currency": dict(zip(cm.country, cm.currency)),
                "cfg": {"model": self.cfg["model"], "drivers": self.cfg["drivers"], "reporting": self.cfg["reporting"]}}
        json.dump(meta, open(kit / "meta.json", "w"), indent=1, default=str)
        rf["model"].booster.save_model(str(kit / "volume_model.txt"))

    def run(self):
        t0 = time.time()
        self.backtest()
        self.cv = self.cv_fold()
        self.rolling()
        print("  explaining drivers and sensitivities ...")
        self.explain()
        self.results()
        self.export()
        print(f"  done in {time.time() - t0:.0f}s")
        return self


def main():
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "forecast.yaml"
    print("Running forecast ...")
    r = ForecastRunner(cfg).run()
    pd.set_option("display.width", 200)
    print("\nBACKTEST ACCURACY (Jan-Sep 2026, Net Sales at actual FX; lower WAPE = better)")
    print(r.bt["acc"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-").to_string())
    print(f"\nForecast error split: model {r.bt['err_model'] / 1e6:+.0f}m | driver assumptions (consensus) {r.bt['err_assumptions'] / 1e6:+.0f}m")
    print("\nGROUP P&L BY YEAR (EUR m; 2026 = 9 months actual + 3 forecast)")
    f = r.fy.copy()
    for col in ["net_sales", "ap", "overheads", "contribution"]:
        f[col] = (f[col] / 1e6).round(0)
    for col in ["gross_margin_pct", "contribution_margin_pct", "ns_growth_reported"]:
        f[col] = (f[col] * 100).round(1)
    print(f.to_string())
    print("\nBRIDGE FY2026 -> FY2027 (EUR m)\n" + r.bridge.round(1).to_string())
    print("\nWHAT DRIVES THE VOLUME MODEL\n" + r.importance.map(lambda x: f"{x:.0%}").to_string())
    print("\nSENSITIVITIES (FY2027)\n" + r.sens.round(2).to_string(index=False))
    print(f"\nOutputs in {r.out}")


if __name__ == "__main__":
    main()
