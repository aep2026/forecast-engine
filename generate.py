"""
generate.py  -  STEP 1: build the synthetic company "Hygea Group" (fictional).

WHAT IT DOES (in finance language)
1. Builds the outside world first: FX, inflation, consumer confidence, oil and
   other commodities, a geopolitical risk index, and market size by category
   and country.
2. Gives the company a market share in each category and country, so our
   sales = market x share (the "outside-in" logic the forecast will reuse).
3. Splits sales across 30 brands and 300 SKUs, then builds the P&L to A&P:
   Gross Sales -> Trade Spend -> Net Sales -> COGS -> A&P.
   Overheads are built separately by country x function.
4. Creates 6 years of department budget submissions with planted biases
   (e.g. Sales sandbagging volume) for the AI challenger to find later.
5. Exports MESSY raw files (like a real ERP extract) plus a hidden answer key
   to check that the cleaning step works.

HOW TO RUN
    python forecast_engine/generate.py            (uses config.yaml)
All business assumptions live in config.yaml - change them there, not here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RECIPE_INPUTS = ["surfactant", "plastics", "fragrance", "aerosol"]   # "other" follows local inflation


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class SyntheticCompany:
    """Builds the whole synthetic company step by step. Call .run() then .save_all()."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg["company"]["seed"])
        tl = cfg["timeline"]
        self.months = pd.date_range(tl["start"], tl["truth_end"], freq="MS")
        self.T = len(self.months)
        self.last_actual = pd.Timestamp(tl["last_actual"])
        self.year = self.months.year.values
        self.moy = self.months.month.values - 1          # month of year, 0 = January
        self.tyears = np.arange(self.T) / 12.0            # years since start
        self.countries = pd.DataFrame(cfg["countries"])
        self.cats = cfg["categories"]
        self.C, self.K = len(self.countries), len(self.cats)
        self.planted = []                                   # log of planted data issues (answer key)

    # ------------------------------------------------------------------ helpers
    def _ix(self, date) -> int:
        """Position of a month in the timeline."""
        return int(np.clip(np.searchsorted(self.months, pd.Timestamp(date)), 0, self.T - 1))

    def _ar1(self, sd: float, phi: float = 0.7, shape: tuple = ()) -> np.ndarray:
        """Smooth random noise over time (each month remembers part of the last one)."""
        e = self.rng.normal(0, sd, (*shape, self.T))
        out = np.zeros_like(e)
        out[..., 0] = e[..., 0]
        for t in range(1, self.T):
            out[..., t] = phi * out[..., t - 1] + e[..., t]
        return out

    def _anchor_path(self, anchors: dict, noise_sd: float = 0.0) -> np.ndarray:
        """Draw a line through the anchor points in config.yaml, then add noise."""
        xs = [self._ix(d) for d in anchors]
        path = np.interp(np.arange(self.T), xs, list(anchors.values()))
        return path * np.exp(self._ar1(noise_sd)) if noise_sd else path

    # ------------------------------------------------------------ 1. macro world
    def build_macro(self):
        m, ctry, T = self.cfg["macro"], self.countries, self.T
        self.eurusd = self._anchor_path(m["eurusd_anchors"], 0.008)

        # Inflation -> CPI index (Jan 2021 = 100)
        cpi = np.zeros((self.C, T))
        for c, row in ctry.iterrows():
            over = m["inflation_overrides"].get(row["name"])
            annual = np.array([over[y] if over else row["inflation"] * m["inflation_profile"][y] for y in self.year])
            monthly = (1 + annual) ** (1 / 12) - 1 + self.rng.normal(0, 0.0008, T)
            cpi[c] = 100 * np.cumprod(1 + monthly)
        self.cpi = cpi

        # FX: local currency per EUR
        fx = np.zeros((self.C, T))
        for c, row in ctry.iterrows():
            if row["currency"] == "EUR":
                fx[c] = 1.0
            elif pd.notna(row.get("peg_usd")):
                fx[c] = row["peg_usd"] * self.eurusd                    # USD and USD-pegged currencies
            else:
                path = row["fx0"] * np.exp(row["fx_drift"] * self.tyears + self._ar1(row["fx_vol"] / 3, 0.9))
                jumps = row.get("fx_jumps")
                if isinstance(jumps, dict):
                    for d, mult in jumps.items():                         # e.g. Argentina devaluation Dec 2023
                        path[self._ix(d):] *= mult
                fx[c] = path
        self.fx = fx
        self.lc_per_usd = fx / self.eurusd

        # Consumer confidence: global path + local noise (+ extra US dip after tariffs)
        base_conf = self._anchor_path(m["confidence_anchors"])
        self.conf = base_conf + self._ar1(1.5, 0.8, (self.C,))
        us = ctry.index[ctry["name"] == "USA"][0]
        self.conf[us, self._ix("2026-04-01"):] -= 3

        # Retailer concentration and private label share (country level)
        self.retailer_conc = np.clip(ctry["retailer_conc"].values[:, None] + 0.005 * self.tyears, 0, 0.95)
        squeeze = np.clip((self.tyears - 1.0) / 2.0, 0, 1) * 0.03            # cost-of-living squeeze 2022-23
        self.pl_country = (ctry["pl_share"].values[:, None] + 0.005 * self.tyears + squeeze
                           + self._ar1(0.002, 0.9, (self.C,)))

    # -------------------------------------------------------- 2. commodities
    def build_commodities(self):
        cm = self.cfg["commodities"]
        self.brent = self._anchor_path(cm["brent_anchors"], 0.03)
        lag = lambda x, k: np.concatenate([np.repeat(x[0], k), x[:-k]])
        self.idx = {
            "surfactant": 100 * (lag(self.brent, 2) / self.brent[0]) ** 0.45 * np.exp(self._ar1(0.01)),
            "plastics":   100 * (lag(self.brent, 3) / self.brent[0]) ** 0.40 * np.exp(self._ar1(0.01)),
            "fragrance":  self._anchor_path(cm["fragrance_anchors"], 0.012),
            "aerosol":    self._anchor_path(cm["aerosol_anchors"], 0.015),
        }
        self.geo = self._anchor_path(cm["geo_risk_anchors"]) + self._ar1(1.5)
        self.freight = self._anchor_path(cm["freight_anchors"], 0.02)

    # ------------------------------------------------- 3. markets and share
    def build_markets(self):
        K, C, T, ctry = self.K, self.C, self.T, self.countries
        market = np.zeros((K, C, T))
        pl = np.zeros((K, C, T))
        for k, cat in enumerate(self.cats):
            for c, row in ctry.iterrows():
                g = cat["real_growth"] + (cat["em_extra_growth"] if row["em"] else 0)
                seas = np.array(cat["seasonality"]) / np.mean(cat["seasonality"])
                if row["south"] and cat["code"] == "SD":
                    seas = np.roll(seas, 6)                                   # flu season flips south
                covid = 1 + cat["covid_boost"] * np.exp(-np.arange(T) / 10)  # disinfection boom fading
                conf = (self.conf[c] / 100) ** cat["conf_elasticity"]
                price = (self.cpi[c] / 100) ** 0.9
                market[k, c] = (1 + g) ** self.tyears * seas[self.moy] * covid * conf * price \
                    * np.exp(self._ar1(0.015, 0.5))
                pl[k, c] = np.clip(self.pl_country[c] + cat["pl_offset"], 0.02, 0.6)
        base_share = self.rng.uniform(0.12, 0.32, (K, C))
        share = base_share[..., None] * np.exp(self._ar1(0.008, 0.95, (K, C))) * (1 - 0.9 * (pl - pl[..., :1]))
        self.market, self.pl, self.share_potential = market, pl, share

    # -------------------------------------------------- 4. brands and SKUs
    def build_portfolio(self):
        p, rng, C, T = self.cfg["portfolio"], self.rng, self.C, self.T
        brands, skus = [], []
        for k, cat in enumerate(self.cats):
            for r, name in enumerate(cat["brands"]):
                b = len(brands)
                brands.append({"brand_id": b, "brand": name, "brand_code": name[:3].upper(),
                               "category": cat["name"], "cat_code": cat["code"], "k": k, "rank": r})
                for s in range(p["skus_per_brand"]):
                    lo, hi = cat["price_range"]
                    rec = np.array([cat["recipe"][i] for i in RECIPE_INPUTS + ["other"]])
                    rec = np.clip(rec + rng.normal(0, 0.03, 5) * (rec > 0), 0, None)
                    rec = rec / rec.sum()
                    skus.append({"sku_code": f"{cat['code']}-{name[:3].upper()}-{s + 1:03d}",
                                 "sku_name": f"{name} {cat['forms'][s]}", "brand": name, "brand_id": b,
                                 "category": cat["name"], "k": k, "sku_rank": s,
                                 "base_price_eur": round(rng.uniform(lo, hi), 2),
                                 **{f"recipe_{i}": round(v, 3) for i, v in zip(RECIPE_INPUTS + ["other"], rec)}})
        self.brands, self.skus = pd.DataFrame(brands), pd.DataFrame(skus)
        B, S = len(self.brands), len(self.skus)

        # Which brands sell where (top brand of each category sells everywhere)
        pres_b = rng.random((B, C)) < np.array(p["brand_presence_by_rank"])[self.brands["rank"].values][:, None]
        pres_b[self.brands["rank"].values == 0] = True
        self.brand_present = pres_b

        # Which SKUs sell where, launch and delist dates
        sku_p = np.array(p["sku_presence_by_rank"])[self.skus["sku_rank"].values]
        pres_s = (rng.random((S, C)) < sku_p[:, None]) & pres_b[self.skus["brand_id"].values]
        launch = pd.Series(pd.NaT, index=self.skus.index)
        delist = pd.Series(pd.NaT, index=self.skus.index)
        for i, r in self.skus.iterrows():
            if r["sku_rank"] in p["launch_ranks"]:
                launch[i] = self.months[rng.integers(self._ix("2022-01-01"), self._ix("2025-12-01") + 1)]
            if r["sku_rank"] == p["delist_rank"] and rng.random() < 0.5:
                delist[i] = self.months[rng.integers(self._ix("2023-06-01"), self._ix("2026-06-01") + 1)]
        self.skus["launch_date"], self.skus["delist_date"] = launch, delist
        acq = p["acquired_brand"]
        self.skus["acquired_date"] = np.where(self.skus["brand"] == acq["name"], acq["date"], None)
        self.sku_present = pres_s

    # ----------------------------------------------------- 5. sales by SKU
    def build_sales(self):
        p, rng, ctry = self.cfg["portfolio"], self.rng, self.countries
        B, S, C, T, K = len(self.brands), len(self.skus), self.C, self.T, self.K
        bk = self.brands["k"].values

        # Brand weights inside each category x country (slowly drifting)
        logw = np.log(1 / (self.brands["rank"].values + 1) ** 0.8)[:, None, None] \
            + rng.normal(0, 0.3, (B, C, 1)) + self._ar1(0.02, 0.98, (B, C))
        w = np.exp(logw) * self.brand_present[..., None]
        for k in range(K):
            w[bk == k] /= w[bk == k].sum(axis=0, keepdims=True)
        acq = p["acquired_brand"]
        acq_b = self.brands.index[self.brands["brand"] == acq["name"]][0]
        w[acq_b, :, : self._ix(acq["date"])] = 0                              # not ours before the deal

        promo = rng.random((B, C, T)) < p["promo_prob"]
        tariff_vol = np.ones((C, T))
        for sh in self.cfg["shocks"]["tariffs"]:
            c = ctry.index[ctry["name"] == sh["country"]][0]
            tariff_vol[c, self._ix(sh["start"]):] *= 1 + sh["volume_impact"]
        # Consumer offtake (sell-out, retail value) = market x our share x brand weight
        brand_sellout = (self.share_potential * self.market)[bk] * w * (1 + p["promo_uplift"] * promo) * tariff_vol[None]
        # Sell-in Net Sales differs from sell-out: retailer margin & VAT (conversion ratio) and inventory timing
        # (retailers load stock the month before a promo and destock after it, plus random stock moves)
        lo, hi = p["rsv_to_ns_range"]
        rsv_to_ns = rng.uniform(lo, hi, (K, C))[bk][..., None] * np.exp(self._ar1(0.004, 0.98, (B, C)))
        nxt, prv = np.roll(promo, -1, axis=2), np.roll(promo, 1, axis=2)
        nxt[..., -1], prv[..., 0] = False, False
        inventory = 1 + self._ar1(p["sellin_noise_sd"], 0.4, (B, C)) + p["promo_loading"] * (nxt - 0.75 * prv)
        # Retailer stocking / destocking years (e.g. loading ahead of price increases, destocking after)
        yrs = np.unique(self.year)
        yr_stock = rng.normal(1, p["annual_stocking_sd"], (K, C, len(yrs)))
        inventory = inventory * yr_stock[:, :, np.searchsorted(yrs, self.year)][bk]
        brand_ns = brand_sellout * rsv_to_ns * inventory

        # SKU mix inside each brand x country
        sb = self.skus["brand_id"].values
        mix = np.exp(np.log(1 / (self.skus["sku_rank"].values + 1) ** 0.9)[:, None, None]
                     + rng.normal(0, 0.35, (S, C, 1)) + self._ar1(0.03, 0.97, (S, C)))
        active = np.repeat(self.sku_present[..., None], T, axis=2)
        mt = self.months.values
        for i, r in self.skus.iterrows():
            if pd.notna(r["launch_date"]):
                since = (np.arange(T) - self._ix(r["launch_date"]))
                active[i] &= since >= 0
                mix[i] *= np.clip((since + 1) / 6, 0.05, 1)[None, :]          # 6-month launch ramp-up
            if pd.notna(r["delist_date"]):
                active[i] &= mt < np.datetime64(r["delist_date"])
        tail = self.skus["sku_rank"].values >= p["intermittent_from_rank"]
        active[tail] &= rng.random((tail.sum(), C, T)) > p["intermittent_prob"]
        mix = mix * active
        denom = np.zeros((B, C, T))
        np.add.at(denom, sb, mix)
        sku_ns = np.where(denom[sb] > 0, mix / np.where(denom[sb] == 0, 1, denom[sb]), 0) * brand_ns[sb]

        # Calibrate: 2025 Net Sales in EUR per category x country hits config weights
        target = float(self.cfg["company"]["target_net_sales_2025_eur"])
        catmix = np.array([c["mix"] for c in self.cats])[:, None] * rng.lognormal(0, 0.15, (K, C))
        catmix /= catmix.sum(axis=0, keepdims=True)
        tgt = target * catmix * (ctry["weight"].values / ctry["weight"].sum())[None]
        y25 = self.year == 2025
        sk = self.skus["k"].values
        ns_eur25 = np.zeros((K, C))
        np.add.at(ns_eur25, sk, (sku_ns[..., y25] / self.fx[None, :, y25]).sum(axis=2))
        scale = tgt / ns_eur25
        self.sku_ns = sku_ns * scale[sk][..., None]
        self.market = self.market * scale[..., None]
        self.brand_ns = brand_ns * scale[bk][..., None]
        self.promo = promo
        self.acq_b = acq_b
        company_sellout = np.zeros((K, C, T))
        np.add.at(company_sellout, bk, brand_sellout * scale[bk][..., None])
        self.company_sellout = company_sellout                     # true consumer offtake of our brands (RSV)

        # External retail panel (Nielsen/Circana-style): measures sell-out, not our invoices.
        # It covers only part of the market (no discounters / e-commerce) and has its own sampling noise.
        lo, hi = p["panel_coverage_range"]
        cov = rng.uniform(lo, hi, C)
        self.panel_market = self.market * cov[None, :, None] * np.exp(self._ar1(0.01, 0.5, (K, C)))
        our_cov = np.clip(cov[None, :] * rng.uniform(0.96, 1.05, (K, C)), 0, 1)
        # Our sales drift into channels the panel does not see (e-commerce, discounters) at different speeds
        drift = rng.normal(p["offpanel_shift_pa"], p["offpanel_shift_sd"], (K, C))[..., None] * self.tyears
        panel_ours = company_sellout * our_cov[..., None] * np.exp(-drift + self._ar1(0.012, 0.5, (K, C))
                                                                   + rng.normal(0, 0.012, (K, C, T)))
        # Panel sample redesigns: a level shift in some category x country series
        for k_, c_ in zip(rng.integers(0, K, 8), rng.integers(0, C, 8)):
            panel_ours[k_, c_, rng.integers(12, T):] *= rng.choice([0.96, 1.04])
        self.our_share = panel_ours / self.panel_market
        self.panel_coverage_est = np.clip(cov + rng.normal(0, 0.02, C), 0, 1)

    # ------------------------------------------------------- 6. P&L lines
    def build_pnl(self):
        p, ctry, cats = self.cfg["portfolio"], self.countries, self.cats
        s, c, t = np.nonzero(self.sku_ns > 0)
        sk = self.skus.iloc[s]
        k = sk["k"].values
        ns = self.sku_ns[s, c, t]

        # Trade spend rate: category base + retailer power + private label + promos + noise
        tr = (np.array([x["trade_rate"] for x in cats])[k] + 0.08 * (self.retailer_conc[c, t] - 0.6)
              + 0.04 * (ctry["region"].values[c] == "North_Am")
              + 0.05 * (self.pl[k, c, t] - self.pl[k, c, 0])
              + p["promo_trade_rate_add"] * self.promo[sk["brand_id"].values, c, t]
              + self.rng.normal(0, 0.008, len(s)))
        gs = ns / (1 - tr)

        # List price per unit in local currency, follows local inflation (partly)
        price0 = sk["base_price_eur"].values * ctry["fx0"].values[c] * ctry["price_premium"].values[c]
        price = price0 * (self.cpi[c, t] / 100) ** 0.95
        volume = np.round(gs / price)

        # COGS per unit: SKU recipe x commodity prices (USD) x FX + local costs x inflation
        tr0 = np.array([x["trade_rate"] for x in cats])[k]
        cost0 = np.array([x["cogs_ratio"] for x in cats])[k] * price0 * (1 - tr0)
        usd_fx = self.lc_per_usd[c, t] / self.lc_per_usd[c, 0]
        mat = p["materials_share_of_cogs"]          # rest of COGS = conversion & labour, follows local inflation
        local = self.cpi[c, t] / 100
        # Hedging & supplier contracts: costs follow a 6-month average of commodity prices, lagged 1 month
        hedged = {i: pd.Series(self.idx[i]).rolling(6, min_periods=1).mean().shift(1).bfill().values for i in RECIPE_INPUTS}
        cost_idx = sum(sk[f"recipe_{i}"].values * hedged[i][t] / hedged[i][0] * usd_fx for i in RECIPE_INPUTS)
        cost_idx = mat * (cost_idx + sk["recipe_other"].values * local) + (1 - mat) * local
        freight = 1 + p["freight_share_of_cogs"] * (self.freight[t] / self.freight[0] - 1)
        tariff = np.zeros(len(s))
        for sh in self.cfg["shocks"]["tariffs"]:
            ci = ctry.index[ctry["name"] == sh["country"]][0]
            tariff += sh["cogs_uplift"] * ((c == ci) & (t >= self._ix(sh["start"])))
        savings = (1 - p["productivity_savings_pa"]) ** self.tyears[t]
        cogs = volume * cost0 * cost_idx * freight * (1 + tariff) * savings * self.rng.normal(1, 0.01, len(s))

        # A&P: % of Net Sales, Q4 heavier, launch bursts
        ap_rate = np.array([x["ap_rate"] for x in cats])[k]
        phasing = np.where(self.moy[t] >= 9, 1.25, np.where(self.moy[t] <= 1, 0.8, 0.98))
        launch_ix = pd.to_datetime(sk["launch_date"]).values
        burst = np.where(pd.notna(launch_ix) & ((self.months.values[t] - launch_ix) < np.timedelta64(180, "D")), 2.0, 1.0)
        ap = ns * ap_rate * phasing * burst * self.rng.lognormal(0, 0.1, len(s))

        keep = volume >= 1
        pnl = pd.DataFrame({
            "date": self.months[t], "region": ctry["region"].values[c], "country": ctry["name"].values[c],
            "currency": ctry["currency"].values[c], "category": sk["category"].values, "brand": sk["brand"].values,
            "sku_code": sk["sku_code"].values, "sku_name": sk["sku_name"].values,
            "volume_units": volume, "gross_sales_lc": gs, "trade_spend_lc": gs - ns, "net_sales_lc": ns,
            "cogs_lc": cogs, "ap_lc": ap,
        })[keep].reset_index(drop=True)
        money = ["gross_sales_lc", "trade_spend_lc", "net_sales_lc", "cogs_lc", "ap_lc"]
        pnl[money] = pnl[money].round(2)
        self.pnl_truth = pnl
        self.fx_df = pd.DataFrame(self.fx.T, index=self.months, columns=ctry["name"])

    # --------------------------------------------------------- 7. overheads
    def build_overheads(self):
        oh, ctry = self.cfg["overheads"], self.countries
        pnl = self.pnl_truth
        ns_c = pnl.groupby(["country", "date"])["net_sales_lc"].sum().unstack().reindex(ctry["name"]).fillna(0)
        base = ns_c.iloc[:, :12].mean(axis=1).values                       # 2021 average month
        rows = []
        for fn, sh in oh["functions"].items():
            growth = (self.cpi / 100) ** 0.8 * (1 + oh["headcount_growth_pa"]) ** self.tyears
            path = base[:, None] * sh * growth * np.exp(self._ar1(0.03, 0.3, (self.C,)))
            path = path * np.where(self.moy == 11, 1 + oh["december_bonus_accrual"], 1)[None]
            rs = oh["restructuring"]
            emea = (ctry["region"] == rs["region"]).values
            path[emea, self._ix(rs["date"]):] *= 1 - rs["saving"]
            for c in range(self.C):
                rows.append(pd.DataFrame({"date": self.months, "country": ctry["name"][c],
                                          "currency": ctry["currency"][c], "function": fn, "overheads_lc": path[c]}))
        ohd = pd.concat(rows, ignore_index=True)
        # Calibrate overheads to the % of Net Sales in config (2025)
        fxm = self.fx_df.stack().rename("fx").reset_index().rename(columns={"level_0": "date", "name": "country"})
        fxm.columns = ["date", "country", "fx"]
        tmp = ohd.merge(fxm, on=["date", "country"])
        oh25 = (tmp.loc[tmp.date.dt.year == 2025, "overheads_lc"] / tmp.loc[tmp.date.dt.year == 2025, "fx"]).sum()
        target = float(self.cfg["company"]["target_net_sales_2025_eur"]) * float(self.cfg["company"]["overheads_pct_of_ns_2025"])
        ohd["overheads_lc"] = (ohd["overheads_lc"] * target / oh25).round(2)
        self.overheads_truth = ohd

    # ----------------------------------------------- 8. department submissions
    def build_submissions(self):
        rng, sub, ctry = self.rng, self.cfg["submissions"], self.countries
        pnl = self.pnl_truth.assign(year=self.pnl_truth.date.dt.year)
        g = pnl.groupby(["country", "year"]).agg(vol=("volume_units", "sum"), gs=("gross_sales_lc", "sum"),
                                                 ns=("net_sales_lc", "sum"), cogs=("cogs_lc", "sum"),
                                                 ap=("ap_lc", "sum"))
        oh = self.overheads_truth.assign(year=self.overheads_truth.date.dt.year).groupby(["country", "year"])["overheads_lc"].sum()
        launches = self.skus.dropna(subset=["launch_date"])
        launch_ns = pnl[pnl.sku_code.isin(launches.sku_code)].merge(launches[["sku_code", "launch_date"]], on="sku_code")
        launch_ns = launch_ns[launch_ns.year == launch_ns.launch_date.dt.year].groupby(["country", "year"])["net_sales_lc"].sum()

        def truth(country, y, metric):
            cur, prev = g.loc[(country, y)], g.loc[(country, y - 1)]
            return {
                "volume_growth": cur.vol / prev.vol - 1,
                "price_increase": (cur.ns / cur.vol) / (prev.ns / prev.vol) - 1,
                "trade_rate": (cur.gs - cur.ns) / cur.gs,
                "ap_budget": cur.ap / 1e6,
                "promo_uplift": rng.normal(0.09, 0.02),
                "launch_net_sales": launch_ns.get((country, y), 0.0) / 1e6,
                "cogs_inflation": (cur.cogs / cur.vol) / (prev.cogs / prev.vol) - 1,
                "cogs_savings": rng.normal(self.cfg["portfolio"]["productivity_savings_pa"], 0.003),
                "overhead_growth": oh.loc[(country, y)] / oh.loc[(country, y - 1)] - 1,
            }[metric]

        metrics = {  # metric: (department, label, unit, noise sd)
            "volume_growth": ("Sales", "Volume growth", "%", 0.01),
            "price_increase": ("Sales", "Net price/mix increase", "%", 0.005),
            "trade_rate": ("Sales", "Trade spend rate", "% of Gross Sales", 0.004),
            "ap_budget": ("Marketing", "A&P budget", "LC m", 0.03),
            "promo_uplift": ("Marketing", "Promo volume uplift", "%", 0.01),
            "launch_net_sales": ("Marketing", "New launch Net Sales", "LC m", 0.10),
            "cogs_inflation": ("Supply Chain", "COGS per unit inflation", "%", 0.006),
            "cogs_savings": ("Supply Chain", "COGS savings delivered", "% of COGS", 0.002),
            "overhead_growth": ("Finance/HR", "Overhead growth", "%", 0.006),
        }
        rel = {"ap_budget", "launch_net_sales"}                               # noise is relative for LC amounts

        def bias_for(dept, metric, country):
            add, mult = 0.0, 1.0
            for b in sub["biases"]:
                if b["department"] == dept and b["metric"] == metric and (b["countries"] == "ALL" or country in b["countries"]):
                    add += b.get("bias", 0.0)
                    mult *= b.get("bias_mult", 1.0)
            return add, mult

        rows = []
        for y in sub["budget_years"]:
            for country in ctry["name"]:
                if (country, y - 1) not in g.index or (country, y) not in g.index:
                    continue
                for metric, (dept, label, unit, sd) in metrics.items():
                    actual = truth(country, y, metric)
                    if metric == "launch_net_sales" and actual == 0:
                        continue
                    add, mult = bias_for(dept, metric, country)
                    noise = actual * rng.normal(0, sd) if metric in rel else rng.normal(0, sd)
                    submitted = actual * mult + add + noise
                    biased = add != 0 or mult != 1
                    rows.append({"budget_year": y, "submitted_on": f"{y - 1}-10-15", "department": dept,
                                 "country": country, "metric": label, "unit": unit,
                                 "submitted_value": round(submitted, 4),
                                 "actual_value": round(actual, 4) if y <= 2025 else np.nan,
                                 "justification": self._justification(metric, biased, country, y),
                                 "submitted_by_role": f"{dept} Director {country}"})
        # Strategy: acquisition contribution (group level)
        pnl_eur = pnl.merge(self._fx_long(), on=["date", "country"])
        acq = self.cfg["portfolio"]["acquired_brand"]["name"]
        for y in (2023, 2024):
            actual = (pnl_eur.loc[(pnl_eur.brand == acq) & (pnl_eur.year == y), "net_sales_lc"]
                      / pnl_eur.loc[(pnl_eur.brand == acq) & (pnl_eur.year == y), "fx"]).sum() / 1e6
            rows.append({"budget_year": y, "submitted_on": f"{y - 1}-10-15", "department": "Strategy",
                         "country": "GROUP", "metric": "Acquisition Net Sales contribution", "unit": "EUR m",
                         "submitted_value": round(actual * 1.3, 1), "actual_value": round(actual, 1),
                         "justification": "Deal model assumes full distribution in all acquired markets within 6 months and cross-selling synergies.",
                         "submitted_by_role": "Head of Strategy & M&A"})
        self.submissions = pd.DataFrame(rows)
        self.planted.append("Department submissions carry biases defined in config.yaml -> submissions.biases "
                            "(e.g. Sales volume sandbagging in Spain/Italy/Brazil/Germany, USA over-optimism, "
                            "Supply Chain savings claimed at 2x delivered, Marketing launch Net Sales at 1.5x).")

    def _justification(self, metric, biased, country, year):
        rng = self.rng
        specific = {
            "volume_growth": [f"Distribution gains of {rng.integers(2, 9)} points at the top 3 retailers from Q{rng.integers(1, 4)}.",
                              "Volume follows category growth with flat share; promo calendar unchanged vs prior year."],
            "price_increase": [f"List price increase of {rng.integers(2, 7)}% agreed with key accounts effective March.",
                               "Pricing follows local CPI consensus with partial pass-through."],
            "trade_rate": ["Trade terms renegotiated with the top 2 retailers; promo depth capped at 30%."],
            "ap_budget": ["Media plan phased to Q4 peak; working media ratio 70%."],
            "promo_uplift": ["Uplift based on last year's post-event analysis for comparable promotions."],
            "launch_net_sales": [f"Launch listed in {rng.integers(60, 95)}% of weighted distribution by month 3."],
            "cogs_inflation": ["Based on supplier contracts for H1 and forward curve for oil-linked inputs."],
            "cogs_savings": ["Savings pipeline validated by project: reformulation, packaging lightweighting, freight tender."],
            "overhead_growth": ["Salary increases per local agreements; headcount flat."],
        }
        vague = {
            "volume_growth": ["Challenging retail environment; we remain cautious on volume.",
                              "Strong momentum expected across the portfolio.",
                              "Continued market share gains drive volume growth."],
            "price_increase": ["Pricing environment remains difficult; limited room for increases."],
            "trade_rate": ["Trade investment stable."],
            "promo_uplift": ["Promotions will be more effective this year thanks to new creative.",
                             "Strong uplift expected from the new promo mechanics."],
            "launch_net_sales": ["Breakthrough innovation with strong consumer test results.",
                                 "Expected to become a top-3 SKU in the category within the year."],
            "cogs_inflation": ["Commodity markets expected to stabilise."],
            "cogs_savings": ["Significant savings expected from ongoing procurement initiatives.",
                             "Savings from supplier consolidation plus price renegotiation plus lightweighting."],
            "overhead_growth": ["Overheads under control; open roles will be filled during the year."],
        }
        pool = vague.get(metric, specific[metric]) if (biased and rng.random() < 0.7) else specific[metric]
        return pool[rng.integers(len(pool))]

    def _fx_long(self):
        f = self.fx_df.copy()
        f.index.name = "date"
        f.columns.name = "country"
        return f.stack().rename("fx").reset_index()

    # ----------------------------------------------- 9. external driver tables
    def build_external(self):
        ctry, K, C = self.countries, self.K, self.C
        act = self.months <= self.last_actual
        rows = []
        for k, cat in enumerate(self.cats):
            for c in range(C):
                rows.append(pd.DataFrame({"date": self.months, "country": ctry["name"][c], "category": cat["name"],
                                          "market_size_rsv_lc": self.panel_market[k, c].round(0),
                                          "our_value_share": self.our_share[k, c].round(4),
                                          "private_label_share": self.pl[k, c].round(4),
                                          "panel_coverage_est": round(float(self.panel_coverage_est[c]), 2),
                                          "source": "Retail panel - sell-out, retail sales value (RSV)"}))
        self.ext_market = pd.concat(rows, ignore_index=True)
        rows = []
        for c in range(C):
            rows.append(pd.DataFrame({"date": self.months, "country": ctry["name"][c], "currency": ctry["currency"][c],
                                      "fx_lc_per_eur": self.fx[c].round(5), "cpi_index": self.cpi[c].round(2),
                                      "consumer_confidence": self.conf[c].round(1),
                                      "retailer_concentration": self.retailer_conc[c].round(3)}))
        self.ext_macro = pd.concat(rows, ignore_index=True)
        self.ext_global = pd.DataFrame({"date": self.months, "eurusd": self.eurusd.round(4), "brent_usd": self.brent.round(2),
                                        **{f"{i}_index": self.idx[i].round(2) for i in RECIPE_INPUTS},
                                        "freight_index": self.freight.round(1), "geopolitical_risk_index": self.geo.round(1)})
        # Consensus outlooks (what forecasters would have expected at each vintage)
        self.outlooks = {v: self._outlook(v) for v in ("2025-12-01", "2026-09-01")}
        # Only actual months are exported
        self.planted.append("Market data is an external retail panel: sell-out in retail value (RSV) with partial coverage "
                            "(no discounters/e-commerce) and sampling noise. Market x share does NOT equal our sell-in Net Sales: "
                            "retailer margin/VAT, promo stock loading and inventory moves sit in between. Share is a valid, "
                            "independent driver - but not a completeness check.")
        cut = lambda d: d[d.date <= self.last_actual].reset_index(drop=True)
        self.ext_market, self.ext_macro, self.ext_global = cut(self.ext_market), cut(self.ext_macro), cut(self.ext_global)

    def _outlook(self, vintage):
        """Naive consensus at a vintage date: trend for growing series, flat for levels. No knowledge of later shocks."""
        v = pd.Timestamp(vintage)
        out = {}
        for name, df, keys, trend_cols in [
            ("market", self.ext_market, ["country", "category"], ["market_size_rsv_lc"]),
            ("macro", self.ext_macro, ["country"], ["fx_lc_per_eur", "cpi_index"]),
            ("global", self.ext_global, [], []),
        ]:
            num = [col for col in df.columns if col not in keys + ["date", "currency", "source"]]
            hist = df[df.date <= v]
            future = pd.date_range(v + pd.offsets.MonthBegin(1), self.months[-1], freq="MS")
            groups = hist.groupby(keys) if keys else [((), hist)]
            parts = []
            for key, h in groups:
                h = h.set_index("date")
                f = pd.DataFrame(index=future)
                for col in num:
                    if col in trend_cols:
                        gr = h[col].iloc[-12:].sum() / h[col].iloc[-24:-12].sum() - 1
                        f[col] = [h[col].loc[d - pd.DateOffset(years=int(np.ceil(((d - v).days / 30.44) / 12)))]
                                  * (1 + gr) ** np.ceil(((d - v).days / 30.44) / 12) for d in future]
                    else:
                        f[col] = h[col].iloc[-3:].mean()
                f = f.round(4).reset_index().rename(columns={"index": "date"})
                for kk, kv in zip(keys, key if isinstance(key, tuple) else (key,)):
                    f[kk] = kv
                parts.append(f)
            o = pd.concat(parts, ignore_index=True)
            o.insert(0, "vintage", v.strftime("%Y-%m"))
            out[name] = o[["vintage", "date"] + keys + num]
        return out

    # ------------------------------------------------ 10. make it messy
    def make_messy(self):
        rng = self.rng
        df = self.pnl_truth[self.pnl_truth.date <= self.last_actual].copy()
        self.pnl_actual_truth = df.copy()

        # a) Acquired brand keeps its legacy coding
        acq = self.cfg["portfolio"]["acquired_brand"]["name"]
        m = df.brand == acq
        legacy = {code: f"ACQ-{i + 1:04d}" for i, code in enumerate(sorted(df.loc[m, "sku_code"].unique()))}
        df.loc[m, "sku_code"] = df.loc[m, "sku_code"].map(legacy)
        df.loc[m, "brand"] = acq.upper()
        df.loc[m, "category"] = "Dishwasher Care"
        self.code_map = [{"source_code": v, "master_code": k, "reason": "Acquired brand legacy code"} for k, v in legacy.items()]
        self.planted.append(f"Acquired brand {acq}: legacy SKU codes ACQ-xxxx, brand in capitals, category 'Dishwasher Care' "
                            f"(master: Water Softeners & Dishwasher). Sales only from {self.cfg['portfolio']['acquired_brand']['date']} (M&A, not organic).")

        # b) SKU re-coding in EMEA with two months of double reporting
        old = "FC-STA-001"
        new = old + "N"
        emea = (df.region == "EMEA") & (df.sku_code == old)
        df.loc[emea & (df.date >= "2024-07-01"), "sku_code"] = new
        dup1 = df[emea & (df.date == "2024-06-01")].assign(sku_code=new)
        dup2 = df[(df.region == "EMEA") & (df.sku_code == new) & (df.date == "2024-07-01")].assign(sku_code=old)
        df = pd.concat([df, dup1, dup2], ignore_index=True)
        self.code_map.append({"source_code": new, "master_code": old, "reason": "Re-coded in EMEA from Jul 2024"})
        self.planted.append(f"SKU {old} re-coded to {new} in EMEA from Jul 2024; Jun and Jul 2024 reported under BOTH codes "
                            f"({len(dup1) + len(dup2)} double-counted rows).")

        # c) Missing data (not 'no sales'): Chile, Mar-Apr 2025 system migration
        gap = (df.country == "Chile") & df.date.isin(pd.to_datetime(["2025-03-01", "2025-04-01"]))
        self.planted.append(f"Data gap: Chile Mar-Apr 2025 rows missing ({gap.sum()} rows) - ERP migration, NOT zero sales. "
                            "Tail SKUs also have genuine no-sale months (rows absent).")
        df = df[~gap]

        # d) Fat-finger errors: Net Sales entered x1000
        ff = rng.choice(df.index, 3, replace=False)
        df.loc[ff, "net_sales_lc"] = df.loc[ff, "net_sales_lc"] * 1000
        self.planted.append("Fat-finger: 3 rows with Net Sales x1000 (Net Sales > Gross Sales). Rows: "
                            + "; ".join(f"{r.country} {r.sku_code} {r.date:%Y-%m}" for r in df.loc[ff].itertuples()))

        # e) Exact duplicate rows
        dups = df.sample(frac=0.003, random_state=1)
        df = pd.concat([df, dups], ignore_index=True)
        self.planted.append(f"{len(dups)} exact duplicate rows.")

        # f) Country name variants
        variants = {"Germany": ["germany", "DE", "Deutschland "], "USA": ["United States", "US", "U.S.A."],
                    "United Kingdom": ["UK", "Great Britain"], "Brazil": ["Brasil"], "Spain": ["España", "spain "],
                    "Netherlands": ["The Netherlands", "Holland"], "Turkey": ["Türkiye"]}
        for name, vs in variants.items():
            idx = df.index[(df.country == name) & (rng.random(len(df)) < 0.10)]
            df.loc[idx, "country"] = rng.choice(vs, len(idx))
        self.planted.append("Country name variants (e.g. 'Deutschland ', 'US', 'Türkiye', 'Brasil') on about 10% of rows.")

        # g) Per-region layouts
        tabs = {}
        layouts = {
            "EMEA": ["Date", "Country", "Curr", "Category", "Brand", "SKU", "SKU Description", "Volume (units)",
                     "Gross Sales", "Trade Spend", "Net Sales", "COGS", "A&P"],
            "Latam": ["Fecha", "Pais", "Moneda", "Categoria", "Marca", "Codigo SKU", "Descripcion", "Volumen",
                      "Ventas Brutas", "Inversion Comercial", "Ventas Netas", "Costo de Ventas", "Publicidad y Promocion"],
            "North_Am": ["Month", "Market", "CCY", "Category", "Brand Name", "Item #", "Item Description", "Units",
                         "Gross Revenue", "Trade Promotion", "Net Revenue", "Cost of Goods Sold", "Advertising & Promotion"],
            "APAC": ["Period", "Country/Market", "Currency", "Cat.", "Brand", "SKU Code", "Desc", "Qty", "GS", "TS",
                     "NS", "COGS", "A&P"],
        }
        std = ["date", "country", "currency", "category", "brand", "sku_code", "sku_name", "volume_units",
               "gross_sales_lc", "trade_spend_lc", "net_sales_lc", "cogs_lc", "ap_lc"]
        money = std[8:]
        for region, cols in layouts.items():
            t = df[df.region == region][std].sample(frac=1, random_state=7).reset_index(drop=True).astype(object)
            n = len(t)
            d = pd.to_datetime(t["date"])
            u = rng.random(n)
            if region == "EMEA":
                dates = d.dt.strftime("%d/%m/%Y").astype(object)
                dates[u < 0.05] = (d[u < 0.05] - pd.Timestamp("1899-12-30")).dt.days.values   # Excel serial numbers
                dates[(u >= 0.05) & (u < 0.08)] = d[(u >= 0.05) & (u < 0.08)].dt.strftime("%Y-%m-%d")
            elif region == "Latam":
                dates = d.dt.strftime("%b-%y").astype(object)
                dates[u < 0.15] = d[u < 0.15].dt.strftime("%d/%m/%Y")
            elif region == "North_Am":
                dates = d.dt.strftime("%m-%d-%y").astype(object)
                dates[u < 0.2] = d[u < 0.2]
                t["trade_spend_lc"] = -t["trade_spend_lc"].astype(float)                    # sign convention
            else:
                dates = d.astype(object)
                india = t["country"] == "India"
                fy = d.dt.year + (d.dt.month >= 4)                                           # FY25 = Apr24-Mar25
                per = (d.dt.month - 4) % 12 + 1
                dates[india] = "FY" + (fy[india] % 100).astype(str).str.zfill(2) + " P" + per[india].astype(str).str.zfill(2)
            t["date"] = dates

            # Numbers stored as text with symbols, and error strings
            for col in money + ["volume_units"]:
                vals = t[col].astype(float)
                t[col] = t[col].astype(object)
                txt = rng.random(n) < 0.04
                if region == "EMEA":
                    sym = np.where(t["currency"] == "GBP", "£", np.where(t["currency"] == "EUR", "€", ""))
                    eu = vals.map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                    t.loc[txt, col] = (pd.Series(sym, index=t.index) + eu)[txt]
                elif region == "North_Am":
                    t.loc[txt, col] = vals[txt].map(lambda x: f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}")
                elif region == "Latam":
                    t.loc[txt, col] = vals[txt].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                else:
                    t.loc[txt, col] = vals[txt].map(lambda x: f" {x:,.0f} ")
                err = rng.random(n) < 0.006
                t.loc[err, col] = rng.choice(["ERROR", "n/a", "-", "#N/A", None], err.sum())
            t.columns = cols
            tabs[region] = t
        self.planted.append("Four regional tabs with different column names and languages (Latam in Spanish).")
        self.planted.append("Dates: EMEA DD/MM/YYYY + Excel serials + ISO; Latam 'Jan-24' + DD/MM/YYYY; "
                            "North_Am MM-DD-YY + real dates; APAC real dates, but India uses fiscal periods 'FY25 P01' (= Apr 2024).")
        self.planted.append("About 4% of numeric cells stored as text with currency symbols or local separators "
                            "(EMEA '€1.234,50', North_Am '$1,234.50', Latam '1.234,50', APAC ' 1,234 ').")
        self.planted.append("About 0.6% of numeric cells contain 'ERROR', 'n/a', '-', '#N/A' or blank.")
        self.planted.append("North_Am reports Trade Spend as NEGATIVE numbers (other regions positive).")
        self.messy_tabs = tabs

        # Overheads: title rows above the header, inconsistent function names, dates as text
        ohd = self.overheads_truth[self.overheads_truth.date <= self.last_actual].copy()
        fn_var = {"HR": ["HR", "Human Resources", "People"], "IT": ["IT", "Information Technology"],
                  "General Management": ["General Management", "Gen. Mgmt"]}
        for fn, vs in fn_var.items():
            mm = ohd.function == fn
            ohd.loc[mm, "function"] = rng.choice(vs, mm.sum())
        ohd["date"] = ohd["date"].dt.strftime("%Y-%m")
        self.messy_overheads = ohd.rename(columns={"date": "Period", "country": "Country", "currency": "Currency",
                                                   "function": "Cost Centre Function", "overheads_lc": "Amount (LC)"})
        self.planted.append("Overheads file: 2 title rows above the header; function names vary ('HR'/'Human Resources'/'People').")

    # ------------------------------------------------------------- run / save
    def run(self):
        steps = [("Macro world", self.build_macro), ("Commodities", self.build_commodities),
                 ("Markets & share", self.build_markets), ("Brands & SKUs", self.build_portfolio),
                 ("Sales by SKU", self.build_sales), ("P&L lines", self.build_pnl),
                 ("Overheads", self.build_overheads), ("Department submissions", self.build_submissions),
                 ("External drivers & outlooks", self.build_external), ("Messy raw extracts", self.make_messy)]
        for i, (label, fn) in enumerate(steps, 1):
            print(f"  [{i:>2}/{len(steps)}] {label} ...", flush=True)
            fn()
        return self

    def save_all(self, base: str | Path = "."):
        base = Path(base)
        raw = base / self.cfg["output"]["raw_dir"]
        key = base / self.cfg["output"]["answer_key_dir"]
        raw.mkdir(parents=True, exist_ok=True)
        key.mkdir(parents=True, exist_ok=True)
        print("  Writing files (the P&L extract is large - this can take 2-4 minutes) ...", flush=True)

        with pd.ExcelWriter(raw / "pnl_actuals_raw.xlsx") as w:
            for region, t in self.messy_tabs.items():
                t.to_excel(w, sheet_name=region, index=False)

        with pd.ExcelWriter(raw / "overheads_raw.xlsx", engine="openpyxl") as w:
            self.messy_overheads.to_excel(w, sheet_name="Overheads", index=False, startrow=3)
            ws = w.sheets["Overheads"]
            ws["A1"] = "Group Controlling - Overheads by country and cost centre function (local currency)"
            ws["A2"] = "Extract date: 2026-10-05 | Source: ERP cost centre report | Unaudited"

        d = lambda x: x.assign(date=x.date.dt.strftime("%Y-%m-%d")) if "date" in x else x
        with pd.ExcelWriter(raw / "external_drivers.xlsx") as w:
            d(self.ext_market).to_excel(w, sheet_name="market_category_country", index=False)
            d(self.ext_macro).to_excel(w, sheet_name="macro_country", index=False)
            d(self.ext_global).to_excel(w, sheet_name="commodities_global", index=False)
            for v, parts in self.outlooks.items():
                for name, o in parts.items():
                    d(o).to_excel(w, sheet_name=f"outlook_{v[:7]}_{name}", index=False)

        sku_master = self.skus.drop(columns=["k", "brand_id"]).copy()
        for col in ["launch_date", "delist_date"]:
            sku_master[col] = pd.to_datetime(sku_master[col]).dt.strftime("%Y-%m-%d")
        with pd.ExcelWriter(raw / "master_data.xlsx") as w:
            sku_master.to_excel(w, sheet_name="sku_master", index=False)
            self.brands[["brand", "brand_code", "category", "cat_code"]].assign(
                acquired_date=lambda x: np.where(x.brand == self.cfg["portfolio"]["acquired_brand"]["name"],
                                                 self.cfg["portfolio"]["acquired_brand"]["date"], "")
            ).to_excel(w, sheet_name="brand_master", index=False)
            self.countries[["name", "iso", "region", "currency", "em", "south"]].assign(
                fiscal_year_start_month=self.countries.get("fiscal_year_start_month", pd.Series(1)).fillna(1).astype(int)
            ).rename(columns={"name": "country"}).to_excel(w, sheet_name="country_master", index=False)
            pd.DataFrame(self.code_map).to_excel(w, sheet_name="sku_code_map", index=False)
            pd.DataFrame(self.cfg["strategic_targets"]).to_excel(w, sheet_name="strategic_targets", index=False)

        self.submissions.to_excel(raw / "department_submissions.xlsx", sheet_name="submissions", index=False)

        # Answer key: clean truth + control totals + list of planted issues
        self.pnl_actual_truth.to_csv(key / "pnl_truth_clean.csv.gz", index=False)
        ctl = self.pnl_actual_truth.merge(self._fx_long(), on=["date", "country"])
        ctl["year"] = ctl.date.dt.year
        for col in ["gross_sales", "trade_spend", "net_sales", "cogs", "ap"]:
            ctl[f"{col}_eur"] = ctl[f"{col}_lc"] / ctl["fx"]
        ohc = self.overheads_truth[self.overheads_truth.date <= self.last_actual].merge(self._fx_long(), on=["date", "country"])
        ohc = ohc.assign(year=ohc.date.dt.year, overheads_eur=ohc.overheads_lc / ohc.fx).groupby("year")["overheads_eur"].sum()
        tot = ctl.groupby("year")[[c for c in ctl.columns if c.endswith("_eur")]].sum().join(ohc)
        tot["gross_margin_pct"] = 1 - tot.cogs_eur / tot.net_sales_eur
        tot["ebit_eur"] = tot.net_sales_eur - tot.cogs_eur - tot.ap_eur - tot.overheads_eur
        tot["ebit_margin_pct"] = tot.ebit_eur / tot.net_sales_eur
        tot.round(4).to_csv(key / "control_totals_by_year.csv")
        (key / "planted_issues.md").write_text(
            "# Planted data issues (answer key - do not feed to the pipeline)\n\n"
            + "\n".join(f"{i}. {p}" for i, p in enumerate(self.planted, 1)) + "\n", encoding="utf-8")
        self.control_totals = tot
        return raw, key


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(cfg_path)
    print(f"Building {cfg['company']['name']} ...")
    co = SyntheticCompany(cfg).run()
    raw, key = co.save_all(Path(cfg_path).resolve().parent)
    t = co.control_totals
    print("\nControl totals (EUR m) - the clean pipeline must reproduce these:")
    show = t[["net_sales_eur", "gross_margin_pct", "ap_eur", "overheads_eur", "ebit_eur", "ebit_margin_pct"]].copy()
    for col in ["net_sales_eur", "ap_eur", "overheads_eur", "ebit_eur"]:
        show[col] = (show[col] / 1e6).round(0)
    for col in ["gross_margin_pct", "ebit_margin_pct"]:
        show[col] = (show[col] * 100).round(1).astype(str) + "%"
    print(show.to_string())
    n_rows = sum(len(x) for x in co.messy_tabs.values())
    n_combos = co.pnl_actual_truth.groupby(["sku_code", "country"]).ngroups
    print(f"\nRaw P&L rows: {n_rows:,} | active SKU x country combinations: {n_combos:,}")
    print(f"Files written to: {raw}\nAnswer key in:    {key}")


if __name__ == "__main__":
    main()
