"""
clean.py  -  STEP 3: reusable, config-driven cleaning pipeline.

Implements the revised (v2) cleaning logic agreed after the controller review.
Every month: drop the new raw files in data/raw, run this script, read the run report.

DESIGN PRINCIPLES (from the cleaning logic)
- Config holds the facts (column names, aliases, formats, thresholds); code holds the rules.
- Unknown means stop: a new country spelling, SKU code or column name never passes silently.
- Raw values are never overwritten: every fix has a reason code and a link to its source row.
- Two layers: CLEAN ACTUALS (certain fixes only) and FORECAST-READY (adds flagged estimates).
- Three severities: BLOCK (run stops), HOLD (publish, marked provisional), FLAG (information).

HOW TO RUN
    python forecast_engine/clean.py               (uses cleaning.yaml)
"""
from __future__ import annotations

import datetime as dt
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MONEY = ["gross_sales_lc", "trade_spend_lc", "net_sales_lc", "cogs_lc", "ap_lc"]


class PipelineBlocked(Exception):
    """Raised when a BLOCK-severity check fails: nothing is published."""


class CleaningPipeline:
    def __init__(self, config_path: str | Path = "cleaning.yaml", base: str | Path | None = None):
        self.cfg_path = Path(config_path).resolve()
        self.base = Path(base) if base else self.cfg_path.parent
        with open(self.cfg_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.th = {k: (v if isinstance(v, (list, dict)) else float(v)) for k, v in self.cfg["thresholds"].items()}
        self.raw = self.base / self.cfg["paths"]["raw_dir"]
        self.out = self.base / self.cfg["paths"]["clean_dir"]
        self.issues: list[dict] = []         # issue log (severity, step, message, count)
        self.row_bridge: list[dict] = []     # rows in / out per step
        self.value_bridge: list[dict] = []   # Net Sales EUR movements per step
        self.quarantine: list[pd.DataFrame] = []
        self.approval_queue: list[dict] = []

    # ================================================================ helpers
    def log(self, severity: str, step: str, message: str, count: int | float | None = None):
        self.issues.append({"severity": severity, "step": step, "message": message, "count": count})
        print(f"    [{severity}] {message}" + (f" ({count:,})" if isinstance(count, (int, np.integer)) else ""))
        if severity == "BLOCK":
            raise PipelineBlocked(f"{step}: {message}")

    def bridge(self, step: str, df: pd.DataFrame, note: str = ""):
        self.row_bridge.append({"step": step, "rows": len(df), "note": note})

    @staticmethod
    def norm_text(s: pd.Series) -> pd.Series:
        """Unicode NFC, non-breaking spaces -> spaces, trim, collapse repeated spaces."""
        def f(v):
            if not isinstance(v, str):
                return v
            v = unicodedata.normalize("NFC", v).replace("\xa0", " ")
            return re.sub(r"\s+", " ", v).strip()
        return s.map(f)

    @staticmethod
    def to_storable(df: pd.DataFrame) -> pd.DataFrame:
        """Audit columns can mix text, numbers and dates - store them as text so files are readable everywhere."""
        df = df.copy()
        for c in df.columns:
            if df[c].dtype == object:
                kinds = df[c].dropna().map(type).nunique()
                if kinds > 1 or (kinds == 1 and not isinstance(df[c].dropna().iloc[0], str)):
                    df[c] = df[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        return df

    def _fx(self, df: pd.DataFrame) -> pd.Series:
        """Monthly FX (local currency per EUR) for each row."""
        return df.merge(self.fx, on=["date", "country"], how="left")["fx_lc_per_eur"].values

    # ======================================================== reference data
    def load_reference(self):
        m = self.raw / self.cfg["files"]["master"]
        self.sku_master = pd.read_excel(m, sheet_name="sku_master")
        self.country_master = pd.read_excel(m, sheet_name="country_master")
        self.code_map = pd.read_excel(m, sheet_name="sku_code_map")
        ext = self.raw / self.cfg["files"]["external"]
        self.fx = pd.read_excel(ext, sheet_name="macro_country", usecols=["date", "country", "fx_lc_per_eur", "cpi_index"])
        self.fx["date"] = pd.to_datetime(self.fx["date"])
        self.alias = {k.lower(): v for k, v in self.cfg["country_alias"].items()}
        for c in self.country_master["country"]:
            self.alias[c.lower()] = c

    # ============================================================ step 1
    def step01_ingest(self) -> pd.DataFrame:
        """Read every cell as-is (no type guessing) and record lineage."""
        path = self.raw / self.cfg["files"]["pnl"]
        sheets = pd.read_excel(path, sheet_name=None, dtype=object)
        frames = []
        for tab, df in sheets.items():
            if tab not in self.cfg["tabs"]:
                self.log("BLOCK", "01 ingest", f"Unexpected sheet '{tab}' - add it to cleaning.yaml -> tabs")
            df = df.copy()
            df["source_file"], df["sheet"] = path.name, tab
            df["excel_row"] = np.arange(len(df)) + 2           # header is row 1
            frames.append(df)
            self.bridge(f"01 read {tab}", df)
        self.sheets = frames
        self.rows_read = sum(len(f) for f in frames)
        print(f"    read {sum(len(f) for f in frames):,} rows from {len(frames)} tabs")
        return frames

    # ============================================================ step 2
    def step02_schema(self, frames) -> pd.DataFrame:
        """Rename columns to the 13 standard names; any missing or extra column blocks the run."""
        std = self.cfg["standard_columns"]
        out = []
        for df in frames:
            tab = df["sheet"].iloc[0]
            cmap = self.cfg["tabs"][tab]["columns"]
            raw_cols = [c for c in df.columns if c not in ("source_file", "sheet", "excel_row")]
            missing, extra = set(cmap) - set(raw_cols), set(raw_cols) - set(cmap)
            if missing or extra:
                self.log("BLOCK", "02 schema", f"{tab}: missing {sorted(missing)}, unexpected {sorted(extra)}")
            df = df.rename(columns=cmap)
            df["region"] = self.cfg["tabs"][tab]["region"]
            out.append(df[std + ["region", "source_file", "sheet", "excel_row"]])
        df = pd.concat(out, ignore_index=True)
        # keep untouched raw copies for audit
        for c in ["date", "country", "sku_code", "brand", "category"]:
            df[f"raw_{c}"] = df[c]
        self.bridge("02 schema stacked", df)
        return df

    # ============================================================ step 3
    def step03_text(self, df):
        for c in ["country", "currency", "category", "brand", "sku_code", "sku_name", "date"]:
            df[c] = self.norm_text(df[c])
        return df

    # ============================================================ step 4
    def step04_countries(self, df):
        """Map aliases to master names; check region and currency against master.
        Runs BEFORE dates because India's fiscal calendar comes from the country master."""
        key = df["country"].astype(str).str.lower()
        df["country"] = key.map(self.alias)
        unknown = df.loc[df["country"].isna(), "raw_country"].unique()
        if len(unknown):
            self.log("BLOCK", "04 countries", f"Unknown country names {list(unknown)[:10]} - add to country_alias")
        n_alias = (df["country"] != df["raw_country"]).sum()
        self.log("FLAG", "04 countries", "Country aliases mapped to master names", int(n_alias))
        cm = self.country_master.set_index("country")
        bad_region = df["region"] != df["country"].map(cm["region"])
        bad_ccy = df["currency"] != df["country"].map(cm["currency"])
        if bad_region.any() or bad_ccy.any():
            self.log("BLOCK", "04 countries", "Region or currency differs from country master",
                     int(bad_region.sum() + bad_ccy.sum()))
        return df

    # ============================================================ step 5
    def step05_dates(self, df):
        """Parse each tab with its configured patterns, in priority order; output = first day of month."""
        fy_start = self.country_master.set_index("country")["fiscal_year_start_month"]
        lo, hi = self.th["serial_range"]
        month = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        pattern = pd.Series(None, index=df.index, dtype=object)
        day = pd.Series(np.nan, index=df.index)
        for tab, conf in self.cfg["tabs"].items():
            idx = df.index[df["sheet"] == tab]
            v = df.loc[idx, "date"]
            is_dt = v.map(lambda x: isinstance(x, (dt.datetime, dt.date, pd.Timestamp)))
            is_num = v.map(lambda x: isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and pd.notna(x))
            s = v.where(~(is_dt | is_num)).astype("string")
            for p in conf["date_patterns"]:
                todo = month.loc[idx].isna()
                if p == "datetime":
                    m = is_dt & todo
                    d = pd.to_datetime(v[m])
                elif p == "excel_serial":
                    m = is_num & todo & v.map(lambda x: lo <= float(x) <= hi if pd.notna(x) and not isinstance(x, str) else False)
                    d = pd.Timestamp("1899-12-30") + pd.to_timedelta(v[m].astype(float), unit="D")
                elif p == "dd/mm/yyyy":
                    m = todo & s.str.fullmatch(r"\d{2}/\d{2}/\d{4}").fillna(False)
                    d = pd.to_datetime(s[m], format="%d/%m/%Y")
                elif p == "mm-dd-yy":
                    m = todo & s.str.fullmatch(r"\d{2}-\d{2}-\d{2}").fillna(False)
                    d = pd.to_datetime(s[m], format="%m-%d-%y")
                elif p == "yyyy-mm-dd":
                    m = todo & s.str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False)
                    d = pd.to_datetime(s[m], format="%Y-%m-%d")
                elif p == "mon-yy":
                    m = todo & s.str.fullmatch(r"[A-Za-z]{3}-\d{2}").fillna(False)
                    d = pd.to_datetime(s[m], format="%b-%y")
                elif p == "fiscal_period":
                    m = todo & s.str.fullmatch(r"FY\d{2} P\d{2}").fillna(False)
                    fy = 2000 + s[m].str[2:4].astype(int)
                    per = s[m].str[-2:].astype(int)
                    start = df.loc[m[m].index, "country"].map(fy_start).astype(int)
                    m0 = (start - 1) + (per - 1)                     # months after January of year before FY end
                    year = fy - 1 + m0 // 12 if self.cfg["fiscal_period_naming"] == "end_year" else fy + m0 // 12
                    d = pd.to_datetime(dict(year=year, month=m0 % 12 + 1, day=1))
                else:
                    self.log("BLOCK", "05 dates", f"Unknown date pattern '{p}' in config")
                if m.any():
                    month.loc[m[m].index] = d.values
                    pattern.loc[m[m].index] = p
                    day.loc[m[m].index] = pd.DatetimeIndex(d).day
        # Day/month guard: dates must be first-of-month, otherwise the convention has flipped
        for p in ["dd/mm/yyyy", "mm-dd-yy", "yyyy-mm-dd"]:
            sel = pattern == p
            if sel.any() and (day[sel] == 1).mean() < self.th["day_one_min_share"]:
                self.log("BLOCK", "05 dates", f"Pattern {p}: day part is not 01 - day/month convention may have flipped")
        failed = month.isna()
        if failed.any():
            self.log("BLOCK", "05 dates", f"Unparseable dates, e.g. {df.loc[failed, 'raw_date'].head(5).tolist()}", int(failed.sum()))
        df["date"] = month.dt.to_period("M").dt.to_timestamp()
        df["date_pattern"] = pattern
        future = df["date"] > self.fx["date"].max()
        if future.any():
            self.log("BLOCK", "05 dates", "Dates after the extract month", int(future.sum()))
        self.extract_month = df["date"].max()
        prov = self.extract_month - pd.DateOffset(months=int(self.th["provisional_months"]) - 1)
        df["month_status"] = np.where(df["date"] >= prov, "provisional", "closed")
        counts = df["date_pattern"].value_counts().to_dict()
        self.log("FLAG", "05 dates", f"Date patterns parsed: {counts}")
        return df

    # ============================================================ step 6
    def step06_numbers(self, df):
        """Convert text to numbers with each source's locale. Excel numbers are left alone.
        Missing tokens become blank with a reason - never 0."""
        tokens = set(self.cfg["missing_tokens"])
        syms = "".join(re.escape(s) for s in self.cfg["currency_symbols"])
        for col in self.cfg["measures"]:
            v = df[col]
            is_num = v.map(lambda x: isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and pd.notna(x))
            is_blank = v.isna()
            s = v.where(~is_num & ~is_blank).astype("string").str.strip()
            is_tok = s.isin(tokens).fillna(False)
            out = pd.Series(np.nan, index=df.index)
            out[is_num] = v[is_num].astype(float)
            txt = s.notna() & ~is_tok
            for tab, conf in self.cfg["tabs"].items():
                sel = txt & (df["sheet"] == tab)
                if not sel.any():
                    continue
                t = s[sel]
                neg = t.str.contains(r"^\(|^-|-\s*[" + syms + r"]", regex=True)
                t = t.str.replace(r"[()\s" + syms + r"]", "", regex=True).str.replace("-", "", regex=False)
                t = t.str.replace(conf["thousands"], "", regex=False).str.replace(conf["decimal"], ".", regex=False)
                num = pd.to_numeric(t, errors="coerce")
                out[sel] = np.where(neg, -num, num)
            bad = txt & out.isna()
            reason = pd.Series(None, index=df.index, dtype=object)
            reason[is_blank] = "blank"
            reason[is_tok] = "token:" + s[is_tok].astype(str)
            reason[bad] = "unparseable"
            df[col] = out
            df[f"{col}__origin"] = np.select([is_num, txt], ["number", "text"], "missing")
            df[f"{col}__missing"] = reason
            if bad.any():
                self.log("HOLD", "06 numbers", f"{col}: text that could not be converted (blanked, reason=unparseable)", int(bad.sum()))
        n_txt = sum((df[f"{c}__origin"] == "text").sum() for c in self.cfg["measures"])
        n_miss = sum(df[f"{c}__missing"].notna().sum() for c in self.cfg["measures"])
        self.log("FLAG", "06 numbers", "Text cells converted to numbers", int(n_txt))
        self.log("FLAG", "06 numbers", "Missing cells (blank / ERROR / - / n/a / #N/A) kept as blank, not 0", int(n_miss))
        df["fx"] = self._fx(df)
        self.add_value("06 parsed raw total", df)
        return df

    def add_value(self, step, df, sign=1.0, label=None):
        """Record Net Sales in EUR by year for the value bridge."""
        v = (df["net_sales_lc"] / df["fx"]).groupby(df["date"].dt.year).sum() * sign
        for y, x in v.items():
            self.value_bridge.append({"step": label or step, "year": int(y), "net_sales_eur_m": x / 1e6})

    # ============================================================ step 7
    def step07_sign(self, df):
        """Apply each source's trade-spend sign, then test GS - TS = NS row by row."""
        for tab, conf in self.cfg["tabs"].items():
            sel = df["sheet"] == tab
            sign = conf["trade_spend_sign"]
            # self-check: the configured sign must match what the data says for the majority of complete rows
            comp = sel & df[["gross_sales_lc", "trade_spend_lc", "net_sales_lc"]].notna().all(axis=1)
            minus = (abs(df.loc[comp, "gross_sales_lc"] - df.loc[comp, "trade_spend_lc"] - df.loc[comp, "net_sales_lc"]) < 1).mean()
            plus = (abs(df.loc[comp, "gross_sales_lc"] + df.loc[comp, "trade_spend_lc"] - df.loc[comp, "net_sales_lc"]) < 1).mean()
            data_sign = 1 if minus >= plus else -1
            if data_sign != sign:
                self.log("BLOCK", "07 sign", f"{tab}: data follows trade_spend_sign={data_sign} but config says {sign}")
            df.loc[sel, "trade_spend_lc"] = df.loc[sel, "trade_spend_lc"] * sign
        neg = (df[self.cfg["measures"]] < 0).any(axis=1)
        df["flag_negative"] = neg
        if neg.any():
            self.log("FLAG", "07 sign", "Rows with a negative value after sign fix (allowed, flagged)", int(neg.sum()))
        df["identity_gap"] = df["gross_sales_lc"] - df["trade_spend_lc"] - df["net_sales_lc"]
        return df

    def _identity_ok(self, df):
        tol = np.maximum(self.th["identity_abs_tol_lc"], self.th["identity_rel_tol"] * df["gross_sales_lc"].abs())
        return (df["gross_sales_lc"] - df["trade_spend_lc"] - df["net_sales_lc"]).abs() <= tol

    # ============================================================ step 8
    def step08_magnitude(self, df):
        """Repair x1000 / x0.001 errors only when exactly one value explains the gap,
        using volume x typical price as tie-breaker. Large fixes go to the approval queue."""
        complete = df[["gross_sales_lc", "trade_spend_lc", "net_sales_lc"]].notna().all(axis=1)
        broken = complete & ~self._identity_ok(df)
        self.log("FLAG", "08 magnitude", "Complete rows breaking GS - TS = NS before repair", int(broken.sum()))
        # typical price per SKU-country (Gross Sales per unit), for the tie-breaker
        ok = complete & ~broken & (df["volume_units"] > 0)
        price = (df.loc[ok, "gross_sales_lc"] / df.loc[ok, "volume_units"]).groupby([df.loc[ok, "sku_code"], df.loc[ok, "country"]]).median()
        df["fix_reason"] = None
        before = df.copy()
        fixed = 0
        for i in df.index[broken]:
            r = df.loc[i]
            implied = {"net_sales_lc": r.gross_sales_lc - r.trade_spend_lc,
                       "gross_sales_lc": r.net_sales_lc + r.trade_spend_lc,
                       "trade_spend_lc": r.gross_sales_lc - r.net_sales_lc}
            cands = [c for c, imp in implied.items() if imp > 0 and any(abs(r[c] / imp - f) / f < 0.01 for f in self.th["scale_factors"])]
            if len(cands) > 1 and pd.notna(r.volume_units):        # tie-breaker: volume x usual price
                p = price.get((r.sku_code, r.country), np.nan)
                cands = [c for c in cands if c != "gross_sales_lc" or abs(r.gross_sales_lc / (r.volume_units * p) - 1) > 0.5]
            if len(cands) == 1:
                c = cands[0]
                df.at[i, c] = implied[c]
                df.at[i, "fix_reason"] = f"scale_fix:{c}"
                fixed += 1
                eur = abs(r[c] - implied[c]) / r.fx
                if eur > self.th["autofix_max_eur"]:
                    self.approval_queue.append({"country": r.country, "date": r.date, "sku": r.sku_code, "field": c,
                                                "raw_value": r[c], "fixed_value": implied[c], "change_eur_m": eur / 1e6,
                                                "status": "auto-applied, needs regional controller sign-off"})
        self.log("FLAG", "08 magnitude", "Magnitude errors repaired (x1000 / x0.001)", fixed)
        if self.approval_queue:
            self.log("HOLD", "08 magnitude", "Fixes above the auto-fix limit waiting for controller sign-off", len(self.approval_queue))
        self.add_value("08 magnitude fixes", df)
        self.value_bridge[-len(df["date"].dt.year.unique()):] = []  # replace with the delta only
        delta = (df["net_sales_lc"] - before["net_sales_lc"]).fillna(0) / df["fx"]
        for y, x in delta.groupby(df["date"].dt.year).sum().items():
            self.value_bridge.append({"step": "08 magnitude fixes", "year": int(y), "net_sales_eur_m": x / 1e6})
        # rows still broken -> quarantine
        still = complete & ~self._identity_ok(df)
        if still.any():
            q = df[still].assign(quarantine_reason="identity_fails_after_repair")
            self.quarantine.append(q)
            self.add_value("08 quarantined (identity)", q, sign=-1)
            df = df[~still]
            self.log("HOLD", "08 magnitude", "Rows quarantined: identity still fails", int(still.sum()))
        # other ratio bands: COGS and A&P must be plausible vs Net Sales (catches locale misreads)
        for col, band in [("cogs_lc", "cogs_to_ns"), ("ap_lc", "ap_to_ns")]:
            lo, hi = self.th["ratio_bands"][band]
            r = df[col] / df["net_sales_lc"]
            bad = r.notna() & ((r < lo) | (r > hi))
            if bad.any():
                df.loc[bad, f"{col}__missing"] = "quarantined_out_of_band"
                self.quarantine.append(df[bad].assign(quarantine_reason=f"{col} out of band"))
                df.loc[bad, col] = np.nan
                self.log("HOLD", "08 magnitude", f"{col} values out of plausible band - blanked and quarantined", int(bad.sum()))
        self.bridge("08 after magnitude + quarantine", df)
        return df

    # ============================================================ step 9
    def step09_sku(self, df):
        """Map legacy / re-coded SKU codes to master; take attributes from master (raw kept)."""
        cmap = dict(zip(self.code_map["source_code"], self.code_map["master_code"]))
        df["sku_master"] = df["sku_code"].map(cmap).fillna(df["sku_code"])
        df["code_mapped"] = df["sku_master"] != df["sku_code"]
        for code, v in (self.cfg.get("sku_code_validity") or {}).items():
            sel = df["sku_code"] == code
            early = sel & (df["date"] < pd.Timestamp(v["valid_from"]))
            if early.any():
                self.log("FLAG", "09 sku", f"{code} used before its valid_from date {v['valid_from']}", int(early.sum()))
            df.loc[(df["sku_master"] == cmap.get(code, code)) & (df["date"] == pd.Timestamp(v["valid_from"])), "event_recode"] = True
        unknown = ~df["sku_master"].isin(self.sku_master["sku_code"])
        if unknown.any():
            q = df[unknown].assign(quarantine_reason="unknown_sku")
            self.quarantine.append(q)
            self.add_value("09 quarantined (unknown SKU)", q, sign=-1)
            df = df[~unknown]
            self.log("HOLD", "09 sku", f"Unknown SKU codes quarantined: {q['sku_code'].unique()[:5].tolist()}", int(unknown.sum()))
        self.log("FLAG", "09 sku", "Rows mapped through sku_code_map", int(df["code_mapped"].sum()))
        m = self.sku_master.set_index("sku_code")
        decisions = {(d["attribute"], d["from"]): d["to"] for d in self.cfg.get("attribute_decisions", [])}
        for a in ["brand", "category"]:
            alias = self.cfg["attribute_alias"].get(a, {})
            mapped = df[a].map(lambda x: alias.get(x, x))
            master = df["sku_master"].map(m[a])
            differs = mapped != master
            pending = differs & ~df[a].map(lambda x: (a, x) in decisions or x in alias)
            if pending.any():
                self.log("HOLD", "09 sku", f"{a} differs from master without a logged decision", int(pending.sum()))
            df[a] = master
        df["sku_name"] = df["sku_master"].map(m["sku_name"])
        df["launch_date"] = pd.to_datetime(df["sku_master"].map(m["launch_date"]))
        df["delist_date"] = pd.to_datetime(df["sku_master"].map(m["delist_date"]))
        df["acquired_date"] = pd.to_datetime(df["sku_master"].map(m["acquired_date"]))
        life = (df["date"] < df["launch_date"]) | (df["date"] >= df["delist_date"]) | (df["date"] < df["acquired_date"])
        if life.any():
            self.log("HOLD", "09 sku", "Sales outside SKU life (before launch / after delist / before acquisition)", int(life.sum()))
        self.bridge("09 after SKU mapping", df)
        return df

    # ============================================================ step 10
    def step10_duplicates(self, df):
        """Merge copies of the same country x month x master SKU field by field.
        Conflicting copies are never summed or picked - they go to quarantine."""
        key = ["country", "date", "sku_master"]
        dup = df.duplicated(key, keep=False)
        groups = df[dup].groupby(key, sort=False)
        keep_rows, drop_idx, conflicts, filled = [], [], [], 0
        types = []
        for _, g in groups:
            ok = True
            for col in self.cfg["measures"]:
                vals = g[col].dropna()
                if len(vals) > 1:
                    tol = max(self.th["duplicate_abs_tol_lc"], self.th["duplicate_rel_tol"] * vals.abs().max())
                    if vals.max() - vals.min() > tol:
                        ok = False
            if not ok:
                conflicts.append(g)
                continue
            # merge: prefer a non-blank original Excel number, then text
            base = g.iloc[0].copy()
            for col in self.cfg["measures"]:
                cand = g[g[col].notna()]
                if len(cand):
                    pick = cand[cand[f"{col}__origin"] == "number"]
                    row = (pick if len(pick) else cand).iloc[0]
                    if pd.isna(base[col]):
                        filled += 1
                    base[col], base[f"{col}__origin"], base[f"{col}__missing"] = row[col], row[f"{col}__origin"], None
            if g["sku_code"].nunique() > 1:
                t = "recode"
            elif g[self.cfg["measures"]].nunique(dropna=True).max() > 1:
                t = "rounding"
            elif g[["raw_date", "raw_country"]].astype(str).nunique().max() > 1:
                t = "format_variant"
            else:
                t = "exact"
            base["dedup_type"], base["dedup_copies"] = t, len(g)
            types.append(t)
            keep_rows.append(base)
            drop_idx.extend(g.index)
        removed = df.loc[drop_idx]
        merged = pd.DataFrame(keep_rows)
        if conflicts:
            q = pd.concat(conflicts).assign(quarantine_reason="conflicting_duplicates")
            self.quarantine.append(q)
            self.add_value("10 quarantined (conflicting duplicates)", q, sign=-1)
            df = df.drop(q.index)
            self.log("HOLD", "10 duplicates", "Conflicting duplicate rows quarantined", len(q))
        out = pd.concat([df.drop(drop_idx), merged]).sort_index()
        self.dup_removed = len(drop_idx) - len(merged)
        # value removed = all copies minus the merged rows kept
        self.add_value("10 duplicates removed", removed, sign=-1)
        self.add_value("10 duplicates merged back", merged)
        out["dedup_type"] = out.get("dedup_type")
        self.log("FLAG", "10 duplicates", f"Duplicate rows removed by merging, by type: {pd.Series(types).value_counts().to_dict()}",
                 int(len(drop_idx) - len(merged)))
        self.log("FLAG", "10 duplicates", "Blank cells filled from a twin copy (exact, not estimated)", filled)
        if out.duplicated(key).any():
            self.log("BLOCK", "10 duplicates", "Key still not unique after merge")
        self.bridge("10 after duplicates", out)
        return out

    # ============================================================ step 11
    def step11_quarantine_gate(self, df):
        self.rows_quarantined = sum(len(q) for q in self.quarantine if q["quarantine_reason"].iloc[0] in
                                    ("identity_fails_after_repair", "unknown_sku", "conflicting_duplicates"))
        q = pd.concat(self.quarantine) if self.quarantine else pd.DataFrame(columns=df.columns)
        if len(q):
            qgs = (q["gross_sales_lc"].fillna(0) / q["fx"]).groupby([q["country"], q["date"]]).sum()
            tot = (df["gross_sales_lc"].fillna(0) / df["fx"]).groupby([df["country"], df["date"]]).sum()
            share = (qgs / tot.reindex(qgs.index)).fillna(1)
            worst = share.max()
            group = qgs.sum() / tot.sum()
            if worst > self.th["quarantine_gate_country_month"] or group > self.th["quarantine_gate_group"]:
                sev = "HOLD"   # prototype: publish but flag. In production set to BLOCK.
                self.log(sev, "11 quarantine gate", f"Quarantined Gross Sales: worst country-month {worst:.3%}, group {group:.4%}")
            else:
                self.log("FLAG", "11 quarantine gate", f"Quarantine below materiality (group {group:.4%})")
        return df

    # ============================================================ step 12
    def step12_recover(self, df):
        """Where exactly one of GS, TS, NS is blank, rebuild it from the other two. Nothing else."""
        trio = ["gross_sales_lc", "trade_spend_lc", "net_sales_lc"]
        one = df[trio].isna().sum(axis=1) == 1
        df["imputed_method"] = None
        before = df["net_sales_lc"].copy()
        for col in trio:
            sel = one & df[col].isna()
            if col == "net_sales_lc":
                df.loc[sel, col] = df.loc[sel, "gross_sales_lc"] - df.loc[sel, "trade_spend_lc"]
            elif col == "gross_sales_lc":
                df.loc[sel, col] = df.loc[sel, "net_sales_lc"] + df.loc[sel, "trade_spend_lc"]
            else:
                df.loc[sel, col] = df.loc[sel, "gross_sales_lc"] - df.loc[sel, "net_sales_lc"]
            df.loc[sel, "imputed_method"] = f"identity:{col}"
            self.log("FLAG", "12 recover", f"{col} rebuilt exactly from the other two", int(sel.sum()))
        delta = (df["net_sales_lc"].fillna(0) - before.fillna(0)) / df["fx"]
        for y, x in delta.groupby(df["date"].dt.year).sum().items():
            self.value_bridge.append({"step": "12 recovered by identity", "year": int(y), "net_sales_eur_m": x / 1e6})
        left = {c: int(df[c].isna().sum()) for c in self.cfg["measures"]}
        self.log("FLAG", "12 recover", f"Blanks remaining in clean layer (estimated only in forecast-ready layer): {left}")
        return df

    # ============================================================ step 13
    def step13_reconcile(self, df):
        path = self.cfg["paths"].get("ledger_control_totals")
        if not path:
            self.log("HOLD", "13 reconcile", "No ledger control totals provided - clean actuals cannot be signed off "
                     "against consolidation (request from Group Controlling)")
            return df
        led = pd.read_csv(self.base / path, parse_dates=["date"])
        ours = df.groupby(["country", "date"])[["net_sales_lc"]].sum().reset_index()
        cmp_ = ours.merge(led, on=["country", "date"], suffixes=("", "_ledger"), how="outer")
        gap = (cmp_["net_sales_lc"] - cmp_["net_sales_lc_ledger"]).abs() > 1
        if gap.any():
            self.log("HOLD", "13 reconcile", "Country-months differing from ledger - explain with reconciling items", int(gap.sum()))
        return df

    # ============================================================ step 14
    def step14_publish(self, df):
        """Add EUR, flags; publish the clean layer; compare with last month's snapshot."""
        df = self.add_fx_and_flags(df)
        cols = (["date", "region", "country", "currency", "category", "brand", "sku_master", "sku_name",
                 "volume_units"] + MONEY + [f"{c.replace('_lc', '_eur')}" for c in MONEY]
                + [f"{c.replace('_lc', '_eur_cc')}" for c in ["net_sales_lc"]]
                + ["month_status", "flag_hyperinflation", "flag_inorganic", "lifecycle_stage", "regime", "hemisphere",
                   "flag_negative", "fix_reason", "imputed_method", "dedup_type", "event_recode",
                   "sku_code", "raw_country", "raw_date", "date_pattern", "source_file", "sheet", "excel_row"]
                + [f"{c}__missing" for c in self.cfg["measures"]])
        clean = df[[c for c in cols if c in df.columns]].rename(columns={"sku_master": "sku_code", "sku_code": "source_sku_code"})
        clean = clean.sort_values(["date", "country", "sku_code"]).reset_index(drop=True)
        self.out.mkdir(parents=True, exist_ok=True)
        clean = self.to_storable(clean)
        clean.to_parquet(self.out / "fact_pnl_clean.parquet", index=False)
        snap = self.base / self.cfg["paths"]["snapshot_dir"]
        snap.mkdir(parents=True, exist_ok=True)
        this = snap / f"fact_pnl_clean_{self.extract_month:%Y-%m}.parquet"
        prev = sorted(p for p in snap.glob("fact_pnl_clean_*.parquet") if p != this)
        if prev:
            old = pd.read_parquet(prev[-1])
            k = ["country", "date", "sku_code"]
            cur = clean[clean.month_status == "closed"]
            m = cur.merge(old[k + ["net_sales_lc"]], on=k, how="outer", suffixes=("", "_prev"), indicator=True)
            m = m[m["date"] <= cur["date"].max()]
            both = m["net_sales_lc"].notna() & m["net_sales_lc_prev"].notna()
            # 1) completeness changes: a value appeared or disappeared (FLAG - explained by extract quality)
            compl = (m["net_sales_lc"].isna() != m["net_sales_lc_prev"].isna()) | (m["_merge"] != "both")
            # 2) value restatements: the same cell has a different value (HOLD - controller approval)
            diff = (m["net_sales_lc"] - m["net_sales_lc_prev"]).where(both, 0)
            month_ns = m.groupby(["country", "date"])["net_sales_lc_prev"].transform("sum")
            m["restate_pct"] = diff.groupby([m["country"], m["date"]]).transform("sum").abs() / month_ns
            restated = m[both & (m["restate_pct"] > self.th["restatement_tol"])]
            self.log("FLAG", "14 publish", f"Completeness changes vs {prev[-1].name} (cells now blank or now filled)", int(compl.sum()))
            if len(restated):
                restated[k + ["net_sales_lc_prev", "net_sales_lc", "restate_pct"]].to_csv(self.out / "restatements.csv", index=False)
                self.log("HOLD", "14 publish", f"Closed-month VALUES restated vs {prev[-1].name} - approval needed",
                         int(restated[["country", "date"]].drop_duplicates().shape[0]))
            else:
                self.log("FLAG", "14 publish", f"No value restatement of closed months vs {prev[-1].name}")
        else:
            self.log("FLAG", "14 publish", "First run: no previous snapshot to compare against")
        clean.to_parquet(this, index=False)
        self.bridge("14 published clean actuals", clean)
        return clean

    def add_fx_and_flags(self, df):
        bf = self.cfg["business_flags"]
        cm = self.country_master.set_index("country")
        df = df.drop(columns=["fx"]).merge(self.fx[["date", "country", "fx_lc_per_eur"]], on=["date", "country"], how="left")
        df = df.rename(columns={"fx_lc_per_eur": "fx"})
        cc = self.fx[self.fx.date.dt.year == bf["constant_currency_year"]].groupby("country")["fx_lc_per_eur"].mean()
        for c in MONEY:
            df[c.replace("_lc", "_eur")] = df[c] / df["fx"]
        df["net_sales_eur_cc"] = df["net_sales_lc"] / df["country"].map(cc)
        start = df["country"].map(lambda c: pd.Timestamp(bf["hyperinflation"].get(c, "2100-01-01")))
        df["flag_hyperinflation"] = df["date"] >= start
        df["flag_inorganic"] = False
        for a in bf["acquisitions"]:
            end = pd.Timestamp(a["date"]) + pd.DateOffset(months=a["inorganic_months"])
            df.loc[(df.brand == a["brand"]) & (df.date < end), "flag_inorganic"] = True
        since = (df["date"].dt.year - df["launch_date"].dt.year) * 12 + (df["date"].dt.month - df["launch_date"].dt.month)
        df["lifecycle_stage"] = np.where(since < bf["launch_window_months"], "launch", "steady")
        df.loc[df["delist_date"].notna(), "lifecycle_stage"] = "delisting"
        df["regime"] = None
        for r in bf["regimes"]:
            sel = (df.date >= pd.Timestamp(r["start"])) & (df.date <= pd.Timestamp(r["end"]))
            if "category" in r:
                sel &= df.category == r["category"]
            if "country" in r:
                sel &= df.country == r["country"]
            df.loc[sel, "regime"] = r["name"]
        df["hemisphere"] = np.where(df["country"].map(cm["south"]), "south", "north")
        return df

    # ============================================== steps 15-16 forecast-ready
    def step15_grid(self, clean):
        """Build the expected SKU x country x month grid and classify every missing cell."""
        bf = self.cfg["business_flags"]
        first = clean.groupby(["sku_code", "country"])["date"].min()
        m = self.sku_master.set_index("sku_code")
        rows = []
        for (sku, ctry), f in first.items():
            start = max(f, pd.Timestamp(m.at[sku, "launch_date"]) if pd.notna(m.at[sku, "launch_date"]) else f)
            end = self.extract_month
            if pd.notna(m.at[sku, "delist_date"]):
                end = min(end, pd.Timestamp(m.at[sku, "delist_date"]) - pd.DateOffset(months=1))
            for d in pd.date_range(start, end, freq="MS"):
                rows.append((sku, ctry, d))
        grid = pd.DataFrame(rows, columns=["sku_code", "country", "date"])
        fr = grid.merge(clean, on=["sku_code", "country", "date"], how="left", indicator=True)
        absent = fr["_merge"] == "left_only"
        cm_present = set(map(tuple, clean[["country", "date"]].drop_duplicates().values))
        whole_missing = absent & pd.Series([(c, d) in cm_present for c, d in zip(fr.country, fr.date)], index=fr.index).eq(False)
        # sell-in vs sell-out check: zero-fill only where the country-category month looks complete
        ext = pd.read_excel(self.raw / self.cfg["files"]["external"], sheet_name="market_category_country",
                            usecols=["date", "country", "category", "market_size_rsv_lc", "our_value_share"])
        ext["date"] = pd.to_datetime(ext["date"])
        ext["sellout"] = ext["market_size_rsv_lc"] * ext["our_value_share"]
        ns = clean.groupby(["date", "country", "category"])["net_sales_lc"].sum().rename("ns").reset_index()
        chk = ext.merge(ns, on=["date", "country", "category"])
        chk["ratio"] = chk["ns"] / chk["sellout"]
        chk["norm"] = chk.groupby(["country", "category"])["ratio"].transform(lambda r: r.rolling(13, center=True, min_periods=6).median())
        chk["suspect"] = (chk["ratio"] / chk["norm"] - 1) < -self.th["sellin_sellout_band"]
        sku_cat = m["category"]
        fr["category"] = fr["category"].fillna(fr["sku_code"].map(sku_cat))
        fr = fr.merge(chk[["date", "country", "category", "suspect"]], on=["date", "country", "category"], how="left")
        fr["grid_status"] = np.select(
            [~absent, whole_missing, absent & fr["suspect"].fillna(False)],
            ["reported", "missing_extract", "suspect_missing"], "zero_sale")
        z = fr["grid_status"] == "zero_sale"
        fr.loc[z, self.cfg["measures"]] = 0.0
        # dormant: long runs of zero months
        fr = fr.sort_values(["sku_code", "country", "date"])
        zero = (fr["grid_status"] == "zero_sale").astype(int)
        run = zero.groupby((zero != zero.shift()).cumsum()).cumsum()
        fr["flag_possibly_dormant"] = run >= self.th["dormant_months"]
        counts = fr["grid_status"].value_counts().to_dict()
        self.log("FLAG", "15 grid", f"Grid cells by status: {counts}")
        miss = fr[fr.grid_status == "missing_extract"][["country", "date"]].drop_duplicates()
        if len(miss):
            self.log("HOLD", "15 grid", f"Whole country-months missing from the extract: "
                     f"{[f'{c} {d:%Y-%m}' for c, d in miss.values]}")
        fr["region"] = fr["region"].fillna(fr["country"].map(self.country_master.set_index("country")["region"]))
        return fr.drop(columns=["_merge"])

    def step16_estimate(self, fr):
        """Forecast-ready layer only: estimate remaining blanks, always flagged."""
        fr = fr.copy()
        fr["estimated"] = None
        ok = fr["grid_status"] == "reported"
        for col in ["cogs_lc", "ap_lc"]:
            r = fr[col] / fr["net_sales_lc"]
            cat_month = r[ok].groupby([fr["country"], fr["category"], fr["date"]]).median()
            sku_rel = (r[ok] / pd.Series(list(zip(fr.country, fr.category, fr.date)), index=fr.index)[ok].map(cat_month)) \
                .groupby([fr["sku_code"], fr["country"]]).median()
            need = ok & fr[col].isna() & fr["net_sales_lc"].notna()
            keys = list(zip(fr.loc[need, "country"], fr.loc[need, "category"], fr.loc[need, "date"]))
            rel = pd.Series(list(zip(fr.loc[need, "sku_code"], fr.loc[need, "country"])), index=fr.index[need]).map(sku_rel).fillna(1)
            fr.loc[need, col] = fr.loc[need, "net_sales_lc"] * pd.Series(keys, index=fr.index[need]).map(cat_month).values * rel
            fr.loc[need, "estimated"] = fr.loc[need, "estimated"].fillna("") + f"{col};"
            self.log("FLAG", "16 estimate", f"{col} estimated (same-month country x category ratio x SKU relative)", int(need.sum()))
        # volume: Gross Sales / same-month country x category price x SKU relative price
        p = fr["gross_sales_lc"] / fr["volume_units"]
        cat_p = p[ok].groupby([fr["country"], fr["category"], fr["date"]]).median()
        rel = (p[ok] / pd.Series(list(zip(fr.country, fr.category, fr.date)), index=fr.index)[ok].map(cat_p)).groupby([fr["sku_code"], fr["country"]]).median()
        need = ok & fr["volume_units"].isna() & fr["gross_sales_lc"].notna()
        price = pd.Series(list(zip(fr.loc[need, "country"], fr.loc[need, "category"], fr.loc[need, "date"])), index=fr.index[need]).map(cat_p) \
            * pd.Series(list(zip(fr.loc[need, "sku_code"], fr.loc[need, "country"])), index=fr.index[need]).map(rel).fillna(1)
        fr.loc[need, "volume_units"] = (fr.loc[need, "gross_sales_lc"] / price).round()
        fr.loc[need, "estimated"] = fr.loc[need, "estimated"].fillna("") + "volume_units;"
        self.log("FLAG", "16 estimate", "volume_units estimated (Gross Sales / same-month price)", int(need.sum()))
        # provisional months down-weighted in model training
        fr["train_weight"] = np.where(fr["date"] >= self.extract_month - pd.DateOffset(months=int(self.th["provisional_months"]) - 1), 0.5, 1.0)
        fr.loc[fr.grid_status.isin(["missing_extract", "suspect_missing"]), "train_weight"] = 0.0
        return fr

    # ===================================================== step 17 overheads
    def step17_overheads(self, fr):
        path = self.raw / self.cfg["files"]["overheads"]
        probe = pd.read_excel(path, header=None, nrows=15)
        hdr = probe.index[probe.astype(str).apply(lambda r: r.str.contains("Country").any() and r.str.contains("Amount").any(), axis=1)]
        if not len(hdr):
            self.log("BLOCK", "17 overheads", "Header row not found in overheads file")
        oh = pd.read_excel(path, header=int(hdr[0]))
        oh.columns = ["date", "country", "currency", "function", "overheads_lc"]
        oh["country"] = self.norm_text(oh["country"]).str.lower().map(self.alias)
        oh["function"] = self.norm_text(oh["function"]).map(lambda x: self.cfg["attribute_alias"]["function"].get(x, x))
        oh["date"] = pd.to_datetime(oh["date"].astype(str), format="%Y-%m")
        if oh.duplicated(["country", "date", "function"]).any():
            self.log("BLOCK", "17 overheads", "Duplicate country x month x function")
        expected = oh["country"].nunique() * oh["function"].nunique() * oh["date"].nunique()
        if len(oh) != expected:
            self.log("HOLD", "17 overheads", f"Overheads grid incomplete: {len(oh)} of {expected}")
        self.log("FLAG", "17 overheads", f"{oh['function'].nunique()} functions after alias mapping: {sorted(oh['function'].unique())}")
        # Allocation to SKU: Net Sales share within country x month (configurable key)
        tot = oh.groupby(["country", "date"])["overheads_lc"].sum().rename("oh_country")
        fr = fr.merge(tot.reset_index(), on=["country", "date"], how="left")
        share = fr["net_sales_lc"] / fr.groupby(["country", "date"])["net_sales_lc"].transform("sum")
        fr["overheads_alloc_lc"] = fr["oh_country"] * share
        miss = fr.groupby(["country", "date"])["grid_status"].transform(lambda s: (s == "missing_extract").all())
        fr["overheads_provisional"] = miss
        self.log("FLAG", "17 overheads", "Overheads allocated to SKUs by Net Sales share; label = contribution after country overheads")
        oh.to_parquet(self.out / "fact_overheads.parquet", index=False)
        return fr.drop(columns=["oh_country"])

    # ======================================== step 18 submissions & external
    def step18_external(self):
        ext = self.raw / self.cfg["files"]["external"]
        x = pd.ExcelFile(ext)
        cat_alias = self.cfg["attribute_alias"]["category"]
        for sheet in ["market_category_country", "macro_country", "commodities_global"]:
            d = x.parse(sheet)
            d["date"] = pd.to_datetime(d["date"])
            if "category" in d:
                d["category"] = d["category"].map(lambda c: cat_alias.get(c, c))
            d.to_parquet(self.out / f"ext_{sheet}.parquet", index=False)
        outl = []
        for sheet in [s for s in x.sheet_names if s.startswith("outlook_")]:
            d = x.parse(sheet)
            d["date"] = pd.to_datetime(d["date"])
            d["table"] = sheet.split("_", 2)[2]
            outl.append(d)
        o = pd.concat(outl, ignore_index=True)
        num = o.select_dtypes("number").columns
        flat = o.groupby(["vintage", "table"])[list(num)].nunique().le(1)
        o.to_parquet(self.out / "ext_outlooks.parquet", index=False)
        self.log("FLAG", "18 external", f"Outlook vintages stored: {sorted(o.vintage.unique())} "
                 "(a forecast made in month T may only use vintages published before T)")
        s = pd.read_excel(self.raw / self.cfg["files"]["submissions"])
        s["actual_status"] = np.where(s["actual_value"].notna(), "closed", "open")
        s[s.country == "GROUP"].to_parquet(self.out / "submissions_group.parquet", index=False)
        s[s.country != "GROUP"].to_parquet(self.out / "submissions_country.parquet", index=False)
        self.log("FLAG", "18 external", f"Submissions: {len(s):,} rows, units per metric = "
                 f"{s.groupby('metric')['unit'].nunique().max()} (must be 1)")

    # ===================================================== step 20 report
    def step20_report(self, clean, fr):
        hard = {
            "unique key (country, month, SKU)": not clean.duplicated(["country", "date", "sku_code"]).any(),
            "GS - TS = NS on all complete rows": bool(self._identity_ok(clean.dropna(subset=["gross_sales_lc", "trade_spend_lc", "net_sales_lc"])).all()),
            "30 countries": clean["country"].nunique() == len(self.country_master),
            "no unknown SKUs": clean["sku_code"].isin(self.sku_master["sku_code"]).all(),
            "row bridge balances (read = clean + duplicates removed + quarantined)": self.rows_read == len(clean) + self.dup_removed + self.rows_quarantined,
        }
        for k, v in hard.items():
            self.log("FLAG" if v else "BLOCK", "20 report", f"Hard check {'PASS' if v else 'FAIL'}: {k}")
        # value bridge must balance: start + movements = clean total
        vb = pd.DataFrame(self.value_bridge).pivot_table(index="step", columns="year", values="net_sales_eur_m", aggfunc="sum")
        clean_tot = clean.groupby(clean.date.dt.year)["net_sales_eur"].sum() / 1e6
        calc = vb.sum()
        diff = (calc - clean_tot).abs().max()
        vb.loc["= clean actuals (check)"] = clean_tot
        self.log("FLAG" if diff < 0.05 else "HOLD", "20 report", f"Value bridge unexplained difference: EUR {diff:.3f}m")
        # key ratios by country (post-clean monitoring)
        y = clean[clean.month_status == "closed"].groupby(["country", clean.date.dt.year])
        ratios = pd.DataFrame({"trade_rate": y.trade_spend_lc.sum() / y.gross_sales_lc.sum(),
                               "gross_margin": 1 - y.cogs_lc.sum() / y.net_sales_lc.sum(),
                               "ap_pct": y.ap_lc.sum() / y.net_sales_lc.sum()}).round(4)
        with pd.ExcelWriter(self.out / "run_report.xlsx") as w:
            pd.DataFrame(self.issues).to_excel(w, sheet_name="issue_log", index=False)
            pd.DataFrame(self.row_bridge).to_excel(w, sheet_name="row_bridge", index=False)
            vb.round(2).to_excel(w, sheet_name="value_bridge_NS_EURm")
            ratios.to_excel(w, sheet_name="key_ratios")
            pd.DataFrame(self.approval_queue).to_excel(w, sheet_name="approval_queue", index=False)
            fr["grid_status"].value_counts().rename_axis("grid_status").reset_index().to_excel(w, sheet_name="grid_status", index=False)
        q = pd.concat(self.quarantine) if self.quarantine else pd.DataFrame()
        q.drop(columns=[c for c in q.columns if c.startswith("raw_") and c != "raw_country"], errors="ignore") \
            .to_csv(self.out / "quarantine.csv", index=False)
        pd.DataFrame(self.issues).to_csv(self.out / "issue_log.csv", index=False)
        self.value_bridge_table = vb
        return vb

    # ===================================================================== run
    def run(self):
        steps_done = []
        def step(name, fn, *a):
            print(f"  {name}", flush=True)
            steps_done.append(name)
            return fn(*a)
        self.load_reference()
        frames = step("01 Ingest raw files (every cell as-is)", self.step01_ingest)
        df = step("02 Map column layouts", self.step02_schema, frames)
        df = step("03 Normalise text", self.step03_text, df)
        df = step("04 Map countries, check region & currency", self.step04_countries, df)
        df = step("05 Parse dates to months", self.step05_dates, df)
        df = step("06 Convert text to numbers", self.step06_numbers, df)
        df = step("07 Fix trade-spend sign", self.step07_sign, df)
        df = step("08 Repair magnitude errors", self.step08_magnitude, df)
        df = step("09 Map SKUs to master", self.step09_sku, df)
        df = step("10 Merge duplicates", self.step10_duplicates, df)
        df = step("11 Quarantine materiality gate", self.step11_quarantine_gate, df)
        df = step("12 Recover blanks exactly (GS - TS = NS)", self.step12_recover, df)
        df = step("13 Reconcile to ledger", self.step13_reconcile, df)
        clean = step("14 Publish clean actuals + snapshot", self.step14_publish, df)
        fr = step("15 Completeness grid", self.step15_grid, clean)
        fr = step("16 Estimate remaining gaps (forecast-ready only)", self.step16_estimate, fr)
        fr = step("17 Overheads: clean and allocate", self.step17_overheads, fr)
        self.to_storable(fr).to_parquet(self.out / "fact_pnl_forecast_ready.parquet", index=False)
        step("18 Submissions & external data", self.step18_external)
        vb = step("20 Validation suite & run report", self.step20_report, clean, fr)
        self.clean, self.forecast_ready = clean, fr
        return self


def main():
    cfg = sys.argv[1] if len(sys.argv) > 1 else "cleaning.yaml"
    print("Running cleaning pipeline ...")
    try:
        p = CleaningPipeline(cfg).run()
    except PipelineBlocked as e:
        print(f"\nRUN BLOCKED - nothing published.\n  {e}")
        sys.exit(1)
    sev = pd.DataFrame(p.issues)["severity"].value_counts().to_dict()
    print(f"\nRun finished. Issues by severity: {sev}")
    print("\nValue bridge - Net Sales EUR m by year:")
    print(p.value_bridge_table.round(1).to_string())
    print(f"\nOutputs in {p.out}")


if __name__ == "__main__":
    main()
