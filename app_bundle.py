"""
app_bundle.py  -  collects the small files the Streamlit app needs into app_data/ (a few MB, safe to put on GitHub).
Run after Steps 5 and 6:   python forecast_engine/app_bundle.py
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def build(base="."):
    base = Path(base).resolve()
    fc, ch, out = base / "data/forecast", base / "data/challenge", base / "app_data"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    shutil.copytree(fc / "scenario_kit", out / "scenario_kit")
    shutil.copy(fc / "group_pnl_monthly.parquet", out)
    shutil.copy(fc / "forecast_meta.json", out)
    x = pd.ExcelFile(fc / "forecast_results_2026.xlsx")
    for sheet in ["backtest_accuracy", "backtest_monthly", "backtest_gross_margin", "group_pnl_by_year"]:
        pd.read_excel(x, sheet, index_col=0).to_parquet(out / f"{sheet}.parquet")
    for f in ["group_views.parquet", "flags.parquet", "challenge_pack.md"]:
        shutil.copy(ch / f, out)
    y = pd.ExcelFile(ch / "challenge_pack.xlsx")
    pd.read_excel(y, "by_department", index_col=0).to_parquet(out / "by_department.parquet")
    pd.read_excel(y, "patterns").to_parquet(out / "patterns.parquet")
    pd.read_excel(base / "data/raw/master_data.xlsx", sheet_name="strategic_targets").to_parquet(out / "targets.parquet")
    g = pd.read_parquet(out / "group_pnl_monthly.parquet")
    json.dump({"built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
               "last_actual_month": str(g[g.Type == "Actual"].index.max().date()),
               "sources": ["data/forecast/forecast_results_2026.xlsx", "data/forecast/scenario_kit",
                           "data/challenge/challenge_pack.xlsx", "data/raw/master_data.xlsx (strategic targets)"]},
              open(out / "build_info.json", "w"), indent=1)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"app_data/ built: {size:.1f} MB, {sum(1 for p in out.rglob('*') if p.is_file())} files")


if __name__ == "__main__":
    build()
