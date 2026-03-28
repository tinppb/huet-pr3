"""
=============================================================
  NHÓM 3 – MODULE A: PHÂN TÍCH XU HƯỚNG & BENCHMARKING
  Tác giả: Phạm Phước Bảo Tín
=============================================================

Phân tích:
  1. Xu hướng chất lượng dịch vụ theo thời gian (visit_date)
  2. Benchmarking so sánh hiệu quả các khách sạn tại Huế
  3. Xuất JSON cho Dashboard HTML

CÁCH CHẠY:
  pip install pandas numpy
  python nhom3_trend_benchmark.py --input "D:\huet-pr3\TripAdvisor_Data_Cleaned_Hotel_English_Hue.json"
"""

import os
import re
import json
import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── CẤU HÌNH ────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("./nhom3_output")

MONTH_MAP = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
}
MONTH_VI = {
    1:"T1",2:"T2",3:"T3",4:"T4",5:"T5",6:"T6",
    7:"T7",8:"T8",9:"T9",10:"T10",11:"T11",12:"T12"
}
STAR_SENTIMENT = {5:"Positive",4:"Positive",3:"Neutral",2:"Negative",1:"Negative"}


# ═════════════════════════════════════════════════════════════════════════════
# LOAD & CHUẨN HÓA
# ═════════════════════════════════════════════════════════════════════════════

def load_and_clean(filepath: str) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("  [1] LOAD & CHUẨN HÓA DỮ LIỆU")
    print(f"{'='*55}")

    fp = Path(filepath)
    try:
        with open(fp, "r", encoding="utf-8") as f:
            raw = json.load(f)
        df = pd.DataFrame(raw) if isinstance(raw, list) else pd.DataFrame([raw])
    except json.JSONDecodeError:
        df = pd.read_json(fp, lines=True)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    print(f"  ✔ Đọc: {len(df):,} records | {df.shape[1]} cột")

    # Trích tên khách sạn từ URL
    if "url" in df.columns:
        df["hotel_name"] = df["url"].apply(_extract_hotel_name)
        print(f"  ✔ Số khách sạn: {df['hotel_name'].nunique()}")

    # Chuẩn hóa visit_date
    if "visit_date" in df.columns:
        parsed = df["visit_date"].apply(_parse_date)
        df["v_month"] = parsed.apply(lambda x: x[0])
        df["v_year"]  = parsed.apply(lambda x: x[1])
        df["v_quarter"] = df["v_month"].apply(
            lambda m: f"Q{((int(m)-1)//3)+1}" if pd.notna(m) else None
        )
        df["month_name"] = df["v_month"].map(MONTH_VI)
        valid = df["v_month"].notna().sum()
        print(f"  ✔ visit_date hợp lệ: {valid:,}/{len(df):,}")

    # Số sao
    if "star" in df.columns:
        df["star"] = pd.to_numeric(df["star"], errors="coerce").clip(1, 5)

    # Sentiment fallback từ star
    df["sentiment"] = df["star"].map(STAR_SENTIMENT).fillna("Neutral")

    # Lọc tỉnh Huế
    if "province" in df.columns:
        before = len(df)
        df = df[df["province"].str.contains("Hue|Huế|Thua Thien", case=False, na=False)]
        print(f"  ✔ Lọc tỉnh Huế: {before:,} → {len(df):,}")

    # Xóa duplicate
    dup_cols = [c for c in ["reviewer_id","hotel_name","comment"] if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=dup_cols).reset_index(drop=True)
    print(f"  ✔ Xóa duplicate: -{before - len(df)}")

    return df


def _extract_hotel_name(url):
    if not isinstance(url, str): return "Unknown"
    m = re.search(r"Reviews-([A-Za-z0-9_]+(?:_[A-Za-z0-9_]+)*)-[A-Z]", url)
    if m:
        return m.group(1).replace("_", " ")
    parts = [p for p in url.split("/") if len(p) > 5 and "Reviews" not in p and "http" not in p]
    return parts[-1].replace("_"," ").replace("-"," ") if parts else "Unknown"


def _parse_date(raw):
    if not isinstance(raw, str) or not raw.strip():
        return (None, None)
    raw = raw.strip()
    # "19-Sep", "Sep-23"
    m = re.match(r"^(\d{1,2})-([A-Za-z]{3})$", raw)
    if m:
        return (MONTH_MAP.get(m.group(2).lower()), None)
    m = re.match(r"^([A-Za-z]{3})-(\d{2,4})$", raw)
    if m:
        yr = int(m.group(2))
        yr = yr + 2000 if yr < 100 else yr
        return (MONTH_MAP.get(m.group(1).lower()), yr)
    # "Sep 2023", "September 2023"
    m = re.match(r"^([A-Za-z]{3,9})\s+(\d{4})$", raw)
    if m:
        mn = MONTH_MAP.get(m.group(1).lower()[:3])
        return (mn, int(m.group(2)))
    # "2023-09"
    m = re.match(r"^(\d{4})-(\d{2})$", raw)
    if m:
        return (int(m.group(2)), int(m.group(1)))
    try:
        dt = pd.to_datetime(raw, dayfirst=True)
        return (dt.month, dt.year)
    except:
        return (None, None)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE A: XU HƯỚNG THEO THỜI GIAN
# ═════════════════════════════════════════════════════════════════════════════

def analyze_trends(df: pd.DataFrame) -> dict:
    print(f"\n{'='*55}")
    print("  [2] PHÂN TÍCH XU HƯỚNG THEO THỜI GIAN")
    print(f"{'='*55}")

    results = {}
    df_t = df.dropna(subset=["v_month"]).copy()

    # ── A1: Xu hướng theo tháng trong năm (mùa vụ) ───────────────────────
    seasonal = df_t.groupby("v_month").agg(
        total_reviews  = ("star","count"),
        avg_star       = ("star","mean"),
        pct_positive   = ("sentiment", lambda x: (x=="Positive").mean()*100),
        pct_negative   = ("sentiment", lambda x: (x=="Negative").mean()*100),
        pct_neutral    = ("sentiment", lambda x: (x=="Neutral").mean()*100),
    ).reset_index().sort_values("v_month")
    seasonal["month_name"] = seasonal["v_month"].map(MONTH_VI)
    seasonal = seasonal.round(2)
    results["seasonal"] = seasonal
    print(f"  ✔ Xu hướng mùa vụ: {len(seasonal)} tháng")

    # ── A2: Xu hướng theo năm (nếu có) ───────────────────────────────────
    df_yr = df_t.dropna(subset=["v_year"])
    if len(df_yr) > 0:
        yearly = df_yr.groupby("v_year").agg(
            total_reviews = ("star","count"),
            avg_star      = ("star","mean"),
            pct_positive  = ("sentiment", lambda x: (x=="Positive").mean()*100),
            pct_negative  = ("sentiment", lambda x: (x=="Negative").mean()*100),
        ).reset_index().sort_values("v_year")
        yearly = yearly.round(2)
        results["yearly"] = yearly
        print(f"  ✔ Xu hướng theo năm: {len(yearly)} năm ({int(yearly['v_year'].min())}–{int(yearly['v_year'].max())})")

    # ── A3: Xu hướng theo quý ─────────────────────────────────────────────
    quarterly = df_t.groupby("v_quarter").agg(
        total_reviews = ("star","count"),
        avg_star      = ("star","mean"),
        pct_positive  = ("sentiment", lambda x: (x=="Positive").mean()*100),
        pct_negative  = ("sentiment", lambda x: (x=="Negative").mean()*100),
    ).reset_index().sort_values("v_quarter")
    quarterly = quarterly.round(2)
    results["quarterly"] = quarterly
    print(f"  ✔ Xu hướng theo quý: {quarterly['v_quarter'].tolist()}")

    # ── A4: Xu hướng theo tháng/năm của từng khách sạn (top 5) ───────────
    if "hotel_name" in df_yr.columns:
        top5 = df_yr["hotel_name"].value_counts().head(5).index.tolist()
        hotel_monthly = df_yr[df_yr["hotel_name"].isin(top5)].groupby(
            ["hotel_name","v_year","v_month"]
        ).agg(avg_star=("star","mean"), total=("star","count")).reset_index()
        hotel_monthly["period"] = (
            hotel_monthly["v_year"].astype(int).astype(str) + "-" +
            hotel_monthly["v_month"].astype(int).astype(str).str.zfill(2)
        )
        hotel_monthly = hotel_monthly.round(2)
        results["hotel_monthly_trend"] = hotel_monthly
        print(f"  ✔ Xu hướng theo tháng của top 5 khách sạn")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# MODULE B: BENCHMARKING
# ═════════════════════════════════════════════════════════════════════════════

def analyze_benchmarking(df: pd.DataFrame) -> dict:
    print(f"\n{'='*55}")
    print("  [3] BENCHMARKING SO SÁNH KHÁCH SẠN")
    print(f"{'='*55}")

    results = {}
    if "hotel_name" not in df.columns:
        print("  ⚠ Không có hotel_name")
        return results

    # Lọc KS có ít nhất 5 đánh giá
    counts = df["hotel_name"].value_counts()
    valid  = counts[counts >= 5].index
    df_b   = df[df["hotel_name"].isin(valid)].copy()
    print(f"  ✔ Khách sạn đủ điều kiện (≥5 đánh giá): {len(valid)}")

    # ── B1: Tổng quan KPI từng khách sạn ─────────────────────────────────
    overview = df_b.groupby("hotel_name").agg(
        total_reviews  = ("star","count"),
        avg_star       = ("star","mean"),
        median_star    = ("star","median"),
        star_5_pct     = ("star", lambda x: (x==5).mean()*100),
        star_1_pct     = ("star", lambda x: (x==1).mean()*100),
        pct_positive   = ("sentiment", lambda x: (x=="Positive").mean()*100),
        pct_negative   = ("sentiment", lambda x: (x=="Negative").mean()*100),
        pct_neutral    = ("sentiment", lambda x: (x=="Neutral").mean()*100),
    ).reset_index()

    # Composite Score = trọng số tổng hợp
    overview["composite_score"] = (
        overview["avg_star"] * 15 +
        overview["pct_positive"] * 0.4 +
        (100 - overview["pct_negative"]) * 0.25 +
        overview["star_5_pct"] * 0.15
    ).round(2)

    overview = overview.round(2)
    overview = overview.sort_values("composite_score", ascending=False)
    overview["rank"] = range(1, len(overview)+1)
    results["overview"] = overview
    print(f"\n  🏆 TOP 10 KHÁCH SẠN:")
    cols = ["rank","hotel_name","total_reviews","avg_star","pct_positive","composite_score"]
    print(overview[cols].head(10).to_string(index=False))

    # ── B2: Phân phối sao từng khách sạn ─────────────────────────────────
    star_dist = (
        df_b.groupby(["hotel_name","star"]).size()
        .reset_index(name="count")
        .pivot(index="hotel_name", columns="star", values="count")
        .fillna(0).astype(int).reset_index()
    )
    star_dist.columns = [f"star_{int(c)}" if c != "hotel_name" else c
                         for c in star_dist.columns]
    results["star_distribution"] = star_dist

    # ── B3: Xu hướng chất lượng theo thời gian từng khách sạn ────────────
    df_bt = df_b.dropna(subset=["v_month"]).copy()
    hotel_season = df_bt.groupby(["hotel_name","v_month"]).agg(
        avg_star     = ("star","mean"),
        total        = ("star","count"),
        pct_positive = ("sentiment", lambda x: (x=="Positive").mean()*100),
    ).reset_index().round(2)
    hotel_season["month_name"] = hotel_season["v_month"].map(MONTH_VI)
    results["hotel_seasonal"] = hotel_season

    # ── B4: Aspect scores (từ Nhóm 2 nếu có, nếu không thì bỏ qua) ───────
    aspect_cols = [c for c in df_b.columns if c.startswith("aspect_")]
    if aspect_cols:
        aspect_bench = df_b.groupby("hotel_name")[aspect_cols].mean().round(2).reset_index()
        results["aspect_scores"] = aspect_bench
        print(f"  ✔ Aspect scores: {aspect_cols}")
    else:
        print("  ℹ Chưa có aspect scores từ Nhóm 2")

    # ── B5: Thống kê tóm tắt ─────────────────────────────────────────────
    best  = overview.iloc[0]
    worst = overview.iloc[-1]
    avg_all = round(float(df_b["star"].mean()), 2)
    print(f"\n  📊 Tóm tắt Benchmarking:")
    print(f"     Tổng reviews phân tích : {len(df_b):,}")
    print(f"     Avg star toàn vùng     : {avg_all}")
    print(f"     Khách sạn tốt nhất     : {best['hotel_name']} ({best['avg_star']}⭐)")
    print(f"     Khách sạn cần cải thiện: {worst['hotel_name']} ({worst['avg_star']}⭐)")

    results["summary"] = {
        "total_hotels": len(valid),
        "total_reviews": len(df_b),
        "overall_avg_star": avg_all,
        "best_hotel": {"name": best["hotel_name"], "avg_star": float(best["avg_star"]),
                       "composite_score": float(best["composite_score"])},
        "worst_hotel": {"name": worst["hotel_name"], "avg_star": float(worst["avg_star"]),
                        "composite_score": float(worst["composite_score"])},
        "overall_pct_positive": round(float((df_b["sentiment"]=="Positive").mean()*100), 1),
        "overall_pct_negative": round(float((df_b["sentiment"]=="Negative").mean()*100), 1),
    }

    return results


# ═════════════════════════════════════════════════════════════════════════════
# XUẤT KẾT QUẢ
# ═════════════════════════════════════════════════════════════════════════════

def export_all(df, trend_res, bench_res, out_dir: Path):
    print(f"\n{'='*55}")
    print(f"  [4] XUẤT FILE → {out_dir}")
    print(f"{'='*55}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSVs
    _csv(df, out_dir / "01_data_clean.csv")

    for key, fname in [
        ("seasonal",           "02_trend_seasonal.csv"),
        ("yearly",             "03_trend_yearly.csv"),
        ("quarterly",          "04_trend_quarterly.csv"),
        ("hotel_monthly_trend","05_trend_hotel_monthly.csv"),
    ]:
        if key in trend_res:
            _csv(trend_res[key], out_dir / fname)

    for key, fname in [
        ("overview",          "06_bench_overview.csv"),
        ("star_distribution", "07_bench_star_dist.csv"),
        ("hotel_seasonal",    "08_bench_hotel_seasonal.csv"),
        ("aspect_scores",     "09_bench_aspect_scores.csv"),
    ]:
        if key in bench_res:
            _csv(bench_res[key], out_dir / fname)

    # JSON tổng hợp cho Dashboard
    dashboard_data = {
        "generated_at": datetime.now().isoformat(),
        "summary": bench_res.get("summary", {}),
        "seasonal_trend": _df_to_records(trend_res.get("seasonal")),
        "yearly_trend": _df_to_records(trend_res.get("yearly")),
        "quarterly_trend": _df_to_records(trend_res.get("quarterly")),
        "hotel_overview": _df_to_records(bench_res.get("overview", pd.DataFrame()).head(20)),
        "hotel_star_dist": _df_to_records(bench_res.get("star_distribution")),
        "hotel_seasonal": _df_to_records(bench_res.get("hotel_seasonal")),
    }
    json_path = out_dir / "dashboard_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  ✔ dashboard_data.json  ← dùng cho Dashboard HTML")

    print(f"\n  ✅ Xong! Tất cả file tại: {out_dir.resolve()}")
    for fp in sorted(out_dir.iterdir()):
        kb = fp.stat().st_size / 1024
        print(f"     {fp.name:<40} {kb:>7.1f} KB")


def _csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  ✔ {path.name}  ({len(df):,} rows)")


def _df_to_records(df):
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return []
    return df.to_dict(orient="records")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Nhóm 3 – Xu hướng & Benchmarking")
    parser.add_argument("--input",  required=True, help="File JSON đầu vào")
    parser.add_argument("--output", default="./nhom3_output", help="Thư mục output")
    parser.add_argument("--absa",   default=None, help="File output Nhóm 2 (aspect scores)")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output)

    print("\n" + "█"*55)
    print("  NHÓM 3 – XU HƯỚNG & BENCHMARKING KHÁCH SẠN HUẾ")
    print("█"*55)

    df = load_and_clean(args.input)

    # Tích hợp aspect từ Nhóm 2 nếu có
    if args.absa and Path(args.absa).exists():
        try:
            absa_df = pd.read_csv(args.absa) if args.absa.endswith(".csv") \
                      else pd.read_json(args.absa, lines=True)
            key = next((c for c in ["reviewer_id","url"] if c in df.columns and c in absa_df.columns), None)
            if key:
                aspect_cols = [c for c in absa_df.columns if c.startswith("aspect_")]
                df = df.merge(absa_df[[key]+aspect_cols], on=key, how="left")
                print(f"  ✔ Tích hợp Nhóm 2: {aspect_cols}")
        except Exception as e:
            print(f"  ⚠ Không đọc được file Nhóm 2: {e}")

    trend_res = analyze_trends(df)
    bench_res = analyze_benchmarking(df)
    export_all(df, trend_res, bench_res, OUTPUT_DIR)


if __name__ == "__main__":
    main()
