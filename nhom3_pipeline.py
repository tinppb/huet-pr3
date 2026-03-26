"""
=============================================================
  NHÓM 3 – TỔNG HỢP TRI THỨC & TRỰC QUAN HÓA
  Pipeline Phân tích Đánh giá Khách sạn Thừa Thiên Huế
  Tác giả: Phạm Phước Bảo Tín
=============================================================

LUỒNG CHẠY:
  1. Load & kiểm tra dữ liệu thô
  2. Làm sạch & chuẩn hóa
  3. Tích hợp output từ Nhóm 1 (sentiment) & Nhóm 2 (ABSA)
  4. Phân tích KPI theo 3 hướng:
       A. Xu hướng đánh giá theo thời gian
       B. Benchmarking so sánh khách sạn
       C. Phân tích theo trip_type
  5. Xuất kết quả (CSV / JSON) cho Dashboard

CÁCH DÙNG:
  pip install pandas numpy matplotlib seaborn plotly
  python nhom3_pipeline.py --input data.json --output ./output/
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
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
PROVINCE_FILTER = "Thua Thien Hue"   # Lọc theo tỉnh mục tiêu (để trống = lấy tất cả)
MIN_REVIEWS_BENCHMARK = 5            # Khách sạn phải có ít nhất N đánh giá mới so sánh
OUTPUT_DIR = Path("./nhom3_output")

# Mapping tháng viết tắt tiếng Anh → số
MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

# Ánh xạ sentiment từ Nhóm 1 (nếu chưa có thì tự suy từ số sao)
STAR_TO_SENTIMENT = {
    5: "Tích cực",
    4: "Tích cực",
    3: "Trung lập",
    2: "Tiêu cực",
    1: "Tiêu cực"
}


# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD & KIỂM TRA DỮ LIỆU
# ═════════════════════════════════════════════════════════════════════════════

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load file JSON (mỗi dòng là 1 object JSON, hoặc 1 mảng JSON lớn).
    Trả về DataFrame thô chưa xử lý.
    """
    filepath = Path(filepath)
    print(f"\n{'='*60}")
    print(f"  [1/5] LOAD DỮ LIỆU: {filepath.name}")
    print(f"{'='*60}")

    if not filepath.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    ext = filepath.suffix.lower()

    if ext == ".json":
        try:
            # Thử đọc JSON array
            with open(filepath, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                df = pd.DataFrame(raw)
            else:
                df = pd.DataFrame([raw])
        except json.JSONDecodeError:
            # Thử đọc JSONL (mỗi dòng là 1 object)
            df = pd.read_json(filepath, lines=True)

    elif ext == ".csv":
        df = pd.read_csv(filepath, encoding="utf-8")
    else:
        raise ValueError(f"Định dạng file không hỗ trợ: {ext}")

    print(f"  ✔ Đọc thành công: {len(df):,} records, {df.shape[1]} cột")
    print(f"  Các cột: {list(df.columns)}")
    _print_missing_report(df)
    return df


def _print_missing_report(df: pd.DataFrame):
    """In báo cáo giá trị thiếu."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("\n  ⚠ Giá trị thiếu:")
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            print(f"     {col:<20} {cnt:>6,} ({pct:.1f}%)")
    else:
        print("  ✔ Không có giá trị thiếu")


# ═════════════════════════════════════════════════════════════════════════════
# 2. LÀM SẠCH & CHUẨN HÓA
# ═════════════════════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch toàn bộ DataFrame:
      - Chuẩn hóa tên cột
      - Xử lý visit_date → year, month, quarter
      - Làm sạch text (comment, title)
      - Chuẩn hóa star, trip_type, province
      - Trích xuất tên khách sạn từ URL
    """
    print(f"\n{'='*60}")
    print(f"  [2/5] LÀM SẠCH & CHUẨN HÓA")
    print(f"{'='*60}")

    df = df.copy()

    # ── 2.1 Tên cột ────────────────────────────────────────────────────────
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ── 2.2 Trích xuất tên khách sạn từ URL ───────────────────────────────
    if "url" in df.columns:
        df["hotel_name"] = df["url"].apply(_extract_hotel_name)
        print(f"  ✔ Trích xuất được {df['hotel_name'].nunique()} khách sạn")

    # ── 2.3 Chuẩn hóa visit_date → year, month, quarter ───────────────────
    if "visit_date" in df.columns:
        parsed = df["visit_date"].apply(_parse_visit_date)
        df["visit_month"] = parsed.apply(lambda x: x[0])
        df["visit_year"]  = parsed.apply(lambda x: x[1])
        df["visit_quarter"] = df["visit_month"].apply(
            lambda m: f"Q{((m-1)//3)+1}" if pd.notna(m) else None
        )
        valid_dates = df["visit_month"].notna().sum()
        print(f"  ✔ Chuẩn hóa visit_date: {valid_dates:,}/{len(df):,} records hợp lệ")

    # ── 2.4 Chuẩn hóa star ────────────────────────────────────────────────
    if "star" in df.columns:
        df["star"] = pd.to_numeric(df["star"], errors="coerce")
        df["star"] = df["star"].clip(1, 5)
        print(f"  ✔ star — trung bình: {df['star'].mean():.2f}, "
              f"null: {df['star'].isna().sum()}")

    # ── 2.5 Chuẩn hóa trip_type ───────────────────────────────────────────
    if "trip_type" in df.columns:
        df["trip_type"] = df["trip_type"].apply(_normalize_trip_type)
        dist = df["trip_type"].value_counts(dropna=False).to_dict()
        print(f"  ✔ trip_type phân bố: {dist}")

    # ── 2.6 Làm sạch text ─────────────────────────────────────────────────
    for col in ["comment", "title"]:
        if col in df.columns:
            df[col] = df[col].apply(_clean_text)

    # ── 2.7 Chuẩn hóa province ────────────────────────────────────────────
    if "province" in df.columns:
        df["province"] = df["province"].str.strip()

    # ── 2.8 Lọc theo tỉnh mục tiêu ────────────────────────────────────────
    if PROVINCE_FILTER and "province" in df.columns:
        before = len(df)
        df = df[df["province"].str.contains(PROVINCE_FILTER, case=False, na=False)]
        print(f"  ✔ Lọc tỉnh '{PROVINCE_FILTER}': {before:,} → {len(df):,} records")

    # ── 2.9 Loại bỏ duplicate ─────────────────────────────────────────────
    dup_cols = [c for c in ["reviewer_id", "hotel_name", "comment"] if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=dup_cols)
    print(f"  ✔ Loại duplicate: {before - len(df)} records bị xóa")

    print(f"\n  → Dữ liệu sạch: {len(df):,} records")
    return df.reset_index(drop=True)


def _extract_hotel_name(url: str) -> str:
    """Trích xuất tên khách sạn từ URL TripAdvisor."""
    if not isinstance(url, str):
        return "Unknown"
    # Ví dụ: .../Reviews-Banyan_Tree_Lang_Co-Cu_Du...
    match = re.search(r"Reviews-([^-]+(?:_[^-]+)*)-[A-Z]", url)
    if match:
        name = match.group(1).replace("_", " ")
        return name
    # Fallback: lấy đoạn cuối URL
    parts = url.rstrip("/").split("/")
    for part in reversed(parts):
        if len(part) > 5 and "Reviews" not in part:
            return part.replace("_", " ").replace("-", " ")
    return "Unknown"


def _parse_visit_date(raw) -> tuple:
    """
    Chuyển đổi nhiều định dạng ngày:
      '19-Sep', 'Sep 2023', 'September 2023', '2023-09', v.v.
    Trả về (month: int, year: int) hoặc (None, None).
    """
    if not isinstance(raw, str) or not raw.strip():
        return (None, None)

    raw = raw.strip()

    # Dạng: "19-Sep", "Sep-19", "Sep 2023", "September 2023"
    for pattern, fmt in [
        (r"^\d{1,2}-([A-Za-z]{3})$", None),       # 19-Sep → chỉ có tháng
        (r"^([A-Za-z]{3})-\d{1,2}$", None),        # Sep-19 → chỉ có tháng
        (r"^([A-Za-z]{3,9})\s+(\d{4})$", "%B %Y"), # Sep 2023
        (r"^(\d{4})-(\d{2})$", "%Y-%m"),            # 2023-09
        (r"^(\d{2})/(\d{4})$", "%m/%Y"),            # 09/2023
    ]:
        m = re.match(pattern, raw)
        if m:
            try:
                if fmt:
                    dt = datetime.strptime(raw, fmt)
                    return (dt.month, dt.year)
                else:
                    # Chỉ có tháng, không có năm
                    month_str = re.search(r"[A-Za-z]+", raw).group().lower()[:3]
                    return (MONTH_MAP.get(month_str), None)
            except Exception:
                pass

    # Thử parse trực tiếp
    try:
        dt = pd.to_datetime(raw, dayfirst=True)
        return (dt.month, dt.year)
    except Exception:
        return (None, None)


def _normalize_trip_type(val) -> str:
    """Chuẩn hóa trip_type về các nhóm chuẩn."""
    if pd.isna(val) or str(val).strip() in ("", "None", "null"):
        return "Unknown"
    val = str(val).strip().lower()
    if any(k in val for k in ["family", "gia đình"]):
        return "Family"
    if any(k in val for k in ["solo", "một mình", "single"]):
        return "Solo"
    if any(k in val for k in ["couple", "cặp đôi", "romance"]):
        return "Couple"
    if any(k in val for k in ["business", "công tác", "work"]):
        return "Business"
    if any(k in val for k in ["friend", "bạn bè", "group"]):
        return "Friends"
    return val.capitalize()


def _clean_text(text) -> str:
    """Làm sạch văn bản cơ bản (giữ nội dung, loại ký tự lạ)."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Loại ký tự điều khiển (giữ newline)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Loại HTML entities cơ bản
    text = re.sub(r"&[a-z]+;", " ", text)
    return text.strip()


# ═════════════════════════════════════════════════════════════════════════════
# 3. TÍCH HỢP OUTPUT TỪ NHÓM 1 & NHÓM 2
# ═════════════════════════════════════════════════════════════════════════════

def integrate_group_outputs(
    df: pd.DataFrame,
    sentiment_file: str = None,
    absa_file: str = None
) -> pd.DataFrame:
    """
    Tích hợp kết quả từ:
      - Nhóm 1: cột 'sentiment' (Tích cực / Tiêu cực / Trung lập)
               và 'is_contradicted' (bình luận mâu thuẫn với số sao)
      - Nhóm 2: các cột aspect như 'aspect_location', 'aspect_service', v.v.

    Nếu chưa có file output từ các nhóm, tự sinh giá trị mặc định từ 'star'.
    """
    print(f"\n{'='*60}")
    print(f"  [3/5] TÍCH HỢP OUTPUT NHÓM 1 & NHÓM 2")
    print(f"{'='*60}")

    df = df.copy()

    # ── 3.1 Tích hợp Nhóm 1 (Sentiment) ──────────────────────────────────
    if sentiment_file and Path(sentiment_file).exists():
        print(f"  ← Đọc Nhóm 1: {sentiment_file}")
        s_df = _load_side_file(sentiment_file)
        key_cols = _find_join_key(df, s_df)
        if key_cols:
            df = df.merge(
                s_df[key_cols + [c for c in s_df.columns if c not in df.columns]],
                on=key_cols, how="left"
            )
            print(f"  ✔ Đã join Nhóm 1 theo: {key_cols}")
        else:
            print("  ⚠ Không tìm thấy key chung với Nhóm 1 — dùng fallback")
            _apply_sentiment_fallback(df)
    else:
        print("  ℹ Chưa có file Nhóm 1 — sinh sentiment từ số sao (fallback)")
        _apply_sentiment_fallback(df)

    # ── 3.2 Tích hợp Nhóm 2 (ABSA) ───────────────────────────────────────
    if absa_file and Path(absa_file).exists():
        print(f"  ← Đọc Nhóm 2: {absa_file}")
        a_df = _load_side_file(absa_file)
        key_cols = _find_join_key(df, a_df)
        if key_cols:
            df = df.merge(
                a_df[key_cols + [c for c in a_df.columns if c not in df.columns]],
                on=key_cols, how="left"
            )
            print(f"  ✔ Đã join Nhóm 2 theo: {key_cols}")
        else:
            print("  ⚠ Không tìm thấy key chung với Nhóm 2 — bỏ qua ABSA")
    else:
        print("  ℹ Chưa có file Nhóm 2 — các cột aspect sẽ để trống")

    print(f"  → Cột hiện tại: {list(df.columns)}")
    return df


def _apply_sentiment_fallback(df: pd.DataFrame):
    """Sinh cột sentiment và is_contradicted từ số sao."""
    if "star" in df.columns:
        df["sentiment"] = df["star"].map(STAR_TO_SENTIMENT).fillna("Trung lập")
        # Phát hiện mâu thuẫn: ví dụ 5 sao nhưng Nhóm 1 cho là Tiêu cực
        # Ở đây chỉ đánh dấu placeholder
        df["is_contradicted"] = False
    else:
        df["sentiment"] = "Trung lập"
        df["is_contradicted"] = False


def _load_side_file(filepath: str) -> pd.DataFrame:
    fp = Path(filepath)
    if fp.suffix == ".csv":
        return pd.read_csv(fp)
    return pd.read_json(fp, lines=True)


def _find_join_key(df1: pd.DataFrame, df2: pd.DataFrame) -> list:
    """Tìm cột chung có thể dùng làm join key."""
    candidates = ["reviewer_id", "url", "comment", "id"]
    common = [c for c in candidates if c in df1.columns and c in df2.columns]
    return common[:1]  # Dùng key đầu tiên tìm được


# ═════════════════════════════════════════════════════════════════════════════
# 4A. PHÂN TÍCH XU HƯỚNG THEO THỜI GIAN
# ═════════════════════════════════════════════════════════════════════════════

def analyze_time_trends(df: pd.DataFrame) -> dict:
    """
    Phân tích xu hướng đánh giá theo thời gian.
    Trả về dict chứa các DataFrame kết quả.
    """
    print(f"\n{'='*60}")
    print(f"  [4A] PHÂN TÍCH XU HƯỚNG THEO THỜI GIAN")
    print(f"{'='*60}")

    results = {}

    # Lọc records có thông tin tháng
    df_time = df.dropna(subset=["visit_month"]).copy()
    print(f"  Records có visit_month: {len(df_time):,}")

    # ── Theo tháng (nếu có năm) ───────────────────────────────────────────
    if "visit_year" in df_time.columns and df_time["visit_year"].notna().sum() > 0:
        df_ym = df_time.dropna(subset=["visit_year"]).copy()
        df_ym["year_month"] = (
            df_ym["visit_year"].astype(int).astype(str) + "-" +
            df_ym["visit_month"].astype(int).astype(str).str.zfill(2)
        )
        monthly = df_ym.groupby("year_month").agg(
            total_reviews=("star", "count"),
            avg_star=("star", "mean"),
            pct_positive=("sentiment", lambda x: (x == "Tích cực").mean() * 100),
            pct_negative=("sentiment", lambda x: (x == "Tiêu cực").mean() * 100),
        ).reset_index().sort_values("year_month")
        monthly["avg_star"] = monthly["avg_star"].round(2)
        monthly["pct_positive"] = monthly["pct_positive"].round(1)
        monthly["pct_negative"] = monthly["pct_negative"].round(1)
        results["monthly_trend"] = monthly
        print(f"  ✔ Xu hướng theo tháng: {len(monthly)} điểm dữ liệu")

    # ── Theo tháng trong năm (mùa vụ) ─────────────────────────────────────
    seasonal = df_time.groupby("visit_month").agg(
        total_reviews=("star", "count"),
        avg_star=("star", "mean"),
        pct_positive=("sentiment", lambda x: (x == "Tích cực").mean() * 100),
    ).reset_index().sort_values("visit_month")
    seasonal["month_name"] = seasonal["visit_month"].map({
        1:"Tháng 1", 2:"Tháng 2", 3:"Tháng 3", 4:"Tháng 4",
        5:"Tháng 5", 6:"Tháng 6", 7:"Tháng 7", 8:"Tháng 8",
        9:"Tháng 9", 10:"Tháng 10", 11:"Tháng 11", 12:"Tháng 12"
    })
    seasonal["avg_star"] = seasonal["avg_star"].round(2)
    seasonal["pct_positive"] = seasonal["pct_positive"].round(1)
    results["seasonal_trend"] = seasonal
    print(f"  ✔ Xu hướng mùa vụ: {len(seasonal)} tháng")

    # ── Theo quý ──────────────────────────────────────────────────────────
    quarterly = df_time.groupby("visit_quarter").agg(
        total_reviews=("star", "count"),
        avg_star=("star", "mean"),
        pct_positive=("sentiment", lambda x: (x == "Tích cực").mean() * 100),
        pct_negative=("sentiment", lambda x: (x == "Tiêu cực").mean() * 100),
    ).reset_index().sort_values("visit_quarter")
    results["quarterly_trend"] = quarterly
    print(f"  ✔ Xu hướng theo quý: {len(quarterly)} quý")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 4B. BENCHMARKING SO SÁNH KHÁCH SẠN
# ═════════════════════════════════════════════════════════════════════════════

def analyze_benchmarking(df: pd.DataFrame) -> dict:
    """
    So sánh hiệu quả hoạt động giữa các khách sạn.
    Trả về dict chứa các DataFrame kết quả.
    """
    print(f"\n{'='*60}")
    print(f"  [4B] BENCHMARKING SO SÁNH KHÁCH SẠN")
    print(f"{'='*60}")

    results = {}

    if "hotel_name" not in df.columns:
        print("  ⚠ Không có cột hotel_name — bỏ qua benchmarking")
        return results

    # Lọc khách sạn đủ đánh giá
    hotel_counts = df["hotel_name"].value_counts()
    valid_hotels = hotel_counts[hotel_counts >= MIN_REVIEWS_BENCHMARK].index
    df_bench = df[df["hotel_name"].isin(valid_hotels)].copy()
    print(f"  Khách sạn đủ điều kiện (≥{MIN_REVIEWS_BENCHMARK} đánh giá): "
          f"{len(valid_hotels)}/{df['hotel_name'].nunique()}")

    # ── Tổng quan từng khách sạn ──────────────────────────────────────────
    agg_dict = {
        "total_reviews": ("star", "count"),
        "avg_star":       ("star", "mean"),
        "pct_positive":   ("sentiment", lambda x: (x == "Tích cực").mean() * 100),
        "pct_negative":   ("sentiment", lambda x: (x == "Tiêu cực").mean() * 100),
        "pct_neutral":    ("sentiment", lambda x: (x == "Trung lập").mean() * 100),
        "contradictions": ("is_contradicted", "sum"),
    }
    hotel_overview = df_bench.groupby("hotel_name").agg(**agg_dict).reset_index()
    for col in ["avg_star", "pct_positive", "pct_negative", "pct_neutral"]:
        hotel_overview[col] = hotel_overview[col].round(2)

    # Xếp hạng tổng hợp (composite score)
    hotel_overview["composite_score"] = (
        hotel_overview["avg_star"] * 20 +          # 0–100
        hotel_overview["pct_positive"] * 0.3 +      # trọng số 30%
        (100 - hotel_overview["pct_negative"]) * 0.2 # trọng số 20%
    ).round(2)
    hotel_overview = hotel_overview.sort_values("composite_score", ascending=False)
    hotel_overview["rank"] = range(1, len(hotel_overview) + 1)
    results["hotel_overview"] = hotel_overview
    print(f"  ✔ Tổng quan {len(hotel_overview)} khách sạn")
    print(f"\n  Top 5 khách sạn:")
    cols_show = ["rank", "hotel_name", "total_reviews", "avg_star", "composite_score"]
    print(hotel_overview[cols_show].head().to_string(index=False))

    # ── Phân phối sao theo khách sạn ──────────────────────────────────────
    star_dist = (
        df_bench.groupby(["hotel_name", "star"])
        .size()
        .reset_index(name="count")
        .pivot(index="hotel_name", columns="star", values="count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    results["hotel_star_distribution"] = star_dist

    # ── Aspect scores (nếu Nhóm 2 đã cung cấp) ───────────────────────────
    aspect_cols = [c for c in df_bench.columns if c.startswith("aspect_")]
    if aspect_cols:
        aspect_scores = df_bench.groupby("hotel_name")[aspect_cols].mean().round(2).reset_index()
        results["hotel_aspect_scores"] = aspect_scores
        print(f"  ✔ Aspect scores: {aspect_cols}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 4C. PHÂN TÍCH THEO TRIP_TYPE
# ═════════════════════════════════════════════════════════════════════════════

def analyze_trip_type(df: pd.DataFrame) -> dict:
    """
    Phân tích sự khác biệt trong ưu tiên của từng loại khách.
    Trả về dict chứa các DataFrame kết quả.
    """
    print(f"\n{'='*60}")
    print(f"  [4C] PHÂN TÍCH THEO TRIP_TYPE")
    print(f"{'='*60}")

    results = {}

    if "trip_type" not in df.columns:
        print("  ⚠ Không có cột trip_type — bỏ qua")
        return results

    df_tt = df[df["trip_type"] != "Unknown"].copy()
    print(f"  Records có trip_type hợp lệ: {len(df_tt):,}")

    # ── Tổng quan theo trip_type ──────────────────────────────────────────
    tt_overview = df_tt.groupby("trip_type").agg(
        total_reviews=("star", "count"),
        avg_star=("star", "mean"),
        pct_positive=("sentiment", lambda x: (x == "Tích cực").mean() * 100),
        pct_negative=("sentiment", lambda x: (x == "Tiêu cực").mean() * 100),
    ).reset_index()
    tt_overview["avg_star"] = tt_overview["avg_star"].round(2)
    tt_overview["pct_positive"] = tt_overview["pct_positive"].round(1)
    tt_overview["pct_negative"] = tt_overview["pct_negative"].round(1)
    tt_overview = tt_overview.sort_values("avg_star", ascending=False)
    results["triptype_overview"] = tt_overview
    print(f"\n  Tổng quan theo trip_type:")
    print(tt_overview.to_string(index=False))

    # ── Phân phối sao theo trip_type ──────────────────────────────────────
    star_by_tt = (
        df_tt.groupby(["trip_type", "star"])
        .size()
        .reset_index(name="count")
    )
    star_by_tt["pct"] = (
        star_by_tt.groupby("trip_type")["count"]
        .transform(lambda x: x / x.sum() * 100)
        .round(1)
    )
    results["triptype_star_dist"] = star_by_tt

    # ── Aspect theo trip_type (nếu có) ────────────────────────────────────
    aspect_cols = [c for c in df_tt.columns if c.startswith("aspect_")]
    if aspect_cols:
        aspect_tt = df_tt.groupby("trip_type")[aspect_cols].mean().round(2).reset_index()
        results["triptype_aspect"] = aspect_tt
        print(f"  ✔ Aspect theo trip_type: {aspect_cols}")

    # ── Khách sạn được yêu thích nhất theo từng loại khách ────────────────
    if "hotel_name" in df_tt.columns:
        fav_hotels = (
            df_tt.groupby(["trip_type", "hotel_name"])
            .agg(reviews=("star", "count"), avg_star=("star", "mean"))
            .reset_index()
        )
        fav_hotels = fav_hotels[fav_hotels["reviews"] >= 3]
        fav_hotels = (
            fav_hotels.sort_values("avg_star", ascending=False)
            .groupby("trip_type")
            .head(3)
            .reset_index(drop=True)
        )
        fav_hotels["avg_star"] = fav_hotels["avg_star"].round(2)
        results["triptype_fav_hotels"] = fav_hotels
        print(f"  ✔ Khách sạn yêu thích theo trip_type")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 5. XUẤT KẾT QUẢ
# ═════════════════════════════════════════════════════════════════════════════

def export_results(
    df_clean: pd.DataFrame,
    time_results: dict,
    bench_results: dict,
    triptype_results: dict,
    output_dir: Path
):
    """
    Xuất toàn bộ kết quả ra thư mục output:
      - data_clean.csv          : Dữ liệu đã làm sạch (đầy đủ)
      - dashboard_summary.json  : Tóm tắt KPI tổng quan cho Dashboard
      - trend_monthly.csv       : Xu hướng theo tháng
      - trend_seasonal.csv      : Xu hướng mùa vụ
      - trend_quarterly.csv     : Xu hướng theo quý
      - bench_overview.csv      : Tổng quan khách sạn
      - bench_star_dist.csv     : Phân phối sao từng khách sạn
      - triptype_overview.csv   : Tổng quan theo loại khách
      - triptype_star_dist.csv  : Phân phối sao theo loại khách
      - triptype_fav_hotels.csv : Khách sạn yêu thích theo loại khách
    """
    print(f"\n{'='*60}")
    print(f"  [5/5] XUẤT KẾT QUẢ → {output_dir}")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data sạch ─────────────────────────────────────────────────────────
    _save_csv(df_clean, output_dir / "data_clean.csv")

    # ── Xu hướng thời gian ────────────────────────────────────────────────
    for key in ["monthly_trend", "seasonal_trend", "quarterly_trend"]:
        if key in time_results:
            fname = key.replace("_trend", "").replace("monthly", "trend_monthly")
            fname = f"trend_{key.split('_')[0]}.csv"
            _save_csv(time_results[key], output_dir / fname)

    # ── Benchmarking ──────────────────────────────────────────────────────
    for key in ["hotel_overview", "hotel_star_distribution", "hotel_aspect_scores"]:
        if key in bench_results:
            fname = key.replace("hotel_", "bench_") + ".csv"
            _save_csv(bench_results[key], output_dir / fname)

    # ── Trip type ─────────────────────────────────────────────────────────
    for key in ["triptype_overview", "triptype_star_dist",
                "triptype_aspect", "triptype_fav_hotels"]:
        if key in triptype_results:
            _save_csv(triptype_results[key], output_dir / f"{key}.csv")

    # ── Dashboard summary JSON ─────────────────────────────────────────────
    summary = _build_dashboard_summary(df_clean, bench_results, triptype_results)
    summary_path = output_dir / "dashboard_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"  ✔ dashboard_summary.json")

    print(f"\n  ✅ Hoàn thành! Tất cả file đã lưu tại: {output_dir.resolve()}")
    _print_file_list(output_dir)


def _save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  ✔ {path.name} ({len(df):,} rows)")


def _build_dashboard_summary(df, bench, triptype) -> dict:
    """Tổng hợp KPI chính cho Dashboard."""
    summary = {
        "generated_at": datetime.now().isoformat(),
        "overview": {
            "total_reviews": len(df),
            "total_hotels": int(df["hotel_name"].nunique()) if "hotel_name" in df.columns else 0,
            "avg_star": round(float(df["star"].mean()), 2) if "star" in df.columns else None,
            "pct_positive": round(float((df["sentiment"] == "Tích cực").mean() * 100), 1)
                            if "sentiment" in df.columns else None,
            "pct_negative": round(float((df["sentiment"] == "Tiêu cực").mean() * 100), 1)
                            if "sentiment" in df.columns else None,
        },
        "top_hotels": [],
        "worst_hotels": [],
        "triptype_avg_star": {},
    }

    if "hotel_overview" in bench:
        ho = bench["hotel_overview"]
        top = ho.head(5)[["hotel_name", "avg_star", "composite_score", "total_reviews"]]
        summary["top_hotels"] = top.to_dict(orient="records")
        worst = ho.tail(5)[["hotel_name", "avg_star", "composite_score", "total_reviews"]]
        summary["worst_hotels"] = worst.to_dict(orient="records")

    if "triptype_overview" in triptype:
        tt = triptype["triptype_overview"]
        summary["triptype_avg_star"] = dict(zip(tt["trip_type"], tt["avg_star"]))

    return summary


def _print_file_list(output_dir: Path):
    print(f"\n  📁 Files trong {output_dir.name}/:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"     {f.name:<40} {size_kb:>8.1f} KB")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global PROVINCE_FILTER, OUTPUT_DIR
    parser = argparse.ArgumentParser(
        description="Nhóm 3 – Pipeline phân tích đánh giá khách sạn"
    )
    parser.add_argument("--input",     required=True, help="File JSON hoặc CSV đầu vào")
    parser.add_argument("--output",    default="./nhom3_output", help="Thư mục output")
    parser.add_argument("--sentiment", default=None, help="File output Nhóm 1 (sentiment)")
    parser.add_argument("--absa",      default=None, help="File output Nhóm 2 (ABSA)")
    parser.add_argument("--province",  default=PROVINCE_FILTER, help="Lọc tỉnh (để trống = tất cả)")
    args = parser.parse_args()

    print("\n" + "█"*60)
    print("  NHÓM 3 – PIPELINE PHÂN TÍCH KHÁCH SẠN THỪA THIÊN HUẾ")
    print("█"*60)

    # Bước 1: Load
    df_raw = load_data(args.input)

    # Bước 2: Làm sạch
    df_clean = clean_data(df_raw)

    # Bước 3: Tích hợp Nhóm 1 & 2
    df_clean = integrate_group_outputs(df_clean, args.sentiment, args.absa)

    # Bước 4: Phân tích
    time_results    = analyze_time_trends(df_clean)
    bench_results   = analyze_benchmarking(df_clean)
    triptype_results = analyze_trip_type(df_clean)

    # Bước 5: Xuất kết quả
    export_results(df_clean, time_results, bench_results, triptype_results, OUTPUT_DIR)


if __name__ == "__main__":
    main()