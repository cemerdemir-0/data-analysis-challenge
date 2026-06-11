import os
import plotly.express as px
from data_preparation import load_and_prepare_data
from normalization import normalize_rfm, remove_outliers_iqr
from clustering import apply_kmeans
from segment_analysis import analyze_segments
from recommendation import compute_all_segment_insights
from time_series import compute_monthly_metrics

DATA_PATH = os.path.join(os.path.dirname(__file__), "online_retail.csv")
KMEANS_K = 3  # silhouette + elbow analizi ile doğrulandı

SEGMENT_COLORS = {
    "Şampiyonlar": "#2563EB",
    "Sadık Müşteriler": "#16A34A",
    "Risk Altındakiler": "#D4845A",
    "Kaybedilenler": "#9CA3AF",
}

_cache = {}


def get_pipeline_results():
    if _cache:
        return _cache

    print("[Pipeline] CSV yükleniyor...")
    raw_df, rfm = load_and_prepare_data(DATA_PATH, return_raw=True)

    print("[Pipeline] Outlier temizleniyor...")
    rfm_clean = remove_outliers_iqr(rfm)

    print("[Pipeline] Normalize ediliyor...")
    rfm_scaled, scaler = normalize_rfm(rfm_clean)

    print("[Pipeline] K-Means uygulanıyor...")
    labels, model = apply_kmeans(rfm_scaled, k=KMEANS_K)

    rfm_copy = rfm_clean.copy()
    rfm_copy["Cluster"] = labels
    stats = analyze_segments(rfm_copy)

    segment_labels = stats["Segment_Adi"].to_dict()
    rfm_copy["Segment"] = rfm_copy["Cluster"].map(segment_labels)

    # Tüm müşterileri (outlier dahil) eğitilen modelle classify et
    # Böylece arama tüm 4338 müşteriyi kapsar
    import pandas as pd
    full_scaled = scaler.transform(rfm[["Recency", "Frequency", "Monetary"]])
    full_labels = model.predict(full_scaled)
    rfm_full = rfm.copy()
    rfm_full["Cluster"] = full_labels
    rfm_full["Segment"] = rfm_full["Cluster"].map(segment_labels)

    print("[Pipeline] Segment insights hesaplanıyor...")
    raw_filtered = raw_df[raw_df["CustomerID"].isin(rfm_copy.index)]
    segment_insights = compute_all_segment_insights(raw_filtered, rfm_copy)

    _cache["rfm"] = rfm_copy          # istatistik + grafik için (temiz)
    _cache["rfm_full"] = rfm_full     # müşteri arama için (tüm)
    _cache["raw_df"] = raw_df         # tüm işlemler (öneri motoru için)
    _cache["stats"] = stats
    _cache["scaler"] = scaler
    _cache["model"] = model
    _cache["segment_insights"] = segment_insights
    print("[Pipeline] Zaman serisi hesaplanıyor...")
    ts_data = compute_monthly_metrics(raw_df, rfm_full)
    _cache["ts_data"] = ts_data

    print("[Pipeline] Hazır.")
    return _cache


def build_landing_chart(rfm_df):
    """Landing sayfası için minimal, marketing odaklı versiyon."""
    fig = px.scatter_3d(
        rfm_df.sample(min(800, len(rfm_df)), random_state=42),
        x="Recency", y="Frequency", z="Monetary",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        opacity=0.65,
        labels={"Recency": "R", "Frequency": "F", "Monetary": "M"},
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans", size=11, color="#ffffff"),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)", color="#888"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)", color="#888"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)", color="#888"),
        ),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def build_3d_chart(rfm_df):
    fig = px.scatter_3d(
        rfm_df,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        opacity=0.75,
        labels={
            "Recency": "Recency (gün)",
            "Frequency": "Fatura Sayısı",
            "Monetary": "Harcama (£)",
        },
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans", size=12),
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(
            bgcolor="rgba(242,237,228,0.95)",
            bordercolor="#0D0D0D",
            borderwidth=1,
            font=dict(size=12),
        ),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#ddd"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#ddd"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#ddd"),
        ),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")
