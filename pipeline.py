import os
import plotly.express as px
from data_preparation import load_and_prepare_data
from normalization import normalize_rfm, remove_outliers_iqr
from clustering import apply_kmeans
from segment_analysis import analyze_segments

DATA_PATH = os.path.join(os.path.dirname(__file__), "online_retail.csv")
KMEANS_K = 3  # silhouette + elbow analizi ile doğrulandı

SEGMENT_COLORS = {
    "Şampiyonlar": "#2563EB",       # mavi — net, öne çıkıyor
    "Sadık Müşteriler": "#16A34A",  # yeşil — pozitif
    "Risk Altındakiler": "#D4845A", # turuncu — uyarı
    "Kaybedilenler": "#9CA3AF",     # gri
}

_cache = {}


def get_pipeline_results():
    if _cache:
        return _cache

    print("[Pipeline] CSV yükleniyor...")
    rfm = load_and_prepare_data(DATA_PATH)
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

    _cache["rfm"] = rfm_copy
    _cache["stats"] = stats
    _cache["scaler"] = scaler
    _cache["model"] = model
    print("[Pipeline] Hazır.")
    return _cache


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
