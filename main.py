from data_preparation import load_and_prepare_data
from normalization import normalize_rfm, remove_outliers_iqr
from clustering import find_optimal_k, plot_elbow, apply_kmeans
from segment_analysis import analyze_segments
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd

print("=== 1. VERİ YÜKLEME ===")
# nrows=None → tüm veri (541K satır, ~4K müşteri)
rfm = load_and_prepare_data("online_retail.csv")
print(f"Toplam müşteri (temizlik sonrası): {rfm.shape[0]}")
print(rfm.describe().round(2))

print("\n=== 2. OUTLIER TEMİZLEME ===")
rfm_clean = remove_outliers_iqr(rfm, multiplier=3.0)
print(f"Outlier sonrası müşteri: {rfm_clean.shape[0]}")

print("\n=== 3. NORMALİZASYON ===")
rfm_scaled_df, scaler = normalize_rfm(rfm_clean)
print("Her kolonda mean≈0, std≈1 olmalı:")
print(rfm_scaled_df.describe().round(3))

print("\n=== 4. OPTİMAL K BULMA ===")
results, best_k = find_optimal_k(rfm_scaled_df, max_k=10)
plot_elbow(results)

print(f"\n=== 5. K-MEANS (k={best_k}) ===")
labels, model = apply_kmeans(rfm_scaled_df, k=best_k)
rfm_clean["Cluster"] = labels

print("\n=== 6. SEGMENT ANALİZİ ===")
segment_stats = analyze_segments(rfm_clean)
print(segment_stats.to_string())

print("\n=== 7. KÜME DAĞILIMI ===")
print(rfm_clean["Cluster"].value_counts().sort_index())

print("\n=== 8. DOĞRULAMA ===")

# Davies-Bouldin: düşük = iyi (kümeler birbirine karışmıyor)
db_score = davies_bouldin_score(rfm_scaled_df, labels)
print(f"Davies-Bouldin Score: {db_score:.4f}  (düşük = iyi, 0'a yakın mükemmel)")

# Stabilite: 5 farklı başlangıçla silhouette aynı mı?
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
print("\nStabilite testi (5 farklı random_state):")
for seed in [0, 1, 42, 99, 123]:
    lbl = KMeans(n_clusters=best_k, random_state=seed, n_init=10).fit_predict(rfm_scaled_df)
    sil = silhouette_score(rfm_scaled_df, lbl)
    print(f"  seed={seed:>3}: silhouette={sil:.4f}")

# 3D Scatter — ham RFM değerleriyle (normalize değil, okunması kolay)
segment_names = analyze_segments(rfm_clean)["Segment_Adi"].to_dict()
rfm_clean["Segment"] = rfm_clean["Cluster"].map(segment_names)

fig_3d = px.scatter_3d(
    rfm_clean,
    x="Recency", y="Frequency", z="Monetary",
    color="Segment",
    title="RFM Kümeleme — 3D Görselleştirme",
    labels={"Recency": "Recency (gün)", "Frequency": "Fatura Sayısı", "Monetary": "Toplam Harcama (£)"},
    opacity=0.7
)
fig_3d.show()

# PCA — 3 boyutu 2'ye indir, kümeler görsel olarak ayrışıyor mu?
pca = PCA(n_components=2)
coords = pca.fit_transform(rfm_scaled_df)
pca_df = pd.DataFrame({
    "PCA_1": coords[:, 0],
    "PCA_2": coords[:, 1],
    "Segment": rfm_clean["Segment"].values
})
explained = pca.explained_variance_ratio_ * 100

fig_pca = px.scatter(
    pca_df,
    x="PCA_1", y="PCA_2",
    color="Segment",
    title=f"PCA 2D — Varyansın %{explained[0]:.1f}+%{explained[1]:.1f}={sum(explained):.1f}'ini açıklıyor",
    opacity=0.6
)
fig_pca.show()
