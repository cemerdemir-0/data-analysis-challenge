import numpy as np
import pandas as pd # Eksik olabilir, eklemekte fayda var
from data_preparation import load_and_prepare_data
from normalization import normalize_rfm
from clustering import plot_elbow, apply_kmeans
from segment_analysis import analyze_segments
from visualizations import plot_3d_clusters # En üste aldık

# 1. Veriyi Oku (30.000 satır ideal bir örneklem)
rfm = load_and_prepare_data("Online Retail.xlsx", nrows=30000)

# 🔥 LOG DÖNÜŞÜMÜ (Üç metrik için de uyguluyoruz)
rfm_log = rfm.copy()
rfm_log["Recency"] = np.log1p(rfm_log["Recency"]) # Bunu ekledik!
rfm_log["Frequency"] = np.log1p(rfm_log["Frequency"])
rfm_log["Monetary"] = np.log1p(rfm_log["Monetary"])

# 2. Normalize et (Artık rfm_log kullanarak uçurumları kapattık)
rfm_scaled_df = normalize_rfm(rfm_log)

# 3. K-Means ve Kümeleme
# plot_elbow(rfm_scaled_df) # K değerinden eminsen bunu yorum satırı yapabilirsin
rfm["Cluster"] = apply_kmeans(rfm_scaled_df, k=4)

# 4. Analiz ve Görselleştirme
print("\n📊 Segmentlerin ortalama RFM değerleri:")
print(analyze_segments(rfm)) # Analizi aktif ettik

print("\n🚀 3D Grafik oluşturuluyor...")
plot_3d_clusters(rfm)