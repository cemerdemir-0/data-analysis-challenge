import pandas as pd


# Segment isimlerini otomatik ata:
# Recency düşük = yakın zamanda alışveriş (iyi)
# Frequency yüksek = çok alışveriş (iyi)
# Monetary yüksek = çok harcama (iyi)

def label_segments(segment_stats):
    """
    Her kümenin R/F/M ortalamalarına bakarak anlamlı isim ata.
    Strateji: ham RFM ortalamalarını rank'le, kombinasyondan isim üret.
    """
    stats = segment_stats.copy()

    # Rank: R için küçük = iyi (rank ascending), F ve M için büyük = iyi
    stats["R_rank"] = stats["Recency"].rank(ascending=True)   # düşük recency = iyi = rank 1
    stats["F_rank"] = stats["Frequency"].rank(ascending=False)
    stats["M_rank"] = stats["Monetary"].rank(ascending=False)
    stats["score"] = stats["R_rank"] + stats["F_rank"] + stats["M_rank"]

    n = len(stats)
    sorted_idx = stats["score"].sort_values().index

    labels = {}
    label_names = {
        1: "Şampiyonlar",        # En iyi: yakın, sık, çok para
        2: "Sadık Müşteriler",   # İyi ama her zaman en iyi değil
        3: "Risk Altındakiler",  # Eskiden iyiydi, uzaklaşıyor
        4: "Kaybedilenler",      # Uzun süre gelmedi
    }

    # 4'ten fazla küme varsa generic isim
    for rank, cluster_id in enumerate(sorted_idx, start=1):
        labels[cluster_id] = label_names.get(rank, f"Segment {rank}")

    return labels


def analyze_segments(rfm_df):
    """
    Her kümenin ham RFM ortalamaları + müşteri sayısı.
    """
    stats = rfm_df.groupby("Cluster").agg(
        Recency_Ort=("Recency", "mean"),
        Frequency_Ort=("Frequency", "mean"),
        Monetary_Ort=("Monetary", "mean"),
        Musteri_Sayisi=("Recency", "count")
    ).round(2)

    segment_labels = label_segments(
        rfm_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    )
    stats["Segment_Adi"] = stats.index.map(segment_labels)

    return stats
