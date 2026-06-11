import pandas as pd

SEGMENT_INSIGHTS = {
    "Şampiyonlar": {
        "color": "#2563EB",
        "icon": "★",
        "insight": "En değerli %18'lik dilim. Yakın zamanda aktif, yüksek frekans ve harcama. Bu müşteriler markayı zaten seviyor.",
        "action": "VIP sadakat programı, erken erişim kampanyaları ve kişisel teşekkür mesajları gönder. Onları elde tutmak, yeni müşteri kazanmaktan 5× ucuz.",
        "risk": "Düşük",
        "priority": "Koru",
        "priority_color": "#16A34A",
    },
    "Sadık Müşteriler": {
        "color": "#16A34A",
        "icon": "◆",
        "insight": "En büyük grup (%57). Düzenli alışveriş yapıyor ama harcama potansiyeli henüz kullanılmamış.",
        "action": "Çapraz satış ve paket teklifleri ile monetary değerini artır. Şampiyonlara terfi ettirmeye odaklan.",
        "risk": "Orta",
        "priority": "Geliştir",
        "priority_color": "#2563EB",
    },
    "Risk Altındakiler": {
        "color": "#D4845A",
        "icon": "▲",
        "insight": "Eskiden aktifti, ortalama 253 gün önce son alışveriş. Kaybedilmek üzere — acil müdahale gerekiyor.",
        "action": "Kişisel geri kazanım e-postası ve özel indirim kuponu gönder. 'Sizi özledik' kampanyası %15-20 geri dönüş sağlar.",
        "risk": "Yüksek",
        "priority": "Geri Kazan",
        "priority_color": "#D97706",
    },
    "Kaybedilenler": {
        "color": "#9CA3AF",
        "icon": "●",
        "insight": "Uzun süredir inaktif. Geri kazanım maliyeti diğer segmentlere yatırımdan yüksek olabilir.",
        "action": "Büyük indirim kampanyası dene, yanıt vermeyenleri listeden çıkar. Kaynakları aktif segmentlere yönlendir.",
        "risk": "Çok Yüksek",
        "priority": "Son Şans",
        "priority_color": "#6B7280",
    },
}


def get_segment_top_products(raw_df: pd.DataFrame, rfm_df: pd.DataFrame, segment_name: str, top_n: int = 8):
    segment_customers = rfm_df[rfm_df["Segment"] == segment_name].index
    segment_txns = raw_df[raw_df["CustomerID"].isin(segment_customers)]

    top = (
        segment_txns
        .groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    return top.reset_index().rename(columns={"Description": "urun", "Quantity": "adet"})


def get_customer_recommendations(customer_id: int, raw_df: pd.DataFrame, rfm_df: pd.DataFrame, top_n: int = 5):
    if customer_id not in rfm_df.index:
        return None, None, None

    customer_row = rfm_df.loc[customer_id]
    segment = customer_row["Segment"]

    segment_top = get_segment_top_products(raw_df, rfm_df, segment, top_n=30)

    already_bought = set(
        raw_df[raw_df["CustomerID"] == customer_id]["Description"]
        .str.strip().str.upper()
    )

    recs = segment_top[
        ~segment_top["urun"].str.strip().str.upper().isin(already_bought)
    ].head(top_n)

    segment_avg = rfm_df[rfm_df["Segment"] == segment][["Recency", "Frequency", "Monetary"]].mean().round(1)

    return customer_row, recs, segment_avg


def compute_all_segment_insights(raw_df: pd.DataFrame, rfm_df: pd.DataFrame):
    result = []
    for segment_name, meta in SEGMENT_INSIGHTS.items():
        if segment_name not in rfm_df["Segment"].values:
            continue
        top_products = get_segment_top_products(raw_df, rfm_df, segment_name, top_n=6)
        seg_df = rfm_df[rfm_df["Segment"] == segment_name]
        result.append({
            **meta,
            "name": segment_name,
            "count": len(seg_df),
            "recency_avg": round(seg_df["Recency"].mean(), 0),
            "monetary_avg": round(seg_df["Monetary"].mean(), 0),
            "products": top_products.to_dict("records"),
        })
    return result
