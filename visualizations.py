import plotly.express as px


def plot_3d_clusters(rfm_df):
    """
    RFM verilerini ve küme etiketlerini kullanarak 3D scatter plot oluşturur.
    """
    # Küme numaralarını isimlere çevirelim (Opsiyonel ama şık durur)
    names = {0: "Yeni", 1: "Uykuda", 2: "Sadık", 3: "Müdavim"}
    rfm_df['Segment'] = rfm_df['Cluster'].map(names)

    fig = px.scatter_3d(
        rfm_df,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Segment',  # Kümelere göre renklendir
        title='Müşteri Segmentasyonu 3D Görünüm',
        opacity=0.7,
        template='plotly_dark'  # M4 Air'in ekranında siyah tema çok iyi durur!
    )

    fig.show()