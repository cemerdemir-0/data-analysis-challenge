import pandas as pd


def load_and_prepare_data(filepath: str, nrows: int = None):
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath, nrows=nrows)
    else:
        df = pd.read_excel(filepath, engine="openpyxl", nrows=nrows)

    # Misafir alışverişleri at (CustomerID yok)
    df = df[pd.notnull(df["CustomerID"])]
    df["CustomerID"] = df["CustomerID"].astype(int)

    # İadeler: "C" ile başlayan invoice numaraları
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Hatalı satırlar: negatif/sıfır Quantity veya UnitPrice
    # (C olmayan ama negatif qty içeren test/hata kayıtları)
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Referans tarihi: veri setindeki son tarih + 1 gün
    # (gerçek "bugün" kullanmak Recency'yi anlamsız büyütür)
    today_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (today_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum")
    )

    # Negatif monetary kalan edge-case'leri temizle
    rfm = rfm[rfm["Monetary"] > 0]

    return rfm
