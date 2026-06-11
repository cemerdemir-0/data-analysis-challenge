import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize_rfm(rfm_df):
    """
    StandardScaler: her kolonu mean=0, std=1 yapar.
    K-Means Euclidean mesafe kullandığı için zorunlu —
    aksi halde Monetary (büyük sayılar) diğer boyutları ezer.
    """
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df)
    return pd.DataFrame(rfm_scaled, columns=rfm_df.columns, index=rfm_df.index), scaler


def remove_outliers_iqr(rfm_df, multiplier=3.0):
    """
    IQR yöntemiyle aşırı uç değerleri kaldır.
    multiplier=3.0 → sadece gerçek outlier'ları atar (1.5 çok agresif olur).

    Neden gerekli: 1 müşteri 200,000₺ harcamışsa kümeleme ona göre şekillenir,
    diğer 4000 müşteri anlamsız kümelere düşer.
    """
    mask = pd.Series(True, index=rfm_df.index)
    for col in rfm_df.columns:
        Q1 = rfm_df[col].quantile(0.25)
        Q3 = rfm_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        mask = mask & rfm_df[col].between(lower, upper)
    removed = (~mask).sum()
    if removed > 0:
        print(f"Outlier temizleme: {removed} müşteri kaldırıldı")
    return rfm_df[mask]
