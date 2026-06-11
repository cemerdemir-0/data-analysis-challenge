import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd


def find_optimal_k(rfm_scaled_df, max_k=10):
    """
    Elbow (SSE) + Silhouette Score ile optimal k bul.

    SSE (inertia): küme içi toplam kare mesafe — düşük = iyi ama
      k arttıkça her zaman düşer, tek başına yeterli değil.

    Silhouette: -1 ile 1 arası. Bir noktanın kendi kümesine ne kadar
      yakın, komşu kümeye ne kadar uzak olduğunu ölçer.
      1'e yakın = mükemmel kümeleme.
    """
    sse = []
    silhouette_scores = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(rfm_scaled_df)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(rfm_scaled_df, labels))

    results = pd.DataFrame({
        "k": range(2, max_k + 1),
        "SSE": sse,
        "Silhouette": silhouette_scores
    })

    best_k_silhouette = results.loc[results["Silhouette"].idxmax(), "k"]
    print("\n=== K Seçim Analizi ===")
    print(results.to_string(index=False))
    print(f"\nSilhouette'e göre en iyi k: {best_k_silhouette}")

    return results, int(best_k_silhouette)


def plot_elbow(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(results["k"], results["SSE"], marker='o', color='steelblue')
    ax1.set_xlabel("Küme Sayısı (k)")
    ax1.set_ylabel("SSE (Inertia)")
    ax1.set_title("Elbow Yöntemi")
    ax1.grid(True)

    ax2.plot(results["k"], results["Silhouette"], marker='o', color='darkorange')
    ax2.set_xlabel("Küme Sayısı (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analizi")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def apply_kmeans(rfm_scaled_df, k):
    """
    n_init=10: 10 farklı başlangıç noktasıyla çalıştır, en iyisini seç.
    random_state=42: tekrarlanabilir sonuç.
    """
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(rfm_scaled_df)
    return labels, model
