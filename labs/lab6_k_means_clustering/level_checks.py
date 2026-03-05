import numpy as np


def check_step_1_manual_kmeans(local_vars):
    """
    Step 1: Manual k-means implementation.
    Expected: 'centroids' (k, d) array, 'labels' (N,) array.
    """
    if "centroids" not in local_vars:
        return (
            False,
            "⚠️ Variable `centroids` not found. Store the final centroid positions here.",
        )

    if "labels" not in local_vars:
        return (
            False,
            "⚠️ Variable `labels` not found. Store the cluster assignments here.",
        )

    centroids = local_vars["centroids"]
    labels = local_vars["labels"]

    if not isinstance(centroids, np.ndarray):
        return False, "⚠️ `centroids` should be a numpy array."

    if not isinstance(labels, np.ndarray):
        return False, "⚠️ `labels` should be a numpy array."

    if centroids.ndim != 2:
        return False, f"⚠️ `centroids` should be 2D (k, d), got {centroids.ndim}D."

    if labels.ndim != 1:
        return False, f"⚠️ `labels` should be 1D (N,), got {labels.ndim}D."

    k = centroids.shape[0]
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return False, "⚠️ Only one cluster found. Check your assignment step."

    if len(unique_labels) != k:
        return (
            False,
            f"⚠️ Found {len(unique_labels)} unique labels but {k} centroids. "
            "Every centroid should have at least one point assigned.",
        )

    return (
        True,
        f"✅ K-means implemented! Found {k} clusters with {len(labels)} assignments.",
    )


def check_step_2_sklearn_kmeans(local_vars):
    """
    Step 2: scikit-learn KMeans.
    Expected: 'kmeans' (fitted KMeans object), 'sk_labels' (N,) array.
    """
    if "kmeans" not in local_vars:
        return (
            False,
            "⚠️ Variable `kmeans` not found. Store the fitted KMeans object here.",
        )

    if "sk_labels" not in local_vars:
        return (
            False,
            "⚠️ Variable `sk_labels` not found. Store the cluster labels here.",
        )

    kmeans = local_vars["kmeans"]
    sk_labels = local_vars["sk_labels"]

    from sklearn.cluster import KMeans

    if not isinstance(kmeans, KMeans):
        return False, "⚠️ `kmeans` should be an instance of sklearn.cluster.KMeans."

    if not hasattr(kmeans, "labels_"):
        return False, "⚠️ KMeans model has not been fitted yet. Did you call `.fit()`?"

    if not isinstance(sk_labels, np.ndarray):
        return False, "⚠️ `sk_labels` should be a numpy array."

    n_clusters = kmeans.n_clusters
    unique = np.unique(sk_labels)

    if len(unique) < 2:
        return False, "⚠️ Only one cluster found. Check n_clusters parameter."

    return (
        True,
        f"✅ scikit-learn KMeans fitted! {n_clusters} clusters, inertia = {kmeans.inertia_:.2f}",
    )


def check_step_3_elbow(local_vars):
    """
    Step 3: Elbow method.
    Expected: 'inertias' list of inertia values for different k, 'k_range' range/list.
    """
    if "inertias" not in local_vars:
        return (
            False,
            "⚠️ Variable `inertias` not found. Store the list of inertia values here.",
        )

    if "k_range" not in local_vars:
        return (
            False,
            "⚠️ Variable `k_range` not found. Store the range of k values here.",
        )

    inertias = local_vars["inertias"]
    k_range = list(local_vars["k_range"])

    if not isinstance(inertias, list) or len(inertias) == 0:
        return False, "⚠️ `inertias` should be a non-empty list."

    if len(inertias) != len(k_range):
        return (
            False,
            f"⚠️ `inertias` has {len(inertias)} values but `k_range` has {len(k_range)}. They should match.",
        )

    if len(inertias) < 3:
        return False, "⚠️ Try at least 3 different values of k to see the elbow."

    if not all(inertias[i] >= inertias[i + 1] for i in range(len(inertias) - 1)):
        return (
            False,
            "⚠️ Inertia should generally decrease as k increases. Check your loop.",
        )

    return True, f"✅ Elbow method complete! Tested k = {k_range[0]}..{k_range[-1]}."


def check_step_4_embedding_clustering(local_vars):
    """
    Step 4: Clustering on real embeddings.
    Expected: 'emb_labels' (N,) array, 'emb_kmeans' fitted KMeans.
    """
    if "emb_kmeans" not in local_vars:
        return (
            False,
            "⚠️ Variable `emb_kmeans` not found. Store the fitted KMeans model here.",
        )

    if "emb_labels" not in local_vars:
        return (
            False,
            "⚠️ Variable `emb_labels` not found. Store the cluster labels here.",
        )

    emb_kmeans = local_vars["emb_kmeans"]
    emb_labels = local_vars["emb_labels"]

    if not hasattr(emb_kmeans, "labels_"):
        return False, "⚠️ KMeans model has not been fitted. Did you call `.fit()`?"

    if not isinstance(emb_labels, np.ndarray):
        return False, "⚠️ `emb_labels` should be a numpy array."

    n_clusters = emb_kmeans.n_clusters
    unique = np.unique(emb_labels)

    if len(unique) < 2:
        return False, "⚠️ Only one cluster found in the embeddings."

    return (
        True,
        f"✅ Embedding clustering done! {n_clusters} clusters over {len(emb_labels)} items.",
    )
