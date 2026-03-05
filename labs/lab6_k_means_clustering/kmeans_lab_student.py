import io
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from labs.lab6_k_means_clustering.level_checks import (
    check_step_1_manual_kmeans,
    check_step_2_sklearn_kmeans,
    check_step_3_elbow,
    check_step_4_embedding_clustering,
)


def _st_image_full_width(path):
    """st.image() wrapper that works across Streamlit versions."""
    try:
        st.image(path, use_container_width=True)
    except TypeError:
        st.image(path, use_column_width=True)


def _run_student_code(code, ctx, console_key):
    """Execute student code, capturing stdout/stderr and displaying a console.

    Returns (ctx, success) where success is True if no exception was raised.
    """
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = cap_out = io.StringIO()
    sys.stderr = cap_err = io.StringIO()

    success = True
    tb_text = ""
    try:
        exec(code, ctx)
    except Exception:
        success = False
        tb_text = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    stdout_text = cap_out.getvalue()
    stderr_text = cap_err.getvalue()

    console_parts = []
    if stdout_text:
        console_parts.append(stdout_text)
    if stderr_text:
        console_parts.append(f"[stderr]\n{stderr_text}")
    if tb_text:
        console_parts.append(f"[error]\n{tb_text}")

    console_output = "\n".join(console_parts) if console_parts else "(no output)"
    st.session_state[console_key] = console_output

    has_error = bool(tb_text or stderr_text)
    with st.expander("🖥️ Console Output", expanded=has_error):
        st.code(st.session_state[console_key], language="text")

    if not success:
        st.error(
            "Your code raised an error — check the console above for the full traceback."
        )

    return ctx, success


DATASET_CONFIG = {
    "TMDB Movies": {
        "path": "data/processed/tmdb_embedded.parquet",
        "title_col": "title",
        "category_col": "genres",
        "label": "movies",
    },
    "Airbnb Listings": {
        "path": "data/processed/airbnb_embedded.parquet",
        "title_col": "name",
        "category_col": "neighbourhood_group_cleansed",
        "label": "listings",
    },
}

_COLUMN_DEFAULTS = {
    "TMDB Movies": {
        "embedding_col": "embedding",
        "title_col": "title",
        "category_col": "genres",
        "latitude_col": "",
        "longitude_col": "",
        "image_embedding_col": "image_embedding",
    },
    "Airbnb Listings": {
        "embedding_col": "embedding",
        "title_col": "name",
        "category_col": "neighbourhood_group_cleansed",
        "latitude_col": "latitude",
        "longitude_col": "longitude",
        "image_embedding_col": "image_embedding",
    },
}

_COLUMN_DESCRIPTIONS = {
    "embedding_col": "Text embedding (array column)",
    "title_col": "Item name / title",
    "category_col": "Category for analysis",
    "latitude_col": "Latitude (for map, leave blank to skip)",
    "longitude_col": "Longitude (for map, leave blank to skip)",
    "image_embedding_col": "Image embedding (optional, leave blank to skip)",
}


def _get_column_mapping(dataset_choice, df_columns):
    """Show a column-mapping UI and return the resolved config.

    Renders text inputs pre-filled with defaults so students whose parquet
    has different column names can override them.
    """
    cfg = DATASET_CONFIG[dataset_choice].copy()
    defaults_for_ds = _COLUMN_DEFAULTS.get(dataset_choice, {})

    mapping_key = f"lab6_col_mapping_{dataset_choice}"
    saved = st.session_state.get(mapping_key)

    with st.expander("⚙️ Column Mapping — click to configure if your columns differ"):
        st.markdown(
            "Your processed parquet may use different column names. "
            "Adjust the mapping below to match **your** file. "
            "Leave a field blank to skip that feature."
        )
        st.caption(f"Detected columns: `{'`, `'.join(df_columns)}`")

        defaults = saved if saved else defaults_for_ds
        new_mapping = {}
        cols = st.columns(2)
        for i, (key, desc) in enumerate(_COLUMN_DESCRIPTIONS.items()):
            with cols[i % 2]:
                val = st.text_input(
                    desc,
                    value=defaults.get(key, ""),
                    key=f"lab6_colmap_{dataset_choice}_{key}",
                )
                new_mapping[key] = val.strip()

        if st.button("Apply column mapping", key=f"lab6_apply_colmap_{dataset_choice}"):
            st.session_state[mapping_key] = new_mapping
            for k in [
                "lab6_text_emb",
                "lab6_image_emb",
                "lab6_titles",
                "lab6_categories",
                "lab6_cat_views",
                "lab6_lat",
                "lab6_lon",
                "lab6_embeddings",
                "lab6_pca_2d",
                "lab6_step_4_done",
                "lab6_emb_labels",
                "lab6_emb_centroids",
                "lab6_emb_elbow",
            ]:
                st.session_state.pop(k, None)
            st.rerun()

    mapping = st.session_state.get(mapping_key, defaults_for_ds)
    cfg["title_col"] = mapping.get("title_col") or defaults_for_ds.get(
        "title_col", "title"
    )
    cfg["category_col"] = mapping.get("category_col") or defaults_for_ds.get(
        "category_col", ""
    )
    cfg["embedding_col"] = mapping.get("embedding_col") or "embedding"
    cfg["latitude_col"] = mapping.get("latitude_col", "")
    cfg["longitude_col"] = mapping.get("longitude_col", "")
    cfg["image_embedding_col"] = mapping.get("image_embedding_col", "")
    return cfg


def render_kmeans_lab():
    st.header("Lab 6: K-Means Clustering")

    st.markdown(
        """
    ### 🎯 The Mission

    So far we've learned to **embed** data into vector spaces and **reduce** dimensions.
    Now we'll learn to **group** similar items together automatically — without any labels!

    **Clustering** is an *unsupervised* learning technique: the algorithm discovers structure
    in the data on its own. **K-Means** is the most widely-used clustering algorithm because
    it's simple, fast, and surprisingly effective.

    We'll work through four steps:
    1. **Implement k-means from scratch** on a toy 2D dataset
    2. **Use scikit-learn's KMeans** for the production version
    3. **Choose the right k** with the Elbow Method
    4. **Cluster real embeddings** and explore what the algorithm discovers
    """
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔧 Step 1: K-Means from Scratch",
            "📦 Step 2: scikit-learn KMeans",
            "📐 Step 3: Elbow Method",
            "🗺️ Step 4: Cluster Real Embeddings",
        ]
    )

    with tab1:
        render_step_1_manual()

    with tab2:
        render_step_2_sklearn()

    with tab3:
        render_step_3_elbow()

    with tab4:
        render_step_4_embeddings()


# ── Shared helpers: animation + metrics ──────────────────────────


def _kmeans_plus_plus_init(data, k, rng):
    """K-means++ initialization: pick centroids that are spread far apart."""
    n = len(data)
    centroids = [data[rng.randint(n)]]

    for _ in range(1, k):
        dists = np.min(
            [np.linalg.norm(data - c, axis=1) ** 2 for c in centroids], axis=0
        )
        probs = dists / dists.sum()
        centroids.append(data[rng.choice(n, p=probs)])

    return np.array(centroids)


def _run_kmeans_history(data, k=3, max_iters=15, seed=0, init="random"):
    """Run k-means and record centroid positions + labels at every iteration.

    init: "random" for random point selection, "kmeans++" for k-means++ seeding.
    """
    rng = np.random.RandomState(seed)

    if init == "kmeans++":
        centroids = _kmeans_plus_plus_init(data, k, rng)
    else:
        indices = rng.choice(len(data), size=k, replace=False)
        centroids = data[indices].copy()

    history = []

    distances = np.linalg.norm(data[:, np.newaxis] - centroids[np.newaxis, :], axis=2)
    labels = np.argmin(distances, axis=1)
    history.append(
        {"centroids": centroids.copy(), "labels": labels.copy(), "iteration": 0}
    )

    for it in range(1, max_iters + 1):
        new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])

        distances = np.linalg.norm(
            data[:, np.newaxis] - new_centroids[np.newaxis, :], axis=2
        )
        labels = np.argmin(distances, axis=1)
        history.append(
            {
                "centroids": new_centroids.copy(),
                "labels": labels.copy(),
                "iteration": it,
            }
        )

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return history


def _build_kmeans_animated_figure(
    data, k=3, max_iters=15, seed=0, init="random", title="K-Means Convergence"
):
    """Build a Plotly figure with Play/Pause animation frames for k-means iterations."""
    history = _run_kmeans_history(data, k=k, max_iters=max_iters, seed=seed, init=init)
    colors = px.colors.qualitative.Set2

    # Stable axis range across all frames
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5

    # Build initial frame (iteration 0) as the base traces
    snap0 = history[0]
    fig = go.Figure()

    for j in range(k):
        mask = snap0["labels"] == j
        fig.add_trace(
            go.Scatter(
                x=data[mask, 0],
                y=data[mask, 1],
                mode="markers",
                marker=dict(color=colors[j % len(colors)], size=6, opacity=0.5),
                name=f"Cluster {j}",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=snap0["centroids"][:, 0],
            y=snap0["centroids"][:, 1],
            mode="markers",
            marker=dict(
                symbol="x", size=18, color="black", line=dict(width=2, color="white")
            ),
            name="Centroids",
        )
    )

    for j in range(k):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="black", width=1.5, dash="dot"),
                showlegend=False,
            )
        )

    # Build animation frames
    frames = []
    for idx, snap in enumerate(history):
        frame_data = []
        centroids = snap["centroids"]
        labels = snap["labels"]

        for j in range(k):
            mask = labels == j
            frame_data.append(
                go.Scatter(
                    x=data[mask, 0],
                    y=data[mask, 1],
                    mode="markers",
                    marker=dict(color=colors[j % len(colors)], size=6, opacity=0.5),
                )
            )

        frame_data.append(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=18,
                    color="black",
                    line=dict(width=2, color="white"),
                ),
            )
        )

        if idx > 0:
            prev_centroids = history[idx - 1]["centroids"]
            for j in range(k):
                frame_data.append(
                    go.Scatter(
                        x=[prev_centroids[j, 0], centroids[j, 0]],
                        y=[prev_centroids[j, 1], centroids[j, 1]],
                        mode="lines",
                        line=dict(color="black", width=1.5, dash="dot"),
                    )
                )
        else:
            for j in range(k):
                frame_data.append(go.Scatter(x=[None], y=[None], mode="lines"))

        iter_label = snap["iteration"]
        suffix = " (converged)" if idx == len(history) - 1 and idx > 0 else ""
        frame_title = f"{title} — Iteration {iter_label}{suffix}"

        frames.append(
            go.Frame(
                data=frame_data,
                name=str(iter_label),
                layout=go.Layout(title_text=frame_title),
            )
        )

    fig.frames = frames

    # Sliders for frame navigation
    sliders = [
        dict(
            active=0,
            steps=[
                dict(
                    args=[
                        [str(snap["iteration"])],
                        dict(frame=dict(duration=0, redraw=True), mode="immediate"),
                    ],
                    label=str(snap["iteration"]),
                    method="animate",
                )
                for snap in history
            ],
            currentvalue=dict(prefix="Iteration: "),
            pad=dict(t=50),
        )
    ]

    fig.update_layout(
        title=f"{title} — Iteration 0",
        xaxis=dict(title="X₁", range=[x_min, x_max]),
        yaxis=dict(title="X₂", range=[y_min, y_max]),
        width=680,
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=600, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=sliders,
    )

    return fig


@st.fragment
def _render_kmeans_animation(data_2d):
    """Render an auto-play k-means animation with hyperparameter controls."""
    col_k, col_seed = st.columns(2)
    with col_k:
        anim_k = st.slider(
            "Number of clusters (k)",
            min_value=2,
            max_value=8,
            value=5,
            key="lab6_anim_k",
        )
    with col_seed:
        anim_seed = st.slider(
            "Random seed (changes initialization)",
            min_value=0,
            max_value=10,
            value=0,
            key="lab6_anim_seed",
        )

    fig = _build_kmeans_animated_figure(
        data_2d, k=anim_k, seed=anim_seed, title="K-Means Demo"
    )
    st.plotly_chart(fig, use_container_width=True)


def _compute_inertia(data, labels, centroids):
    """Compute within-cluster sum of squared distances."""
    total = 0.0
    for j in range(len(centroids)):
        mask = labels == j
        if mask.any():
            total += np.sum(np.linalg.norm(data[mask] - centroids[j], axis=1) ** 2)
    return total


def _display_metrics(data, labels, centroids):
    """Display Inertia and Silhouette Score in a two-column metric layout."""
    inertia = _compute_inertia(data, labels, centroids)

    n_unique = len(np.unique(labels))
    if n_unique >= 2 and n_unique < len(data):
        sil = silhouette_score(data, labels)
    else:
        sil = None

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Inertia (WCSS)", f"{inertia:,.2f}")
        st.caption(
            "Sum of squared distances to nearest centroid. Lower = tighter clusters."
        )
    with col2:
        if sil is not None:
            st.metric("Silhouette Score", f"{sil:.4f}")
            st.caption(
                "Ranges from -1 to 1. Higher = better-separated, more cohesive clusters."
            )
        else:
            st.metric("Silhouette Score", "N/A")
            st.caption("Requires at least 2 clusters with multiple points each.")


def _build_iteration_figure(data, history, idx, k, title, col_x="X₁", col_y="X₂"):
    """Build a single Plotly figure for one k-means iteration snapshot."""
    colors = px.colors.qualitative.Set2
    snap = history[idx]
    centroids = snap["centroids"]
    labels = snap["labels"]

    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5

    fig = go.Figure()

    for j in range(k):
        mask = labels == j
        fig.add_trace(
            go.Scatter(
                x=data[mask, 0],
                y=data[mask, 1],
                mode="markers",
                marker=dict(color=colors[j % len(colors)], size=6, opacity=0.5),
                name=f"Cluster {j}",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode="markers",
            marker=dict(
                symbol="x", size=18, color="black", line=dict(width=2, color="white")
            ),
            name="Centroids",
        )
    )

    if idx > 0:
        prev_centroids = history[idx - 1]["centroids"]
        for j in range(k):
            fig.add_trace(
                go.Scatter(
                    x=[prev_centroids[j, 0], centroids[j, 0]],
                    y=[prev_centroids[j, 1], centroids[j, 1]],
                    mode="lines",
                    line=dict(color="black", width=1.5, dash="dot"),
                    showlegend=False,
                )
            )

    iter_label = snap["iteration"]
    is_last = idx == len(history) - 1
    suffix = " — Converged!" if is_last and idx > 0 else ""
    frame_title = f"{title} — Iteration {iter_label}{suffix}"

    fig.update_layout(
        title=frame_title,
        xaxis=dict(title=col_x, range=[x_min, x_max]),
        yaxis=dict(title=col_y, range=[y_min, y_max]),
        width=680,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _render_live_fitting(
    data,
    k,
    seed=0,
    init="random",
    title="K-Means Fitting",
    col_x="X₁",
    col_y="X₂",
    delay=0.5,
):
    """Run k-means and update a chart in real time, iteration by iteration."""
    history = _run_kmeans_history(data, k=k, max_iters=20, seed=seed, init=init)

    chart_placeholder = st.empty()
    status_placeholder = st.empty()

    for idx in range(len(history)):
        fig = _build_iteration_figure(data, history, idx, k, title, col_x, col_y)
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        is_last = idx == len(history) - 1
        if is_last and idx > 0:
            status_placeholder.success(
                f"Converged after {history[idx]['iteration']} iterations."
            )
        elif is_last:
            status_placeholder.info("Single iteration — already converged.")
        else:
            status_placeholder.info(f"Iteration {history[idx]['iteration']}...")
            time.sleep(delay)

    return history[-1]["labels"], history[-1]["centroids"]


def _render_live_fitting_hd(
    data_hd,
    data_2d,
    k,
    seed=42,
    init="kmeans++",
    title="K-Means Fitting",
    col_x="PC1",
    col_y="PC2",
    delay=0.4,
):
    """Run k-means on high-dimensional data but visualize on a 2D projection.

    Centroids are projected to 2D by averaging the 2D positions of assigned points
    (since the actual HD centroids can't be plotted directly).
    """
    history = _run_kmeans_history(data_hd, k=k, max_iters=30, seed=seed, init=init)

    x_min, x_max = data_2d[:, 0].min() - 0.5, data_2d[:, 0].max() + 0.5
    y_min, y_max = data_2d[:, 1].min() - 0.5, data_2d[:, 1].max() + 0.5

    colors = px.colors.qualitative.Set2
    chart_placeholder = st.empty()
    status_placeholder = st.empty()

    for idx, snap in enumerate(history):
        labels = snap["labels"]

        # Project centroids to 2D as the mean of assigned 2D points
        centroids_2d = np.array(
            [
                data_2d[labels == j].mean(axis=0)
                if (labels == j).any()
                else data_2d.mean(axis=0)
                for j in range(k)
            ]
        )

        fig = go.Figure()
        for j in range(k):
            mask = labels == j
            fig.add_trace(
                go.Scatter(
                    x=data_2d[mask, 0],
                    y=data_2d[mask, 1],
                    mode="markers",
                    marker=dict(color=colors[j % len(colors)], size=5, opacity=0.4),
                    name=f"Cluster {j}",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=centroids_2d[:, 0],
                y=centroids_2d[:, 1],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=16,
                    color="black",
                    line=dict(width=2, color="white"),
                ),
                name="Centroids",
            )
        )

        if idx > 0:
            prev_labels = history[idx - 1]["labels"]
            prev_centroids_2d = np.array(
                [
                    data_2d[prev_labels == j].mean(axis=0)
                    if (prev_labels == j).any()
                    else data_2d.mean(axis=0)
                    for j in range(k)
                ]
            )
            for j in range(k):
                fig.add_trace(
                    go.Scatter(
                        x=[prev_centroids_2d[j, 0], centroids_2d[j, 0]],
                        y=[prev_centroids_2d[j, 1], centroids_2d[j, 1]],
                        mode="lines",
                        line=dict(color="black", width=1.5, dash="dot"),
                        showlegend=False,
                    )
                )

        iter_label = snap["iteration"]
        is_last = idx == len(history) - 1
        suffix = " — Converged!" if is_last and idx > 0 else ""
        fig.update_layout(
            title=f"{title} — Iteration {iter_label}{suffix}",
            xaxis=dict(title=col_x, range=[x_min, x_max]),
            yaxis=dict(title=col_y, range=[y_min, y_max]),
            width=680,
            height=500,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        chart_placeholder.plotly_chart(fig, use_container_width=True)

        if is_last and idx > 0:
            status_placeholder.success(f"Converged after {iter_label} iterations.")
        elif is_last:
            status_placeholder.info("Single iteration — already converged.")
        else:
            status_placeholder.info(f"Iteration {iter_label}...")
            time.sleep(delay)


# ── Step 1: K-Means from Scratch ─────────────────────────────────


def _generate_toy_data(seed=42):
    """Generate a 2D dataset with 5 clusters of varying size and proximity.

    Some clusters are close together so that random initialization can fail
    while k-means++ handles them correctly.
    """
    rng = np.random.RandomState(seed)
    centers = np.array([[1, 1], [3, 2.5], [6, 1.5], [8, 5], [4, 7]])
    spreads = [0.6, 0.8, 0.7, 1.0, 0.9]
    sizes = [60, 90, 50, 100, 70]

    points = []
    for center, spread, n in zip(centers, spreads, sizes):
        pts = rng.randn(n, 2) * spread + center
        points.append(pts)

    return np.vstack(points)


def render_step_1_manual():
    st.subheader("Step 1: Implement K-Means from Scratch")

    st.markdown(
        r"""
    ### 📐 The K-Means Algorithm

    K-means is beautifully simple. It alternates between two steps until convergence:

    **Input:** Data points $X = \{x_1, x_2, \ldots, x_n\}$ and number of clusters $k$.

    **Step 0 — Initialize:** Pick $k$ random data points as initial centroids $\mu_1, \ldots, \mu_k$.

    **Step 1 — Assign:** For each point $x_i$, find the nearest centroid:
    $$c_i = \arg\min_{j} \| x_i - \mu_j \|^2$$

    **Step 2 — Update:** Move each centroid to the mean of its assigned points:
    $$\mu_j = \frac{1}{|S_j|} \sum_{x_i \in S_j} x_i$$

    **Repeat** Steps 1–2 until the centroids stop moving (or a max number of iterations).

    ---

    ### 🧪 Toy Dataset

    We'll start with a simple 2D dataset so you can **see** the algorithm work.
    """
    )

    data_2d = _generate_toy_data()
    st.session_state["lab6_toy_data"] = data_2d

    fig_data = px.scatter(
        x=data_2d[:, 0],
        y=data_2d[:, 1],
        title="Toy 2D Dataset (5 hidden clusters)",
        labels={"x": "X₁", "y": "X₂"},
        opacity=0.6,
    )
    fig_data.update_layout(width=600, height=450)
    st.plotly_chart(fig_data, use_container_width=True)

    # ── Interactive K-Means animation ────────────────────────────
    st.markdown("### 🎬 Watch K-Means in Action")
    st.markdown("Press **Play** to watch centroids move iteration by iteration.")

    _render_kmeans_animation(data_2d)

    st.markdown("---")

    st.markdown(
        """
    ### 💻 Your Task

    Implement k-means from scratch:
    1. Initialize `k` centroids by randomly selecting `k` data points.
    2. **Assign** each point to the nearest centroid (use Euclidean distance).
    3. **Update** each centroid to the mean of its assigned points.
    4. Repeat for `max_iters` iterations.
    5. Store final centroids in `centroids` and assignments in `labels`.
    """
    )

    default_code = """# 'data' is available — shape (N, 2)
k = 5
max_iters = 20

# Step 0: Initialize centroids by picking k random points
rng = np.random.RandomState(3)
indices = rng.choice(len(data), size=k, replace=False)
centroids = data[indices].copy()

for iteration in range(max_iters):
    # TODO Step 1: Assign each point to the nearest centroid.
    # Hint: compute the Euclidean distance from every point to every centroid,
    #       then use np.argmin to find the closest one.
    # distances = ...   # shape (N, k)
    # labels = ...      # shape (N,)

    # TODO Step 2: Update each centroid to the mean of its assigned points.
    # new_centroids = ...   # shape (k, 2)

    # Check convergence (do not modify)
    if np.allclose(centroids, new_centroids):
        print(f"Converged at iteration {iteration + 1}")
        break
    centroids = new_centroids

print(f"Final centroids:\\n{centroids}")
print(f"Cluster sizes: {[int((labels == j).sum()) for j in range(k)]}")"""

    code = st.text_area(
        "K-Means Code:", value=default_code, height=380, key="kmeans_manual_code"
    )

    if st.button("Run K-Means", key="run_manual_kmeans"):
        ctx = {"data": data_2d, "np": np, "__builtins__": __builtins__}
        ctx, success = _run_student_code(code, ctx, "lab6_console_step1")

        if success:
            passed, msg = check_step_1_manual_kmeans(ctx)
            if passed:
                st.success(msg)
                st.session_state["lab6_step_1_done"] = True
                st.session_state["lab6_manual_labels"] = ctx["labels"]
                st.session_state["lab6_manual_centroids"] = ctx["centroids"]

                st.markdown("---")
                st.markdown("### 🎬 Live Fitting Visualization (Random Init)")
                k = len(ctx["centroids"])
                _render_live_fitting(
                    data_2d, k=k, seed=3, init="random", title="Step 1: Random Init"
                )

                st.markdown("### 📏 Evaluation Metrics")
                _display_metrics(data_2d, ctx["labels"], ctx["centroids"])
            else:
                st.error(msg)


# ── Step 2: scikit-learn KMeans ──────────────────────────────────


def render_step_2_sklearn():
    st.subheader("Step 2: K-Means with scikit-learn")

    st.markdown(
        """
    Now that you understand the algorithm, let's use the **production-quality** implementation
    from scikit-learn. It includes important improvements over our scratch version:

    | Feature | Our version | scikit-learn |
    |---------|------------|--------------|
    | Initialization | Random points | **k-means++** (smarter seeding) |
    | Runs | 1 | **10 restarts** (picks best) |
    | Convergence | Simple check | Tolerance-based + max iterations |
    | Speed | Pure Python/NumPy | Optimized C/Cython |

    ### 🎯 What is k-means++?

    Bad initialization can lead to poor clusters. **k-means++** picks initial centroids
    that are spread far apart:
    1. Pick the first centroid randomly.
    2. For each remaining centroid, pick a point with probability proportional to its
       **squared distance** from the nearest existing centroid.

    This simple trick dramatically improves convergence and final quality.

    ---

    ### 💻 Your Task

    Use `sklearn.cluster.KMeans` to cluster the same toy data:
    1. Import `KMeans` from `sklearn.cluster`.
    2. Create a `KMeans` instance with `n_clusters=5` and `random_state=42`.
    3. Fit it on `data`.
    4. Store the model in `kmeans` and the labels in `sk_labels`.
    """
    )

    data_2d = _generate_toy_data()

    default_code = """# 'data' is available — shape (N, 2)
from sklearn.cluster import KMeans

# TODO: Create a KMeans instance with n_clusters=5, random_state=42, n_init=10
# kmeans = ...

# TODO: Fit the model on 'data'
# kmeans.fit(...)

# TODO: Extract the cluster labels into 'sk_labels'
# sk_labels = ...

print(f"Inertia (sum of squared distances): {kmeans.inertia_:.2f}")
print(f"Cluster sizes: {[int((sk_labels == j).sum()) for j in range(5)]}")"""

    code = st.text_area(
        "scikit-learn Code:", value=default_code, height=250, key="kmeans_sklearn_code"
    )

    if st.button("Run scikit-learn KMeans", key="run_sklearn_kmeans"):
        ctx = {"data": data_2d, "np": np, "__builtins__": __builtins__}
        ctx, success = _run_student_code(code, ctx, "lab6_console_step2")

        if success:
            passed, msg = check_step_2_sklearn_kmeans(ctx)
            if passed:
                st.success(msg)
                st.session_state["lab6_step_2_done"] = True
                st.session_state["lab6_sk_labels"] = ctx["sk_labels"]
                st.session_state["lab6_sk_centroids"] = ctx["kmeans"].cluster_centers_

                st.markdown("---")
                st.markdown("### 🎬 Side-by-Side: Random Init vs. k-means++")
                st.markdown(
                    "Watch both initializations run on the **same data**. "
                    "Notice how k-means++ starts with well-spread centroids "
                    "and typically converges in fewer iterations."
                )

                k = ctx["kmeans"].n_clusters
                hist_random = _run_kmeans_history(
                    data_2d, k=k, max_iters=20, seed=3, init="random"
                )
                hist_kpp = _run_kmeans_history(
                    data_2d, k=k, max_iters=20, seed=42, init="kmeans++"
                )

                col_left, col_right = st.columns(2)
                with col_left:
                    ph_left = st.empty()
                    status_left = st.empty()
                with col_right:
                    ph_right = st.empty()
                    status_right = st.empty()

                n_frames = max(len(hist_random), len(hist_kpp))
                for idx in range(n_frames):
                    ri = min(idx, len(hist_random) - 1)
                    ki = min(idx, len(hist_kpp) - 1)

                    fig_r = _build_iteration_figure(
                        data_2d,
                        hist_random,
                        ri,
                        k,
                        title=f"Random Init (iter {hist_random[ri]['iteration']})",
                    )
                    fig_k = _build_iteration_figure(
                        data_2d,
                        hist_kpp,
                        ki,
                        k,
                        title=f"k-means++ (iter {hist_kpp[ki]['iteration']})",
                    )

                    ph_left.plotly_chart(
                        fig_r, use_container_width=True, key=f"s2_left_{idx}"
                    )
                    ph_right.plotly_chart(
                        fig_k, use_container_width=True, key=f"s2_right_{idx}"
                    )

                    r_done = ri == len(hist_random) - 1
                    k_done = ki == len(hist_kpp) - 1

                    if r_done and ri > 0:
                        status_left.success(
                            f"Converged after {hist_random[ri]['iteration']} iterations"
                        )
                    elif not r_done:
                        status_left.info(f"Iteration {hist_random[ri]['iteration']}...")

                    if k_done and ki > 0:
                        status_right.success(
                            f"Converged after {hist_kpp[ki]['iteration']} iterations"
                        )
                    elif not k_done:
                        status_right.info(f"Iteration {hist_kpp[ki]['iteration']}...")

                    if idx < n_frames - 1:
                        time.sleep(0.5)

                r_inertia = _compute_inertia(
                    data_2d, hist_random[-1]["labels"], hist_random[-1]["centroids"]
                )
                k_inertia = _compute_inertia(
                    data_2d, hist_kpp[-1]["labels"], hist_kpp[-1]["centroids"]
                )

                st.markdown("### 📏 Comparison Metrics")
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.markdown("**Random Init**")
                    _display_metrics(
                        data_2d, hist_random[-1]["labels"], hist_random[-1]["centroids"]
                    )
                with mc2:
                    st.markdown("**k-means++**")
                    _display_metrics(
                        data_2d, ctx["sk_labels"], ctx["kmeans"].cluster_centers_
                    )

                if k_inertia < r_inertia:
                    st.info(
                        f"k-means++ achieved **{((r_inertia - k_inertia) / r_inertia * 100):.1f}% lower inertia** "
                        f"than random init, demonstrating the benefit of smarter initialization."
                    )
            else:
                st.error(msg)


# ── Step 3: Elbow Method ─────────────────────────────────────────


def render_step_3_elbow():
    st.subheader("Step 3: Choosing k — The Elbow Method")

    st.markdown(
        r"""
    K-means requires you to specify $k$ (the number of clusters) in advance.
    But how do you know the right $k$?

    ### 📐 Inertia (Within-Cluster Sum of Squares)

    **Inertia** measures how tight the clusters are:

    $$\text{Inertia} = \sum_{j=1}^{k} \sum_{x_i \in S_j} \| x_i - \mu_j \|^2$$

    - Lower inertia = tighter clusters
    - Inertia **always** decreases as $k$ increases (at $k = n$, inertia = 0)
    - So we can't just pick the $k$ with lowest inertia!

    ### 📉 The Elbow Method

    Plot inertia vs. $k$ and look for an **"elbow"** — the point where adding more
    clusters stops giving much improvement:

    - Before the elbow: each new cluster captures real structure → big drop in inertia
    - After the elbow: new clusters just split existing groups → diminishing returns

    ---

    ### 💻 Your Task

    1. Loop over `k_range = range(1, 11)`.
    2. For each `k`, fit `KMeans(n_clusters=k)` and record `kmeans.inertia_`.
    3. Store the range in `k_range` and the inertia values in `inertias`.
    """
    )

    data_2d = _generate_toy_data()

    default_code = """# 'data' is available — shape (N, 2)
from sklearn.cluster import KMeans

k_range = range(1, 11)
inertias = []

# TODO: Loop over each value of k in k_range.
# For each k:
#   1. Create and fit a KMeans model with n_clusters=k, random_state=42, n_init=10
#   2. Append the model's inertia_ to the 'inertias' list
#   3. (Optional) Print k and inertia for debugging

"""

    code = st.text_area(
        "Elbow Method Code:", value=default_code, height=250, key="kmeans_elbow_code"
    )

    if st.button("Run Elbow Method", key="run_elbow"):
        ctx = {"data": data_2d, "np": np, "__builtins__": __builtins__}
        ctx, success = _run_student_code(code, ctx, "lab6_console_step3")

        if success:
            passed, msg = check_step_3_elbow(ctx)
            if passed:
                st.success(msg)
                st.session_state["lab6_step_3_done"] = True
                st.session_state["lab6_inertias"] = ctx["inertias"]
                st.session_state["lab6_k_range"] = list(ctx["k_range"])

                sil_scores = []
                k_list = list(ctx["k_range"])
                for k_val in k_list:
                    if k_val < 2:
                        sil_scores.append(None)
                    else:
                        km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
                        km.fit(data_2d)
                        sil_scores.append(silhouette_score(data_2d, km.labels_))
                st.session_state["lab6_silhouettes"] = sil_scores
            else:
                st.error(msg)

    if st.session_state.get("lab6_step_3_done"):
        st.markdown("---")
        st.markdown("### 📉 Elbow Plot with Silhouette Score")

        inertias = st.session_state["lab6_inertias"]
        k_range = st.session_state["lab6_k_range"]
        sil_scores = st.session_state.get("lab6_silhouettes", [])

        from plotly.subplots import make_subplots

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=k_range,
                y=inertias,
                mode="lines+markers",
                marker=dict(size=10, color="#4A90D9"),
                line=dict(width=2, color="#4A90D9"),
                name="Inertia",
            ),
            secondary_y=False,
        )

        if sil_scores:
            valid_k = [k for k, s in zip(k_range, sil_scores) if s is not None]
            valid_sil = [s for s in sil_scores if s is not None]
            fig.add_trace(
                go.Scatter(
                    x=valid_k,
                    y=valid_sil,
                    mode="lines+markers",
                    marker=dict(size=10, color="#E74C3C"),
                    line=dict(width=2, color="#E74C3C"),
                    name="Silhouette Score",
                ),
                secondary_y=True,
            )

        if 5 in k_range:
            fig.add_vline(
                x=5,
                line_dash="dash",
                line_color="gray",
                annotation_text="k=5 (true clusters)",
                annotation_position="top right",
            )

        fig.update_xaxes(title_text="k (number of clusters)")
        fig.update_yaxes(title_text="Inertia (WCSS)", secondary_y=False)
        fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)
        fig.update_layout(
            title="Elbow Method: Inertia & Silhouette Score vs. k",
            width=700,
            height=480,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "💡 The **elbow** in inertia (blue) is around **k=5**, and the **silhouette score** (red) "
            "peaks near there as well — both confirm the true number of clusters. "
            "After k=5, adding more clusters gives diminishing returns."
        )


# ── Step 4: Cluster Real Embeddings ──────────────────────────────

_HIGHLIGHT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
]


def _parse_labels(x):
    """Parse label values from different dataset formats."""
    try:
        if isinstance(x, str) and x.strip().startswith("[") and x.strip().endswith("]"):
            data = json.loads(x)
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict) and "name" in data[0]:
                    return [d["name"] for d in data]
                return [str(d) for d in data]
        if isinstance(x, list):
            return [str(d) for d in x]
        if isinstance(x, str):
            return [x]
    except Exception:
        pass
    return []


NYC_CENTER = {"lat": 40.7128, "lon": -73.95}
NYC_ZOOM = 10
MAP_HEIGHT = 600


def _display_cluster_map(labels, titles, categories):
    """Scatter-mapbox of Airbnb listings coloured by cluster assignment."""
    lat = st.session_state.get("lab6_lat")
    lon = st.session_state.get("lab6_lon")
    if lat is None or lon is None:
        return

    map_df = pd.DataFrame(
        {
            "latitude": lat,
            "longitude": lon,
            "Cluster": labels.astype(str),
            "Title": titles,
            "Category": categories,
        }
    )

    fig = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Set2,
        mapbox_style="carto-positron",
        center=NYC_CENTER,
        zoom=NYC_ZOOM,
        opacity=0.6,
        hover_data=["Title", "Category"],
        title="K-Means Clusters on NYC Map",
    )
    fig.update_layout(
        height=MAP_HEIGHT,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_step_4_embeddings():
    st.subheader("Step 4: Cluster Real Embeddings")

    st.markdown(
        """
    Now let's apply k-means to **real data**! We'll cluster the high-dimensional embeddings
    from our datasets and see what groups the algorithm discovers.

    Since embeddings live in 768 dimensions, we'll use **PCA** (from Lab 5!) to project
    down to 2D for visualization — but the clustering happens in the full 768D space.
    """
    )

    dataset_choice = st.selectbox(
        "📂 Choose a dataset",
        list(DATASET_CONFIG.keys()),
        key="lab6_dataset_choice",
    )

    _ALL_LAB6_KEYS = [
        "lab6_text_emb",
        "lab6_image_emb",
        "lab6_titles",
        "lab6_categories",
        "lab6_cat_views",
        "lab6_lat",
        "lab6_lon",
        "lab6_poster_paths",
        "lab6_embeddings",
        "lab6_emb_mode",
        "lab6_step_4_done",
        "lab6_emb_labels",
        "lab6_emb_centroids",
        "lab6_pca_2d",
        "lab6_emb_elbow",
    ]

    needs_reload = (
        st.session_state.get("lab6_loaded_dataset") != dataset_choice
        or "lab6_text_emb" not in st.session_state
    )
    if needs_reload:
        for key in _ALL_LAB6_KEYS:
            st.session_state.pop(key, None)
        st.session_state["lab6_loaded_dataset"] = dataset_choice

    # Pre-read column names so the mapping UI can display them
    base_cfg = DATASET_CONFIG[dataset_choice]
    try:
        df_preview = pd.read_parquet(base_cfg["path"], columns=None)
        df_columns = list(df_preview.columns)
    except Exception as e:
        st.error(f"Could not read parquet file: {e}")
        return

    cfg = _get_column_mapping(dataset_choice, df_columns)

    if "lab6_text_emb" not in st.session_state:
        with st.spinner(f"Loading {dataset_choice}..."):
            try:
                df = df_preview

                emb_col = cfg.get("embedding_col", "embedding")
                if emb_col not in df.columns:
                    st.error(
                        f"Embedding column `{emb_col}` not found. "
                        f"Available columns: {', '.join(df.columns)}"
                    )
                    return
                text_emb = np.stack(df[emb_col].values)
                st.session_state["lab6_text_emb"] = text_emb

                title_col = cfg["title_col"]
                if title_col not in df.columns:
                    st.error(
                        f"Title column `{title_col}` not found. "
                        f"Available columns: {', '.join(df.columns)}"
                    )
                    return
                st.session_state["lab6_titles"] = df[title_col].values

                img_emb_col = cfg.get("image_embedding_col", "")
                if (
                    img_emb_col
                    and img_emb_col in df.columns
                    and df[img_emb_col].iloc[0] is not None
                ):
                    try:
                        img_emb = np.stack(df[img_emb_col].values)
                        st.session_state["lab6_image_emb"] = img_emb
                    except Exception:
                        pass

                cat_views = {}
                cat_col = cfg["category_col"]

                if cat_col == "genres" and cat_col in df.columns:
                    primary = []
                    for x in df["genres"].values:
                        parsed = _parse_labels(x)
                        primary.append(parsed[0] if parsed else "Unknown")
                    cat_views["Primary Genre"] = np.array(primary)

                    if "vote_average" in df.columns:
                        ratings = pd.to_numeric(
                            df["vote_average"], errors="coerce"
                        ).fillna(0)
                        bins = [0, 5, 6.5, 7.5, 10.1]
                        tier_labels = [
                            "Low (0–5)",
                            "Medium (5–6.5)",
                            "Good (6.5–7.5)",
                            "Great (7.5+)",
                        ]
                        cat_views["Rating Tier"] = (
                            pd.cut(ratings, bins=bins, labels=tier_labels, right=False)
                            .astype(str)
                            .values
                        )

                    if "popularity" in df.columns:
                        pop = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
                        quartiles = pop.quantile([0.25, 0.5, 0.75]).values

                        def _pop_tier(v):
                            if v < quartiles[0]:
                                return "Niche"
                            elif v < quartiles[1]:
                                return "Moderate"
                            elif v < quartiles[2]:
                                return "Popular"
                            return "Blockbuster"

                        cat_views["Popularity Tier"] = np.array(
                            [_pop_tier(v) for v in pop]
                        )
                elif cat_col in df.columns:
                    cat_views[cat_col.replace("_", " ").title()] = df[cat_col].values
                else:
                    st.warning(
                        f"Category column `{cat_col}` not found — "
                        f"using a placeholder. Adjust the column mapping above."
                    )
                    cat_views["(no category)"] = np.array(["N/A"] * len(df))

                st.session_state["lab6_cat_views"] = cat_views
                st.session_state["lab6_categories"] = list(cat_views.values())[0]

                lat_col = cfg.get("latitude_col", "")
                lon_col = cfg.get("longitude_col", "")
                if (
                    lat_col
                    and lon_col
                    and lat_col in df.columns
                    and lon_col in df.columns
                ):
                    st.session_state["lab6_lat"] = df[lat_col].values
                    st.session_state["lab6_lon"] = df[lon_col].values

                if "local_poster_path" in df.columns:
                    st.session_state["lab6_poster_paths"] = df[
                        "local_poster_path"
                    ].values
            except Exception as e:
                st.error(f"Could not load data: {e}")
                return

    # ── Embedding mode selector ──────────────────────────────────
    has_image_emb = "lab6_image_emb" in st.session_state
    emb_modes = ["Text"]
    if has_image_emb:
        emb_modes += ["Image", "Text + Image"]

    if len(emb_modes) > 1:
        emb_mode = st.radio(
            "🧬 Embedding modality for clustering:",
            emb_modes,
            horizontal=True,
            key="lab6_emb_mode_radio",
            help="**Text**: clusters by plot/description similarity. "
            "**Image**: clusters by poster visual style. "
            "**Text + Image**: combines both signals.",
        )
    else:
        emb_mode = "Text"

    if st.session_state.get("lab6_emb_mode") != emb_mode:
        for key in [
            "lab6_embeddings",
            "lab6_pca_2d",
            "lab6_step_4_done",
            "lab6_emb_labels",
            "lab6_emb_centroids",
            "lab6_emb_elbow",
        ]:
            st.session_state.pop(key, None)
        st.session_state["lab6_emb_mode"] = emb_mode

    if "lab6_embeddings" not in st.session_state:
        text_emb = st.session_state["lab6_text_emb"]
        if emb_mode == "Text":
            active_emb = text_emb
        elif emb_mode == "Image":
            active_emb = st.session_state["lab6_image_emb"]
        else:
            img_emb = st.session_state["lab6_image_emb"]
            from sklearn.preprocessing import normalize

            active_emb = np.hstack(
                [
                    normalize(text_emb),
                    normalize(img_emb),
                ]
            )

        st.session_state["lab6_embeddings"] = active_emb
        pca = PCA(n_components=2, random_state=42)
        st.session_state["lab6_pca_2d"] = pca.fit_transform(active_emb)

    embeddings = st.session_state["lab6_embeddings"]
    emb_dim = embeddings.shape[1]

    mode_desc = {
        "Text": "text (plot/description)",
        "Image": "image (poster visual style)",
        "Text + Image": "text + image (combined)",
    }
    st.markdown(
        f"""
    **Loaded {len(embeddings)} {cfg['label']}** — clustering on **{mode_desc[emb_mode]}** embeddings ({emb_dim}D).

    Unlike the toy dataset, we don't know how many clusters exist.
    Use the button below to run an elbow analysis and pick a good `k`.
    """
    )

    # ── Elbow analysis for real embeddings ────────────────────────
    if st.button("Run Elbow Analysis on Embeddings", key="run_emb_elbow"):
        with st.spinner("Fitting KMeans for k = 2..30 (this may take a moment)..."):
            emb_k_range = list(range(2, 31))
            emb_inertias = []
            emb_sils = []
            for k_val in emb_k_range:
                km = KMeans(n_clusters=k_val, random_state=42, n_init=5)
                km.fit(embeddings)
                emb_inertias.append(km.inertia_)
                emb_sils.append(silhouette_score(embeddings, km.labels_))

            st.session_state["lab6_emb_elbow"] = {
                "k_range": emb_k_range,
                "inertias": emb_inertias,
                "silhouettes": emb_sils,
            }

    if "lab6_emb_elbow" in st.session_state:
        elbow = st.session_state["lab6_emb_elbow"]

        from plotly.subplots import make_subplots

        fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])

        fig_elbow.add_trace(
            go.Scatter(
                x=elbow["k_range"],
                y=elbow["inertias"],
                mode="lines+markers",
                marker=dict(size=8, color="#4A90D9"),
                line=dict(width=2, color="#4A90D9"),
                name="Inertia",
            ),
            secondary_y=False,
        )

        fig_elbow.add_trace(
            go.Scatter(
                x=elbow["k_range"],
                y=elbow["silhouettes"],
                mode="lines+markers",
                marker=dict(size=8, color="#E74C3C"),
                line=dict(width=2, color="#E74C3C"),
                name="Silhouette Score",
            ),
            secondary_y=True,
        )

        fig_elbow.update_xaxes(title_text="k (number of clusters)")
        fig_elbow.update_yaxes(title_text="Inertia", secondary_y=False)
        fig_elbow.update_yaxes(title_text="Silhouette Score", secondary_y=True)
        fig_elbow.update_layout(
            title=f"Elbow Analysis on {cfg['label'].title()} Embeddings",
            width=700,
            height=450,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

        best_k = elbow["k_range"][int(np.argmax(elbow["silhouettes"]))]
        st.info(
            f"💡 The **silhouette score** peaks at **k={best_k}**. "
            "Use this as a starting point for your clustering below — "
            "but feel free to experiment with other values!"
        )

    st.markdown("---")

    st.markdown(
        """
    ### 💻 Your Task

    1. Pick a value of `k` based on the elbow analysis above.
    2. Fit `KMeans` on the **full** embedding matrix.
    3. Store the model in `emb_kmeans` and labels in `emb_labels`.
    """
    )

    default_code = f"""# 'embeddings' is available — shape ({len(embeddings)}, {emb_dim})
from sklearn.cluster import KMeans

# TODO: Choose a value of k based on the elbow analysis above
# k = ...

# TODO: Create a KMeans model with your chosen k, random_state=42, n_init=10
# emb_kmeans = ...

# TODO: Fit the model on 'embeddings'

# TODO: Extract the cluster labels into 'emb_labels'
# emb_labels = ...

print(f"Clustering {{len(embeddings)}} items into {{k}} clusters")
print(f"Inertia: {{emb_kmeans.inertia_:.2f}}")
for j in range(k):
    print(f"  Cluster {{j}}: {{int((emb_labels == j).sum())}} items")"""

    code = st.text_area(
        "Embedding Clustering Code:",
        value=default_code,
        height=280,
        key="kmeans_emb_code",
    )

    if st.button("Run Embedding Clustering", key="run_emb_cluster"):
        ctx = {"embeddings": embeddings, "np": np, "__builtins__": __builtins__}
        ctx, success = _run_student_code(code, ctx, "lab6_console_step4")

        if success:
            passed, msg = check_step_4_embedding_clustering(ctx)
            if passed:
                st.success(msg)
                st.session_state["lab6_step_4_done"] = True
                st.session_state["lab6_emb_labels"] = ctx["emb_labels"]
                st.session_state["lab6_emb_centroids"] = ctx[
                    "emb_kmeans"
                ].cluster_centers_

                st.markdown("---")
                st.markdown("### 🎬 Live Fitting (768D, projected to PCA 2D)")
                st.markdown(
                    "K-means runs in the full high-dimensional space. "
                    "Here's each iteration projected to 2D via PCA:"
                )
                pca_2d = st.session_state["lab6_pca_2d"]
                n_clusters = ctx["emb_kmeans"].n_clusters
                _render_live_fitting_hd(
                    embeddings,
                    pca_2d,
                    k=n_clusters,
                    seed=42,
                    init="kmeans++",
                    title=f"Embedding Clustering (k={n_clusters})",
                )
            else:
                st.error(msg)

    if st.session_state.get("lab6_step_4_done"):
        st.markdown("---")
        st.markdown("### 🗺️ Cluster Visualization (PCA Projection)")

        emb_labels = st.session_state["lab6_emb_labels"]
        pca_2d = st.session_state["lab6_pca_2d"]
        titles = st.session_state["lab6_titles"]
        categories = st.session_state["lab6_categories"]

        plot_df = pd.DataFrame(pca_2d, columns=["PC1", "PC2"])
        plot_df["Cluster"] = emb_labels.astype(str)
        plot_df["Title"] = titles
        plot_df["Category"] = categories

        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            hover_data=["Title", "Category"],
            title=f"K-Means Clusters on {cfg['label'].title()} Embeddings (PCA 2D view)",
            color_discrete_sequence=px.colors.qualitative.Set2,
            opacity=0.6,
        )
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        if "lab6_lat" in st.session_state:
            st.markdown("### 🗺️ Cluster Map — NYC Listings")
            st.markdown(
                "Each dot is an Airbnb listing, coloured by its cluster assignment. "
                "Do the clusters correspond to geographic patterns?"
            )
            _display_cluster_map(emb_labels, titles, categories)

        st.markdown("### 📏 Evaluation Metrics")
        st.markdown(
            "Computed on the **full** high-dimensional embeddings (not the 2D projection)."
        )
        _display_metrics(embeddings, emb_labels, st.session_state["lab6_emb_centroids"])

        # Category selector
        cat_views = st.session_state.get("lab6_cat_views", {})
        cat_options = list(cat_views.keys())
        if len(cat_options) > 1:
            selected_cat_view = st.selectbox(
                "🏷️ Analyze clusters by:",
                cat_options,
                key="lab6_cat_view_select",
            )
        else:
            selected_cat_view = cat_options[0] if cat_options else None

        active_categories = (
            cat_views[selected_cat_view] if selected_cat_view else categories
        )

        # Cluster exploration
        st.markdown("### 🔍 Explore Clusters")
        st.markdown(
            "Select a cluster to see its most representative items (closest to centroid)."
        )

        n_clusters = len(np.unique(emb_labels))
        selected_cluster = st.selectbox(
            "Choose cluster:",
            range(n_clusters),
            format_func=lambda x: f"Cluster {x} ({int((emb_labels == x).sum())} items)",
            key="lab6_explore_cluster",
        )

        cluster_mask = emb_labels == selected_cluster
        cluster_embeddings = embeddings[cluster_mask]
        cluster_titles = titles[cluster_mask]
        cluster_cats = np.array(active_categories)[cluster_mask]
        centroid = st.session_state["lab6_emb_centroids"][selected_cluster]

        dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_indices = np.argsort(dists)[:10]

        explore_df = pd.DataFrame(
            {
                "Title": cluster_titles[closest_indices],
                selected_cat_view: cluster_cats[closest_indices],
                "Distance to Centroid": dists[closest_indices].round(4),
            }
        )
        st.dataframe(explore_df, use_container_width=True)

        poster_paths = st.session_state.get("lab6_poster_paths")
        if poster_paths is not None:
            cluster_posters = poster_paths[cluster_mask]
            top_posters = cluster_posters[closest_indices]
            top_titles = cluster_titles[closest_indices]
            top_cats = cluster_cats[closest_indices]

            cols = st.columns(5)
            for i, (path, title, cat) in enumerate(
                zip(top_posters, top_titles, top_cats)
            ):
                with cols[i % 5]:
                    if pd.notna(path) and os.path.exists(path):
                        _st_image_full_width(path)
                    else:
                        st.markdown("*No poster*")
                    st.caption(f"**{title}**\n{cat}")

        # Category distribution per cluster — Hinton diagram
        st.markdown("### 📊 Category Distribution per Cluster (Hinton Diagram)")
        st.markdown(
            f"How does **{selected_cat_view}** distribute across clusters? "
            "Square size and colour intensity show the **percentage** of each "
            "category within a cluster — larger/brighter squares mean higher concentration."
        )

        all_labels_flat = []
        for i, cat in enumerate(active_categories):
            all_labels_flat.append(
                {"Cluster": str(emb_labels[i]), "Category": str(cat)}
            )

        if all_labels_flat:
            cat_df = pd.DataFrame(all_labels_flat)
            top_cats = cat_df["Category"].value_counts().head(12).index.tolist()
            cat_df_filtered = cat_df[cat_df["Category"].isin(top_cats)]

            counts = (
                cat_df_filtered.groupby(["Cluster", "Category"])
                .size()
                .reset_index(name="Count")
            )
            cluster_totals = counts.groupby("Cluster")["Count"].transform("sum")
            counts["Percentage"] = (counts["Count"] / cluster_totals * 100).round(1)

            pivot = counts.pivot(
                index="Cluster", columns="Category", values="Percentage"
            ).fillna(0)
            pivot = pivot[top_cats]
            clusters_sorted = sorted(pivot.index, key=lambda x: int(x))
            pivot = pivot.loc[clusters_sorted]

            z = pivot.values
            y_labels = [f"Cluster {c}" for c in pivot.index]
            x_labels = list(pivot.columns)

            max_pct = z.max() if z.max() > 0 else 1
            marker_sizes = z / max_pct * 45
            marker_sizes = np.clip(marker_sizes, 3, 45)

            xs, ys, sizes, colors, hovers = [], [], [], [], []
            for row_i in range(len(y_labels)):
                for col_j in range(len(x_labels)):
                    pct = z[row_i, col_j]
                    xs.append(x_labels[col_j])
                    ys.append(y_labels[row_i])
                    sizes.append(float(marker_sizes[row_i, col_j]))
                    colors.append(float(pct))
                    hovers.append(
                        f"<b>{y_labels[row_i]}</b><br>" f"{x_labels[col_j]}: {pct:.1f}%"
                    )

            fig_hinton = go.Figure(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=dict(
                        size=sizes,
                        color=colors,
                        colorscale="Blues",
                        cmin=0,
                        cmax=float(max_pct),
                        symbol="square",
                        line=dict(width=0.5, color="white"),
                        colorbar=dict(title="% of cluster", thickness=15, len=0.8),
                    ),
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hovers,
                    showlegend=False,
                )
            )

            fig_hinton.update_layout(
                title=f"{selected_cat_view} Profile per Cluster",
                xaxis=dict(title=selected_cat_view, tickangle=-30, side="bottom"),
                yaxis=dict(title="", autorange="reversed"),
                height=max(350, 80 * len(y_labels) + 120),
                plot_bgcolor="white",
                margin=dict(l=80, r=40, t=60, b=100),
            )
            st.plotly_chart(fig_hinton, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏆 Bonus Question")
    st.markdown(
        "K-means assumes clusters are **spherical** (equal spread in all directions). "
        "What kind of data would violate this assumption, and what algorithm could handle it better? "
        "*(Hint: think about the next topic in the syllabus)*"
    )

    answer = st.text_area("Type your answer here:", key="bonus_q_kmeans", height=100)
    if answer:
        st.info("Great thinking! Discuss your answer with your classmates or TA.")


if __name__ == "__main__":
    render_kmeans_lab()
