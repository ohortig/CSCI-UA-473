import io
import json
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

# Import checkers
from labs.lab5_pca_autoencoder.level_checks import (
    check_step_1_pca,
    check_step_3_training_loop,
)

# Hardcode config to make lab standalone but use real data
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
        "category_col": "neighbourhood_cleansed",
        "label": "listings",
    },
}


def render_pca_lab():
    st.header("Lab 5: Dimensionality Reduction (PCA & Autoencoders)")

    # Dataset selector
    dataset_choice = st.selectbox(
        "📂 Choose a dataset",
        list(DATASET_CONFIG.keys()),
        key="lab5_dataset_choice",
    )
    cfg = DATASET_CONFIG[dataset_choice]

    # Embedding type selector
    embedding_choice = st.radio(
        "🧠 Choose Embedding Type",
        ["Text Embeddings (768-dim)", "Image Embeddings (384-dim)"],
        horizontal=True,
        key="lab5_embedding_choice",
    )

    is_image = "Image" in embedding_choice
    emb_col = "image_embedding" if is_image else "embedding"
    emb_dim = 384 if is_image else 768

    st.markdown(
        f"""
    ### 🎯 The Mission

    We have {emb_dim}-dimensional embeddings for our {cfg["label"]}. That's too many dimensions for our human brains to visualize!

    **Goal**: Compress these {emb_dim} dimensions down to just **2 dimensions** (X, Y) so we can plot them on a scatter plot.

    We will try two methods:
    1.  **PCA (Principal Component Analysis)**: The classic linear method.
    2.  **Autoencoder**: A deep learning approach (non-linear).
    """
    )

    # Clear cached data when dataset or embedding type changes
    if (
        st.session_state.get("lab5_loaded_dataset") != dataset_choice
        or st.session_state.get("lab5_loaded_embedding_type") != embedding_choice
    ):
        for key in [
            "lab5_data",
            "lab5_titles",
            "lab5_genres",
            "lab5_vars",
            "lab5_reduced_data",
            "step_1_done",
            "step_2_done",
            "pca_fig",
            "Autoencoder_Class",
        ]:
            st.session_state.pop(key, None)
        st.session_state["lab5_loaded_dataset"] = dataset_choice
        st.session_state["lab5_loaded_embedding_type"] = embedding_choice

    # Load Data Once
    if "lab5_data" not in st.session_state:
        with st.spinner(f"Loading {dataset_choice} ({embedding_choice})..."):
            try:
                df = pd.read_parquet(cfg["path"])
                # Stack embeddings into (N, emb_dim) matrix
                embeddings = np.stack(df[emb_col].values)
                st.session_state["lab5_data"] = embeddings
                st.session_state["lab5_titles"] = df[cfg["title_col"]].values
                st.session_state["lab5_genres"] = df[cfg["category_col"]].values
            except Exception as e:
                st.error(f"Could not load data: {e}")
                return

    embeddings = st.session_state["lab5_data"]

    tab1, tab2, tab3 = st.tabs(
        ["📉 Step 1: PCA", "🧠 Step 2: Autoencoder Arch", "🔥 Step 3: Training"]
    )

    with tab1:
        render_step_1_pca(embeddings, cfg, emb_dim)

    with tab2:
        render_step_2_arch(emb_dim)

    with tab3:
        render_step_3_train(embeddings, cfg, emb_dim)


# ── Helpers for feature-highlighted scatter plots ─────────────────


def _parse_labels(x):
    """Parse label values from different dataset formats.

    TMDB genres come as JSON strings like '[{"id": 28, "name": "Action"}, ...]'.
    Airbnb neighbourhoods are plain strings like 'Manhattan'.
    This function normalises both into a list of strings.
    """
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


@st.fragment
def _render_highlighted_scatter(reduced_data, cfg, *, col_x, col_y, title, key_prefix):
    """Render a Plotly scatter with a multiselect to highlight by category."""

    titles = st.session_state["lab5_titles"]
    categories = st.session_state["lab5_genres"]

    # Build DataFrame
    plot_df = pd.DataFrame(reduced_data, columns=[col_x, col_y])
    plot_df["Title"] = titles
    plot_df["_raw_cat"] = categories

    # Parse labels
    plot_df["label_list"] = plot_df["_raw_cat"].apply(_parse_labels)
    plot_df["label_readable"] = plot_df["label_list"].apply(lambda ls: ", ".join(ls))

    # All unique labels
    all_labels = sorted({lbl for sublist in plot_df["label_list"] for lbl in sublist})

    label_name = cfg["category_col"].replace("_", " ").title()
    selected = st.multiselect(
        f"Highlight by {label_name}",
        all_labels,
        key=f"{key_prefix}_highlight_select",
    )

    if selected:

        def _assign(labels):
            for s in selected:
                if s in labels:
                    return s
            return "Other"

        plot_df["highlight"] = plot_df["label_list"].apply(_assign)

        # Sort so "Other" draws first (behind highlighted points)
        plot_df["_sort"] = plot_df["highlight"].apply(
            lambda h: 0 if h == "Other" else 1
        )
        plot_df = plot_df.sort_values("_sort")

        # Color map
        color_map = {
            s: _HIGHLIGHT_COLORS[i % len(_HIGHLIGHT_COLORS)]
            for i, s in enumerate(selected)
        }
        color_map["Other"] = "lightgray"

        fig = px.scatter(
            plot_df,
            x=col_x,
            y=col_y,
            color="highlight",
            color_discrete_map=color_map,
            hover_data=["Title", "label_readable"],
            title=title,
            labels={"highlight": label_name, "label_readable": label_name},
            category_orders={"highlight": selected + ["Other"]},
        )
        # Dim "Other" points
        for trace in fig.data:
            if trace.name == "Other":
                trace.marker.opacity = 0.15
            else:
                trace.marker.opacity = 0.8
    else:
        fig = px.scatter(
            plot_df,
            x=col_x,
            y=col_y,
            hover_data=["Title", "label_readable"],
            title=title,
            labels={"label_readable": label_name},
            opacity=0.5,
        )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_step_1_pca(embeddings, cfg, emb_dim):
    st.subheader("Step 1: The Linear approach (PCA)")

    st.markdown(
        r"""
    ### 📐 The Math Behind PCA

    **Principal Component Analysis (PCA)** finds the directions (axes) in your data that capture the most variance.

    **Step 1: Center the Data**
    - Subtract the mean from each dimension: $X_{centered} = X - \mu$
    - This ensures the data is centered at the origin.

    **Step 2: Compute Covariance Matrix**
    - For data matrix $X$ (shape: $n \times d$), compute: $C = \frac{1}{n-1} X^T X$
    - This tells us how dimensions vary together.

    **Step 3: Eigenvalue Decomposition**
    - Find eigenvectors and eigenvalues of $C$: $C v = \lambda v$
    - Eigenvectors = **Principal Components** (the new axes)
    - Eigenvalues = **Variance** along each axis (larger = more important)

    **Step 4: Project to Lower Dimensions**
    - Keep only the top $k$ eigenvectors (with largest eigenvalues)
    - Project data: $X_{reduced} = X \cdot W_k$ where $W_k$ are the top $k$ eigenvectors

    **Intuition**: PCA finds the "best" 2D view of your {emb_dim}D data by rotating it to show the most spread.

    ---

    ### 💻 Your Task

    Now implement it using scikit-learn:
    1. Import `PCA` from `sklearn.decomposition`.
    2. Initialize `pca = PCA(n_components=2)`.
    3. Fit and transform the `embeddings`.
    4. Store result in `reduced_data`.
    """
    )

    answer_code_1 = f"""# 'embeddings' is available (shape: N, {emb_dim})
from sklearn.decomposition import PCA

# Initialize PCA to reduce to 2 dimensions
pca = PCA(n_components=2)

# Fit and Transform
reduced_data = pca.fit_transform(embeddings)"""  # noqa: F841

    def_code = f"""# 'embeddings' is available (shape: N, {emb_dim})"""  # noqa: F841

    code = st.text_area("PCA Code:", value=def_code, height=200, key="pca_code")

    if st.button("Run PCA"):
        st.session_state["lab5_vars"] = {"embeddings": embeddings}

        try:
            exec(code, {}, st.session_state["lab5_vars"])

            passed, msg = check_step_1_pca(st.session_state["lab5_vars"])

            if passed:
                st.success(msg)
                st.session_state["step_1_done"] = True
                # Persist reduced_data so it survives reruns from widgets
                rd = st.session_state["lab5_vars"].get("reduced_data")
                if rd is not None:
                    st.session_state["lab5_reduced_data"] = rd
            else:
                st.error(msg)
        except Exception as e:
            st.error(f"Runtime Error: {e}")

    # Show visualization section if step 1 is done (persists across reruns)
    if st.session_state.get("step_1_done", False):
        st.markdown("---")
        st.markdown("### 📊 Visualize the Results")
        st.markdown(
            """
            Now create a scatter plot to visualize your 2D projection!

            **Your Task:**
            1. Create a pandas DataFrame with columns `PC1` and `PC2` from `reduced_data`.
            2. Add a `Title` column using the item titles (available as `titles`).
            3. Use `plotly.express.scatter()` to create a scatter plot.
            4. Store the figure in a variable named `fig`.
            """
        )

        default_viz_code = """# 'reduced_data' is available (shape: N, 2)
# 'titles' is available (array of item titles)"""  # noqa: F841

        answer_viz_code = """# 'reduced_data' is available (shape: N, 2)
# 'titles' is available (array of item titles)

import plotly.express as px
import pandas as pd

# Create DataFrame
df_vis = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])
df_vis["Title"] = titles

# Create scatter plot
fig = px.scatter(df_vis, x="PC1", y="PC2", hover_data=["Title"],
                 title="PCA Projection (2D)", opacity=0.6)"""  # noqa: F841

        viz_code = st.text_area(
            "Visualization Code:",
            value=default_viz_code,
            height=250,
            key="pca_viz_code",
        )

        if st.button("Run Visualization", key="run_viz_btn"):
            # Ensure lab5_vars exists
            if "lab5_vars" not in st.session_state:
                st.session_state["lab5_vars"] = {"embeddings": embeddings}

            # Make sure necessary modules and data are available
            st.session_state["lab5_vars"]["px"] = px
            st.session_state["lab5_vars"]["pd"] = pd
            st.session_state["lab5_vars"]["titles"] = st.session_state["lab5_titles"]
            st.session_state["lab5_vars"]["reduced_data"] = st.session_state[
                "lab5_vars"
            ].get("reduced_data")

            if st.session_state["lab5_vars"]["reduced_data"] is None:
                st.error("⚠️ `reduced_data` not found. Please run PCA first.")
            else:
                try:
                    exec(viz_code, {}, st.session_state["lab5_vars"])

                    # Check if 'fig' was created
                    if "fig" in st.session_state["lab5_vars"]:
                        # Store figure in session state so it persists
                        st.session_state["pca_fig"] = st.session_state["lab5_vars"][
                            "fig"
                        ]
                        st.success("✅ Visualization created!")
                    else:
                        st.warning(
                            "⚠️ Variable `fig` not found. Make sure you create a figure and assign it to `fig`."
                        )
                except Exception as e:
                    st.error(f"Runtime Error: {e}")

        # Display the figure if it exists in session state
        if "pca_fig" in st.session_state:
            st.plotly_chart(st.session_state["pca_fig"], use_container_width=True)

        # ── Feature Highlighting ──────────────────────────────
        reduced_data = st.session_state.get("lab5_reduced_data")
        if reduced_data is not None:
            st.markdown("---")
            st.markdown("### 🎨 Explore by Feature")
            st.markdown(
                "Use the selector below to highlight points by category and see how they cluster."
            )
            _render_highlighted_scatter(
                reduced_data,
                cfg,
                col_x="PC1",
                col_y="PC2",
                title="PCA Projection — Highlighted",
                key_prefix="pca",
            )


def render_step_2_arch(emb_dim):
    st.subheader("Step 2: The Neural Approach (Autoencoder)")

    st.markdown(
        """
    An **Autoencoder** is a neural network that learns to **copy its input to its output** — but with a twist:
    it must squeeze all the information through a tiny **bottleneck** in the middle.

    Before we build one, let's understand the building blocks.
    """
    )

    # ── Concept 1: Linear Layers ──────────────────────────────────
    with st.expander("📐 Concept 1: What is a Linear Layer?", expanded=True):
        st.markdown(
            r"""
        A **Linear Layer** (also called a *fully connected* or *dense* layer) is the most fundamental building block
        of a neural network. It computes:

        $$y = xW^T + b$$

        | Symbol | Meaning | Shape |
        |--------|---------|-------|
        | $x$ | Input vector | $(1, \text{in\_features})$ |
        | $W$ | Learnable weight matrix | $(\text{out\_features}, \text{in\_features})$ |
        | $b$ | Learnable bias vector | $(\text{out\_features},)$ |
        | $y$ | Output vector | $(1, \text{out\_features})$ |

        **In plain English:** each output neuron takes a *weighted sum* of all the inputs and adds a bias.
        The weights and biases are the **parameters** the network learns during training.

        ```python
        # Example: compress {emb_dim} dimensions down to 128
        layer = nn.Linear(in_features={emb_dim}, out_features=128)
        # This layer has {emb_dim} × 128 + 128 = {emb_dim * 128 + 128:,} learnable parameters
        ```
        """
        )

    # ── Concept 2: Activation Functions / ReLU ────────────────────
    with st.expander("⚡ Concept 2: Activation Functions — ReLU", expanded=True):
        st.markdown(
            r"""
        If we stack two linear layers without anything in between, the result is just **another linear transformation**
        (because a linear function of a linear function is still linear). That would be no better than PCA!

        **Activation functions** break this linearity by applying a non-linear transformation after each layer.

        **ReLU (Rectified Linear Unit)** is the most popular choice:

        $$\text{ReLU}(x) = \max(0, x)$$

        - If the input is **positive** → keep it as-is
        - If the input is **negative** → set it to zero
        """
        )

        # Interactive ReLU visualization
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Input values:**")
            example_input = np.array([-2.5, -1.0, 0.0, 0.7, 3.2])
            st.code(f"x = {example_input.tolist()}", language="python")
        with col2:
            st.markdown("**After ReLU:**")
            example_output = np.maximum(0, example_input)
            st.code(f"ReLU(x) = {example_output.tolist()}", language="python")

        st.markdown(
            """
        **Why ReLU?** It's simple, fast, and gives the network the power to learn complex non-linear patterns
        — like curved boundaries between clusters of movies vs. listings.
        """
        )

    # ── Concept 3: Shape Walkthrough ──────────────────────────────
    with st.expander(
        "🔢 Concept 3: Following the Shapes — A Concrete Example", expanded=True
    ):
        st.markdown(
            r"""
        Let's trace **one embedding** through the entire autoencoder to see how the shapes change.

        Say we have a single item with a {emb_dim}-dimensional embedding:
        """
        )

        st.code(
            f"""# Original embedding (e.g. for the movie "Inception")
embedding = [0.023, -0.107, 0.891, 0.045, ..., -0.334]   # {emb_dim} numbers""",
            language="python",
        )

        st.markdown("---")

        st.markdown(f"**🔽 ENCODER** — compresses {emb_dim} → 2")

        enc_data = {
            "Layer": [
                "Input",
                f"① nn.Linear({emb_dim}, 128)",
                "② nn.ReLU()",
                "③ nn.Linear(128, 2)",
            ],
            "What it does": [
                "Raw embedding",
                "Weighted sum → 128 features",
                "Zero out negatives",
                "Weighted sum → 2 features",
            ],
            "Output shape": [
                f"(1, {emb_dim})",
                "(1, 128)",
                "(1, 128)",
                "(1, 2)",
            ],
            "Example values": [
                "[0.023, -0.107, 0.891, ...]",
                "[1.42, -0.83, 0.27, ...]",
                "[1.42,  0.00, 0.27, ...]",
                "[3.15, -1.07]  ← the 2D point!",
            ],
        }
        st.table(pd.DataFrame(enc_data))

        st.markdown(f"**🔼 DECODER** — expands 2 → {emb_dim}")

        dec_data = {
            "Layer": [
                "Input (latent)",
                "④ nn.Linear(2, 128)",
                "⑤ nn.ReLU()",
                f"⑥ nn.Linear(128, {emb_dim})",
            ],
            "What it does": [
                "2D bottleneck point",
                "Weighted sum → 128 features",
                "Zero out negatives",
                f"Weighted sum → {emb_dim} features",
            ],
            "Output shape": [
                "(1, 2)",
                "(1, 128)",
                "(1, 128)",
                f"(1, {emb_dim})",
            ],
            "Example values": [
                "[3.15, -1.07]",
                "[0.56, -1.23, 2.01, ...]",
                "[0.56,  0.00, 2.01, ...]",
                "[0.019, -0.112, 0.887, ...] ≈ input!",
            ],
        }
        st.table(pd.DataFrame(dec_data))

        st.info(
            "💡 The network is trained to make the output (step ⑥) as close "
            "as possible to the original input. The **loss** = difference between input and output (MSE)."
        )

    # ── Concept 4: Encoder-Decoder Architecture Diagram ───────────
    with st.expander("🏗️ Concept 4: The Hourglass Architecture", expanded=True):
        st.markdown(
            f"""
        The full architecture looks like an **hourglass** — wide at both ends ({emb_dim} dims) and narrow in the middle (2 dims):
        """
        )

        # Build a visual diagram using columns
        cols = st.columns([1, 1, 1, 1, 1, 1, 1])
        layers = [
            (str(emb_dim), "Input", "#4A90D9"),
            ("→", "Linear", "#666"),
            ("128", "Hidden", "#F5A623"),
            ("→", "ReLU + Linear", "#666"),
            ("2", "Latent\n(bottleneck)", "#E74C3C"),
            ("→", "Linear + ReLU →", "#666"),
            (str(emb_dim), "Output", "#2ECC71"),
        ]
        for col, (size, label, color) in zip(cols, layers):
            with col:
                if size == "→":
                    st.markdown(
                        f"<div style='text-align:center; padding-top:35px; color:#888; font-size:12px;'>"
                        f"{label}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='text-align:center; background:{color}; color:white; "
                        f"border-radius:10px; padding:16px 4px; font-size:22px; font-weight:bold;'>"
                        f"{size}</div>"
                        f"<div style='text-align:center; font-size:11px; color:#888; margin-top:4px;'>{label}</div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("")

        col_enc, col_dec = st.columns(2)
        with col_enc:
            st.markdown(
                "<div style='text-align:center; font-size:14px;'>"
                "◀──── <b>ENCODER</b> ────▶</div>",
                unsafe_allow_html=True,
            )
        with col_dec:
            st.markdown(
                "<div style='text-align:center; font-size:14px;'>"
                "◀──── <b>DECODER</b> ────▶</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
        - **Encoder**: Squeezes {emb_dim} dims into just **2 numbers** — a point on a 2D map.
        - **Decoder**: Tries to reconstruct the original {emb_dim} dims from those 2 numbers.
        - **Training goal**: Minimize the reconstruction error (MSE between input and output).

        If the network can reconstruct well, it means those **2 numbers capture the essence** of the original embedding!
        """
        )

    st.markdown("---")

    st.markdown(
        f"""
    ### 💻 Your Task

    Now put it all together! Define the `Autoencoder` class below:
    1. Define a class `Autoencoder(nn.Module)`.
    2. In `__init__`, define `self.encoder`: `Linear({emb_dim}→128)` → `ReLU` →  `Linear(128→2)`. Hint: you can use `nn.Sequential` to chain the layers.
    3. Define `self.decoder`: `Linear(2→128)` → `ReLU` → `Linear(128→{emb_dim})`.
    4. In `forward`, pass `x` through encoder then decoder. Return the decoded output and the encoded latent in this order (named decoded and encoded respectively).
    """
    )

    answer_code_2 = f"""import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim={emb_dim}, latent_dim=2):
        super().__init__()
        # Encoder: Compress
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Decoder: Expand
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
"""  # noqa: F841

    default_code = """#TODO: fill in the code:
"""  # noqa: F841

    code = st.text_area(
        "Model Architecture:", value=default_code, height=300, key="ae_code"
    )

    if st.button("Check Architecture"):
        # Pre-populate globals with torch and nn so they're always available
        # This ensures class methods can access them even if user code re-imports
        exec_globals = {"torch": torch, "nn": nn, "__builtins__": __builtins__}
        local_scope = {}

        try:
            # Execute user code
            # Pass exec_globals as globals so class __globals__ points to it
            # This ensures class methods can find 'nn' when they execute
            exec(code, exec_globals, local_scope)

            # Merge local_scope into exec_globals so we can access everything
            exec_globals.update(local_scope)

            # Check if Autoencoder was defined (could be in local_scope or exec_globals)
            if "Autoencoder" not in exec_globals and "Autoencoder" not in local_scope:
                st.error("⚠️ Class `Autoencoder` not found in your code.")
                return

            # Get Autoencoder from wherever it ended up
            Autoencoder = exec_globals.get("Autoencoder") or local_scope.get(
                "Autoencoder"
            )

            # Try to instantiate HERE in the same scope where it was defined
            try:
                test_model = Autoencoder(input_dim=emb_dim, latent_dim=2)

                # If we got here, instantiation worked! Now check structure
                if not hasattr(test_model, "encoder") or not hasattr(
                    test_model, "decoder"
                ):
                    st.error(
                        "⚠️ Model must have `self.encoder` and `self.decoder` attributes."
                    )
                    return

                if not isinstance(test_model.encoder, torch.nn.Module):
                    st.error(
                        "⚠️ `encoder` must be a PyTorch Module (e.g. nn.Sequential)."
                    )
                    return

                st.success(
                    "✅ Architecture defined! You have a neural network structure."
                )
                st.session_state["Autoencoder_Class"] = Autoencoder
                st.session_state["step_2_done"] = True

            except NameError as e:
                st.error(
                    f"⚠️ NameError during instantiation: {e}. Make sure `import torch.nn as nn` is at the TOP of your code, before the class definition."
                )
            except Exception as e:
                st.error(f"⚠️ Could not instantiate Autoencoder: {e}")

        except Exception as e:
            st.error(f"Runtime Error: {e}")


def render_step_3_train(embeddings, cfg, emb_dim):
    if not st.session_state.get("step_2_done", False):
        st.warning("Please complete Step 2 first.")
        return

    st.subheader("Step 3: Training the Autoencoder")

    st.markdown(
        "Now we need to **train** the autoencoder so it actually learns to compress and reconstruct. "
        "Before we write the training loop, let's understand each ingredient."
    )

    # ── Concept 1: Hyperparameters ────────────────────────────────
    with st.expander("🎛️ Concept 1: Hyperparameters", expanded=True):
        st.markdown(
            r"""
        **Hyperparameters** are settings you choose *before* training — the network never learns them.
        They control *how* training happens.

        | Hyperparameter | What it controls | Typical values |
        |----------------|-----------------|----------------|
        | **Learning rate** (`lr`) | Step size for each weight update | 0.0001 – 0.01 |
        | **Epochs** | Number of full passes over the data | 20 – 200 |
        | **Latent dim** | Size of the bottleneck | 2 (for visualization) |
        | **Hidden dim** | Width of intermediate layers | 64 – 512 |

        **Learning rate** is the most important one:
        - **Too high** → the model overshoots and loss oscillates wildly
        - **Too low** → training is painfully slow and may get stuck
        - **Just right** → smooth, steady decrease in loss

        ```python
        lr = 0.001      # A safe default for Adam optimizer
        epochs = 50     # Enough to see convergence for this dataset
        ```
        """
        )

    # ── Concept 2: Optimizer & SGD ────────────────────────────────
    with st.expander("🧭 Concept 2: Optimizers — From SGD to Adam", expanded=True):
        st.markdown(
            r"""
        An **optimizer** is the algorithm that updates the model\'s weights to reduce the loss.

        #### Stochastic Gradient Descent (SGD)

        The simplest optimizer. Recall from our optimization class:

        $$w_{new} = w_{old} - \eta \cdot \nabla L$$

        | Symbol | Meaning |
        |--------|---------|
        | $w$ | A weight in the network |
        | $\eta$ | Learning rate |
        | $\nabla L$ | Gradient of the loss w.r.t. that weight |

        **In English:** compute how much the loss would change if you nudged each weight,
        then nudge it in the opposite direction (downhill) by a `learning_rate`-sized step.

        ```python
        # Plain SGD
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        ```

        #### Adam (Adaptive Moment Estimation)

        Adam improves on SGD by maintaining **per-parameter learning rates** that adapt over time.
        It tracks both the average gradient (momentum) and the average squared gradient (scaling):

        - **Momentum**: Smooths out noisy gradients → steadier updates
        - **Adaptive scaling**: Parameters with large gradients get smaller steps → more stable

        ```python
        # Adam — the go-to optimizer for most deep learning
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        ```

        > 💡 **Rule of thumb**: Start with Adam at `lr=0.001`. Switch to SGD only if you need fine-grained control.
        """
        )

    # ── Concept 3: Train / Validation Split ───────────────────────
    with st.expander("✂️ Concept 3: Train / Validation Split", expanded=True):
        st.markdown(
            r"""
        **Why split?** If we only measure loss on the training data, we can't tell if the model
        is actually *learning patterns* or just *memorizing* the data (overfitting).

        We split the data into two parts:

        | Set | Purpose | Used for weight updates? |
        |-----|---------|--------------------------|
        | **Training set** (~80%) | The model learns from this | ✅ Yes |
        | **Validation set** (~20%) | We evaluate on this to check generalization | ❌ No |

        **What to watch for:**
        """
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**✅ Good fit**")
            st.markdown("Both losses decrease together")
        with col2:
            st.markdown("**⚠️ Overfitting**")
            st.markdown("Train loss ↓ but val loss ↑")
        with col3:
            st.markdown("**❌ Underfitting**")
            st.markdown("Both losses stay high")

        st.markdown(
            r"""
        ---

        ```python
        from sklearn.model_selection import train_test_split

        # 80% train, 20% validation
        train_data, val_data = train_test_split(
            embeddings, test_size=0.2, random_state=42
        )
        ```

        During training, we compute loss on **both** sets each epoch, but only call
        `loss.backward()` on the training loss.
        """
        )

    # ── Concept 4: The Training Loop ──────────────────────────────
    with st.expander("🔄 Concept 4: The Training Loop — Step by Step", expanded=True):
        st.markdown(
            r"""
        Each **epoch** follows the same recipe:

        | Step | Code | What happens |
        |------|------|-------------|
        | **1. Forward pass** | `output, latent = model(data)` | Data flows through the network |
        | **2. Compute loss** | `loss = criterion(output, data)` | Measure reconstruction error (MSE) |
        | **3. Zero gradients** | `optimizer.zero_grad()` | Clear old gradients (PyTorch accumulates them!) |
        | **4. Backward pass** | `loss.backward()` | Compute gradients via backpropagation |
        | **5. Update weights** | `optimizer.step()` | Optimizer nudges weights using the gradients |

        For the **validation** step, we skip steps 3–5 (no learning, just measuring):

        ```python
        # Validation — no gradient computation needed
        with torch.no_grad():
            val_output, _ = model(val_tensor)
            val_loss = criterion(val_output, val_tensor)
        ```

        > 💡 `torch.no_grad()` tells PyTorch to skip tracking gradients — saves memory and is faster.
        """
        )

    st.markdown("---")

    st.markdown(
        """
    ### 💻 Your Task

    Fill in the training code below. The key steps:
    1. Split data into train (80%) and validation (20%) sets.
    2. Set hyperparameters (learning rate, epochs).
    3. Initialize the model, loss function (`MSELoss`), and optimizer (`Adam`).
    4. Write the training loop with both train and validation loss tracking.
    """
    )

    default_code = f"""# 'embeddings' (numpy) and 'Autoencoder' (class) are available
import torch.optim as optim
from sklearn.model_selection import train_test_split

# ───── 1. Train / Validation Split ─────
train_emb, val_emb = # TODO: split the embeddings into train and validation sets

train_tensor = # TODO: convert the train embeddings to a PyTorch tensor
val_tensor   = # TODO: convert the validation embeddings to a PyTorch tensor

# ───── 2. Hyperparameters ─────
input_dim  = {emb_dim}
latent_dim = 2
lr         = 0.001
epochs     = 50

# ───── 3. Initialize ─────
model     = # TODO: initialize the autoencoder model
criterion = # TODO: initialize the mean squared error loss function
optimizer = # TODO: initialize the Adam optimizer

# ───── 4. Training Loop ─────
losses     = []   # training losses
val_losses = []   # validation losses

for epoch in range(epochs):
    # -- Train --
    model.train()
    reconstructed, latent = # TODO: pass the train tensor through the model and get the reconstructed and latent outputs
    loss = criterion(reconstructed, train_tensor)

    optimizer.zero_grad()
    loss.backward() # TODO: compute the gradients
    optimizer.step() # TODO: update the model parameters

    # -- Validate (no gradients!) --
    # TODO: skip the gradient computation and compute the validation loss

    losses.append(loss.item())
    val_losses.append(val_loss.item())

# Keep full data tensor for latent-space visualization
data_tensor = torch.tensor(embeddings, dtype=torch.float32)
print("Training finished!")
"""

    answer_code_3 = f"""# 'embeddings' (numpy) and 'Autoencoder' (class) are available
import torch.optim as optim
from sklearn.model_selection import train_test_split

# ───── 1. Train / Validation Split ─────
train_emb, val_emb = train_test_split(embeddings, test_size=0.2, random_state=42)

train_tensor = torch.tensor(train_emb, dtype=torch.float32)
val_tensor   = torch.tensor(val_emb,   dtype=torch.float32)

# ───── 2. Hyperparameters ─────
input_dim  = {emb_dim}
latent_dim = 2
lr         = 0.001
epochs     = 50

# ───── 3. Initialize ─────
model     = Autoencoder(input_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ───── 4. Training Loop ─────
losses     = []   # training losses
val_losses = []   # validation losses

for epoch in range(epochs):
    # -- Train --
    model.train()
    reconstructed, latent = model(train_tensor)
    loss = criterion(reconstructed, train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # -- Validate (no gradients!) --
    model.eval()
    with torch.no_grad():
        val_reconstructed, _ = model(val_tensor)
        val_loss = criterion(val_reconstructed, val_tensor)

    losses.append(loss.item())
    val_losses.append(val_loss.item())

# Keep full data tensor for latent-space visualization
data_tensor = torch.tensor(embeddings, dtype=torch.float32)
print("Training finished!")
"""  # noqa: F841

    code = st.text_area(
        "Training Loop:", value=default_code, height=500, key="train_code"
    )

    if st.button("Train Model"):
        # Prepare context
        Autoencoder = st.session_state["Autoencoder_Class"]
        ctx = {
            "embeddings": embeddings,
            "Autoencoder": Autoencoder,
            "torch": torch,
            "nn": nn,
            "optim": optim,
            "np": np,
            "st": st,
        }

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()

        try:
            exec(code, {"__builtins__": __builtins__}, ctx)

            sys.stdout = old_stdout
            output = mystdout.getvalue()
            if output:
                st.code(output)

            passed, msg = check_step_3_training_loop(ctx)

            if passed:
                st.success(msg)

                # ── Live loss chart ──────────────────────────────
                st.markdown("### 📉 Training & Validation Loss")

                losses = ctx["losses"]
                val_losses = ctx.get("val_losses", [])

                loss_df = pd.DataFrame(
                    {"Epoch": range(1, len(losses) + 1), "Train Loss": losses}
                )
                if val_losses:
                    loss_df["Val Loss"] = val_losses

                fig_loss = px.line(
                    loss_df,
                    x="Epoch",
                    y=[c for c in loss_df.columns if c != "Epoch"],
                    title="Loss Curve",
                    labels={"value": "MSE Loss", "variable": ""},
                    color_discrete_map={"Train Loss": "#4A90D9", "Val Loss": "#E74C3C"},
                )
                fig_loss.update_layout(
                    hovermode="x unified",
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                )
                st.plotly_chart(fig_loss, use_container_width=True)

                # Quick diagnostics
                if val_losses:
                    final_train = losses[-1]
                    final_val = val_losses[-1]
                    if final_val > final_train * 1.5:
                        st.warning(
                            f"⚠️ Possible **overfitting**: val loss ({final_val:.4f}) is much higher "
                            f"than train loss ({final_train:.4f}). Try fewer epochs or a wider hidden layer."
                        )
                    elif final_train > losses[0] * 0.9:
                        st.warning(
                            "⚠️ Possible **underfitting**: loss barely decreased. "
                            "Try a higher learning rate or more epochs."
                        )
                    else:
                        st.success(
                            f"✅ Looks good! Train loss: {final_train:.4f}, Val loss: {final_val:.4f}"
                        )

                # ── Latent-space scatter with highlighting ──────
                st.markdown("### 🗺️ Autoencoder Latent Space (2D)")

                model = ctx["model"]
                # Use full dataset for visualization
                data_tensor = ctx.get(
                    "data_tensor", torch.tensor(embeddings, dtype=torch.float32)
                )

                with torch.no_grad():
                    model.eval()
                    _, latents = model(data_tensor)
                    latents_np = latents.numpy()

                _render_highlighted_scatter(
                    latents_np,
                    cfg,
                    col_x="Dim1",
                    col_y="Dim2",
                    title="Autoencoder Latent Space — Highlighted",
                    key_prefix="ae",
                )

            else:
                st.error(msg)

        except Exception as e:
            sys.stdout = old_stdout
            st.error(f"Runtime Error: {e}")

    st.markdown("---")
    st.markdown("### 🏆 Bonus Question")
    st.markdown(
        "What is the main theoretical difference between doing PCA with scikit-learn vs training an autoencoder? *(Hint: think about the activation functions)*"
    )

    answer = st.text_input(
        "Type your answer here to reveal the solution:", key="bonus_q_pca_ae"
    )
    if answer:
        st.success(
            "**Answer:** PCA is strictly a **linear** transformation. It finds directions of maximum variance using only linear combinations of features. An Autoencoder with non-linear activation functions (like ReLU) can learn **non-linear** transformations, allowing it to compress complex, curved manifolds in the data that PCA might completely miss. You can make them equivalent by using linear activation functions in the autoencoder (removing ReLU)."
        )


if __name__ == "__main__":
    render_pca_lab()
