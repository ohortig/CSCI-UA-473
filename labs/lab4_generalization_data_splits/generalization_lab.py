# TODO: neighborhood group: borough
# condense the texts - students do not read them! use more visualizations over the map.
# rmse overlay on map (Lavender has the nyc map and code)
"""
Lab 4: Generalization & Data Splits

Students train a Ridge regression model to predict Airbnb one-night room rates
and compare three data-splitting strategies:

1. Random split       - baseline, easiest
2. Host-based split   - predict prices for *new hosts*
3. Neighborhood-based - predict prices for *new neighborhoods*

The progressive drop in test-set performance teaches the concept of
generalization and distribution shift.
"""

import io
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from labs.lab4_generalization_data_splits.level_checks import (
    check_step_1_load,
    check_step_2_features,
    check_step_3_random_split,
    check_step_4_host_split,
    check_step_5_neighborhood_split,
    check_step_6_borough_split,
)
from labs.lab4_generalization_data_splits.map_visualization import (
    display_comparison_maps,
    display_error_choropleth,
    display_error_scatter_map,
    display_price_map,
    display_split_choropleth,
    load_neighbourhood_geojson,
)

# ======================================================================
# Helper: train + evaluate + visualize
# ======================================================================


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test):
    """Fit a Ridge regression, return metrics dict and predictions.

    Features are standardised (fit on train, transform val/test).
    Target ``y`` is assumed to be ``log1p(price)``.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)

    preds = {
        "train": model.predict(X_train_s),
        "val": model.predict(X_val_s),
        "test": model.predict(X_test_s),
    }
    actuals = {"train": y_train, "val": y_val, "test": y_test}

    metrics = {}
    for split in ("train", "val", "test"):
        y_true = actuals[split]
        y_pred = preds[split]
        r2 = r2_score(y_true, y_pred)
        mae_dollar = mean_absolute_error(np.expm1(y_true), np.expm1(y_pred))
        metrics[split] = {
            "R2": r2,
            "MAE ($)": mae_dollar,
            "n": len(y_true),
        }

    return metrics, preds, actuals


def display_metrics(metrics, strategy_name):
    """Display a metrics table and predicted-vs-actual scatter plot."""
    # --- Metrics table ---
    rows = []
    for split in ("train", "val", "test"):
        m = metrics[split]
        rows.append(
            {
                "Split": split.title(),
                "N": f"{m['n']:,}",
                "R\u00b2": f"{m['R2']:.3f}",
                "MAE ($)": f"{m['MAE ($)']:.1f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def display_scatter(preds, actuals, title="Predicted vs Actual"):
    """Show a scatter plot of predicted vs actual prices (dollar space)."""
    frames = []
    for split in ("train", "val", "test"):
        n = len(actuals[split])
        frames.append(
            pd.DataFrame(
                {
                    "Actual ($)": np.expm1(actuals[split]),
                    "Predicted ($)": np.expm1(preds[split]),
                    "Split": [split.title()] * n,
                }
            )
        )
    scatter_df = pd.concat(frames, ignore_index=True)

    fig = px.scatter(
        scatter_df,
        x="Actual ($)",
        y="Predicted ($)",
        color="Split",
        opacity=0.4,
        title=title,
        color_discrete_map={
            "Train": "#636EFA",
            "Val": "#EF553B",
            "Test": "#00CC96",
        },
    )
    # Perfect-prediction line
    max_val = scatter_df[["Actual ($)", "Predicted ($)"]].max().max()
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(dash="dash", color="grey"),
            name="Perfect",
            showlegend=True,
        )
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# Main render function
# ======================================================================


def render_generalization_lab():  # noqa: C901
    st.header("Lab 4: Generalization & Data Splits")

    st.markdown(
        """
    ### The Mission

    You have a dataset of **NYC Airbnb listings**. Your goal is to build a
    model that predicts the **one-night room rate** from listing features.

    But here is the twist: **how you split your data changes what
    "generalization" means**.

    | Split Strategy | Question it Answers |
    |---|---|
    | **Random** | Can the model predict prices for a random held-out listing? |
    | **Host-based** | Can the model predict prices for a **brand-new host**? |
    | **Neighborhood-based** | Can the model predict prices in an **entirely new neighborhood**? |
    | **Borough-based** | Can the model predict prices in an **entirely new borough**? |

    As we move down the table, the task gets progressively harder
    because the test set becomes more *different* from the training set.

    ### Roadmap

    1. **Load & Embed** -- load raw data, generate text embeddings (recap of Lab 3)
    2. **Prepare** features (`X` = embeddings, `y` = log price)
    3. **Random split** -- train & evaluate
    4. **Host-based split** -- train & evaluate
    5. **Neighborhood-based split** -- train & evaluate
    6. **Borough-based split** -- train on 4 boroughs, test on the 5th
    7. **Compare** all four strategies
    """
    )

    # ------------------------------------------------------------------
    # Session state initialisation
    # ------------------------------------------------------------------
    if "lab4_vars" not in st.session_state:
        st.session_state["lab4_vars"] = {}
    if "lab4_results" not in st.session_state:
        st.session_state["lab4_results"] = {}
    if "lab4_map_data" not in st.session_state:
        st.session_state["lab4_map_data"] = {}

    st.sidebar.divider()
    show_solutions = st.sidebar.checkbox(
        "Show solution code", value=False, key="lab4_show_solutions"
    )

    _render_step_1(show_solutions)

    if st.session_state.get("lab4_step_1_done"):
        _render_step_2(show_solutions)

    if st.session_state.get("lab4_step_2_done"):
        _render_step_3(show_solutions)

    if st.session_state.get("lab4_step_3_done"):
        _render_step_4(show_solutions)

    if st.session_state.get("lab4_step_4_done"):
        _render_step_5(show_solutions)

    if st.session_state.get("lab4_step_5_done"):
        _render_step_6(show_solutions)

    if st.session_state.get("lab4_step_6_done"):
        _render_comparison()


# ======================================================================
# Step 1 – Load data
# ======================================================================


def _render_step_1(show_solutions=False):
    st.divider()
    st.subheader("Step 1: Load Raw Data & Generate Embeddings")
    st.info(
        "**Goal**: Load your raw Airbnb CSV, build a text representation for "
        "each listing, generate embeddings, and save the result as `df`."
    )

    st.markdown(
        """
    #### Recap: How Embeddings Work (from Lab 3)

    An **embedding model** converts text into a dense numerical vector
    (e.g. 768 dimensions). Similar texts end up with similar vectors.
    In this lab, each listing's embedding becomes our **feature vector**
    for predicting price.

    #### What Goes Into the Embedding?

    You decide! Combine whichever columns you think are
    most informative about price. For example:

    | Column | Example Value | Why It Might Help |
    |---|---|---|
    | `name` | *"Cozy studio in SoHo"* | Describes the vibe/type |
    | `description` | *"Bright 1BR with city views…"* | Detailed listing info |
    | `neighborhood_overview` | *"Steps from Central Park…"* | Location character |
    | `room_type` | *"Entire home/apt"* | Listing category |
    | `property_type` | *"Apartment"* | Building type |
    | `accommodates` | *4* | Capacity → price driver |
    | `bedrooms` | *2* | Size → price driver |

    Concatenate them into a single string per listing, then encode with
    the Nomic model from Lab 3.

    #### Columns to Keep

    After generating embeddings, make sure your DataFrame keeps these
    columns — they are needed for the strategic splits in later steps:

    | Column | Used In | Purpose |
    |---|---|---|
    | `price` | Step 2 | Prediction target |
    | `embedding` | Step 2 | Feature vector (768-dim list per row) |
    | `host_id` | Step 4 | Host-based splitting |
    | `neighbourhood_cleansed` | Step 5 | Neighbourhood-based splitting |
    | `neighbourhood_group_cleansed` | Step 6 | Borough-based splitting |
    | `latitude`, `longitude` | Maps | Map visualizations |

    **Your Task**:
    1. Load your raw CSV into a DataFrame.
    2. Build a text string for each listing (combine columns of your choice).
    3. Generate embeddings using `SentenceTransformer`.
    4. Store the embeddings in a column named `embedding`.
    5. **Save** the result to a file (e.g. parquet) so you don't have to
       re-run embedding generation every time.
    6. The final DataFrame should be named `df`.

    > **Tip**: The code below checks for a cached processed file first.
    > If it exists, it loads directly and skips the slow embedding step.
    > On first run you'll generate embeddings (~7 min); after that it's instant.
    """
    )

    student_code = """import os, torch
from sentence_transformers import SentenceTransformer

PROCESSED_PATH = "data/processed/airbnb_embedded.parquet"

if os.path.exists(PROCESSED_PATH):
    print(f"Found cached file: {PROCESSED_PATH}")
    df = pd.read_parquet(PROCESSED_PATH)
else:
    print("No cached file found — generating embeddings from scratch...")

    df = pd.read_csv("data/raw/airbnb_listings.csv")

    # Clean price: "$1,234.00" -> 1234.0, then filter outliers
    df["price"] = df["price"].replace("[\\$,]", "", regex=True).astype(float)
    df = df.dropna(subset=["price"])
    df = df[(df["price"] >= 10) & (df["price"] <= 2000)]

    df = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)
    print(f"Sampled {len(df):,} listings")

    # TODO: Write a function that takes a row and returns a single text
    # string combining whichever columns you think matter for pricing.
    # Think about: name, description, room_type, bedrooms, neighbourhood, ...
    def build_text(row):
        parts = []
        # --- fill in: append relevant column values to parts ---
        return " ".join(parts).strip() or "Airbnb listing"

    texts = df.apply(build_text, axis=1).tolist()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    # TODO: Encode the texts into embeddings using model.encode().
    # Remember to prepend "search_document: " to each text (Nomic requirement).
    embeddings = ...

    df["embedding"] = [emb.tolist() for emb in embeddings]

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"Saved to {PROCESSED_PATH}")

print(f"Loaded {len(df):,} listings")
print(f"Columns: {list(df.columns)}")
print(f"Embedding dim: {len(df['embedding'].iloc[0])}")"""

    solution_code = """import os, torch
from sentence_transformers import SentenceTransformer

PROCESSED_PATH = "data/processed/airbnb_embedded.parquet"

if os.path.exists(PROCESSED_PATH):
    print(f"Found cached file: {PROCESSED_PATH}")
    df = pd.read_parquet(PROCESSED_PATH)
else:
    print("No cached file found — generating embeddings from scratch...")
    df = pd.read_csv("data/raw/airbnb_listings.csv")

    # Clean price: "$1,234.00" -> 1234.0, then filter outliers
    df["price"] = df["price"].replace("[\\$,]", "", regex=True).astype(float)
    df = df.dropna(subset=["price"])
    df = df[(df["price"] >= 10) & (df["price"] <= 2000)]

    df = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)
    print(f"Sampled {len(df):,} listings")

    def build_text(row):
        parts = []
        if pd.notna(row.get("name")):
            parts.append(str(row["name"]))
        if pd.notna(row.get("description")):
            parts.append(str(row["description"])[:500])
        if pd.notna(row.get("neighborhood_overview")):
            parts.append(str(row["neighborhood_overview"])[:200])
        if pd.notna(row.get("room_type")):
            parts.append(str(row["room_type"]))
        if pd.notna(row.get("accommodates")):
            parts.append(f"accommodates {int(row['accommodates'])} guests")
        if pd.notna(row.get("bedrooms")):
            parts.append(f"{int(row['bedrooms'])} bedrooms")
        if pd.notna(row.get("neighbourhood_cleansed")):
            borough = row.get("neighbourhood_group_cleansed", "")
            parts.append(f"in {row['neighbourhood_cleansed']}, {borough}")
        return " ".join(parts).strip() or "Airbnb listing"

    texts = df.apply(build_text, axis=1).tolist()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    embeddings = model.encode(
        ["search_document: " + t for t in texts],
        normalize_embeddings=True, show_progress_bar=True,
        batch_size=128, device=device,
    )
    df["embedding"] = [emb.tolist() for emb in embeddings]

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"Saved to {PROCESSED_PATH}")

print(f"Loaded {len(df):,} listings")
print(f"Columns: {list(df.columns)}")
print(f"Embedding dim: {len(df['embedding'].iloc[0])}")"""

    mode = "sol" if show_solutions else "stu"
    default_code = solution_code if show_solutions else student_code
    code = st.text_area(
        "Step 1 Code:",
        value=default_code,
        height=480 if show_solutions else 420,
        key=f"lab4_code_1_{mode}",
    )

    if st.button("Run Step 1", key="lab4_run_1"):
        st.session_state["lab4_vars"] = {"pd": pd, "np": np}
        st.session_state["lab4_results"] = {}
        st.session_state["lab4_map_data"] = {}

        try:
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()

            with st.spinner(
                "Running Step 1 (may take 1-2 min if generating embeddings)..."
            ):
                exec(code, st.session_state["lab4_vars"])

            sys.stdout = old_stdout
            output = mystdout.getvalue()
            if output:
                st.code(output)

            passed, msg = check_step_1_load(st.session_state["lab4_vars"])
            if passed:
                st.success(msg)
                st.session_state["lab4_step_1_done"] = True

                df = st.session_state["lab4_vars"]["df"]
                # Quick overview
                metric_cols = st.columns(3)
                metric_cols[0].metric("Listings", f"{len(df):,}")
                if "host_id" in df.columns:
                    metric_cols[1].metric(
                        "Unique Hosts", f"{df['host_id'].nunique():,}"
                    )
                if "neighbourhood_cleansed" in df.columns:
                    metric_cols[2].metric(
                        "Neighborhoods",
                        f"{df['neighbourhood_cleansed'].nunique()}",
                    )

                has_price = "price" in df.columns
                has_latlon = "latitude" in df.columns and "longitude" in df.columns

                if has_price and has_latlon:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("**Price Distribution**")
                        fig = px.histogram(
                            df,
                            x="price",
                            nbins=80,
                            title="Nightly Price ($)",
                            labels={"price": "Price ($)"},
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    with col_b:
                        st.write("**Prices on the Map**")
                        display_price_map(df)
                elif has_price:
                    st.write("**Price Distribution**")
                    fig = px.histogram(
                        df,
                        x="price",
                        nbins=80,
                        title="Nightly Price ($)",
                        labels={"price": "Price ($)"},
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(msg)
        except Exception as e:
            sys.stdout = old_stdout
            st.error(f"Runtime Error: {e}")


# ======================================================================
# Step 2 – Prepare features
# ======================================================================


def _render_step_2(show_solutions=False):
    st.divider()
    st.subheader("Step 2: Prepare Features")
    st.info("**Goal**: Build the feature matrix `X` and target vector `y`.")

    st.markdown(
        r"""
    Your embedding already captures the important information about each
    listing (description, location, room type, etc.), so we use it
    directly as our feature matrix `X`.

    For the target, we use **log-transformed price** (`log1p`) because
    prices are right-skewed. This helps linear models and makes metrics
    more meaningful.

    $$y = \log(1 + \text{price})$$

    **Your Task**:
    1. Stack the `embedding` column into a 2-D numpy matrix → `X`.
    2. Create `y = np.log1p(df["price"].values)`.
    """
    )

    student_code = """# 'df' is available from Step 1

# TODO: Stack the embedding column into a 2-D matrix (N, 768).
# Hint: np.stack(df["embedding"].values)
X = ...

# TODO: Create the target variable y (log-transformed price).
y = ...

print(f"X shape: {X.shape}  |  y shape: {y.shape}")"""

    solution_code = """# 'df' is available from Step 1

X = np.stack(df["embedding"].values)
y = np.log1p(df["price"].values)

print(f"X shape: {X.shape}  |  y shape: {y.shape}")"""

    mode = "sol" if show_solutions else "stu"
    default_code = solution_code if show_solutions else student_code
    code = st.text_area(
        "Step 2 Code:",
        value=default_code,
        height=220 if show_solutions else 260,
        key=f"lab4_code_2_{mode}",
    )

    if st.button("Run Step 2", key="lab4_run_2"):
        if "df" not in st.session_state["lab4_vars"]:
            st.error("DataFrame not found. Please run Step 1 first.")
            return

        try:
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()

            exec(code, st.session_state["lab4_vars"])

            sys.stdout = old_stdout
            output = mystdout.getvalue()
            if output:
                st.code(output)

            passed, msg = check_step_2_features(st.session_state["lab4_vars"])
            if passed:
                st.success(msg)
                st.session_state["lab4_step_2_done"] = True
            else:
                st.error(msg)
        except Exception as e:
            sys.stdout = old_stdout
            st.error(f"Runtime Error: {e}")


# ======================================================================
# Step 3 – Random split
# ======================================================================


def _render_step_3(show_solutions=False):
    st.divider()
    st.subheader("Step 3: Random Split")
    st.info(
        "**Goal**: Split the data randomly into train / val / test "
        "and see how the model performs."
    )

    st.markdown(
        """
    The simplest strategy: **randomly** assign each listing to one of three
    sets (70 % train, 15 % validation, 15 % test).

    This is the standard approach in many tutorials, and it typically gives
    the *best* test-set numbers -- but it may **overestimate** how well
    the model generalises in the real world. Why?

    Because a random split means some listings from the **same host** or
    **same neighborhood** can appear in both training and testing.
    The model may learn host-specific or location-specific patterns and
    "cheat" on the test set.

    **Your Task**:
    Use `sklearn.model_selection.train_test_split` to create:
    - `X_train, y_train` (70 %)
    - `X_val, y_val` (15 %)
    - `X_test, y_test` (15 %)
    """
    )

    student_code = """# 'X' and 'y' are available from Step 2
from sklearn.model_selection import train_test_split

# TODO: Split indices into train (70%), val (15%), test (15%).
# Hint: use train_test_split twice on np.arange(len(X)).
#   1) Split off 30% as temp  2) Split temp 50/50 into val and test.
all_idx = np.arange(len(X))


# TODO: Convert indices to boolean masks so we can map back to the DataFrame.
# Hint: create np.zeros(len(X), dtype=bool) then set mask[idx] = True.
train_mask = ...
val_mask   = ...
test_mask  = ...

# TODO: Use the masks to create X_train, y_train, X_val, y_val, X_test, y_test.


print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")"""

    solution_code = """# 'X' and 'y' are available from Step 2
from sklearn.model_selection import train_test_split

all_idx = np.arange(len(X))
train_idx, temp_idx = train_test_split(all_idx, test_size=0.3, random_state=42)
val_idx, test_idx   = train_test_split(temp_idx, test_size=0.5, random_state=42)

train_mask = np.zeros(len(X), dtype=bool)
val_mask   = np.zeros(len(X), dtype=bool)
test_mask  = np.zeros(len(X), dtype=bool)
train_mask[train_idx] = True
val_mask[val_idx]     = True
test_mask[test_idx]   = True

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")"""

    mode = "sol" if show_solutions else "stu"
    default_code = solution_code if show_solutions else student_code
    code = st.text_area(
        "Step 3 Code:",
        value=default_code,
        height=320,
        key=f"lab4_code_3_{mode}",
    )

    if st.button("Run Step 3", key="lab4_run_3"):
        if "X" not in st.session_state["lab4_vars"]:
            st.error("Features not found. Please run Step 2 first.")
            return

        try:
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()

            exec(code, st.session_state["lab4_vars"])

            sys.stdout = old_stdout
            output = mystdout.getvalue()
            if output:
                st.code(output)

            passed, msg = check_step_3_random_split(st.session_state["lab4_vars"])
            if passed:
                st.success(msg)
                st.session_state["lab4_step_3_done"] = True

                lv = st.session_state["lab4_vars"]
                with st.spinner("Training Ridge regression on random split ..."):
                    metrics, preds, actuals = train_and_evaluate(
                        lv["X_train"],
                        lv["y_train"],
                        lv["X_val"],
                        lv["y_val"],
                        lv["X_test"],
                        lv["y_test"],
                    )
                st.session_state["lab4_results"]["random"] = metrics

                # Store map data for comparison later
                if "test_mask" in lv:
                    st.session_state["lab4_map_data"]["random"] = {
                        "test_preds": preds["test"].copy(),
                        "test_actuals": actuals["test"].copy(),
                        "test_mask": lv["test_mask"].copy(),
                    }

                st.write("### Results: Random Split")
                display_metrics(metrics, "Random")
                display_scatter(preds, actuals, "Random Split: Predicted vs Actual ($)")

                # Error map
                if "test_mask" in lv:
                    with st.expander("View prediction errors on the NYC map"):
                        display_error_scatter_map(
                            st.session_state["lab4_vars"]["df"],
                            lv["test_mask"],
                            actuals["test"],
                            preds["test"],
                            title="Random Split — Test Error Map",
                        )

                st.info(
                    "**Note the test R\u00b2 and MAE.** They look reasonable -- "
                    "but can the model really handle *new hosts* or *new neighborhoods*? "
                    "Let's find out in Steps 4 and 5."
                )
            else:
                st.error(msg)
        except Exception as e:
            sys.stdout = old_stdout
            st.error(f"Runtime Error: {e}")


# ======================================================================
# Step 4 – Host-based split
# ======================================================================


def _render_step_4(show_solutions=False):
    st.divider()
    st.subheader("Step 4: Host-Based Split")
    st.info(
        "**Goal**: Split so that each host appears in *only one* split. "
        "Can the model predict prices for a brand-new host?"
    )

    st.markdown(
        """
    Many Airbnb hosts have **multiple listings**. In a random split, one
    host's listings can end up in both the training and test sets.
    The model might memorize host-specific pricing patterns (e.g., a
    host who always prices 20 % above average) and appear to generalise
    when it's really just recognising the host.

    A **host-based split** ensures that *all* listings from the same host
    are in the same split. This tests whether the model can predict prices
    for hosts it has **never seen before**.

    **Your Task**:
    1. Get the unique host IDs and shuffle them.
    2. Assign 70 % of *hosts* to train, 15 % to val, 15 % to test.
    3. Create boolean masks and use them to split X and y.
    """
    )

    student_code = """# 'X', 'y', and 'df' are available
host_ids = df["host_id"].values
unique_hosts = np.unique(host_ids)

# TODO: Shuffle the unique hosts (not individual listings!).
# Hint: np.random.default_rng(42).shuffle(unique_hosts)


# TODO: Assign 70% of *hosts* to train, 15% to val, 15% to test.
# Hint: slice unique_hosts and wrap each slice in set() for fast lookup.
n = len(unique_hosts)
train_hosts = ...
val_hosts   = ...
test_hosts  = ...

# TODO: Build boolean masks over all listings.
# For each listing, check whether its host_id is in train_hosts, val_hosts, or test_hosts.
# Hint: np.array([h in train_hosts for h in host_ids])
train_mask = ...
val_mask   = ...
test_mask  = ...

# TODO: Use the masks to create X_train, y_train, X_val, y_val, X_test, y_test.


print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"Hosts: {len(train_hosts):,} / {len(val_hosts):,} / {len(test_hosts):,}")"""

    solution_code = """# 'X', 'y', and 'df' are available
host_ids = df["host_id"].values
unique_hosts = np.unique(host_ids)

rng = np.random.default_rng(42)
rng.shuffle(unique_hosts)

n = len(unique_hosts)
train_hosts = set(unique_hosts[: int(0.7 * n)])
val_hosts   = set(unique_hosts[int(0.7 * n) : int(0.85 * n)])
test_hosts  = set(unique_hosts[int(0.85 * n) :])

train_mask = np.array([h in train_hosts for h in host_ids])
val_mask   = np.array([h in val_hosts   for h in host_ids])
test_mask  = np.array([h in test_hosts  for h in host_ids])

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"Hosts: {len(train_hosts):,} / {len(val_hosts):,} / {len(test_hosts):,}")"""

    mode = "sol" if show_solutions else "stu"
    default_code = solution_code if show_solutions else student_code
    code = st.text_area(
        "Step 4 Code:",
        value=default_code,
        height=420,
        key=f"lab4_code_4_{mode}",
    )

    if st.button("Run Step 4", key="lab4_run_4"):
        if "X" not in st.session_state["lab4_vars"]:
            st.error("Features not found. Please run Step 2 first.")
            return

        try:
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()

            exec(code, st.session_state["lab4_vars"])

            sys.stdout = old_stdout
            output = mystdout.getvalue()
            if output:
                st.code(output)

            passed, msg = check_step_4_host_split(st.session_state["lab4_vars"])
            if passed:
                st.success(msg)
                st.session_state["lab4_step_4_done"] = True

                lv = st.session_state["lab4_vars"]
                with st.spinner("Training Ridge regression on host-based split ..."):
                    metrics, preds, actuals = train_and_evaluate(
                        lv["X_train"],
                        lv["y_train"],
                        lv["X_val"],
                        lv["y_val"],
                        lv["X_test"],
                        lv["y_test"],
                    )
                st.session_state["lab4_results"]["host"] = metrics

                # Store map data for comparison later
                if "test_mask" in lv:
                    st.session_state["lab4_map_data"]["host"] = {
                        "test_preds": preds["test"].copy(),
                        "test_actuals": actuals["test"].copy(),
                        "test_mask": lv["test_mask"].copy(),
                    }

                st.write("### Results: Host-Based Split")
                display_metrics(metrics, "Host-Based")
                display_scatter(
                    preds, actuals, "Host-Based Split: Predicted vs Actual ($)"
                )

                # Error map
                if "test_mask" in lv:
                    with st.expander("View prediction errors on the NYC map"):
                        display_error_scatter_map(
                            st.session_state["lab4_vars"]["df"],
                            lv["test_mask"],
                            actuals["test"],
                            preds["test"],
                            title="Host-Based Split — Test Error Map",
                        )

                # Compare with random
                if "random" in st.session_state["lab4_results"]:
                    r_test = st.session_state["lab4_results"]["random"]["test"]
                    h_test = metrics["test"]
                    delta_r2 = h_test["R2"] - r_test["R2"]
                    st.info(
                        f"**Compared to random split:** "
                        f"Test R\u00b2 changed by {delta_r2:+.3f}. "
                        f"If it dropped, that means the random split was "
                        f"benefiting from host leakage."
                    )
            else:
                st.error(msg)
        except Exception as e:
            sys.stdout = old_stdout
            st.error(f"Runtime Error: {e}")


# ======================================================================
# Step 5 – Neighbourhood-based split
# ======================================================================


def _render_step_5(show_solutions=False):
    st.divider()
    st.subheader("Step 5: Neighbourhood-Based Split")
    st.info(
        "**Goal**: Hold out entire neighborhoods. "
        "Can the model predict prices in a neighborhood it has never seen?"
    )

    st.markdown(
        """
    Neighbourhood is one of the strongest predictors of price. A listing in
    SoHo costs very differently from one in the South Bronx.

    In a **neighborhood-based split**, we hold out *entire neighborhoods*
    for validation and testing. The model must learn pricing patterns that
    transfer across locations -- a much harder test of generalization.

    **Your Task**:
    1. Get the unique neighborhood names and shuffle them.
    2. Assign 70 % of *neighborhoods* to train, 15 % to val, 15 % to test.
    3. Create boolean masks and split.
    """
    )

    student_code = """# 'X', 'y', and 'df' are available
neighborhoods = df["neighbourhood_cleansed"].values
unique_hoods = np.unique(neighborhoods)

# TODO: Follow the same pattern as the host-based split (Step 4),
# but split by *neighbourhood* instead of host_id.
# Assign 70% of neighbourhoods to train, 15% to val, 15% to test.


print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"Neighborhoods: {len(train_hoods)} / {len(val_hoods)} / {len(test_hoods)}")"""

    solution_code = """# 'X', 'y', and 'df' are available
neighborhoods = df["neighbourhood_cleansed"].values
unique_hoods = np.unique(neighborhoods)

rng = np.random.default_rng(42)
rng.shuffle(unique_hoods)

n = len(unique_hoods)
train_hoods = set(unique_hoods[: int(0.7 * n)])
val_hoods   = set(unique_hoods[int(0.7 * n) : int(0.85 * n)])
test_hoods  = set(unique_hoods[int(0.85 * n) :])

train_mask = np.array([h in train_hoods for h in neighborhoods])
val_mask   = np.array([h in val_hoods   for h in neighborhoods])
test_mask  = np.array([h in test_hoods  for h in neighborhoods])

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"Neighborhoods: {len(train_hoods)} / {len(val_hoods)} / {len(test_hoods)}")"""

    mode = "sol" if show_solutions else "stu"
    default_code = solution_code if show_solutions else student_code
    code = st.text_area(
        "Step 5 Code:",
        value=default_code,
        height=320 if show_solutions else 280,
        key=f"lab4_code_5_{mode}",
    )

    if st.button("Run Step 5", key="lab4_run_5"):
        if "X" not in st.session_state["lab4_vars"]:
            st.error("Features not found. Please run Step 2 first.")
            return

        try:
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()

            exec(code, st.session_state["lab4_vars"])

            sys.stdout = old_stdout
            output = mystdout.getvalue()
            if output:
                st.code(output)

            passed, msg = check_step_5_neighborhood_split(st.session_state["lab4_vars"])
            if passed:
                st.success(msg)
                st.session_state["lab4_step_5_done"] = True

                lv = st.session_state["lab4_vars"]
                with st.spinner(
                    "Training Ridge regression on neighborhood-based split ..."
                ):
                    metrics, preds, actuals = train_and_evaluate(
                        lv["X_train"],
                        lv["y_train"],
                        lv["X_val"],
                        lv["y_val"],
                        lv["X_test"],
                        lv["y_test"],
                    )
                st.session_state["lab4_results"]["neighborhood"] = metrics

                # Store map data for comparison later
                if "test_mask" in lv:
                    st.session_state["lab4_map_data"]["neighborhood"] = {
                        "test_preds": preds["test"].copy(),
                        "test_actuals": actuals["test"].copy(),
                        "test_mask": lv["test_mask"].copy(),
                        "train_mask": lv["train_mask"].copy(),
                        "val_mask": lv["val_mask"].copy(),
                    }

                st.write("### Results: Neighbourhood-Based Split")
                display_metrics(metrics, "Neighbourhood")
                display_scatter(
                    preds,
                    actuals,
                    "Neighbourhood-Based Split: Predicted vs Actual ($)",
                )

                # Show which neighborhoods are in each split
                df = st.session_state["lab4_vars"]["df"]
                hoods = df["neighbourhood_cleansed"].values
                test_hoods_list = sorted(
                    set(hoods[st.session_state["lab4_vars"]["test_mask"]])
                )
                with st.expander("Held-out test neighborhoods"):
                    st.write(", ".join(test_hoods_list))

                # Neighbourhood maps
                if "test_mask" in lv:
                    geojson = load_neighbourhood_geojson()
                    if geojson is not None:
                        st.write("### NYC Neighbourhood Maps")
                        map_col1, map_col2 = st.columns(2)
                        with map_col1:
                            display_split_choropleth(
                                df,
                                lv["train_mask"],
                                lv["val_mask"],
                                lv["test_mask"],
                                geojson,
                            )
                        with map_col2:
                            display_error_choropleth(
                                df,
                                lv["test_mask"],
                                actuals["test"],
                                preds["test"],
                                geojson,
                            )
                    else:
                        display_error_scatter_map(
                            df,
                            lv["test_mask"],
                            actuals["test"],
                            preds["test"],
                            title="Neighbourhood Split — Test Error Map",
                        )
            else:
                st.error(msg)
        except Exception as e:
            sys.stdout = old_stdout
            st.error(f"Runtime Error: {e}")


# ======================================================================
# Step 6 – Borough-based split (leave-one-borough-out)
# ======================================================================


def _render_step_6(show_solutions=False):
    st.divider()
    st.subheader("Step 6: Borough-Based Split (Leave-One-Borough-Out)")
    st.info(
        "**Goal**: Train on four boroughs and test on the fifth. "
        "This is an *out-of-distribution* evaluation — can the model "
        "price listings in a borough it has never seen?"
    )

    st.markdown(
        """
    NYC has five boroughs, each with a very different character:

    | Borough | Character |
    |---|---|
    | **Manhattan** | Dense, expensive, tourist-heavy |
    | **Brooklyn** | Diverse, rapidly gentrifying |
    | **Queens** | Suburban feel, near airports |
    | **Bronx** | Affordable, fewer listings |
    | **Staten Island** | Suburban, very few listings |

    Holding out an entire borough is a more extreme version of the
    neighbourhood split. The model must transfer pricing knowledge
    across boroughs that may have fundamentally different markets.

    This is a form of **out-of-distribution (OOD) evaluation** — the
    test data comes from a systematically different distribution than
    the training data.

    **Your Task**:
    1. Pick a borough to hold out for testing (default: Manhattan).
    2. Split the remaining four boroughs into train and validation.
    3. Create boolean masks and split X and y.
    """
    )

    student_code = """# 'X', 'y', and 'df' are available
boroughs = df["neighbourhood_group_cleansed"].values

# TODO: Pick one borough to hold out as the test set.
test_borough = "Manhattan"

# TODO: Collect the other boroughs and split them into train (~75%) and val (~25%).
# Hint: there are only 4 other boroughs, so 3 for train, 1 for val.
other_boroughs = [b for b in np.unique(boroughs) if b != test_borough]


# TODO: Create boolean masks for train, val, and test.
# Hint: test_mask can be simply (boroughs == test_borough).
train_mask = ...
val_mask   = ...
test_mask  = ...

# TODO: Use the masks to create X_train, y_train, X_val, y_val, X_test, y_test.


print(f"Test borough: {test_borough}")
print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")"""

    solution_code = """# 'X', 'y', and 'df' are available
boroughs = df["neighbourhood_group_cleansed"].values

test_borough = "Manhattan"

other_boroughs = [b for b in np.unique(boroughs) if b != test_borough]
rng = np.random.default_rng(42)
rng.shuffle(other_boroughs)

n_other = len(other_boroughs)
train_boroughs = set(other_boroughs[: int(0.75 * n_other)])
val_boroughs   = set(other_boroughs[int(0.75 * n_other) :])

train_mask = np.array([b in train_boroughs for b in boroughs])
val_mask   = np.array([b in val_boroughs   for b in boroughs])
test_mask  = (boroughs == test_borough)

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"Train boroughs: {sorted(train_boroughs)}")
print(f"Val boroughs:   {sorted(val_boroughs)}")
print(f"Test borough:   {test_borough}")
print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")"""

    mode = "sol" if show_solutions else "stu"
    default_code = solution_code if show_solutions else student_code
    code = st.text_area(
        "Step 6 Code:",
        value=default_code,
        height=420 if show_solutions else 380,
        key=f"lab4_code_6_{mode}",
    )

    if st.button("Run Step 6", key="lab4_run_6"):
        if "X" not in st.session_state["lab4_vars"]:
            st.error("Features not found. Please run Step 2 first.")
            return

        try:
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()

            exec(code, st.session_state["lab4_vars"])

            sys.stdout = old_stdout
            output = mystdout.getvalue()
            if output:
                st.code(output)

            passed, msg = check_step_6_borough_split(st.session_state["lab4_vars"])
            if passed:
                st.success(msg)
                st.session_state["lab4_step_6_done"] = True

                lv = st.session_state["lab4_vars"]
                with st.spinner("Training Ridge regression on borough-based split ..."):
                    metrics, preds, actuals = train_and_evaluate(
                        lv["X_train"],
                        lv["y_train"],
                        lv["X_val"],
                        lv["y_val"],
                        lv["X_test"],
                        lv["y_test"],
                    )
                st.session_state["lab4_results"]["borough"] = metrics

                # Store map data for comparison later
                if "test_mask" in lv:
                    st.session_state["lab4_map_data"]["borough"] = {
                        "test_preds": preds["test"].copy(),
                        "test_actuals": actuals["test"].copy(),
                        "test_mask": lv["test_mask"].copy(),
                        "train_mask": lv["train_mask"].copy(),
                        "val_mask": lv["val_mask"].copy(),
                    }

                st.write("### Results: Borough-Based Split")
                display_metrics(metrics, "Borough")
                display_scatter(
                    preds,
                    actuals,
                    "Borough-Based Split: Predicted vs Actual ($)",
                )

                # Borough maps
                if "test_mask" in lv:
                    geojson = load_neighbourhood_geojson()
                    if geojson is not None:
                        st.write("### NYC Borough Maps")
                        map_col1, map_col2 = st.columns(2)
                        df = st.session_state["lab4_vars"]["df"]
                        with map_col1:
                            display_split_choropleth(
                                df,
                                lv["train_mask"],
                                lv["val_mask"],
                                lv["test_mask"],
                                geojson,
                            )
                        with map_col2:
                            display_error_choropleth(
                                df,
                                lv["test_mask"],
                                actuals["test"],
                                preds["test"],
                                geojson,
                            )

                # Compare with neighbourhood split
                if "neighborhood" in st.session_state["lab4_results"]:
                    n_test = metrics["test"]
                    nb_test = st.session_state["lab4_results"]["neighborhood"]["test"]
                    delta_r2 = n_test["R2"] - nb_test["R2"]
                    st.info(
                        f"**Compared to neighbourhood split:** "
                        f"Test R\u00b2 changed by {delta_r2:+.3f}. "
                        f"Holding out an entire borough is an even harder "
                        f"out-of-distribution challenge."
                    )
            else:
                st.error(msg)
        except Exception as e:
            sys.stdout = old_stdout
            st.error(f"Runtime Error: {e}")


# ======================================================================
# Comparison
# ======================================================================


def _render_comparison():
    st.divider()
    st.header("Step 7: Compare All Four Strategies")

    results = st.session_state.get("lab4_results", {})
    if len(results) < 4:
        st.warning("Complete all four splits to see the comparison.")
        return

    # Build comparison DataFrame
    strategy_order = [
        ("random", "Random"),
        ("host", "Host"),
        ("neighborhood", "Neighbourhood"),
        ("borough", "Borough"),
    ]
    rows = []
    for key, label in strategy_order:
        m = results[key]["test"]
        rows.append(
            {
                "Strategy": label,
                "Test R\u00b2": m["R2"],
                "Test MAE ($)": m["MAE ($)"],
                "Test N": m["n"],
            }
        )
    comp_df = pd.DataFrame(rows)

    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # Bar charts (separate scales for R² and MAE)
    chart_col1, chart_col2 = st.columns(2)
    strategy_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

    with chart_col1:
        fig_r2 = px.bar(
            comp_df,
            x="Strategy",
            y="Test R\u00b2",
            color="Strategy",
            color_discrete_sequence=strategy_colors,
            title="Test R\u00b2 (higher is better)",
        )
        fig_r2.update_layout(height=350, showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig_r2, use_container_width=True)

    with chart_col2:
        fig_mae = px.bar(
            comp_df,
            x="Strategy",
            y="Test MAE ($)",
            color="Strategy",
            color_discrete_sequence=strategy_colors,
            title="Test MAE $ (lower is better)",
        )
        fig_mae.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_mae, use_container_width=True)

    # Train vs test gap
    gap_rows = []
    for key, label in strategy_order:
        train_r2 = results[key]["train"]["R2"]
        test_r2 = results[key]["test"]["R2"]
        gap_rows.append(
            {
                "Strategy": label,
                "Train R\u00b2": train_r2,
                "Test R\u00b2": test_r2,
                "Gap": train_r2 - test_r2,
            }
        )
    gap_df = pd.DataFrame(gap_rows)

    st.write("### Generalization Gap (Train R\u00b2 - Test R\u00b2)")
    st.dataframe(
        gap_df.style.format(
            {"Train R\u00b2": "{:.3f}", "Test R\u00b2": "{:.3f}", "Gap": "{:.3f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )

    # ------------------------------------------------------------------
    # NYC error maps for all three strategies
    # ------------------------------------------------------------------
    map_data = st.session_state.get("lab4_map_data", {})
    if map_data and "df" in st.session_state.get("lab4_vars", {}):
        st.write("### Prediction Errors on the NYC Map")
        st.caption(
            "Each dot is a test-set listing coloured by absolute prediction "
            "error (green = low, red = high). Compare how errors cluster "
            "differently across the three splitting strategies."
        )
        geojson = load_neighbourhood_geojson()
        display_comparison_maps(
            st.session_state["lab4_vars"]["df"],
            map_data,
            geojson,
        )

    st.markdown(
        """
    ### Key Takeaways

    1. **Random splits** give the most optimistic performance because
       train and test share hosts and neighborhoods. The model can
       "cheat" by memorising host- or location-specific patterns.

    2. **Host-based splits** test whether the model can price a listing
       from a host it has never seen. Performance typically drops because
       host-specific pricing strategies are no longer exploitable.

    3. **Neighbourhood-based splits** are harder still. Entire
       neighbourhoods are unseen at test time, so the model must
       generalise pricing patterns across locations.

    4. **Borough-based splits** are the hardest — an extreme form of
       out-of-distribution (OOD) evaluation. An entire borough (e.g.
       all of Manhattan) is unseen at test time, forcing the model to
       transfer knowledge across fundamentally different housing markets.

    5. **The gap between train and test performance** (the
       "generalisation gap") tends to grow as the split becomes more
       realistic. A large gap signals that the model is overfitting to
       patterns specific to the training distribution.

    ### Which Split Should You Use?

    It depends on your **deployment scenario**:

    | Scenario | Best Split |
    |---|---|
    | Predict price for a new listing from an *existing* host | Random |
    | Predict price for a listing from a *new* host | Host-based |
    | Expand to a *new neighbourhood* | Neighbourhood-based |
    | Expand to a *new borough* or city | Borough-based |

    **Always choose the split that matches how your model will be used
    in the real world.** Optimistic evaluations lead to nasty surprises
    in production!
    """
    )

    st.balloons()
