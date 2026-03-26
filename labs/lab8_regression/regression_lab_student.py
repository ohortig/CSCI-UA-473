"""
Lab 8: Regression — Point Estimation, MDN, and Quantile Regression

Students use Airbnb listing embeddings (text, image, or joint) to predict
nightly price with three approaches:
  1. MLP point regression (MSE)
  2. Mixture Density Network (MDN)
  3. Simultaneous Quantile Regression (SQR)
"""

import io
import sys
import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from streamlit_monaco import st_monaco

from labs.lab8_regression.level_checks import (
    check_step_0_explore,
    check_step_1_load,
    check_step_2_mlp_architecture,
    check_step_3_mlp_training,
    check_step_4_standardize,
    check_step_5_mdn_architecture,
    check_step_6_mdn_training,
    check_step_7_sqr,
)

# ======================================================================
# Helpers
# ======================================================================

_PRE_INJECTED = {
    "pd": pd,
    "np": np,
    "torch": torch,
    "nn": nn,
    "optim": optim,
    "F": F,
    "math": __import__("math"),
}


def _exec_with_capture(code, local_vars):
    """Execute code, capturing stdout. Returns (output_str, traceback_str_or_None)."""
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    tb_str = None
    try:
        exec(code, local_vars)
    except Exception:
        tb_str = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
    return mystdout.getvalue(), tb_str


def _run_and_save(step_key, code, local_vars, check_fn, spinner_msg="Running..."):
    """Execute code, run check, save result to session state. Returns (passed, lv)."""
    with st.spinner(spinner_msg):
        output, tb = _exec_with_capture(code, local_vars)

    result = {"output": output, "tb": tb, "passed": False, "msg": ""}

    if tb is not None:
        result["msg"] = ""
    else:
        passed, msg = check_fn(local_vars)
        result["passed"] = passed
        result["msg"] = msg

    st.session_state[step_key] = result
    return result


def _show_result(step_key):
    """Render persisted output/check result for a step. Returns (passed, shown)."""
    result = st.session_state.get(step_key)
    if result is None:
        return False, False

    if result["output"]:
        st.code(result["output"], language="text")
    if result["tb"]:
        st.error("Runtime Error")
        st.code(result["tb"], language="python")
        return False, True
    if result["passed"]:
        st.success(result["msg"])
    else:
        st.error(result["msg"])
    return result["passed"], True


def _plot_loss_curve(train_key, val_key, title, ylabel):
    """Plot a loss curve from lab8_vars if data exists."""
    lv = st.session_state.get("lab8_vars", {})
    train_losses = lv.get(train_key, [])
    val_losses = lv.get(val_key, [])
    if not train_losses:
        return
    data = {"Epoch": range(1, len(train_losses) + 1), "Train": train_losses}
    if val_losses and len(val_losses) == len(train_losses):
        data["Val"] = val_losses
    loss_df = pd.DataFrame(data)
    y_cols = [c for c in loss_df.columns if c != "Epoch"]
    fig = px.line(
        loss_df,
        x="Epoch",
        y=y_cols,
        title=title,
        labels={"value": ylabel, "variable": ""},
        color_discrete_map={"Train": "#4A90D9", "Val": "#E74C3C"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# Step 0: Load Data & Explore Price Distribution
# ======================================================================


def _render_step_0(show_solutions=False):
    st.subheader("Step 0: Load Data & Explore Price Distribution")
    st.info(
        "**Goal**: Load the Airbnb dataset and explore the price distribution. "
        "What do you notice about its shape?"
    )

    emb_dim = st.session_state.get("lab8_vars", {}).get("EMB_DIM", 768)
    emb_col = st.session_state.get("lab8_vars", {}).get("EMB_COL", "embedding")

    st.markdown(
        f"""
    We'll use the same Airbnb embedding dataset from Lab 4. Each listing has a
    {emb_dim}-dimensional embedding (`{emb_col}`) that captures listing
    features.

    **Your Task**:
    1. Load `data/processed/airbnb_embedded.parquet` into `df`.
    2. Filter to listings with price between $10 and $2,000.
    3. Compute a `stats` dict with keys `"mean"`, `"median"`, `"std"`, and
       `"skewness"` of the `price` column.

    > **Hint**: `pd.Series.skew()` computes the skewness.
    """
    )

    student_code = """\
# Load the Airbnb dataset
df = pd.read_parquet("data/processed/airbnb_embedded.parquet")

# Filter price range
df = df[(df["price"] >= 10) & (df["price"] <= 2000)].reset_index(drop=True)

# TODO: Compute descriptive statistics of the price column
stats = {
    "mean": ...,
    "median": ...,
    "std": ...,
    "skewness": ...,
}

print(f"Loaded {len(df):,} listings")
print(f"Price range: ${df['price'].min():.0f} – ${df['price'].max():.0f}")
for k, v in stats.items():
    print(f"  {k}: {v:.2f}")"""

    solution_code = """\
# Load the Airbnb dataset
df = pd.read_parquet("data/processed/airbnb_embedded.parquet")

# Filter price range
df = df[(df["price"] >= 10) & (df["price"] <= 2000)].reset_index(drop=True)

# Compute descriptive statistics of the price column
stats = {
    "mean": df["price"].mean(),
    "median": df["price"].median(),
    "std": df["price"].std(),
    "skewness": df["price"].skew(),
}

print(f"Loaded {len(df):,} listings")
print(f"Price range: ${df['price'].min():.0f} – ${df['price'].max():.0f}")
for k, v in stats.items():
    print(f"  {k}: {v:.2f}")"""

    default_code = solution_code if show_solutions else student_code
    code = st_monaco(
        value=default_code,
        height="300px",
        language="python",
        theme="vs-dark",
    )

    if st.button("Run Step 0", key="lab8_run_0"):
        st.session_state["lab8_vars"] = dict(_PRE_INJECTED)
        result = _run_and_save(
            "lab8_step_0_result",
            code,
            st.session_state["lab8_vars"],
            check_step_0_explore,
            "Loading data...",
        )
        if result["passed"]:
            st.session_state["lab8_step_0_done"] = True

    # --- Persistent output ---
    passed, shown = _show_result("lab8_step_0_result")
    if passed:
        lv = st.session_state["lab8_vars"]
        df = lv["df"]

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(
                df,
                x="price",
                nbins=80,
                title="Raw Price Distribution",
                labels={"price": "Price ($)"},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            log_price = np.log1p(df["price"].values)
            fig = px.histogram(
                x=log_price,
                nbins=80,
                title="log(1 + price) Distribution",
                labels={"x": "log(1 + price)"},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        if "neighbourhood_group_cleansed" in df.columns:
            fig = px.box(
                df,
                x="neighbourhood_group_cleansed",
                y="price",
                title="Price by Borough",
                labels={
                    "neighbourhood_group_cleansed": "Borough",
                    "price": "Price ($)",
                },
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            r"""
            **Why does distribution shape matter for regression?**
            """
        )


# ======================================================================
# Step 1: Prepare Features
# ======================================================================


def _render_step_1(show_solutions=False):
    st.divider()
    st.subheader("Step 1: Prepare Features")

    lv = st.session_state.get("lab8_vars", {})
    emb_col = lv.get("EMB_COL", "embedding")
    emb_dim = lv.get("EMB_DIM", 768)

    st.info(
        "**Goal**: Create feature matrix `X` from embeddings and target vector "
        "`y` — applying what you learned about the price distribution."
    )

    st.markdown(
        f"""
    Now that you've seen the price distribution, let's prepare the data for
    modeling.

    **Your Task**:
    1. Stack the `{emb_col}` column into a numpy matrix `X` with shape `(N, {emb_dim})`.
    2. Create a target vector `y` from the price column. Think about what
       transformation (if any) would help based on what you saw in Step 0.
    """
    )

    student_code = f"""\
# TODO: Stack embeddings into a 2-D numpy matrix (N, {emb_dim})
X = ...

# TODO: Create target vector y from the price column.
# Think about the distribution you saw in Step 0 —
# what transformation would make this a better regression target?
y = ...

print(f"X shape: {{X.shape}}")
print(f"y shape: {{y.shape}}")
print(f"y range: {{y.min():.2f}} – {{y.max():.2f}}")"""

    solution_code = f"""\
# Stack embeddings into a 2-D numpy matrix (N, {emb_dim})
X = np.stack(df["{emb_col}"].values)

# Create log-price target — log(1+price) to handle skewness
y = np.log1p(df["price"].values)

print(f"X shape: {{X.shape}}")
print(f"y shape: {{y.shape}}")
print(f"y range: {{y.min():.2f}} – {{y.max():.2f}}")"""

    default_code = solution_code if show_solutions else student_code
    code = st_monaco(
        value=default_code,
        height="240px",
        language="python",
        theme="vs-dark",
    )

    if st.button("Run Step 1", key="lab8_run_1"):
        result = _run_and_save(
            "lab8_step_1_result",
            code,
            st.session_state["lab8_vars"],
            check_step_1_load,
            "Preparing features...",
        )
        if result["passed"]:
            st.session_state["lab8_step_1_done"] = True

    _show_result("lab8_step_1_result")


# ======================================================================
# Step 2: Define MLP Point Regressor
# ======================================================================


def _render_step_2(show_solutions=False):
    st.divider()
    st.subheader("Step 2: Define MLP Point Regressor")

    emb_dim = st.session_state.get("lab8_vars", {}).get("EMB_DIM", 768)

    st.info(
        f"**Goal**: Build a simple MLP that maps a {emb_dim}-dim embedding to a "
        "single predicted log-price."
    )

    with st.expander("Why MSE = Gaussian MLE"):
        st.markdown(
            r"""
        Minimizing **Mean Squared Error** is equivalent to maximizing the
        log-likelihood of a Gaussian with fixed variance:

        $$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}(y_i - f_\theta(x_i))^2$$

        This means a point-regression MLP implicitly assumes **constant noise**
        (homoscedasticity) — it predicts the conditional *mean* but says nothing
        about uncertainty. We'll fix that with MDN and quantile regression!
        """
        )

    st.markdown(
        f"""
    **Architecture**: `{emb_dim} → 256 → ReLU → 128 → ReLU → 1`

    **Your Task**: Define `MLPRegressor(nn.Module)` with:
    - `self.net = nn.Sequential(...)` with the layers above.
    - `forward(self, x)` returns `self.net(x).squeeze(-1)` so output is `(N,)`.
    """
    )

    student_code = f"""\
class MLPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define self.net as nn.Sequential
        # {emb_dim} -> 256 -> ReLU -> 128 -> ReLU -> 1
        self.net = nn.Sequential(
            # ... fill in layers ...
        )

    def forward(self, x):
        # TODO: Return predictions with shape (N,)
        return ..."""

    solution_code = f"""\
class MLPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear({emb_dim}, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)"""

    mode = "sol" if show_solutions else "stu"
    default_code = solution_code if show_solutions else student_code
    code = st_monaco(
        value=default_code,
        height="280px",
        language="python",
        theme="vs-dark",
    )

    if st.button("Check Architecture", key="lab8_run_2"):
        exec_vars = dict(st.session_state["lab8_vars"])
        result = _run_and_save(
            "lab8_step_2_result",
            code,
            exec_vars,
            check_step_2_mlp_architecture,
            "Checking architecture...",
        )
        if result["passed"]:
            st.session_state["lab8_vars"].update(exec_vars)
            st.session_state["lab8_step_2_done"] = True

    _show_result("lab8_step_2_result")


# ======================================================================
# Step 3: Train MLP
# ======================================================================


def _render_step_3(show_solutions=False):
    st.divider()
    st.subheader("Step 3: Train MLP Regressor")
    st.info("**Goal**: Train the MLP with MSE loss and evaluate on held-out data.")

    st.markdown(
        """
    **Your Task**:
    1. Split data 80/10/10 using `train_test_split` (twice).
    2. Convert to float32 tensors.
    3. Train with Adam (lr=1e-3) for 300 epochs, collecting train & val loss.
    4. Store: `model_mlp`, `mlp_train_losses`, `mlp_val_losses`.
    """
    )

    student_code = """\
from sklearn.model_selection import train_test_split

# Split 80/10/10
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# TODO: Instantiate model, optimizer, and loss function
model_mlp = MLPRegressor()
optimizer = optim.Adam(model_mlp.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# TODO: Training loop — 300 epochs
mlp_train_losses = []
mlp_val_losses = []

for epoch in range(300):
    # --- Train ---
    model_mlp.train()
    # ... forward, loss, backward, step ...

    # --- Validate ---
    model_mlp.eval()
    with torch.no_grad():
        pass  # ... compute val loss ...

    # Append losses
    # mlp_train_losses.append(...)
    # mlp_val_losses.append(...)

print(f"Final train loss: {mlp_train_losses[-1]:.4f}")
print(f"Final val loss:   {mlp_val_losses[-1]:.4f}")"""

    solution_code = """\
from sklearn.model_selection import train_test_split

# Split 80/10/10
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

model_mlp = MLPRegressor()
optimizer = optim.Adam(model_mlp.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

mlp_train_losses = []
mlp_val_losses = []

for epoch in range(300):
    model_mlp.train()
    pred = model_mlp(X_train_t)
    loss = loss_fn(pred, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    mlp_train_losses.append(loss.item())

    model_mlp.eval()
    with torch.no_grad():
        val_pred = model_mlp(X_val_t)
        val_loss = loss_fn(val_pred, y_val_t)
    mlp_val_losses.append(val_loss.item())

print(f"Final train loss: {mlp_train_losses[-1]:.4f}")
print(f"Final val loss:   {mlp_val_losses[-1]:.4f}")"""

    mode = "sol" if show_solutions else "stu"
    default_code = solution_code if show_solutions else student_code
    code = st_monaco(
        value=default_code,
        height="420px",
        language="python",
        theme="vs-dark",
    )

    if st.button("Train MLP", key="lab8_run_3"):
        if "MLPRegressor" not in st.session_state["lab8_vars"]:
            st.error("Run Step 2 first to define MLPRegressor.")
            return

        result = _run_and_save(
            "lab8_step_3_result",
            code,
            st.session_state["lab8_vars"],
            check_step_3_mlp_training,
            "Training MLP (300 epochs)...",
        )
        if result["passed"]:
            st.session_state["lab8_step_3_done"] = True
            lv = st.session_state["lab8_vars"]
            lv["mlp_train_losses_step3"] = list(lv["mlp_train_losses"])
            lv["mlp_val_losses_step3"] = list(lv["mlp_val_losses"])
            # Snapshot model predictions so Step 4 doesn't overwrite them
            model_mlp = lv["model_mlp"]
            model_mlp.eval()
            with torch.no_grad():
                lv["step3_test_preds"] = model_mlp(lv["X_test_t"]).numpy()
            lv["step3_y_test"] = lv["y_test"]

    # --- Persistent output ---
    _plot_loss_curve(
        "mlp_train_losses_step3", "mlp_val_losses_step3", "MLP Training Loss (MSE)", "MSE Loss"
    )

    passed, shown = _show_result("lab8_step_3_result")
    if passed:
        lv = st.session_state["lab8_vars"]
        test_preds = lv["step3_test_preds"]
        y_test = lv["step3_y_test"]

        pred_dollars = np.expm1(test_preds)
        true_dollars = np.expm1(y_test)

        from sklearn.metrics import mean_absolute_error, r2_score

        r2 = r2_score(true_dollars, pred_dollars)
        mae = mean_absolute_error(true_dollars, pred_dollars)

        col1, col2 = st.columns(2)
        col1.metric("R² (test, $)", f"{r2:.3f}")
        col2.metric("MAE (test, $)", f"${mae:.2f}")

        scatter_df = pd.DataFrame(
            {
                "True Price ($)": true_dollars,
                "Predicted Price ($)": pred_dollars,
            }
        )
        fig = px.scatter(
            scatter_df,
            x="True Price ($)",
            y="Predicted Price ($)",
            title="MLP: Predicted vs Actual (Test Set)",
            opacity=0.4,
        )
        max_val = max(true_dollars.max(), pred_dollars.max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Perfect",
                line=dict(dash="dash", color="gray"),
            )
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# Step 4: Standardize Features & Retrain
# ======================================================================


def _render_step_4(show_solutions=False):
    st.divider()
    st.subheader("Step 4: Standardize Features & Retrain")
    st.info(
        "**Goal**: Improve the MLP by standardizing the input features. "
        "Compare the results to your Step 3 model."
    )

    st.markdown(
        r"""
    Neural networks are sensitive to the **scale** of their inputs. If some
    features have large values while others are small, gradients flow unevenly
    and training is slow or unstable.

    `StandardScaler` transforms each feature to have **zero mean** and **unit
    variance**:

    $$x'_j = \frac{x_j - \mu_j}{\sigma_j}$$

    **Your Task**:
    1. Fit a `StandardScaler` on `X_train` and transform train/val/test.
    2. Re-create tensors from the scaled arrays.
    3. Train a fresh `MLPRegressor` for 300 epochs (same as Step 3).
    4. Store: `scaler`, `model_mlp`, `mlp_train_losses`, `mlp_val_losses`.

    > **Important**: Fit the scaler on `X_train` only — never on val/test!
    """
    )

    student_code = """\
from sklearn.preprocessing import StandardScaler

# TODO: Fit scaler on training data, transform all splits
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = ...
X_test_s = ...

# Re-create tensors from scaled data
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# Train a fresh MLP on standardized features
model_mlp = MLPRegressor()
optimizer = optim.Adam(model_mlp.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

mlp_train_losses = []
mlp_val_losses = []

for epoch in range(300):
    model_mlp.train()
    # ... forward, loss, backward, step ...

    model_mlp.eval()
    with torch.no_grad():
        pass  # ... compute val loss ...

    # mlp_train_losses.append(...)
    # mlp_val_losses.append(...)

print(f"Final train loss: {mlp_train_losses[-1]:.4f}")
print(f"Final val loss:   {mlp_val_losses[-1]:.4f}")"""

    solution_code = """\
from sklearn.preprocessing import StandardScaler

# Fit scaler on training data, transform all splits
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Re-create tensors from scaled data
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# Train a fresh MLP on standardized features
model_mlp = MLPRegressor()
optimizer = optim.Adam(model_mlp.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

mlp_train_losses = []
mlp_val_losses = []

for epoch in range(300):
    model_mlp.train()
    pred = model_mlp(X_train_t)
    loss = loss_fn(pred, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    mlp_train_losses.append(loss.item())

    model_mlp.eval()
    with torch.no_grad():
        val_pred = model_mlp(X_val_t)
        val_loss = loss_fn(val_pred, y_val_t)
    mlp_val_losses.append(val_loss.item())

print(f"Final train loss: {mlp_train_losses[-1]:.4f}")
print(f"Final val loss:   {mlp_val_losses[-1]:.4f}")"""

    default_code = solution_code if show_solutions else student_code
    code = st_monaco(
        value=default_code,
        height="450px",
        language="python",
        theme="vs-dark",
    )

    if st.button("Train Standardized MLP", key="lab8_run_4"):
        if "MLPRegressor" not in st.session_state["lab8_vars"]:
            st.error("Run Step 2 first to define MLPRegressor.")
            return

        result = _run_and_save(
            "lab8_step_4_result",
            code,
            st.session_state["lab8_vars"],
            check_step_4_standardize,
            "Training standardized MLP (300 epochs)...",
        )
        if result["passed"]:
            st.session_state["lab8_step_4_done"] = True
            lv = st.session_state["lab8_vars"]
            lv["mlp_train_losses_step4"] = list(lv["mlp_train_losses"])
            lv["mlp_val_losses_step4"] = list(lv["mlp_val_losses"])
            # Snapshot model predictions so later steps don't overwrite them
            model_mlp = lv["model_mlp"]
            model_mlp.eval()
            with torch.no_grad():
                lv["step4_test_preds"] = model_mlp(lv["X_test_t"]).numpy()
            lv["step4_y_test"] = lv["y_test"]

    # --- Persistent output ---
    _plot_loss_curve(
        "mlp_train_losses_step4", "mlp_val_losses_step4",
        "Standardized MLP Training Loss (MSE)", "MSE Loss",
    )

    passed, shown = _show_result("lab8_step_4_result")
    if passed:
        lv = st.session_state["lab8_vars"]
        test_preds = lv["step4_test_preds"]
        y_test = lv["step4_y_test"]

        pred_dollars = np.expm1(test_preds)
        true_dollars = np.expm1(y_test)

        from sklearn.metrics import mean_absolute_error, r2_score

        r2 = r2_score(true_dollars, pred_dollars)
        mae = mean_absolute_error(true_dollars, pred_dollars)

        col1, col2 = st.columns(2)
        col1.metric("R² (test, $)", f"{r2:.3f}")
        col2.metric("MAE (test, $)", f"${mae:.2f}")

        scatter_df = pd.DataFrame(
            {
                "True Price ($)": true_dollars,
                "Predicted Price ($)": pred_dollars,
            }
        )
        fig = px.scatter(
            scatter_df,
            x="True Price ($)",
            y="Predicted Price ($)",
            title="Standardized MLP: Predicted vs Actual (Test Set)",
            opacity=0.4,
        )
        max_val = max(true_dollars.max(), pred_dollars.max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Perfect",
                line=dict(dash="dash", color="gray"),
            )
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# Step 5: Define MDN Architecture
# ======================================================================


def _render_step_5(show_solutions=False):
    st.divider()
    st.subheader("Step 5: Define Mixture Density Network (MDN)")
    st.info(
        "**Goal**: Build an MDN that outputs a *mixture of Gaussians* — "
        "modeling the full conditional distribution, not just the mean."
    )

    with st.expander("Why point predictions fail"):
        st.markdown(
            """
        A point regressor gives one number per listing. But real prices have
        **uncertainty**: two identical-looking listings can have very different
        prices due to hidden factors (seasonality, host strategy, photos, etc.).

        The MLP's single prediction can't tell you *how confident* it is.
        A Mixture Density Network solves this by predicting a full probability
        distribution over possible prices.
        """
        )

    with st.expander("Mixture of Gaussians formula"):
        st.markdown(
            r"""
        The MDN models the conditional distribution as:

        $$p(y \mid x) = \sum_{k=1}^{K} \pi_k(x)\;\mathcal{N}\!\bigl(y \mid \mu_k(x),\,\sigma_k^2(x)\bigr)$$

        where for each input $x$, the network predicts:
        - $\pi_k(x)$: mixing coefficients (sum to 1 via softmax)
        - $\mu_k(x)$: component means
        - $\sigma_k(x)$: component standard deviations (positive via softplus)
        """
        )

    with st.expander("NLL loss with log-sum-exp trick"):
        st.markdown(
            r"""
        We minimize the **negative log-likelihood**:

        $$\text{NLL} = -\frac{1}{N}\sum_{i=1}^{N}\log\!\left[\sum_{k=1}^{K}\pi_k(x_i)\;\mathcal{N}\!\bigl(y_i \mid \mu_k(x_i),\sigma_k^2(x_i)\bigr)\right]$$

        For numerical stability we use the **log-sum-exp trick**:

        $$\log\sum_k e^{a_k} = c + \log\sum_k e^{a_k - c}, \quad c = \max_k a_k$$

        where $a_k = \log\pi_k - \frac{1}{2}\log(2\pi\sigma_k^2) - \frac{(y-\mu_k)^2}{2\sigma_k^2}$.
        """
        )

    emb_dim = st.session_state.get("lab8_vars", {}).get("EMB_DIM", 768)

    st.markdown(
        f"""
    **Architecture**: Shared trunk `{emb_dim} → 256 → ReLU → 128 → ReLU`, then 3 heads:
    - `pi_head`: `128 → K` + softmax → mixing coefficients
    - `mu_head`: `128 → K` → component means
    - `sigma_head`: `128 → K` + softplus + ε → component std devs

    **Your Task**: Define `MDN(nn.Module)` with `__init__(self, n_components=5)`.
    `forward` returns `(pi, mu, sigma)` each with shape `(N, K)`.
    """
    )

    student_code = f"""\
class MDN(nn.Module):
    def __init__(self, n_components=5):
        super().__init__()
        self.K = n_components

        # TODO: Define shared trunk
        self.trunk = nn.Sequential(
            # ... {emb_dim} -> 256 -> ReLU -> 128 -> ReLU ...
        )

        # TODO: Define three heads
        # self.pi_head = ...
        # self.mu_head = ...
        # self.sigma_head = ...

    def forward(self, x):
        h = self.trunk(x)

        # TODO: Compute pi (softmax), mu, sigma (softplus + epsilon)
        pi = ...     # shape (N, K), sums to 1
        mu = ...     # shape (N, K)
        sigma = ...  # shape (N, K), strictly positive

        return pi, mu, sigma"""

    solution_code = f"""\
class MDN(nn.Module):
    def __init__(self, n_components=5):
        super().__init__()
        self.K = n_components

        self.trunk = nn.Sequential(
            nn.Linear({emb_dim}, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.pi_head = nn.Linear(128, n_components)
        self.mu_head = nn.Linear(128, n_components)
        self.sigma_head = nn.Linear(128, n_components)

    def forward(self, x):
        h = self.trunk(x)
        pi = F.softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) + 1e-6
        return pi, mu, sigma"""

    mode = "sol" if show_solutions else "stu"
    default_code = solution_code if show_solutions else student_code
    code = st_monaco(
        value=default_code,
        height="340px",
        language="python",
        theme="vs-dark",
    )

    if st.button("Check MDN Architecture", key="lab8_run_5"):
        exec_vars = dict(st.session_state["lab8_vars"])
        result = _run_and_save(
            "lab8_step_5_result",
            code,
            exec_vars,
            check_step_5_mdn_architecture,
            "Checking architecture...",
        )
        if result["passed"]:
            st.session_state["lab8_vars"].update(exec_vars)
            st.session_state["lab8_step_5_done"] = True

    _show_result("lab8_step_5_result")


# ======================================================================
# Step 6: Train MDN
# ======================================================================


def _render_step_6(show_solutions=False):
    st.divider()
    st.subheader("Step 6: Train MDN")
    st.info(
        "**Goal**: Implement the MDN NLL loss and train. Then visualize the "
        "predicted mixture distributions."
    )

    st.markdown(
        r"""
    **Your Task**:
    1. Write `mdn_loss(pi, mu, sigma, y)` that returns NLL using `torch.logsumexp`.
    2. Train for 300 epochs with Adam (lr=1e-3).
    3. Store: `model_mdn`, `mdn_train_losses`, `mdn_val_losses`.

    > **Hint**: Compute per-component log-prob $a_k = \log\pi_k - \log\sigma_k
    > - \frac{(y-\mu_k)^2}{2\sigma_k^2} - \frac{1}{2}\log(2\pi)$,
    > then `torch.logsumexp(a, dim=-1)`.
    """
    )

    student_code = """\
def mdn_loss(pi, mu, sigma, y):
    \"\"\"Negative log-likelihood for mixture of Gaussians.\"\"\"
    y = y.unsqueeze(-1)  # (N, 1) for broadcasting

    # TODO: Compute log-prob for each component
    # log_component_prob = log(pi) + log(N(y | mu, sigma^2))
    # Use torch.logsumexp for numerical stability
    # Return: mean negative log-likelihood

    return ...

# Train MDN
model_mdn = MDN(n_components=5)
optimizer = optim.Adam(model_mdn.parameters(), lr=1e-3)

mdn_train_losses = []
mdn_val_losses = []

for epoch in range(300):
    model_mdn.train()
    pi, mu, sigma = model_mdn(X_train_t)
    # TODO: compute loss, backprop, step
    loss = ...

    model_mdn.eval()
    with torch.no_grad():
        pi_v, mu_v, sigma_v = model_mdn(X_val_t)
        # val_loss = ...

    # mdn_train_losses.append(...)
    # mdn_val_losses.append(...)

print(f"Final train NLL: {mdn_train_losses[-1]:.4f}")
print(f"Final val NLL:   {mdn_val_losses[-1]:.4f}")"""

    solution_code = """\
def mdn_loss(pi, mu, sigma, y):
    \"\"\"Negative log-likelihood for mixture of Gaussians.\"\"\"
    y = y.unsqueeze(-1)  # (N, 1)
    log_pi = torch.log(pi + 1e-10)
    log_normal = (
        -0.5 * math.log(2 * math.pi)
        - torch.log(sigma)
        - 0.5 * ((y - mu) / sigma) ** 2
    )
    log_prob = torch.logsumexp(log_pi + log_normal, dim=-1)
    return -log_prob.mean()

model_mdn = MDN(n_components=5)
optimizer = optim.Adam(model_mdn.parameters(), lr=1e-3)

mdn_train_losses = []
mdn_val_losses = []

for epoch in range(300):
    model_mdn.train()
    pi, mu, sigma = model_mdn(X_train_t)
    loss = mdn_loss(pi, mu, sigma, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    mdn_train_losses.append(loss.item())

    model_mdn.eval()
    with torch.no_grad():
        pi_v, mu_v, sigma_v = model_mdn(X_val_t)
        val_loss = mdn_loss(pi_v, mu_v, sigma_v, y_val_t)
    mdn_val_losses.append(val_loss.item())

print(f"Final train NLL: {mdn_train_losses[-1]:.4f}")
print(f"Final val NLL:   {mdn_val_losses[-1]:.4f}")"""

    mode = "sol" if show_solutions else "stu"
    default_code = solution_code if show_solutions else student_code
    code = st_monaco(
        value=default_code,
        height="400px",
        language="python",
        theme="vs-dark",
    )

    if st.button("Train MDN", key="lab8_run_6"):
        if "MDN" not in st.session_state["lab8_vars"]:
            st.error("Run Step 5 first to define MDN.")
            return

        result = _run_and_save(
            "lab8_step_6_result",
            code,
            st.session_state["lab8_vars"],
            check_step_6_mdn_training,
            "Training MDN (300 epochs)...",
        )
        if result["passed"]:
            st.session_state["lab8_step_6_done"] = True

    # --- Persistent output ---
    _plot_loss_curve(
        "mdn_train_losses", "mdn_val_losses", "MDN Training Loss (NLL)", "NLL"
    )

    passed, shown = _show_result("lab8_step_6_result")
    if passed:
        lv = st.session_state["lab8_vars"]

        # --- Component means visualization ---
        st.markdown("**Component means across test listings (sorted by true price):**")
        model_mdn = lv["model_mdn"]
        model_mdn.eval()
        with torch.no_grad():
            pi_all, mu_all, sigma_all = model_mdn(lv["X_test_t"])

        y_test = lv["y_test"]
        pi_np = pi_all.numpy()
        mu_np = mu_all.numpy()
        K = mu_np.shape[1]

        # Sort listings by true price
        sort_idx = np.argsort(y_test)
        x_axis = np.arange(len(sort_idx))
        true_sorted = np.expm1(y_test[sort_idx])

        comp_colors = [
            "#E74C3C", "#2ECC71", "#9B59B6", "#F39C12", "#1ABC9C",
            "#3498DB", "#E67E22", "#8E44AD",
        ]

        fig = go.Figure()
        for k in range(K):
            mu_k_sorted = np.expm1(mu_np[sort_idx, k])
            pi_k_sorted = pi_np[sort_idx, k]
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=mu_k_sorted,
                    mode="markers",
                    name=f"Component {k + 1}",
                    marker=dict(
                        size=pi_k_sorted * 12 + 2,
                        color=comp_colors[k % len(comp_colors)],
                        opacity=0.6,
                    ),
                    hovertemplate=(
                        "mean=$%{y:.0f}<br>"
                        f"weight=%{{customdata:.2f}}"
                        "<extra></extra>"
                    ),
                    customdata=pi_k_sorted,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=true_sorted,
                mode="markers",
                name="True Price",
                marker=dict(size=3, color="black"),
            )
        )
        fig.update_layout(
            title="MDN Component Means (dot size = mixing weight)",
            xaxis_title="Listings (sorted by true price)",
            yaxis_title="Price ($)",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Uncertainty scatter ---
        st.markdown("**Predicted mean vs. uncertainty:**")
        with torch.no_grad():
            pi_t, mu_t, sigma_t = model_mdn(lv["X_test_t"])
        expected_y = (pi_t * mu_t).sum(dim=1).numpy()
        expected_std = torch.sqrt(
            (pi_t * (sigma_t**2 + mu_t**2)).sum(dim=1) - ((pi_t * mu_t).sum(dim=1)) ** 2
        ).numpy()

        unc_df = pd.DataFrame(
            {
                "Predicted Mean ($)": np.expm1(expected_y),
                "Predicted Std ($)": np.expm1(expected_y + expected_std)
                - np.expm1(expected_y),
                "True Price ($)": np.expm1(y_test),
            }
        )
        fig = px.scatter(
            unc_df,
            x="Predicted Mean ($)",
            y="Predicted Std ($)",
            color="True Price ($)",
            title="MDN: Predicted Mean vs Uncertainty",
            opacity=0.5,
            color_continuous_scale="Viridis",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# Step 7: Simultaneous Quantile Regression
# ======================================================================


def _render_step_7(show_solutions=False):
    st.divider()
    st.subheader("Step 7: Simultaneous Quantile Regression (SQR)")
    st.info(
        "**Goal**: Predict *multiple quantiles* simultaneously — "
        "another way to capture uncertainty without assuming Gaussian noise."
    )

    with st.expander("What is quantile regression?"):
        st.markdown(
            r"""
        Instead of predicting the **mean** (MSE) or a full distribution (MDN),
        quantile regression predicts specific **percentiles** of the distribution.

        The **pinball loss** (aka quantile loss) for quantile $\tau \in (0,1)$:

        $$L_\tau(y, \hat{q}) = \max\!\bigl[\tau\,(y - \hat{q}),\;(\tau - 1)(y - \hat{q})\bigr]$$

        This asymmetric loss "pulls" the prediction toward the $\tau$-th percentile.
        For $\tau=0.5$ it's equivalent to median regression (MAE).
        """
        )

    st.markdown(
        r"""
    **Quantiles**: `[0.1, 0.25, 0.5, 0.75, 0.9]` (5 outputs)

    **Your Task**:
    1. Define `SQRModel(nn.Module)`: same MLP but outputs 5 values (one per quantile).
    2. Implement `pinball_loss(preds, targets, quantiles)`.
    3. Train for 300 epochs with Adam (lr=1e-3, weight_decay=1e-4).
    4. Store: `model_sqr`, `sqr_train_losses`, `sqr_val_losses`.
    """
    )

    emb_dim = st.session_state.get("lab8_vars", {}).get("EMB_DIM", 768)

    student_code = f"""\
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

class SQRModel(nn.Module):
    def __init__(self, n_quantiles=5):
        super().__init__()
        # TODO: Same trunk as MLP, but output n_quantiles instead of 1
        self.net = nn.Sequential(
            # ... fill in ...
        )

    def forward(self, x):
        return self.net(x)  # shape (N, n_quantiles)

def pinball_loss(preds, targets, quantiles):
    \"\"\"
    preds: (N, Q) predicted quantiles
    targets: (N,) true values
    quantiles: list of Q quantile levels
    \"\"\"
    # TODO: Implement pinball loss
    # Hint: errors = targets.unsqueeze(-1) - preds
    # Then apply asymmetric weighting per quantile
    return ...

model_sqr = SQRModel(n_quantiles=len(QUANTILES))
optimizer = optim.Adam(model_sqr.parameters(), lr=1e-3, weight_decay=1e-4)
quantiles_t = torch.tensor(QUANTILES, dtype=torch.float32)

sqr_train_losses = []
sqr_val_losses = []

for epoch in range(300):
    model_sqr.train()
    # TODO: forward, loss, backward, step

    model_sqr.eval()
    with torch.no_grad():
        pass  # TODO: compute val loss

    # sqr_train_losses.append(...)
    # sqr_val_losses.append(...)

print(f"Final train loss: {{sqr_train_losses[-1]:.4f}}")
print(f"Final val loss:   {{sqr_val_losses[-1]:.4f}}")"""

    solution_code = f"""\
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

class SQRModel(nn.Module):
    def __init__(self, n_quantiles=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear({emb_dim}, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_quantiles),
        )

    def forward(self, x):
        return self.net(x)

def pinball_loss(preds, targets, quantiles):
    \"\"\"Pinball (quantile) loss.\"\"\"
    errors = targets.unsqueeze(-1) - preds  # (N, Q)
    loss = torch.where(
        errors >= 0,
        quantiles * errors,
        (quantiles - 1) * errors,
    )
    return loss.mean()

model_sqr = SQRModel(n_quantiles=len(QUANTILES))
optimizer = optim.Adam(model_sqr.parameters(), lr=1e-3, weight_decay=1e-4)
quantiles_t = torch.tensor(QUANTILES, dtype=torch.float32)

sqr_train_losses = []
sqr_val_losses = []

for epoch in range(300):
    model_sqr.train()
    preds = model_sqr(X_train_t)
    loss = pinball_loss(preds, y_train_t, quantiles_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    sqr_train_losses.append(loss.item())

    model_sqr.eval()
    with torch.no_grad():
        val_preds = model_sqr(X_val_t)
        val_loss = pinball_loss(val_preds, y_val_t, quantiles_t)
    sqr_val_losses.append(val_loss.item())

print(f"Final train loss: {{sqr_train_losses[-1]:.4f}}")
print(f"Final val loss:   {{sqr_val_losses[-1]:.4f}}")"""

    mode = "sol" if show_solutions else "stu"
    default_code = solution_code if show_solutions else student_code
    code = st_monaco(
        value=default_code,
        height="450px",
        language="python",
        theme="vs-dark",
    )

    if st.button("Train SQR", key="lab8_run_7"):
        if "X_train_t" not in st.session_state["lab8_vars"]:
            st.error("Run Step 3 first to split and tensorize data.")
            return

        result = _run_and_save(
            "lab8_step_7_result",
            code,
            st.session_state["lab8_vars"],
            check_step_7_sqr,
            "Training SQR (300 epochs)...",
        )
        if result["passed"]:
            st.session_state["lab8_step_7_done"] = True

    # --- Persistent output ---
    QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
    _plot_loss_curve(
        "sqr_train_losses",
        "sqr_val_losses",
        "SQR Training Loss (Pinball)",
        "Pinball Loss",
    )

    passed, shown = _show_result("lab8_step_7_result")
    if passed:
        lv = st.session_state["lab8_vars"]

        # --- Quantile fan plot ---
        st.markdown("**Quantile Fan Plot (test set, sorted by median):**")
        model_sqr = lv["model_sqr"]
        model_sqr.eval()
        with torch.no_grad():
            q_preds = model_sqr(lv["X_test_t"]).numpy()  # (N, 5)

        y_test = lv["y_test"]
        q_dollars = np.expm1(q_preds)
        true_dollars = np.expm1(y_test)

        sort_idx = np.argsort(q_preds[:, 2])
        q_sorted = q_dollars[sort_idx]
        true_sorted = true_dollars[sort_idx]
        x_axis = np.arange(len(sort_idx))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_axis, x_axis[::-1]]),
                y=np.concatenate([q_sorted[:, 4], q_sorted[::-1, 0]]),
                fill="toself",
                fillcolor="rgba(74,144,217,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Q10–Q90",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_axis, x_axis[::-1]]),
                y=np.concatenate([q_sorted[:, 3], q_sorted[::-1, 1]]),
                fill="toself",
                fillcolor="rgba(74,144,217,0.3)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Q25–Q75",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=q_sorted[:, 2],
                mode="lines",
                name="Median (Q50)",
                line=dict(color="#4A90D9", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=true_sorted,
                mode="markers",
                name="True Price",
                marker=dict(color="#E74C3C", size=3, opacity=0.5),
            )
        )
        fig.update_layout(
            title="Quantile Fan Plot",
            xaxis_title="Listings (sorted by median prediction)",
            yaxis_title="Price ($)",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Median predicted vs actual ---
        from sklearn.metrics import mean_absolute_error, r2_score

        median_dollars = q_dollars[:, 2]
        r2 = r2_score(true_dollars, median_dollars)
        mae = mean_absolute_error(true_dollars, median_dollars)

        col1, col2 = st.columns(2)
        col1.metric("R² (median, $)", f"{r2:.3f}")
        col2.metric("MAE (median, $)", f"${mae:.2f}")

        scatter_df = pd.DataFrame(
            {
                "True Price ($)": true_dollars,
                "Predicted Median ($)": median_dollars,
                "80% Interval Width ($)": q_dollars[:, 4] - q_dollars[:, 0],
            }
        )
        fig = px.scatter(
            scatter_df,
            x="True Price ($)",
            y="Predicted Median ($)",
            color="80% Interval Width ($)",
            title="SQR: Predicted Median vs Actual (color = prediction interval width)",
            opacity=0.5,
            color_continuous_scale="Viridis",
        )
        max_val = max(true_dollars.max(), median_dollars.max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Perfect",
                line=dict(dash="dash", color="gray"),
                showlegend=False,
            )
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        # --- Coverage calibration ---
        st.markdown("**Coverage Calibration:**")
        coverage_data = []
        for i, q in enumerate(QUANTILES):
            below = (y_test <= q_preds[:, i]).mean()
            coverage_data.append(
                {
                    "Quantile": f"Q{int(q * 100)}",
                    "Expected Coverage": f"{q:.0%}",
                    "Actual Coverage": f"{below:.1%}",
                }
            )
        st.table(pd.DataFrame(coverage_data))

        # --- 3-model comparison ---
        st.markdown("**Model Comparison (Test Set):**")

        model_mlp = lv["model_mlp"]
        model_mdn = lv["model_mdn"]

        model_mlp.eval()
        model_mdn.eval()
        with torch.no_grad():
            mlp_preds = model_mlp(lv["X_test_t"]).numpy()
            pi, mu, sigma = model_mdn(lv["X_test_t"])
            mdn_mean = (pi * mu).sum(dim=1).numpy()
            sqr_median = q_preds[:, 2]

        results = []
        for name, preds_log in [
            ("MLP (point)", mlp_preds),
            ("MDN (expected value)", mdn_mean),
            ("SQR (median)", sqr_median),
        ]:
            p_dollar = np.expm1(preds_log)
            t_dollar = np.expm1(y_test)
            results.append(
                {
                    "Model": name,
                    "R²": f"{r2_score(t_dollar, p_dollar):.3f}",
                    "MAE ($)": f"${mean_absolute_error(t_dollar, p_dollar):.2f}",
                }
            )
        st.table(pd.DataFrame(results))


# ======================================================================
# Main render function
# ======================================================================


_EMBEDDING_OPTIONS = {
    "Text Embeddings (768-dim)": {"col": "embedding", "dim": 768},
    "Image Embeddings (384-dim)": {"col": "image_embedding", "dim": 384},
    "Joint Embeddings (1152-dim)": {"col": "joint_embedding", "dim": 1152},
}


def render_regression_lab(show_solutions=False):
    st.header("Lab 8: Regression — Point, MDN, and Quantile")
    st.markdown(
        """
    In this lab you'll predict Airbnb nightly prices using three approaches:
    1. **MLP Point Regression** — predict the mean (MSE loss)
    2. **Mixture Density Network** — predict a full distribution
    3. **Quantile Regression** — predict specific percentiles

    Each approach reveals different aspects of the prediction problem!
    """
    )

    embedding_choice = st.radio(
        "Choose Embedding Type",
        list(_EMBEDDING_OPTIONS.keys()),
        horizontal=True,
        key="lab8_embedding_choice",
    )
    emb_cfg = _EMBEDDING_OPTIONS[embedding_choice]

    # Clear all lab state when embedding type changes
    if st.session_state.get("lab8_loaded_embedding_type") != embedding_choice:
        for key in list(st.session_state.keys()):
            if key.startswith("lab8_") and key not in (
                "lab8_embedding_choice",
                "lab8_loaded_embedding_type",
            ):
                del st.session_state[key]
        st.session_state["lab8_loaded_embedding_type"] = embedding_choice

    if "lab8_vars" not in st.session_state:
        st.session_state["lab8_vars"] = dict(_PRE_INJECTED)

    st.session_state["lab8_vars"]["EMB_COL"] = emb_cfg["col"]
    st.session_state["lab8_vars"]["EMB_DIM"] = emb_cfg["dim"]

    _render_step_0(show_solutions)

    if st.session_state.get("lab8_step_0_done"):
        _render_step_1(show_solutions)

    if st.session_state.get("lab8_step_1_done"):
        _render_step_2(show_solutions)

    if st.session_state.get("lab8_step_2_done"):
        _render_step_3(show_solutions)

    if st.session_state.get("lab8_step_3_done"):
        _render_step_4(show_solutions)

    if st.session_state.get("lab8_step_4_done"):
        _render_step_5(show_solutions)

    if st.session_state.get("lab8_step_5_done"):
        _render_step_6(show_solutions)

    if st.session_state.get("lab8_step_6_done"):
        _render_step_7(show_solutions)
