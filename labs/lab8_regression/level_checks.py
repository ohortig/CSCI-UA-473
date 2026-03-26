import numpy as np
import torch


def check_step_0_explore(local_vars):
    """
    Step 0: Load data & explore price distribution.
    Expected: 'df' (DataFrame with 'price' column), 'stats' (dict with descriptive stats).
    """
    if "df" not in local_vars:
        return False, "Variable `df` not found. Load the parquet file."

    df = local_vars["df"]

    if "price" not in df.columns:
        return False, "DataFrame `df` doesn't have a `price` column."

    if len(df) < 100:
        return False, f"Only {len(df)} rows — did you filter too aggressively?"

    if "stats" not in local_vars:
        return False, "Variable `stats` not found. Compute descriptive statistics."

    stats = local_vars["stats"]
    if not isinstance(stats, dict):
        return False, "`stats` should be a dict."

    for key in ("mean", "median", "std", "skewness"):
        if key not in stats:
            return False, f"`stats` is missing key `'{key}'`."

    skew = stats["skewness"]
    if not isinstance(skew, (int, float)):
        return False, f"`stats['skewness']` should be a number, got {type(skew).__name__}."

    if abs(skew) < 1.0:
        return (
            False,
            f"Skewness = {skew:.2f} — that seems low. "
            "Make sure you're computing skewness of the raw price column.",
        )

    return True, "Data loaded! Take a look at the distribution — what do you notice?"


def check_step_1_load(local_vars):
    """
    Step 1: Load data & prepare X, y.
    Expected: 'df' (DataFrame), 'X' (ndarray N×D), 'y' (ndarray N,)
    """
    if "df" not in local_vars:
        return False, "⚠️ Variable `df` not found. Load the parquet file."

    if "X" not in local_vars:
        return False, "⚠️ Variable `X` not found. Stack embeddings into a matrix."

    if "y" not in local_vars:
        return False, "⚠️ Variable `y` not found. Create the log-price target."

    df = local_vars["df"]
    X = local_vars["X"]
    y = local_vars["y"]
    expected_dim = local_vars.get("EMB_DIM", 768)

    if X.ndim != 2:
        return False, f"⚠️ `X` should be 2-D, got {X.ndim}-D."

    if X.shape[1] != expected_dim:
        return False, f"⚠️ `X` should have {expected_dim} columns (embedding dim), got {X.shape[1]}."

    if y.ndim != 1:
        return False, f"⚠️ `y` should be 1-D, got {y.ndim}-D."

    if X.shape[0] != y.shape[0]:
        return (
            False,
            f"⚠️ Row count mismatch: X has {X.shape[0]} rows, y has {y.shape[0]}.",
        )

    if X.shape[0] != len(df):
        return (
            False,
            f"⚠️ Row count mismatch: X has {X.shape[0]} rows, df has {len(df)}.",
        )

    if np.any(y < 0):
        return False, "⚠️ `y` contains negative values. Did you apply `np.log1p` to price?"

    return True, "✅ Data loaded! X and y are ready for modeling."


def check_step_2_mlp_architecture(local_vars):
    """
    Step 2: Define MLPRegressor.
    Expected: 'MLPRegressor' class producing shape (N,) from (N, D).
    """
    if "MLPRegressor" not in local_vars:
        return False, "⚠️ Class `MLPRegressor` not found."

    MLPRegressor = local_vars["MLPRegressor"]
    emb_dim = local_vars.get("EMB_DIM", 768)

    try:
        model = MLPRegressor()
    except Exception as e:
        return False, f"⚠️ Could not instantiate MLPRegressor(): {e}"

    if not isinstance(model, torch.nn.Module):
        return False, "⚠️ `MLPRegressor` must inherit from `nn.Module`."

    try:
        test_input = torch.randn(4, emb_dim)
        output = model(test_input)
    except Exception as e:
        return False, f"⚠️ Forward pass failed on input (4, {emb_dim}): {e}"

    if output.shape != (4,):
        return (
            False,
            f"⚠️ Expected output shape (4,), got {tuple(output.shape)}. "
            "Make sure to `.squeeze(-1)` so the output is (N,), not (N, 1).",
        )

    return True, "✅ MLPRegressor architecture looks good!"


def check_step_3_mlp_training(local_vars):
    """
    Step 3: Train MLP.
    Expected: 'model_mlp', 'mlp_train_losses', 'mlp_val_losses'.
    """
    if "model_mlp" not in local_vars:
        return False, "⚠️ Variable `model_mlp` not found."

    if "mlp_train_losses" not in local_vars:
        return False, "⚠️ Variable `mlp_train_losses` not found."

    if "mlp_val_losses" not in local_vars:
        return False, "⚠️ Variable `mlp_val_losses` not found."

    train_losses = local_vars["mlp_train_losses"]
    val_losses = local_vars["mlp_val_losses"]

    if len(train_losses) == 0:
        return False, "⚠️ `mlp_train_losses` is empty. Did training run?"

    if len(train_losses) != len(val_losses):
        return (
            False,
            f"⚠️ Length mismatch: train_losses has {len(train_losses)} entries, "
            f"val_losses has {len(val_losses)}.",
        )

    if train_losses[-1] > train_losses[0] * 0.95:
        return (
            False,
            "⚠️ Training loss didn't decrease. Check optimizer or learning rate.",
        )

    return True, "✅ MLP training complete!"


def check_step_4_standardize(local_vars):
    """
    Step 4: Standardize features and retrain.
    Expected: 'scaler', standardized tensors, retrained 'model_mlp',
    updated loss lists.
    """
    if "scaler" not in local_vars:
        return False, "Variable `scaler` not found. Create a StandardScaler."

    scaler = local_vars["scaler"]
    scaler_type = type(scaler).__name__
    if scaler_type != "StandardScaler":
        return False, f"`scaler` should be a StandardScaler, got {scaler_type}."

    if "model_mlp" not in local_vars:
        return False, "Variable `model_mlp` not found."

    if "mlp_train_losses" not in local_vars:
        return False, "Variable `mlp_train_losses` not found."

    train_losses = local_vars["mlp_train_losses"]
    val_losses = local_vars.get("mlp_val_losses", [])

    if len(train_losses) == 0:
        return False, "`mlp_train_losses` is empty. Did training run?"

    if len(train_losses) != len(val_losses):
        return (
            False,
            f"Length mismatch: train has {len(train_losses)}, val has {len(val_losses)}.",
        )

    if train_losses[-1] > train_losses[0] * 0.95:
        return False, "Training loss didn't decrease. Check optimizer or learning rate."

    return True, "Standardized model trained! Compare the metrics to Step 3."


def check_step_5_mdn_architecture(local_vars):
    """
    Step 5: Define MDN.
    Expected: 'MDN' class returning (pi, mu, sigma) with correct shapes.
    """
    if "MDN" not in local_vars:
        return False, "⚠️ Class `MDN` not found."

    MDN = local_vars["MDN"]
    emb_dim = local_vars.get("EMB_DIM", 768)

    try:
        K = 5
        model = MDN(n_components=K)
    except Exception as e:
        return False, f"⚠️ Could not instantiate MDN(n_components=5): {e}"

    if not isinstance(model, torch.nn.Module):
        return False, "⚠️ `MDN` must inherit from `nn.Module`."

    try:
        test_input = torch.randn(4, emb_dim)
        result = model(test_input)
    except Exception as e:
        return False, f"⚠️ Forward pass failed: {e}"

    if not isinstance(result, tuple) or len(result) != 3:
        return False, "⚠️ MDN.forward() should return a 3-tuple (pi, mu, sigma)."

    pi, mu, sigma = result

    if pi.shape != (4, K):
        return False, f"⚠️ `pi` shape should be (4, {K}), got {tuple(pi.shape)}."

    if mu.shape != (4, K):
        return False, f"⚠️ `mu` shape should be (4, {K}), got {tuple(mu.shape)}."

    if sigma.shape != (4, K):
        return False, f"⚠️ `sigma` shape should be (4, {K}), got {tuple(sigma.shape)}."

    # Check pi sums to 1
    pi_sum = pi.sum(dim=1)
    if not torch.allclose(pi_sum, torch.ones(4), atol=1e-4):
        return False, "⚠️ `pi` should sum to 1 along dim=1 (use softmax)."

    # Check sigma > 0
    if (sigma <= 0).any():
        return False, "⚠️ `sigma` must be strictly positive (use softplus)."

    return True, "✅ MDN architecture looks good!"


def check_step_6_mdn_training(local_vars):
    """
    Step 6: Train MDN.
    Expected: 'model_mdn', 'mdn_train_losses', 'mdn_val_losses', 'mdn_loss' function.
    """
    if "model_mdn" not in local_vars:
        return False, "⚠️ Variable `model_mdn` not found."

    if "mdn_train_losses" not in local_vars:
        return False, "⚠️ Variable `mdn_train_losses` not found."

    if "mdn_val_losses" not in local_vars:
        return False, "⚠️ Variable `mdn_val_losses` not found."

    if "mdn_loss" not in local_vars:
        return False, "⚠️ Function `mdn_loss` not found."

    train_losses = local_vars["mdn_train_losses"]
    val_losses = local_vars["mdn_val_losses"]

    if len(train_losses) == 0:
        return False, "⚠️ `mdn_train_losses` is empty. Did training run?"

    if len(train_losses) != len(val_losses):
        return (
            False,
            f"⚠️ Length mismatch: train has {len(train_losses)}, val has {len(val_losses)}.",
        )

    if train_losses[-1] > train_losses[0] * 0.95:
        return False, "⚠️ MDN loss didn't decrease. Check your loss function or lr."

    # Verify model still produces valid outputs
    model_mdn = local_vars["model_mdn"]
    emb_dim = local_vars.get("EMB_DIM", 768)
    model_mdn.eval()
    with torch.no_grad():
        test_x = torch.randn(2, emb_dim)
        pi, mu, sigma = model_mdn(test_x)
    if (sigma <= 0).any():
        return False, "⚠️ Trained model produces non-positive sigma."

    return True, "✅ MDN training complete!"


def check_step_7_sqr(local_vars):
    """
    Step 7: Simultaneous Quantile Regression.
    Expected: 'model_sqr', 'sqr_train_losses', 'sqr_val_losses', 'pinball_loss'.
    """
    if "model_sqr" not in local_vars:
        return False, "⚠️ Variable `model_sqr` not found."

    if "sqr_train_losses" not in local_vars:
        return False, "⚠️ Variable `sqr_train_losses` not found."

    if "sqr_val_losses" not in local_vars:
        return False, "⚠️ Variable `sqr_val_losses` not found."

    if "pinball_loss" not in local_vars:
        return False, "⚠️ Function `pinball_loss` not found."

    train_losses = local_vars["sqr_train_losses"]
    val_losses = local_vars["sqr_val_losses"]

    if len(train_losses) == 0:
        return False, "⚠️ `sqr_train_losses` is empty."

    if len(train_losses) != len(val_losses):
        return (
            False,
            f"⚠️ Length mismatch: train has {len(train_losses)}, val has {len(val_losses)}.",
        )

    if train_losses[-1] > train_losses[0] * 0.95:
        return False, "⚠️ SQR loss didn't decrease."

    # Check quantile ordering on the actual training data
    model_sqr = local_vars["model_sqr"]
    model_sqr.eval()
    X_test_t = local_vars.get("X_test_t")
    if X_test_t is None:
        return False, "⚠️ `X_test_t` not found. Run the data split step first."
    with torch.no_grad():
        preds = model_sqr(X_test_t)  # (N, Q)
    avg_preds = preds.mean(dim=0)  # (Q,)
    diffs = avg_preds[1:] - avg_preds[:-1]
    if (diffs < 0).any():
        return (
            True,
            "✅ Quantile regression complete! ⚠️ Note: average quantile predictions are not "
            "in ascending order (quantile crossing). This is a known limitation of "
            "unconstrained SQR — monotonicity can be enforced architecturally.",
        )

    return True, "✅ Quantile regression complete! All three models trained."
