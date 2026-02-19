"""
Checker functions for Lab 4: Generalization & Data Splits

Each check_step_* function receives the exec namespace (a dict) produced
by running the student's code and returns ``(passed: bool, message: str)``.
"""

import numpy as np

# -----------------------------------------------------------------------
# Step 1 – Load data
# -----------------------------------------------------------------------


def check_step_1_load(local_vars):
    """Verify that the student loaded their Airbnb data into ``df``."""
    if "df" not in local_vars:
        return False, "Variable `df` not found. Did you load the data?"

    df = local_vars["df"]
    if not hasattr(df, "shape"):
        return False, "`df` should be a pandas DataFrame."

    if len(df) == 0:
        return False, "`df` is empty. Check the file path."

    required = {"price", "embedding"}
    missing = required - set(df.columns)
    if missing:
        return (
            False,
            f"Missing columns in `df`: {missing}. These are needed for Step 2.",
        )

    # Warn (but don't fail) about columns needed in later steps
    warnings = []
    if "host_id" not in df.columns:
        warnings.append("`host_id` (needed for Step 4: host-based split)")
    if "neighbourhood_cleansed" not in df.columns:
        warnings.append(
            "`neighbourhood_cleansed` (needed for Step 5: neighbourhood split)"
        )
    if "neighbourhood_group_cleansed" not in df.columns:
        warnings.append(
            "`neighbourhood_group_cleansed` (needed for Step 6: borough split)"
        )

    msg = f"Data loaded! {len(df):,} listings with {len(df.columns)} columns."
    if warnings:
        msg += " **Warning** — missing optional columns: " + "; ".join(warnings)
    return True, msg


# -----------------------------------------------------------------------
# Step 2 – Prepare features
# -----------------------------------------------------------------------


def check_step_2_features(local_vars):
    """Verify that X (feature matrix) and y (target) are prepared."""
    if "X" not in local_vars:
        return False, "Variable `X` not found."
    if "y" not in local_vars:
        return False, "Variable `y` not found."

    X = local_vars["X"]
    y = local_vars["y"]

    if not hasattr(X, "shape") or len(X.shape) != 2:
        return False, "`X` should be a 2-D numpy array."
    if not hasattr(y, "shape") or len(y.shape) != 1:
        return False, "`y` should be a 1-D numpy array."

    if X.shape[0] != y.shape[0]:
        return (
            False,
            f"X has {X.shape[0]} rows but y has {y.shape[0]} elements. They must match.",
        )

    if X.shape[1] < 100:
        return (
            False,
            f"X has only {X.shape[1]} features — that seems too few. "
            "Did you stack the embedding column? Expected ~768 dimensions.",
        )

    return (
        True,
        f"Features ready! X shape: {X.shape}, y shape: {y.shape}",
    )


# -----------------------------------------------------------------------
# Generic split check (reused by steps 3-5)
# -----------------------------------------------------------------------


def _check_split_basics(local_vars, expected_total):
    """Shared checks for any train/val/test split."""
    for name in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
        if name not in local_vars:
            return False, f"Variable `{name}` not found."

    X_train = local_vars["X_train"]
    X_val = local_vars["X_val"]
    X_test = local_vars["X_test"]
    y_train = local_vars["y_train"]
    y_val = local_vars["y_val"]
    y_test = local_vars["y_test"]

    total = len(X_train) + len(X_val) + len(X_test)
    if total != expected_total:
        return (
            False,
            f"Split sizes add up to {total}, but total data has {expected_total} "
            f"samples. Every sample should appear in exactly one split.",
        )

    if X_train.shape[0] != y_train.shape[0]:
        return False, "X_train and y_train have different lengths."
    if X_val.shape[0] != y_val.shape[0]:
        return False, "X_val and y_val have different lengths."
    if X_test.shape[0] != y_test.shape[0]:
        return False, "X_test and y_test have different lengths."

    # Rough ratio check (allow flexibility)
    train_frac = len(X_train) / total
    if train_frac < 0.5:
        return (
            False,
            f"Training set is only {train_frac:.0%} of the data. "
            "It should be the majority (e.g. 70%).",
        )

    return True, None


# -----------------------------------------------------------------------
# Step 3 – Random split
# -----------------------------------------------------------------------


def check_step_3_random_split(local_vars):
    """Verify random train/val/test split."""
    X = local_vars.get("X")
    if X is None:
        return False, "X not found. Run Step 2 first."

    ok, msg = _check_split_basics(local_vars, len(X))
    if not ok:
        return False, msg

    n_train = len(local_vars["X_train"])
    n_val = len(local_vars["X_val"])
    n_test = len(local_vars["X_test"])

    return (
        True,
        f"Random split done! Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}",
    )


# -----------------------------------------------------------------------
# Step 4 – Host-based split
# -----------------------------------------------------------------------


def check_step_4_host_split(local_vars):
    """Verify host-based split: no host should appear in multiple splits."""
    X = local_vars.get("X")
    if X is None:
        return False, "X not found. Run Step 2 first."

    ok, msg = _check_split_basics(local_vars, len(X))
    if not ok:
        return False, msg

    # Check host overlap via masks
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        if mask_name not in local_vars:
            return (
                False,
                f"Variable `{mask_name}` not found. "
                "Please keep the boolean masks so we can verify host separation.",
            )

    df = local_vars.get("df")
    if df is None:
        return False, "df not found. Did you remove it?"

    host_ids = df["host_id"].values
    train_hosts = set(host_ids[local_vars["train_mask"]])
    val_hosts = set(host_ids[local_vars["val_mask"]])
    test_hosts = set(host_ids[local_vars["test_mask"]])

    overlap_tv = train_hosts & val_hosts
    overlap_tt = train_hosts & test_hosts
    overlap_vt = val_hosts & test_hosts

    total_overlap = len(overlap_tv) + len(overlap_tt) + len(overlap_vt)
    if total_overlap > 0:
        return (
            False,
            f"Host leakage detected! "
            f"{len(overlap_tv)} hosts in train & val, "
            f"{len(overlap_tt)} in train & test, "
            f"{len(overlap_vt)} in val & test. "
            "All listings from one host must be in the same split.",
        )

    n_train = len(local_vars["X_train"])
    n_val = len(local_vars["X_val"])
    n_test = len(local_vars["X_test"])

    return (
        True,
        f"Host-based split done! No host leakage. "
        f"Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,} "
        f"({len(train_hosts):,} / {len(val_hosts):,} / {len(test_hosts):,} unique hosts)",
    )


# -----------------------------------------------------------------------
# Step 5 – Neighbourhood-based split
# -----------------------------------------------------------------------


def check_step_5_neighborhood_split(local_vars):
    """Verify neighborhood-based split: no neighborhood in multiple splits."""
    X = local_vars.get("X")
    if X is None:
        return False, "X not found. Run Step 2 first."

    ok, msg = _check_split_basics(local_vars, len(X))
    if not ok:
        return False, msg

    for mask_name in ("train_mask", "val_mask", "test_mask"):
        if mask_name not in local_vars:
            return (
                False,
                f"Variable `{mask_name}` not found. "
                "Please keep the boolean masks so we can verify neighborhood separation.",
            )

    df = local_vars.get("df")
    if df is None:
        return False, "df not found. Did you remove it?"

    hoods = df["neighbourhood_cleansed"].values
    train_hoods = set(hoods[local_vars["train_mask"]])
    val_hoods = set(hoods[local_vars["val_mask"]])
    test_hoods = set(hoods[local_vars["test_mask"]])

    overlap_tv = train_hoods & val_hoods
    overlap_tt = train_hoods & test_hoods
    overlap_vt = val_hoods & test_hoods

    total_overlap = len(overlap_tv) + len(overlap_tt) + len(overlap_vt)
    if total_overlap > 0:
        return (
            False,
            f"Neighborhood leakage detected! "
            f"{len(overlap_tv)} neighborhoods in train & val, "
            f"{len(overlap_tt)} in train & test, "
            f"{len(overlap_vt)} in val & test. "
            "All listings from one neighborhood must be in the same split.",
        )

    n_train = len(local_vars["X_train"])
    n_val = len(local_vars["X_val"])
    n_test = len(local_vars["X_test"])

    return (
        True,
        f"Neighborhood-based split done! No neighborhood leakage. "
        f"Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,} "
        f"({len(train_hoods)} / {len(val_hoods)} / {len(test_hoods)} unique neighborhoods)",
    )


# -----------------------------------------------------------------------
# Step 6 – Borough-based split (leave-one-borough-out)
# -----------------------------------------------------------------------


def check_step_6_borough_split(local_vars):
    """Verify borough-based split: test set is exactly one borough."""
    X = local_vars.get("X")
    if X is None:
        return False, "X not found. Run Step 2 first."

    ok, msg = _check_split_basics(local_vars, len(X))
    if not ok:
        return False, msg

    for mask_name in ("train_mask", "val_mask", "test_mask"):
        if mask_name not in local_vars:
            return (
                False,
                f"Variable `{mask_name}` not found. "
                "Please keep the boolean masks so we can verify the split.",
            )

    df = local_vars.get("df")
    if df is None:
        return False, "df not found. Did you remove it?"

    if "neighbourhood_group_cleansed" not in df.columns:
        return False, "Column `neighbourhood_group_cleansed` (borough) not found in df."

    boroughs = df["neighbourhood_group_cleansed"].values
    train_boroughs = set(boroughs[local_vars["train_mask"]])
    val_boroughs = set(boroughs[local_vars["val_mask"]])
    test_boroughs = set(boroughs[local_vars["test_mask"]])

    # Test set should contain exactly one borough
    if len(test_boroughs) != 1:
        return (
            False,
            f"Test set contains {len(test_boroughs)} boroughs ({test_boroughs}). "
            "It should contain exactly one held-out borough.",
        )

    test_borough = next(iter(test_boroughs))

    # The test borough should not appear in train or val
    if test_borough in train_boroughs:
        return (
            False,
            f"Borough leakage: {test_borough} appears in both train and test sets.",
        )
    if test_borough in val_boroughs:
        return (
            False,
            f"Borough leakage: {test_borough} appears in both val and test sets.",
        )

    n_train = len(local_vars["X_train"])
    n_val = len(local_vars["X_val"])
    n_test = len(local_vars["X_test"])

    return (
        True,
        f"Borough split done! Held-out borough: **{test_borough}** ({n_test:,} listings). "
        f"Train: {n_train:,} ({sorted(train_boroughs)}) | "
        f"Val: {n_val:,} ({sorted(val_boroughs)})",
    )
