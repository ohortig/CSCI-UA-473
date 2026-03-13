import numpy as np


def check_step_1_softmax(local_vars):
    """
    Step 1: Softmax implementation.
    Expected: 'softmax' function, 'probs' array that sums to ~1.
    """
    if "softmax" not in local_vars:
        return (
            False,
            "Variable `softmax` not found. Define a function called `softmax`.",
        )

    if "probs" not in local_vars:
        return (
            False,
            "Variable `probs` not found. Compute softmax probabilities and store in `probs`.",
        )

    softmax_fn = local_vars["softmax"]
    probs = local_vars["probs"]

    if not callable(softmax_fn):
        return False, "`softmax` should be a callable function."

    if not isinstance(probs, np.ndarray):
        return False, "`probs` should be a numpy array."

    if probs.ndim != 1:
        return False, f"`probs` should be 1D, got {probs.ndim}D."

    if len(probs) != 3:
        return (
            False,
            f"`probs` should have 3 elements (Sell/Hold/Buy), got {len(probs)}.",
        )

    if not np.allclose(probs.sum(), 1.0, atol=1e-5):
        return False, f"Softmax probabilities should sum to 1, got {probs.sum():.6f}."

    if not np.all(probs > 0):
        return False, "All softmax probabilities should be positive."

    # Check on a known input: softmax([0, 1, 2])
    test_input = np.array([0.0, 1.0, 2.0])
    test_output = softmax_fn(test_input)
    expected = np.exp(test_input) / np.exp(test_input).sum()

    if not np.allclose(test_output, expected, atol=1e-5):
        return (
            False,
            f"softmax([0, 1, 2]) should be ~{expected}, got {test_output}. "
            "Check your implementation.",
        )

    return (
        True,
        f"Softmax implemented! probs = [{', '.join(f'{p:.4f}' for p in probs)}], sum = {probs.sum():.4f}",
    )


def check_step_2_cross_entropy(local_vars):
    """
    Step 2: Cross-entropy loss.
    Expected: 'cross_entropy_loss' function, 'loss_ours' and 'loss_uniform' floats.
    """
    if "cross_entropy_loss" not in local_vars:
        return (
            False,
            "Variable `cross_entropy_loss` not found. Define the loss function.",
        )

    if "loss_ours" not in local_vars:
        return False, "Variable `loss_ours` not found. Compute the loss for our model."

    if "loss_uniform" not in local_vars:
        return (
            False,
            "Variable `loss_uniform` not found. Compute the loss for the uniform model.",
        )

    ce_fn = local_vars["cross_entropy_loss"]
    loss_ours = local_vars["loss_ours"]
    loss_uniform = local_vars["loss_uniform"]

    if not callable(ce_fn):
        return False, "`cross_entropy_loss` should be a callable function."

    for name, val in [("loss_ours", loss_ours), ("loss_uniform", loss_uniform)]:
        if not isinstance(val, (int, float, np.floating)):
            return False, f"`{name}` should be a number, got {type(val).__name__}."
        if val < 0:
            return False, f"`{name}` should be non-negative (loss >= 0), got {val:.4f}."

    # The model with logits (0,1,2) and y*=Buy should have lower loss than uniform
    if loss_ours >= loss_uniform:
        return (
            False,
            f"Our model's loss ({loss_ours:.4f}) should be LOWER than the uniform model's "
            f"loss ({loss_uniform:.4f}). Check your loss computation.",
        )

    # Verify against known values
    # uniform loss for K=3: -log(1/3) = log(3) ~ 1.0986
    if not np.isclose(loss_uniform, np.log(3), atol=0.05):
        return (
            False,
            f"Uniform model loss should be ~{np.log(3):.4f} (= log 3), got {loss_uniform:.4f}.",
        )

    return (
        True,
        f"Cross-entropy loss implemented! "
        f"Our model: {loss_ours:.4f}, Uniform: {loss_uniform:.4f}. "
        f"Our model is {loss_uniform - loss_ours:.4f} nats better.",
    )


def check_step_3_sigmoid_detection(local_vars):
    """
    Step 3: Sigmoid and multi-label detection.
    Expected: 'sigmoid' function, 'signal_probs' array with independent probabilities.
    """
    if "sigmoid" not in local_vars:
        return (
            False,
            "Variable `sigmoid` not found. Define a function called `sigmoid`.",
        )

    if "signal_probs" not in local_vars:
        return (
            False,
            "Variable `signal_probs` not found. Compute sigmoid probabilities for each signal.",
        )

    sigmoid_fn = local_vars["sigmoid"]
    signal_probs = local_vars["signal_probs"]

    if not callable(sigmoid_fn):
        return False, "`sigmoid` should be a callable function."

    if not isinstance(signal_probs, np.ndarray):
        return False, "`signal_probs` should be a numpy array."

    if signal_probs.ndim != 1:
        return False, f"`signal_probs` should be 1D, got {signal_probs.ndim}D."

    if not np.all((signal_probs >= 0) & (signal_probs <= 1)):
        return False, "All sigmoid outputs should be in [0, 1]."

    # Verify sigmoid on known input
    test_val = sigmoid_fn(np.array([0.0]))
    if not np.isclose(test_val, 0.5, atol=1e-5):
        return False, f"sigmoid(0) should be 0.5, got {float(test_val):.6f}."

    test_val2 = sigmoid_fn(np.array([100.0]))
    if not np.isclose(test_val2, 1.0, atol=1e-3):
        return False, f"sigmoid(100) should be ~1.0, got {float(test_val2):.6f}."

    # Note: signal_probs do NOT need to sum to 1 (that's the whole point!)
    return (
        True,
        f"Sigmoid detection implemented! Signal probabilities: "
        f"[{', '.join(f'{p:.4f}' for p in signal_probs)}]. "
        f"Sum = {signal_probs.sum():.4f} (doesn't need to be 1 -- signals are independent!).",
    )


def check_step_4_bce_loss(local_vars):
    """
    Step 4: Binary cross-entropy with negative weighting.
    Expected: 'bce_loss' function, 'loss_standard' and 'loss_weighted' floats.
    """
    if "bce_loss" not in local_vars:
        return False, "Variable `bce_loss` not found. Define the BCE loss function."

    if "loss_standard" not in local_vars:
        return (
            False,
            "Variable `loss_standard` not found. Compute the standard BCE loss.",
        )

    if "loss_weighted" not in local_vars:
        return (
            False,
            "Variable `loss_weighted` not found. Compute the weighted BCE loss.",
        )

    bce_fn = local_vars["bce_loss"]
    loss_standard = local_vars["loss_standard"]
    loss_weighted = local_vars["loss_weighted"]

    if not callable(bce_fn):
        return False, "`bce_loss` should be a callable function."

    for name, val in [
        ("loss_standard", loss_standard),
        ("loss_weighted", loss_weighted),
    ]:
        if not isinstance(val, (int, float, np.floating)):
            return False, f"`{name}` should be a number, got {type(val).__name__}."
        if val < 0:
            return False, f"`{name}` should be non-negative, got {val:.4f}."

    # Weighted loss (with neg_weight < 1) should be smaller than standard
    if loss_weighted >= loss_standard:
        return (
            False,
            f"Weighted loss ({loss_weighted:.4f}) should be smaller than standard loss "
            f"({loss_standard:.4f}) when neg_weight < 1 down-weights negative terms.",
        )

    return (
        True,
        f"BCE loss implemented! Standard: {loss_standard:.4f}, "
        f"Weighted (neg_weight=0.1): {loss_weighted:.4f}. "
        f"Down-weighting negatives reduced the loss by {loss_standard - loss_weighted:.4f}.",
    )
