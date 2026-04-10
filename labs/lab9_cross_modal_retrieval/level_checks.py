"""Autograder checks for Lab 9 cross-modal retrieval steps."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _ref_symmetric_infonce(
    text_z: np.ndarray, image_z: np.ndarray, temperature: float
) -> float:
    t = torch.tensor(text_z, dtype=torch.float32)
    i = torch.tensor(image_z, dtype=torch.float32)
    t = F.normalize(t, p=2, dim=1)
    i = F.normalize(i, p=2, dim=1)
    logits = (t @ i.T) / temperature
    labels = torch.arange(logits.size(0), dtype=torch.long)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    return float(loss.item())


def check_step_2_symmetric_infonce(local_vars: dict) -> tuple[bool, str]:
    if "symmetric_infonce" not in local_vars:
        return (
            False,
            "Define a function `symmetric_infonce(text_z, image_z, temperature)`.",
        )
    fn = local_vars["symmetric_infonce"]
    if not callable(fn):
        return False, "`symmetric_infonce` must be a function."

    rng = np.random.RandomState(0)
    text_z = rng.randn(8, 16).astype(np.float32)
    image_z = rng.randn(8, 16).astype(np.float32)
    tau = 0.07
    try:
        got = float(fn(text_z, image_z, tau))
    except Exception as e:  # noqa: BLE001
        return False, f"Calling your function failed: {e}"

    ref = _ref_symmetric_infonce(text_z, image_z, tau)
    if not np.isfinite(got):
        return False, "Loss should be a finite number."

    if not np.isclose(got, ref, rtol=1e-4, atol=1e-4):
        return (
            False,
            f"Expected loss ~{ref:.6f} (reference), got {got:.6f}. "
            "Check L2 normalization, logits = text @ image.T / temperature, "
            "and 0.5 * (CE(logits, labels) + CE(logits.T, labels)).",
        )

    return True, f"Step 2 passed — symmetric InfoNCE matches reference (~{ref:.4f})."


def check_step_1_projections(local_vars: dict) -> tuple[bool, str]:
    for name in ("project_text", "project_image", "text_out", "image_out"):
        if name not in local_vars:
            return False, f"Missing `{name}` — implement projections and store outputs."

    pt, pi = local_vars["project_text"], local_vars["project_image"]
    if not callable(pt) or not callable(pi):
        return False, "`project_text` and `project_image` must be functions."

    text_out = local_vars["text_out"]
    image_out = local_vars["image_out"]
    if not isinstance(text_out, np.ndarray) or not isinstance(image_out, np.ndarray):
        return False, "`text_out` and `image_out` must be numpy arrays."

    if text_out.shape != image_out.shape:
        return (
            False,
            f"Shapes must match; got text_out {text_out.shape} vs image_out {image_out.shape}.",
        )

    if text_out.ndim != 2:
        return False, "Outputs should be 2D (N, shared_dim)."

    norms_t = np.linalg.norm(text_out, axis=1)
    norms_i = np.linalg.norm(image_out, axis=1)
    if not np.allclose(norms_t, 1.0, atol=1e-3) or not np.allclose(
        norms_i, 1.0, atol=1e-3
    ):
        return (
            False,
            "After projection, L2-normalize each row so norms are ~1.0 (cosine dot product).",
        )

    return True, (
        f"Step 1 passed — projected shape {text_out.shape}, unit row norms OK."
    )


def check_step_3_bidirectional(local_vars: dict) -> tuple[bool, str]:
    required = ("logits", "loss_t2i", "loss_i2t", "loss_symmetric", "loss_from_step1")
    for r in required:
        if r not in local_vars:
            return False, f"Missing variable `{r}`."

    logits = local_vars["logits"]
    if not isinstance(logits, torch.Tensor):
        return False, "`logits` should be a torch.Tensor (N, N)."

    if logits.dim() != 2 or logits.size(0) != logits.size(1):
        return False, "`logits` must be a square matrix (N, N)."

    n = logits.size(0)
    labels = torch.arange(n, dtype=torch.long)
    ref_t2i = F.cross_entropy(logits, labels)
    ref_i2t = F.cross_entropy(logits.T, labels)
    ref_sym = 0.5 * (ref_t2i + ref_i2t)

    lt2i = local_vars["loss_t2i"]
    li2t = local_vars["loss_i2t"]
    ls = local_vars["loss_symmetric"]
    l1 = local_vars["loss_from_step1"]

    if not isinstance(lt2i, torch.Tensor):
        lt2i = torch.tensor(float(lt2i))
    if not isinstance(li2t, torch.Tensor):
        li2t = torch.tensor(float(li2t))
    if not isinstance(ls, torch.Tensor):
        ls = torch.tensor(float(ls))
    if not isinstance(l1, torch.Tensor):
        l1 = torch.tensor(float(l1))

    if not torch.isclose(lt2i, ref_t2i, rtol=1e-4, atol=1e-4):
        return False, "`loss_t2i` should equal cross_entropy(logits, labels)."

    if not torch.isclose(li2t, ref_i2t, rtol=1e-4, atol=1e-4):
        return False, "`loss_i2t` should equal cross_entropy(logits.T, labels)."

    if not torch.isclose(ls, ref_sym, rtol=1e-4, atol=1e-4):
        return False, "`loss_symmetric` should be 0.5 * (loss_t2i + loss_i2t)."

    if not torch.isclose(ls, l1, rtol=1e-4, atol=1e-4):
        return (
            False,
            "`loss_from_step1` should match `loss_symmetric` (same symmetric InfoNCE).",
        )

    return (
        True,
        "Bidirectional logits check passed — text→image and image→text CE combine correctly.",
    )


def recall_at_1_text_to_image(text_z: torch.Tensor, image_z: torch.Tensor) -> float:
    """In-batch retrieval: fraction where argmax_j sim(i,j) == i."""
    sim = text_z @ image_z.T
    pred = sim.argmax(dim=1)
    labels = torch.arange(text_z.size(0), device=text_z.device)
    return (pred == labels).float().mean().item()


def recall_at_1_image_to_text(text_z: torch.Tensor, image_z: torch.Tensor) -> float:
    sim = text_z @ image_z.T
    pred = sim.argmax(dim=0)
    labels = torch.arange(text_z.size(0), device=text_z.device)
    return (pred == labels).float().mean().item()


def check_step_3_train_retrieval(local_vars: dict) -> tuple[bool, str]:
    for name in ("recall_t2i", "recall_i2t"):
        if name not in local_vars:
            return False, f"Define `{name}` (Recall@1 on validation, float 0–1)."

    r1 = float(local_vars["recall_t2i"])
    r2 = float(local_vars["recall_i2t"])
    if not (0.0 <= r1 <= 1.0 and 0.0 <= r2 <= 1.0):
        return False, "Recalls must be between 0 and 1."

    # Above chance (1/N); real paired embeddings usually exceed this after training
    threshold = 0.08
    if r1 < threshold or r2 < threshold:
        return (
            False,
            f"Recall@1 should exceed {threshold:.2f} for both directions "
            f"(got t2i={r1:.3f}, i2t={r2:.3f}). Train longer, tune lr, or check loss.",
        )

    if "final_train_loss" in local_vars:
        fl = float(local_vars["final_train_loss"])
        if not np.isfinite(fl):
            return False, "`final_train_loss` should be finite."

    return True, (f"Step 3 passed — Recall@1 text→image={r1:.3f}, image→text={r2:.3f}.")


def check_step_4_temperature(local_vars: dict) -> tuple[bool, str]:
    if "mean_top1_per_tau" not in local_vars:
        return False, "Provide `mean_top1_per_tau` — list parallel to lab `taus` order."

    vals = local_vars["mean_top1_per_tau"]
    if not isinstance(vals, (list, tuple)) or len(vals) < 3:
        return (
            False,
            "`mean_top1_per_tau` should be a list with one float per temperature.",
        )

    arr = np.array([float(v) for v in vals], dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return False, "All values must be finite."

    if not np.all((arr > 0) & (arr <= 1)):
        return False, "Mean top-1 softmax probability should lie in (0, 1]."

    # Increasing temperature -> softer softmax -> mean top-1 should not increase
    if not np.all(arr[:-1] >= arr[1:] - 1e-3):
        return (
            False,
            "As temperature increases, mean top-1 probability should generally decrease "
            f"(monotone non-increasing). Got: {arr}.",
        )

    return (
        True,
        "Step 4 (optional) passed — temperature ordering matches expected softer distributions.",
    )
