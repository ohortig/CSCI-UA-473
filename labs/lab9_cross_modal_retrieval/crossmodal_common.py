"""
Shared Lab 9 UI and sandboxed execution (student + teacher both use this module).

Teacher wrapper passes show_solutions=True via sidebar checkbox.
"""

from __future__ import annotations

import copy
import io
import os
import sys
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from streamlit_monaco import st_monaco

from labs.lab9_cross_modal_retrieval.data_utils import (
    IMAGE_DIM,
    TEXT_DIM,
    load_paired_subset,
    make_synthetic_paired,
    project_root,
)
from labs.lab9_cross_modal_retrieval.level_checks import (
    check_step_1_projections,
    check_step_2_symmetric_infonce,
    check_step_3_train_retrieval,
    check_step_4_temperature,
)
from utils.dataset_config import get_dataset_config
from utils.embedding import get_embedder
from utils.image_embedding import get_image_embedder

_LAB9_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_LAB9_ASSETS = os.path.join(_LAB9_PACKAGE_DIR, "assets")
_DEMO_T2I_FIGURE = os.path.join(_LAB9_ASSETS, "demo_text_to_image_retrieval.png")
_AIRBNB_PAIRED_FIGURE = os.path.join(_LAB9_ASSETS, "airbnb_paired_listings.png")
_ARCHITECTURE_FIGURE = os.path.join(_LAB9_ASSETS, "dual_encoder_cross_modal.png")

# Step 3: dual MLP projection heads + training (frozen Nomic/DINOv2 inputs)
LAB9_SHARED_DIM = 128
LAB9_PROJ_HIDDEN_DIM = 256
LAB9_TRAIN_BATCH_SIZE = 64
LAB9_TRAIN_LR = 0.001

# Steps 5–6: no custom query UI until Step 3 stores `lab9_retrieval_model` (Lesson 8–style gate).
LAB9_MSG_STEP3_REQUIRED = "Custom retrieval requires Step 3:"


def _lab9_try_store_step3_model(ctx: dict) -> None:
    """After Step 3 passes, keep a CPU copy of `model` for Steps 5–6 (Lesson 8–style retrieval)."""
    m = ctx.get("model")
    if isinstance(m, nn.Module) and hasattr(m, "mlp_t") and hasattr(m, "mlp_i"):
        try:
            st.session_state["lab9_retrieval_model"] = copy.deepcopy(m).cpu().eval()
            st.session_state["lab9_retrieval_model_version"] = int(
                st.session_state.get("lab9_retrieval_model_version", 0)
            ) + int(1)
            for k in ("lab9_pg_zt", "lab9_pg_zi", "lab9_pg_meta"):
                st.session_state.pop(k, None)
        except Exception:
            st.session_state.pop("lab9_retrieval_model", None)
            st.warning(
                "Step 3 passed but the trained **model** could not be copied for the playground. "
                "Keep the variable name **`model`** and attributes **`mlp_t`** / **`mlp_i`** as in the template."
            )
    # If no `model` in ctx, leave any previous playground weights unchanged.


def _lab9_encode_query_text(model: nn.Module, t: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return F.normalize(model.mlp_t(t), p=2, dim=1)


def _lab9_encode_query_image(model: nn.Module, i: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return F.normalize(model.mlp_i(i), p=2, dim=1)


def _lab9_project_pairs_batched(
    model: nn.Module,
    text_np: np.ndarray,
    image_np: np.ndarray,
    batch_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project paired rows through DualProj.forward (batch)."""
    model.eval()
    chunks_t, chunks_i = [], []
    n = int(text_np.shape[0])
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tb = torch.tensor(text_np[start:end], dtype=torch.float32)
            ib = torch.tensor(image_np[start:end], dtype=torch.float32)
            zt, zi = model(tb, ib)
            chunks_t.append(zt)
            chunks_i.append(zi)
    return torch.cat(chunks_t, dim=0), torch.cat(chunks_i, dim=0)


def _lab9_playground_ensure_index(
    model: nn.Module,
    text_np: np.ndarray,
    image_np: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    ver = int(st.session_state.get("lab9_retrieval_model_version", 0))
    n = int(text_np.shape[0])
    syn = bool(st.session_state.get("lab9_synthetic"))
    meta = (ver, n, syn)
    if (
        st.session_state.get("lab9_pg_meta") != meta
        or "lab9_pg_zt" not in st.session_state
    ):
        with st.spinner("Indexing gallery with your Step 3 model…"):
            zt, zi = _lab9_project_pairs_batched(model, text_np, image_np)
            st.session_state["lab9_pg_zt"] = zt
            st.session_state["lab9_pg_zi"] = zi
            st.session_state["lab9_pg_meta"] = meta
    return st.session_state["lab9_pg_zt"], st.session_state["lab9_pg_zi"]


# ---------------------------------------------------------------------------
# Sandboxed execution (numpy + torch only)
# ---------------------------------------------------------------------------

_BLOCKED_IMPORTS = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pathlib",
        "socket",
        "http",
        "urllib",
        "requests",
        "ctypes",
        "importlib",
        "code",
        "codeop",
        "compile",
        "compileall",
        "streamlit",
        "st",
        "gspread",
        "google",
        "pickle",
        "shelve",
        "signal",
        "threading",
        "multiprocessing",
        "asyncio",
        "builtins",
        "__builtin__",
        "pandas",
        "pd",
        "sklearn",
        "matplotlib",
        "PIL",
        "cv2",
    }
)


def _safe_import(name, *args, **kwargs):
    base = name.split(".")[0]
    if base in _BLOCKED_IMPORTS:
        raise ImportError(
            f"Module '{name}' is not allowed. Use numpy (np) and torch only."
        )
    import builtins as _b

    return _b.__import__(name, *args, **kwargs)


def _make_safe_builtins():
    import builtins as _builtins_mod

    safe = {}
    _ALLOWED = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hasattr",
        "hash",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        "True",
        "False",
        "None",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "RuntimeError",
        "StopIteration",
        "ZeroDivisionError",
        "AttributeError",
        "Exception",
        "ArithmeticError",
    }
    for name in _ALLOWED:
        if hasattr(_builtins_mod, name):
            safe[name] = getattr(_builtins_mod, name)

    # Required for `class` statements (e.g. nn.Module subclasses in Step 3 training).
    safe["__build_class__"] = getattr(_builtins_mod, "__build_class__")
    safe["super"] = _builtins_mod.super
    safe["object"] = _builtins_mod.object

    safe["__import__"] = _safe_import
    return safe


def _run_student_code(code: str, ctx: dict, console_key: str) -> tuple[dict, bool]:
    # Exec globals normally include module metadata; class bodies / torch.nn expect __name__.
    ctx.setdefault("__name__", "__lab_student__")
    ctx.setdefault("__doc__", None)
    ctx.setdefault("__package__", None)

    ctx["__builtins__"] = _make_safe_builtins()

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = cap_out = io.StringIO()
    sys.stderr = cap_err = io.StringIO()

    success = True
    tb_text = ""
    try:
        exec(code, ctx)  # noqa: S102
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
    with st.expander("Console output", expanded=has_error):
        st.code(st.session_state[console_key], language="text")

    if not success:
        st.error("Your code raised an error — see console above.")

    return ctx, success


# ---------------------------------------------------------------------------
# Default code: stubs vs solutions
# ---------------------------------------------------------------------------

STEP_INFONCE_STUB = '''import numpy as np

# Fixed mini-batch (rows = items, cols = embedding dim)
text_z = text_z_fixed
image_z = image_z_fixed
temperature = 0.07

def symmetric_infonce(text_z, image_z, temperature):
    """Symmetric CLIP-style InfoNCE on a batch of paired embeddings.
    Steps:
    1. L2-normalize each row of text_z and image_z
    2. logits[i,j] = dot(text_i, image_j) / temperature  → shape (N, N)
    3. labels = [0, 1, ..., N-1]
    4. loss = 0.5 * (CE(logits, labels) + CE(logits.T, labels))
    Use torch.nn.functional.cross_entropy or implement CE in numpy.
    """
    # TODO
    return float("nan")

loss = symmetric_infonce(text_z, image_z, temperature)
print("loss:", float(loss))
'''

STEP_INFONCE_SOL = """import numpy as np
import torch
import torch.nn.functional as F

text_z = text_z_fixed
image_z = image_z_fixed
temperature = 0.07

def symmetric_infonce(text_z, image_z, temperature):
    t = torch.tensor(text_z, dtype=torch.float32)
    i = torch.tensor(image_z, dtype=torch.float32)
    t = F.normalize(t, p=2, dim=1)
    i = F.normalize(i, p=2, dim=1)
    logits = (t @ i.T) / temperature
    labels = torch.arange(logits.size(0), dtype=torch.long)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )

loss = symmetric_infonce(text_z, image_z, temperature)
print("loss:", float(loss))
"""

STEP_PROJECTION_STUB = """import numpy as np

# Linear maps (given): text 768 -> 32, image 384 -> 32
W_text = W_text_fixed   # shape (768, 32)
b_text = b_text_fixed   # shape (32,)
W_image = W_image_fixed  # shape (384, 32)
b_image = b_image_fixed  # shape (32,)

text_emb = text_emb_batch    # (N, 768)
image_emb = image_emb_batch  # (N, 384)


def project_text(x, W, b):
    # x: (N, 768) → linear map → L2 normalize rows → (N, 32)
    # TODO
    pass


def project_image(x, W, b):
    # x: (N, 384) → linear map → L2 normalize rows → (N, 32)
    # TODO
    pass


text_out = project_text(text_emb, W_text, b_text)
image_out = project_image(image_emb, W_image, b_image)
print("text_out shape", text_out.shape, "row norm", np.linalg.norm(text_out[0]))
print("image_out shape", image_out.shape)
"""

STEP_PROJECTION_SOL = """import numpy as np

W_text = W_text_fixed
b_text = b_text_fixed
W_image = W_image_fixed
b_image = b_image_fixed

text_emb = text_emb_batch
image_emb = image_emb_batch


def _l2_rows(a):
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    return a / n


def project_text(x, W, b):
    z = x @ W + b
    return _l2_rows(z)


def project_image(x, W, b):
    z = x @ W + b
    return _l2_rows(z)


text_out = project_text(text_emb, W_text, b_text)
image_out = project_image(image_emb, W_image, b_image)
print("text_out shape", text_out.shape, "row norm", np.linalg.norm(text_out[0]))
print("image_out shape", image_out.shape)
"""

STEP3_TRAIN_STUB = '''import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

text_dim = text_dim_ctx    # 768
image_dim = image_dim_ctx  # 384
hidden_dim = hidden_dim_ctx  # 256
shared_dim = shared_dim_ctx  # 128
train_batch = train_batch_ctx
train_lr = train_lr_ctx

Xtr_t = Xtr_t_ctx  # (n_train, 768)
Xtr_i = Xtr_i_ctx  # (n_train, 384)
Xva_t = Xva_t_ctx  # (n_val, 768)
Xva_i = Xva_i_ctx  # (n_val, 384)


class DualProj(nn.Module):
    """One hidden ReLU layer per modality, then shared_dim (L2-normalized)."""

    def __init__(self, dt, di, dh, ds):
        super().__init__()
        self.mlp_t = nn.Sequential(
            nn.Linear(dt, dh),
            nn.ReLU(),
            nn.Linear(dh, ds),
        )
        self.mlp_i = nn.Sequential(
            nn.Linear(di, dh),
            nn.ReLU(),
            nn.Linear(dh, ds),
        )

    def forward(self, t, i):
        zt = F.normalize(self.mlp_t(t), p=2, dim=1)
        zi = F.normalize(self.mlp_i(i), p=2, dim=1)
        return zt, zi


model = DualProj(text_dim, image_dim, hidden_dim, shared_dim)
opt = torch.optim.Adam(model.parameters(), lr=train_lr)

# TODO: training loop (25 epochs, batch size train_batch), symmetric InfoNCE per batch.
# Then compute:
#   final_train_loss  — last epoch mean training loss (float)
#   recall_t2i        — Recall@1 text→image on validation set
#   recall_i2t        — Recall@1 image→text on validation set

final_train_loss = 0.0
recall_t2i = 0.0
recall_i2t = 0.0

print("final_train_loss", final_train_loss)
print("recall_t2i", recall_t2i, "recall_i2t", recall_i2t)
'''

STEP3_TRAIN_SOL = """import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

text_dim = text_dim_ctx
image_dim = image_dim_ctx
hidden_dim = hidden_dim_ctx
shared_dim = shared_dim_ctx
train_batch = train_batch_ctx
train_lr = train_lr_ctx

Xtr_t = Xtr_t_ctx
Xtr_i = Xtr_i_ctx
Xva_t = Xva_t_ctx
Xva_i = Xva_i_ctx


class DualProj(nn.Module):
    def __init__(self, dt, di, dh, ds):
        super().__init__()
        self.mlp_t = nn.Sequential(
            nn.Linear(dt, dh),
            nn.ReLU(),
            nn.Linear(dh, ds),
        )
        self.mlp_i = nn.Sequential(
            nn.Linear(di, dh),
            nn.ReLU(),
            nn.Linear(dh, ds),
        )

    def forward(self, t, i):
        zt = F.normalize(self.mlp_t(t), p=2, dim=1)
        zi = F.normalize(self.mlp_i(i), p=2, dim=1)
        return zt, zi


def batch_loss(zt, zi, temperature=0.07):
    logits = (zt @ zi.T) / temperature
    labels = torch.arange(zt.size(0), device=zt.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )


model = DualProj(text_dim, image_dim, hidden_dim, shared_dim)
opt = torch.optim.Adam(model.parameters(), lr=train_lr)

ds = torch.utils.data.TensorDataset(Xtr_t, Xtr_i)
loader = torch.utils.data.DataLoader(
    ds, batch_size=train_batch, shuffle=True, num_workers=0
)

epochs = 25
last_loss = 0.0
for ep in range(epochs):
    model.train()
    tot = 0.0
    n = 0
    for bt, bi in loader:
        opt.zero_grad()
        zt, zi = model(bt, bi)
        loss = batch_loss(zt, zi)
        loss.backward()
        opt.step()
        tot += loss.item() * bt.size(0)
        n += bt.size(0)
    last_loss = tot / max(n, 1)
    if ep % 5 == 0:
        print("epoch", ep, "loss", last_loss)

model.eval()
with torch.no_grad():
    zt, zi = model(Xva_t, Xva_i)
    sim = zt @ zi.T
    pred_t2i = sim.argmax(dim=1)
    pred_i2t = sim.argmax(dim=0)
    lab = torch.arange(zt.size(0))
    recall_t2i = (pred_t2i == lab).float().mean().item()
    recall_i2t = (pred_i2t == lab).float().mean().item()

final_train_loss = last_loss
print("final_train_loss", final_train_loss)
print("recall_t2i", recall_t2i, "recall_i2t", recall_i2t)
"""

STEP4_TEMP_STUB = """import numpy as np

# Fixed normalized embeddings (N x d), unit rows
text_z = text_z_temp    # (N, d)
image_z = image_z_temp  # (N, d)

taus = [0.03, 0.07, 0.15, 0.4, 1.0]

# For each tau:
#   logits = text_z @ image_z.T / tau   → (N, N)
#   softmax rows (subtract row max for stability)
#   mean_top1 = mean of diagonal probabilities  (confidence on correct match)
# TODO: fill mean_top1_per_tau in taus order

mean_top1_per_tau = []

print("mean_top1_per_tau", mean_top1_per_tau)
"""

STEP4_TEMP_SOL = """import numpy as np

text_z = text_z_temp
image_z = image_z_temp

taus = [0.03, 0.07, 0.15, 0.4, 1.0]

mean_top1_per_tau = []
for tau in taus:
    logits = (text_z @ image_z.T) / tau
    # Subtract row max for numerical stability before exp
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = probs.shape[0]
    top1 = np.mean([probs[i, i] for i in range(n)])
    mean_top1_per_tau.append(float(top1))

print("mean_top1_per_tau", mean_top1_per_tau)
"""


# ---------------------------------------------------------------------------
# Data loading (cached in session state)
# ---------------------------------------------------------------------------


def _ensure_data_pack() -> None:
    if st.session_state.get("lab9_data_ready"):
        return

    df, tnp, inp, err = load_paired_subset("airbnb", max_items=1200, seed=42)
    synthetic = False
    if err is not None or df is None:
        tnp, inp = make_synthetic_paired(n=512, seed=42)
        df = None
        synthetic = True
        st.session_state["lab9_data_warning"] = (
            "Using synthetic paired embeddings (Airbnb parquet missing or incomplete). "
            "For the full NYC demo, run `python process_airbnb.py` and image processing."
        )
    else:
        st.session_state["lab9_data_warning"] = None

    st.session_state["lab9_df"] = df
    st.session_state["lab9_text_np"] = tnp
    st.session_state["lab9_image_np"] = inp
    st.session_state["lab9_synthetic"] = synthetic
    st.session_state["lab9_data_ready"] = True


def _lab9_bonus_train_one_tau(
    tau: float,
    Xtr_t: torch.Tensor,
    Xtr_i: torch.Tensor,
    Xva_t: torch.Tensor,
    Xva_i: torch.Tensor,
    text_dim: int,
    image_dim: int,
    shared_dim: int,
    epochs: int,
    hidden_dim: int = LAB9_PROJ_HIDDEN_DIM,
    batch_size: int = LAB9_TRAIN_BATCH_SIZE,
    learning_rate: float = LAB9_TRAIN_LR,
    seed: int = 42,
) -> dict[str, float]:
    """Instructor-side short train for Bonus: same architecture as Step 3 training, variable τ."""
    torch.manual_seed(seed)

    class DualProj(nn.Module):
        def __init__(self, dt: int, di: int, dh: int, ds: int) -> None:
            super().__init__()
            self.mlp_t = nn.Sequential(
                nn.Linear(dt, dh),
                nn.ReLU(),
                nn.Linear(dh, ds),
            )
            self.mlp_i = nn.Sequential(
                nn.Linear(di, dh),
                nn.ReLU(),
                nn.Linear(dh, ds),
            )

        def forward(
            self, t: torch.Tensor, i: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            zt = F.normalize(self.mlp_t(t), p=2, dim=1)
            zi = F.normalize(self.mlp_i(i), p=2, dim=1)
            return zt, zi

    def batch_loss(
        zt: torch.Tensor, zi: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        logits = (zt @ zi.T) / temperature
        labels = torch.arange(zt.size(0), dtype=torch.long, device=zt.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )

    model = DualProj(text_dim, image_dim, hidden_dim, shared_dim)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ds = torch.utils.data.TensorDataset(Xtr_t, Xtr_i)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0
    )

    last_loss = 0.0
    for _ep in range(epochs):
        model.train()
        tot = 0.0
        n = 0
        for bt, bi in loader:
            opt.zero_grad()
            zt, zi = model(bt, bi)
            loss = batch_loss(zt, zi, tau)
            loss.backward()
            opt.step()
            tot += loss.item() * bt.size(0)
            n += bt.size(0)
        last_loss = tot / max(n, 1)

    model.eval()
    with torch.no_grad():
        zt, zi = model(Xva_t, Xva_i)
        sim = zt @ zi.T
        lab = torch.arange(zt.size(0), device=zt.device)
        pred_t2i = sim.argmax(dim=1)
        pred_i2t = sim.argmax(dim=0)
        recall_t2i = (pred_t2i == lab).float().mean().item()
        recall_i2t = (pred_i2t == lab).float().mean().item()

        lg = sim / tau
        row_max = lg.max(dim=1, keepdim=True).values
        expv = torch.exp(lg - row_max)
        probs = expv / expv.sum(dim=1, keepdim=True)
        mean_diag_softmax = float(torch.diag(probs).mean().item())

    return {
        "tau": tau,
        "final_train_loss": float(last_loss),
        "recall_t2i": float(recall_t2i),
        "recall_i2t": float(recall_i2t),
        "mean_val_diag_softmax": mean_diag_softmax,
    }


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------


def render_crossmodal_lab_impl(*, show_solutions: bool = False) -> None:  # noqa: C901
    st.header("Lab 9: Cross-Modal Retrieval (InfoNCE + Dual Projection)")

    st.markdown(
        "End goal: a custom text→image and image→text search on NYC Airbnb listings."
    )
    if os.path.isfile(_DEMO_T2I_FIGURE):
        st.image(_DEMO_T2I_FIGURE, use_container_width=True)
        st.caption(
            "Example: describe a place in words → rank listing photos by similarity."
        )

    st.subheader("Data")
    if os.path.isfile(_AIRBNB_PAIRED_FIGURE):
        st.image(_AIRBNB_PAIRED_FIGURE, use_container_width=True)
        st.caption(
            "Each listing = **photo** + **title/description**. We learn to pair them in one vector space."
        )

    st.subheader("How we get there")
    st.markdown(
        """
1. **Embeddings + linear heads** — load **768-D** Nomic text and **384-D** DINOv2 image vectors; **linear** maps project both to the **same** $D$, then **L2-normalize** (dot product = cosine similarity).
2. **Symmetric InfoNCE** — implement the batch loss (average text→image and image→text InfoNCE) **before** the full training loop.
3. **Train + Recall@1** — optimize **dual MLP projection heads** (one hidden ReLU layer → **128-D** shared space) on **CPU**, report **Recall@1** on validation (both directions).
4. **Bonus** — retrain at several **τ** in the loss and compare metrics; optional autograded drill on **softmax sharpness** (fixed embeddings) is in Step 4.
5–6. **Custom retrieval** — text→image and image→text playground.
"""
    )
    if os.path.isfile(_ARCHITECTURE_FIGURE):
        st.image(_ARCHITECTURE_FIGURE, use_container_width=True)

    st.markdown("---")

    _ensure_data_pack()
    if st.session_state.get("lab9_data_warning"):
        st.warning(st.session_state["lab9_data_warning"])

    # -----------------------------------------------------------------------
    # Fixed arrays shared by all coding steps
    # -----------------------------------------------------------------------
    rng = np.random.RandomState(123)
    text_np = st.session_state["lab9_text_np"]
    image_np = st.session_state["lab9_image_np"]
    df = st.session_state["lab9_df"]

    # Step 2 (InfoNCE): small random batch, same dim for symmetric loss
    n_fix = 8
    d_small = 16
    text_z_fixed = rng.randn(n_fix, d_small).astype(np.float32)
    image_z_fixed = rng.randn(n_fix, d_small).astype(np.float32)

    # Step 1 (projections): batch at real Nomic / DINOv2 dimensions
    n_batch = 4
    text_emb_batch = rng.randn(n_batch, TEXT_DIM).astype(np.float32) * 0.02
    image_emb_batch = rng.randn(n_batch, IMAGE_DIM).astype(np.float32) * 0.02
    W_text_fixed = rng.randn(TEXT_DIM, 32).astype(np.float32) * 0.01
    b_text_fixed = rng.randn(32).astype(np.float32) * 0.01
    W_image_fixed = rng.randn(IMAGE_DIM, 32).astype(np.float32) * 0.01
    b_image_fixed = rng.randn(32).astype(np.float32) * 0.01

    # Step 4 optional autograded: correlated unit-norm pairs so τ softmax sweep is monotone
    n_t = 16
    z_base = rng.randn(n_t, 8).astype(np.float32)
    text_z_temp = np.concatenate(
        [z_base, rng.randn(n_t, d_small - 8).astype(np.float32) * 0.1], axis=1
    )
    image_z_temp = np.concatenate(
        [z_base, rng.randn(n_t, d_small - 8).astype(np.float32) * 0.1], axis=1
    )
    text_z_temp = text_z_temp / (
        np.linalg.norm(text_z_temp, axis=1, keepdims=True) + 1e-8
    )
    image_z_temp = image_z_temp / (
        np.linalg.norm(image_z_temp, axis=1, keepdims=True) + 1e-8
    )

    # Train/val split for Step 3 (training) and Bonus
    n_total = text_np.shape[0]
    idx_all = np.arange(n_total)
    rng.shuffle(idx_all)
    n_train = int(0.8 * n_total)
    tr_idx, va_idx = idx_all[:n_train], idx_all[n_train:]
    Xtr_t = torch.tensor(text_np[tr_idx], dtype=torch.float32)
    Xtr_i = torch.tensor(image_np[tr_idx], dtype=torch.float32)
    Xva_t = torch.tensor(text_np[va_idx], dtype=torch.float32)
    Xva_i = torch.tensor(image_np[va_idx], dtype=torch.float32)
    shared_dim_ctx = LAB9_SHARED_DIM

    # Toy gallery for synthetic-mode playground tabs
    rng_q = np.random.RandomState(7)
    d_q = 32
    gallery_image_vecs = rng_q.randn(40, d_q).astype(np.float32)
    gallery_image_vecs /= (
        np.linalg.norm(gallery_image_vecs, axis=1, keepdims=True) + 1e-8
    )
    gallery_text_vecs = rng_q.randn(40, d_q).astype(np.float32)
    gallery_text_vecs /= np.linalg.norm(gallery_text_vecs, axis=1, keepdims=True) + 1e-8

    # -----------------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Step 1: Embeddings + linear heads",
            "Step 2: Symmetric InfoNCE",
            "Step 3: Train + Recall@1",
            "Step 4: Temperature (bonus)",
            "Step 5: Text → image",
            "Step 6: Image → text",
        ]
    )

    # -----------------------------------------------------------------------
    # Step 1 — Load-style batch + project to shared dim
    # -----------------------------------------------------------------------
    with tab1:
        st.subheader("Step 1: Text & image embeddings → shared space")
        st.markdown(
            r"""
You already have **768-D** Nomic text and **384-D** DINOv2 image vectors per listing (`text_emb_batch`, `image_emb_batch`).
Implement two **linear** maps into the **same** dimension (here **32**), then **L2-normalize** rows so dot products are cosine similarity.

$$
\hat{z}_\text{text} = \text{L2-norm}(x_\text{text}\,W_\text{text} + b_\text{text})
\quad\in\mathbb{R}^{32}
\qquad
\hat{z}_\text{image} = \text{L2-norm}(x_\text{image}\,W_\text{image} + b_\text{image})
\quad\in\mathbb{R}^{32}
$$

Weights `W_text_fixed`, `W_image_fixed` and biases are **provided** — your job is the projection + normalization logic.
"""
        )
        code_proj = st_monaco(
            value=STEP_PROJECTION_SOL if show_solutions else STEP_PROJECTION_STUB,
            height="380px",
            language="python",
            theme="vs-dark",
        )
        if st.button("Run & check Step 1", key="lab9_b1"):
            ctx = {
                "np": np,
                "W_text_fixed": W_text_fixed,
                "b_text_fixed": b_text_fixed,
                "W_image_fixed": W_image_fixed,
                "b_image_fixed": b_image_fixed,
                "text_emb_batch": text_emb_batch,
                "image_emb_batch": image_emb_batch,
            }
            ctx, ok = _run_student_code(code_proj, ctx, "lab9_c1")
            if ok:
                passed, msg = check_step_1_projections(ctx)
                if passed:
                    st.success(msg)
                    st.session_state["lab9_s1"] = True
                else:
                    st.error(msg)

    # -----------------------------------------------------------------------
    # Step 2 — Symmetric InfoNCE (same-dim batch)
    # -----------------------------------------------------------------------
    with tab2:
        st.subheader("Step 2: Symmetric InfoNCE loss")
        st.markdown(
            r"""
On a **small paired batch** (`text_z_fixed`, `image_z_fixed`, already the same inner dimension), implement **symmetric CLIP-style InfoNCE**.

1. **L2-normalize** rows of `text_z` and `image_z`.
2. **Logits** $\mathbf{L}_{ij} = \hat{t}_i \cdot \hat{v}_j \;/\; \tau$ — shape $(N, N)$.
3. **Labels** = `[0, 1, …, N-1]`.
4. **Loss** = $\frac{1}{2}\bigl(\text{CE}(\mathbf{L},\text{labels}) + \text{CE}(\mathbf{L}^\top,\text{labels})\bigr)$.

Tip: `torch.nn.functional.cross_entropy` takes raw logits. You can use numpy if you implement CE yourself.
"""
        )
        code_infonce = st_monaco(
            value=STEP_INFONCE_SOL if show_solutions else STEP_INFONCE_STUB,
            height="320px",
            language="python",
            theme="vs-dark",
        )
        if st.button("Run & check Step 2", key="lab9_b2"):
            ctx = {
                "np": np,
                "torch": torch,
                "F": F,
                "text_z_fixed": text_z_fixed,
                "image_z_fixed": image_z_fixed,
            }
            ctx, ok = _run_student_code(code_infonce, ctx, "lab9_c2")
            if ok:
                passed, msg = check_step_2_symmetric_infonce(ctx)
                if passed:
                    st.success(msg)
                    st.session_state["lab9_s2"] = True
                else:
                    st.error(msg)

    # -----------------------------------------------------------------------
    # Step 3 — Train model + Recall@1
    # -----------------------------------------------------------------------
    with tab3:
        st.subheader("Step 3: Train the dual projector and measure Recall@1")
        st.markdown(
            f"""
Train **DualProj** — per modality, **Linear → ReLU → Linear** into a **128-D** shared space
(**256** hidden units), then **L2-normalize** — on **CPU** with **Adam** (learning rate **0.001**).
The dataset has **{n_total}** paired listings: **{len(tr_idx)} train / {len(va_idx)} val**.

Use **symmetric InfoNCE** ($\\tau=0.07$) per mini-batch (batch size **{LAB9_TRAIN_BATCH_SIZE}** in the provided template).

After training, compute **Recall@1** on the full validation set:
- Project all val items through the trained model.
- For each text query: rank all val images by cosine similarity → did the correct image rank #1?
- `recall_t2i` = fraction correct text→image; `recall_i2t` = fraction correct image→text.

**Why Recall@1?** It is the strictest retrieval metric: the correct item must be the very
first result. Even a well-trained model on this small dataset typically achieves 15–40%,
far above the random baseline of ~{100/max(len(va_idx),1):.1f}%.
"""
        )
        code_train = st_monaco(
            value=STEP3_TRAIN_SOL if show_solutions else STEP3_TRAIN_STUB,
            height="520px",
            language="python",
            theme="vs-dark",
        )
        if st.button("Run & check Step 3", key="lab9_b3"):
            ctx = {
                "np": np,
                "torch": torch,
                "nn": nn,
                "F": F,
                "text_dim_ctx": TEXT_DIM,
                "image_dim_ctx": IMAGE_DIM,
                "shared_dim_ctx": shared_dim_ctx,
                "hidden_dim_ctx": LAB9_PROJ_HIDDEN_DIM,
                "train_batch_ctx": LAB9_TRAIN_BATCH_SIZE,
                "train_lr_ctx": LAB9_TRAIN_LR,
                "Xtr_t_ctx": Xtr_t,
                "Xtr_i_ctx": Xtr_i,
                "Xva_t_ctx": Xva_t,
                "Xva_i_ctx": Xva_i,
            }
            ctx, ok = _run_student_code(code_train, ctx, "lab9_c3")
            if ok:
                if any(isinstance(v, torch.Tensor) and v.is_cuda for v in ctx.values()):
                    st.warning(
                        "This lab targets **CPU** training. Your session used at least one "
                        "**CUDA** tensor — prefer `cpu` tensors for reproducibility with the class setup."
                    )
                passed, msg = check_step_3_train_retrieval(ctx)
                m0 = ctx.get("model")
                had_dualproj = (
                    isinstance(m0, nn.Module)
                    and hasattr(m0, "mlp_t")
                    and hasattr(m0, "mlp_i")
                )
                # Steps 5–6: save `model` whenever training runs, even if recall is below the autograder threshold.
                _lab9_try_store_step3_model(ctx)
                if passed:
                    st.success(msg)
                    st.session_state["lab9_s3"] = True
                    # Show a mini results card
                    r1 = float(ctx.get("recall_t2i", 0))
                    r2 = float(ctx.get("recall_i2t", 0))
                    fl = float(ctx.get("final_train_loss", float("nan")))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Recall@1 text→image", f"{r1:.1%}")
                    c2.metric("Recall@1 image→text", f"{r2:.1%}")
                    c3.metric(
                        "Final train loss", f"{fl:.4f}" if np.isfinite(fl) else "n/a"
                    )
                else:
                    st.error(msg)
                    if (
                        had_dualproj
                        and st.session_state.get("lab9_retrieval_model") is not None
                    ):
                        st.info(
                            "You can still use **Steps 5–6** custom retrieval with this run's weights "
                            "while you tune training (epochs, learning rate, batch size, or loss)."
                        )

    # -----------------------------------------------------------------------
    # Step 4 — Bonus: τ in training loss; optional autograded softmax drill
    # -----------------------------------------------------------------------
    with tab4:
        st.subheader("Step 4 (bonus): Train at different temperatures")
        st.markdown(
            """
**Not autograded.** The same **DualProj** MLP architecture as Step 3 is retrained **from scratch** on **CPU** for each τ you pick
(fresh init every run). Compare **Recall@1** (both directions), **final train loss**, and **mean diagonal softmax**
on validation (how confident the model is on the correct image for each text when scoring against all val images).

You can also copy your Step 3 training code, change the loss temperature, and experiment in the sandbox.
"""
        )
        bonus_taus = st.multiselect(
            "τ values to compare",
            [0.03, 0.07, 0.15, 0.2],
            default=[0.03, 0.07, 0.15],
            key="lab9_bonus_taus",
        )
        bonus_epochs = st.slider(
            "Epochs per run (fewer = faster)",
            min_value=5,
            max_value=30,
            value=12,
            key="lab9_bonus_epochs",
        )
        if st.button("Run temperature comparison", key="lab9_bonus_run"):
            if not bonus_taus:
                st.warning("Select at least one τ.")
            else:
                rows = []
                with st.spinner("Training (CPU)…"):
                    for tau in sorted(bonus_taus):
                        row = _lab9_bonus_train_one_tau(
                            float(tau),
                            Xtr_t,
                            Xtr_i,
                            Xva_t,
                            Xva_i,
                            TEXT_DIM,
                            IMAGE_DIM,
                            shared_dim_ctx,
                            epochs=int(bonus_epochs),
                        )
                        rows.append(row)
                bonus_df = pd.DataFrame(rows).rename(
                    columns={
                        "tau": "τ",
                        "final_train_loss": "Final train loss",
                        "recall_t2i": "Recall@1 t→i",
                        "recall_i2t": "Recall@1 i→t",
                        "mean_val_diag_softmax": "Mean val P(correct)",
                    }
                )
                # Buttons are only True for one rerun; later reruns would skip the block and
                # drop the table. Persist so the comparison stays visible.
                st.session_state["lab9_bonus_comparison_df"] = bonus_df
                st.session_state["lab9_bonus_comparison_meta"] = (
                    tuple(sorted(float(t) for t in bonus_taus)),
                    int(bonus_epochs),
                )

        bonus_df_cached = st.session_state.get("lab9_bonus_comparison_df")
        if bonus_df_cached is not None:
            meta = st.session_state.get("lab9_bonus_comparison_meta")
            cur_key = (
                tuple(sorted(float(t) for t in bonus_taus)),
                int(bonus_epochs),
            )
            if meta is not None and meta != cur_key:
                st.caption(
                    "τ selection or epochs changed since this table was computed — "
                    "click **Run temperature comparison** again to refresh."
                )
            st.dataframe(
                bonus_df_cached,
                hide_index=True,
                use_container_width=True,
            )

        with st.expander(
            "Optional — autograded: softmax sharpness vs τ (fixed embeddings)",
            expanded=False,
        ):
            st.markdown(
                r"""
For **fixed** unit-norm pairs `text_z_temp` and `image_z_temp`, sweep
`taus = [0.03, 0.07, 0.15, 0.4, 1.0]`. For each $\tau$:

1. `logits = text_z @ image_z.T / tau`
2. Row-softmax (subtract row max for stability).
3. `mean_top1 = mean(diag(softmax))` — confidence on the **correct** match.

Store the five floats in `mean_top1_per_tau` (same order as `taus`). Values should **decrease** as $\tau$ increases.
This isolates how τ scales logits **without** retraining.
"""
            )
            code_temp = st_monaco(
                value=STEP4_TEMP_SOL if show_solutions else STEP4_TEMP_STUB,
                height="320px",
                language="python",
                theme="vs-dark",
            )
            if st.button("Run & check (optional)", key="lab9_b4"):
                ctx = {
                    "np": np,
                    "text_z_temp": text_z_temp,
                    "image_z_temp": image_z_temp,
                }
                ctx, ok = _run_student_code(code_temp, ctx, "lab9_c4")
                if ok:
                    passed, msg = check_step_4_temperature(ctx)
                    if passed:
                        st.success(msg)
                        st.session_state["lab9_s4"] = True
                        taus = [0.03, 0.07, 0.15, 0.4, 1.0]
                        vals = ctx["mean_top1_per_tau"]
                        fig = go.Figure(
                            data=[
                                go.Bar(
                                    x=[str(t) for t in taus],
                                    y=vals,
                                    text=[f"{v:.3f}" for v in vals],
                                    textposition="auto",
                                    marker_color=[
                                        "#1565C0",
                                        "#1976D2",
                                        "#42A5F5",
                                        "#90CAF9",
                                        "#BBDEFB",
                                    ],
                                )
                            ]
                        )
                        fig.update_layout(
                            title="Mean diagonal softmax probability vs temperature τ",
                            xaxis_title="tau (τ)",
                            yaxis_title="Mean P(correct match)",
                            yaxis=dict(range=[0, 1]),
                            height=400,
                        )
                        st.session_state["lab9_temp_softmax_fig"] = fig
                    else:
                        st.error(msg)
                        st.session_state.pop("lab9_temp_softmax_fig", None)
                else:
                    st.session_state.pop("lab9_temp_softmax_fig", None)

            fig_cached = st.session_state.get("lab9_temp_softmax_fig")
            if fig_cached is not None:
                st.plotly_chart(fig_cached, use_container_width=True)
                st.caption(
                    "Lower τ sharpens the distribution on the diagonal; larger τ softens it. "
                    "CLIP often uses τ ≈ 0.07 in the loss."
                )

    # -----------------------------------------------------------------------
    # Step 5 — Text → image (playground)
    # -----------------------------------------------------------------------
    with tab5:
        st.subheader("Step 5: Text → image (playground)")
        st.markdown(
            "**No deliverable** — describe what you want in plain English and see which listing photos "
            "score highest."
        )
        st.subheader(LAB9_MSG_STEP3_REQUIRED)

        if df is not None and not st.session_state["lab9_synthetic"]:
            cfg = get_dataset_config("airbnb")
            img_col = cfg["image_col"]
            tcol = cfg["text_embedding_col"]
            icol = cfg["image_embedding_col"]
            sample = df.dropna(subset=[img_col]).head(200)
            if len(sample) >= 10:
                retrieval_model = st.session_state.get("lab9_retrieval_model")
                if retrieval_model is None:
                    st.info(LAB9_MSG_STEP3_REQUIRED, icon="🔒")
                else:
                    root = project_root()
                    k_show = st.slider(
                        "How many images to show",
                        min_value=3,
                        max_value=12,
                        value=6,
                        key="lab9_t2i_k",
                    )
                    ncols = min(k_show, 6)
                    st.caption(
                        "Nomic (`task_type='search_query'`) → **your Step 3 `mlp_t`**; "
                        "gallery images → **your `mlp_i`** (first 200 listings with local photos)."
                    )
                    user_text = st.text_area(
                        "What are you looking for?",
                        placeholder="e.g. sunny room near Central Park with a desk",
                        height=96,
                        key="lab9_t2i_custom_text",
                    )
                    if st.button("Retrieve images", key="lab9_t2i_custom_btn"):
                        if not (user_text or "").strip():
                            st.warning("Enter a description first.")
                        else:
                            try:
                                raw_emb = get_embedder().embed(
                                    [(user_text or "").strip()],
                                    task_type="search_query",
                                )
                                q_vec = np.asarray(raw_emb, dtype=np.float32).reshape(
                                    -1
                                )
                                if q_vec.size != TEXT_DIM:
                                    st.error(
                                        f"Expected {TEXT_DIM}-D text embedding; got {q_vec.size}-D."
                                    )
                                else:
                                    text_raw = np.stack(sample[tcol].values).astype(
                                        np.float32
                                    )
                                    img_raw = np.stack(sample[icol].values).astype(
                                        np.float32
                                    )
                                    _, z_image = _lab9_playground_ensure_index(
                                        retrieval_model, text_raw, img_raw
                                    )
                                    q_tensor = torch.tensor(
                                        q_vec.reshape(1, -1), dtype=torch.float32
                                    )
                                    q_proj = _lab9_encode_query_text(
                                        retrieval_model, q_tensor
                                    )
                                    sims = (z_image @ q_proj.T).squeeze().cpu().numpy()
                                    order = np.argsort(-sims)[:k_show]
                                    st.subheader("Retrieved images")
                                    cols = st.columns(ncols)
                                    for j, ridx in enumerate(order):
                                        path = sample.iloc[ridx][img_col]
                                        fp = (
                                            path
                                            if os.path.isabs(path)
                                            else os.path.join(root, path)
                                        )
                                        with cols[j % ncols]:
                                            st.caption(
                                                f"#{j + 1} (score {sims[ridx]:.3f})"
                                            )
                                            if isinstance(path, str) and os.path.isfile(
                                                fp
                                            ):
                                                st.image(fp, use_container_width=True)
                                            else:
                                                st.caption(
                                                    f"Row {ridx} (no local image file)"
                                                )
                            except Exception as e:  # noqa: BLE001
                                st.error(
                                    "Could not run the text embedder (network / Hugging Face / "
                                    f"dependencies). Details: {e}"
                                )
            else:
                st.caption(f"Need ≥10 listings with image paths; have {len(sample)}.")
        else:
            st.info(
                "**Live text search (Nomic)** needs real NYC Airbnb paired data. "
                "Run `python process_airbnb.py` and the image embedding pipeline, then reload this app."
            )
            with st.expander("Toy mode (synthetic data only)", expanded=False):
                st.caption(
                    "Random 32-D unit vectors — pick a text row, rank image rows by dot product."
                )
                n_toy = int(gallery_text_vecs.shape[0])
                toy_opts = list(range(n_toy))
                q_i = st.selectbox(
                    "Text row index",
                    toy_opts,
                    index=min(12, n_toy - 1),
                    format_func=lambda i: f"Row {i}",
                    key="lab9_t2i_toy_q",
                )
                k_max = max(3, n_toy - 1)
                k_show = st.slider(
                    "How many image rows",
                    3,
                    min(12, k_max),
                    min(5, k_max),
                    key="lab9_t2i_toy_k",
                )
                q = gallery_text_vecs[q_i]
                sims = gallery_image_vecs @ q.astype(np.float64)
                sims[q_i] = -np.inf
                order = np.argsort(-sims)[:k_show]
                for rank, ridx in enumerate(order, start=1):
                    st.write(
                        f"{rank}. Image row **{ridx}** — score **{float(sims[ridx]):.4f}**"
                    )

    # -----------------------------------------------------------------------
    # Step 6 — Image → text (playground)
    # -----------------------------------------------------------------------
    with tab6:
        st.subheader("Step 6: Image → text (playground)")
        st.markdown(
            "**No deliverable** — upload a photo and see which listing descriptions match best."
        )
        st.subheader(LAB9_MSG_STEP3_REQUIRED)

        if df is not None and not st.session_state["lab9_synthetic"]:
            cfg = get_dataset_config("airbnb")
            img_col = cfg["image_col"]
            title_col = cfg["title_col"]
            text_col = cfg.get("text_col", title_col)
            tcol = cfg["text_embedding_col"]
            icol = cfg["image_embedding_col"]
            sample = df.dropna(subset=[img_col]).head(200)
            if len(sample) >= 10:
                retrieval_model = st.session_state.get("lab9_retrieval_model")
                if retrieval_model is None:
                    st.info(LAB9_MSG_STEP3_REQUIRED, icon="🔒")
                else:
                    k_show = st.slider(
                        "How many text results to show",
                        min_value=3,
                        max_value=12,
                        value=8,
                        key="lab9_i2t_k",
                    )
                    st.caption(
                        "DINOv2 on your upload → **your Step 3 `mlp_i`**; "
                        "corpus texts → **your `mlp_t`** (first 200 listings with local photos)."
                    )
                    uploaded = st.file_uploader(
                        "Upload an image (jpg / png / jpeg)",
                        type=["jpg", "png", "jpeg"],
                        key="lab9_i2t_upload",
                    )
                    if uploaded is not None:
                        up_bytes = uploaded.getvalue()
                        preview = Image.open(io.BytesIO(up_bytes)).convert("RGB")
                        st.image(preview, width=240, caption="Query image")

                    if st.button("Retrieve texts", key="lab9_i2t_custom_btn"):
                        if uploaded is None:
                            st.warning("Upload an image first.")
                        else:
                            try:
                                up_bytes = uploaded.getvalue()
                                pil_im = Image.open(io.BytesIO(up_bytes)).convert("RGB")
                                raw_emb = get_image_embedder().embed(pil_im)
                                q_vec = np.asarray(raw_emb, dtype=np.float32).reshape(
                                    -1
                                )
                                if q_vec.size != IMAGE_DIM:
                                    st.error(
                                        f"Expected {IMAGE_DIM}-D image embedding; "
                                        f"got {q_vec.size}-D."
                                    )
                                else:
                                    text_raw = np.stack(sample[tcol].values).astype(
                                        np.float32
                                    )
                                    img_raw = np.stack(sample[icol].values).astype(
                                        np.float32
                                    )
                                    z_text, _ = _lab9_playground_ensure_index(
                                        retrieval_model, text_raw, img_raw
                                    )
                                    q_tensor = torch.tensor(
                                        q_vec.reshape(1, -1), dtype=torch.float32
                                    )
                                    q_proj = _lab9_encode_query_image(
                                        retrieval_model, q_tensor
                                    )
                                    sims = (z_text @ q_proj.T).squeeze().cpu().numpy()
                                    order = np.argsort(-sims)[:k_show]
                                    st.subheader("Retrieved texts")
                                    for rank, ridx in enumerate(order, start=1):
                                        r = sample.iloc[ridx]
                                        title = str(r[title_col])[:120]
                                        desc = str(r.get(text_col, ""))[:160].replace(
                                            "\n", " "
                                        )
                                        st.markdown(
                                            f"**{rank}. {title}** (score {sims[ridx]:.3f})"
                                        )
                                        if desc and desc != title:
                                            st.caption(f"{desc}…")
                                        st.divider()
                            except Exception as e:  # noqa: BLE001
                                st.error(
                                    "Could not run the image embedder (memory, transformers, "
                                    f"etc.). Details: {e}"
                                )
            else:
                st.caption(f"Need ≥10 listings with image paths; have {len(sample)}.")
        else:
            st.info(
                "**Upload search (DINOv2)** needs real NYC Airbnb paired data. "
                "Run `python process_airbnb.py` and the image embedding pipeline, then reload."
            )
            with st.expander("Toy mode (synthetic data only)", expanded=False):
                st.caption(
                    "Pick an image row; rank text rows by dot product in 32-D toy space."
                )
                n_toy = int(gallery_image_vecs.shape[0])
                toy_opts = list(range(n_toy))
                q_i = st.selectbox(
                    "Image row index",
                    toy_opts,
                    index=min(7, n_toy - 1),
                    format_func=lambda i: f"Row {i}",
                    key="lab9_i2t_toy_q",
                )
                k_max = max(3, n_toy - 1)
                k_show = st.slider(
                    "How many text rows",
                    3,
                    min(12, k_max),
                    min(5, k_max),
                    key="lab9_i2t_toy_k",
                )
                q = gallery_image_vecs[q_i]
                sims = gallery_text_vecs @ q.astype(np.float64)
                sims[q_i] = -np.inf
                order = np.argsort(-sims)[:k_show]
                for rank, ridx in enumerate(order, start=1):
                    st.write(
                        f"{rank}. Text row **{ridx}** — score **{float(sims[ridx]):.4f}**"
                    )

    # -----------------------------------------------------------------------
    # Math foundation expander (path from repo root — not cwd — so it loads in Streamlit)
    # -----------------------------------------------------------------------
    _math_lesson = os.path.join(
        project_root(), "pages", "math", "lesson_8_cross_modal.md"
    )
    with st.expander("Mathematical foundation: shared space & InfoNCE", expanded=False):
        if os.path.isfile(_math_lesson):
            with open(_math_lesson, encoding="utf-8") as f:
                st.markdown(f.read(), unsafe_allow_html=True)
        else:
            st.info(f"Math notes not found (expected `{_math_lesson}`).")
