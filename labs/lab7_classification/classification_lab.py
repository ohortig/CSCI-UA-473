import io
import os
import sys
import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import f1_score as _sklearn_f1
from sklearn.metrics import precision_score, recall_score
from streamlit_monaco import st_monaco

from labs.lab7_classification.level_checks import (
    check_step_1_softmax,
    check_step_2_cross_entropy,
    check_step_3_sigmoid_detection,
    check_step_4_bce_loss,
)

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
    }
)


def _safe_import(name, *args, **kwargs):
    """Import hook that blocks dangerous modules."""
    base = name.split(".")[0]
    if base in _BLOCKED_IMPORTS:
        raise ImportError(
            f"Module '{name}' is not allowed in student code. "
            f"You have access to numpy (np) -- that's all you need!"
        )
    import builtins as _b

    return _b.__import__(name, *args, **kwargs)


def _make_safe_builtins():
    """Create a restricted builtins dict for student code execution."""
    import builtins as _builtins_mod

    safe = {}
    # Allow safe built-in functions
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

    # Provide a safe import that blocks dangerous modules
    safe["__import__"] = _safe_import
    return safe


def _run_student_code(code, ctx, console_key):
    """Execute student code in a sandboxed environment."""
    # Inject safe builtins (blocks os, sys, streamlit, etc.)
    ctx["__builtins__"] = _make_safe_builtins()

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
    with st.expander("Console Output", expanded=has_error):
        st.code(st.session_state[console_key], language="text")

    if not success:
        st.error(
            "Your code raised an error -- check the console above for the full traceback."
        )

    return ctx, success


# ── Visualization helpers ────────────────────────────────────────


def _plot_softmax_bars(probs, labels, title="Softmax Probabilities"):
    """Bar chart showing softmax probabilities for classification actions."""
    colors = ["#d62728", "#7f7f7f", "#2ca02c"]  # red, gray, green
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=probs,
                marker_color=colors[: len(labels)],
                text=[f"{p:.4f}" for p in probs],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title=title,
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=350,
        width=500,
    )
    return fig


def _plot_sigmoid_bars(probs, labels, title="Signal Detection (Sigmoid)"):
    """Bar chart showing independent sigmoid probabilities."""
    colors = ["#17becf", "#9467bd", "#bcbd22"]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=probs,
                marker_color=colors[: len(labels)],
                text=[f"{p:.4f}" for p in probs],
                textposition="auto",
            )
        ]
    )
    fig.add_hline(
        y=0.5, line_dash="dash", line_color="gray", annotation_text="threshold = 0.5"
    )
    fig.update_layout(
        title=title,
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=350,
        width=500,
    )
    return fig


def _plot_loss_comparison(loss_ours, loss_uniform):
    """Side-by-side bar chart comparing losses."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Our Model (u=[0,1,2])", "Uniform (u=[0,0,0])"],
                y=[loss_ours, loss_uniform],
                marker_color=["#2ca02c", "#d62728"],
                text=[f"{loss_ours:.4f}", f"{loss_uniform:.4f}"],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title="Cross-Entropy Loss Comparison (lower is better)",
        yaxis_title="Loss (nats)",
        height=350,
        width=500,
    )
    return fig


def _plot_temperature_scaling(logits, temperatures):
    """Show how temperature affects softmax distribution."""
    action_labels = ["Sell", "Hold", "Buy"]
    fig = go.Figure()

    colors = px.colors.qualitative.Set2
    for i, T in enumerate(temperatures):
        scaled = logits / T
        exp_vals = np.exp(scaled - scaled.max())
        probs = exp_vals / exp_vals.sum()
        fig.add_trace(
            go.Bar(
                name=f"T={T}",
                x=action_labels,
                y=probs,
                marker_color=colors[i % len(colors)],
                text=[f"{p:.3f}" for p in probs],
                textposition="auto",
            )
        )

    fig.update_layout(
        title="Temperature Scaling: Effect on Softmax Distribution",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        barmode="group",
        height=400,
        width=700,
    )
    return fig


def _plot_softmax_vs_sigmoid():
    """Two side-by-side figures: classification (softmax) vs detection (sigmoid)."""
    # Classification (softmax): one action, probs sum to 1
    class_probs = [0.10, 0.15, 0.75]
    class_labels = ["Sell", "Hold", "Buy"]
    fig_class = go.Figure(
        data=[
            go.Bar(
                x=class_labels,
                y=class_probs,
                marker_color=["#d62728", "#7f7f7f", "#2ca02c"],
                text=[f"{p:.2f}" for p in class_probs],
                textposition="auto",
            )
        ]
    )
    fig_class.update_layout(
        title="Classification",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=380,
        width=340,
        showlegend=False,
    )

    # Detection (sigmoid): independent signals, each in [0, 1]
    detect_probs = [0.85, 0.20, 0.70]
    detect_labels = ["Price Rising", "High Volume", "In the News"]
    fig_detect = go.Figure(
        data=[
            go.Bar(
                x=detect_labels,
                y=detect_probs,
                marker_color=["#17becf", "#9467bd", "#bcbd22"],
                text=[f"{p:.2f}" for p in detect_probs],
                textposition="auto",
            )
        ]
    )
    fig_detect.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.7)
    fig_detect.update_layout(
        title="Detection",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=380,
        width=340,
        showlegend=False,
    )

    return fig_class, fig_detect


# ── Main renderer ────────────────────────────────────────────────


def render_classification_lab():
    st.header("Lab 7: The Wolf of Wall Street")

    st.warning(
        "**Disclaimer:** This lab is for **educational purposes only** and does **not** "
        "constitute financial advice. The trading simulation uses real historical factor data "
        "(1960-1992) from the Kenneth French Data Library. Past performance does not predict "
        "future results -- these alpha signals have largely decayed since publication. "
        "Do not use anything from this lab to make actual investment decisions.",
    )

    # -- Block 1 + 2: Charts with paired questions ----------------------
    st.markdown(
        """
    ### The Big Idea

    You're a quant trader in the 1980s. Your model outputs raw scores (**logits**)
    and you need to turn them into **decisions**. Today you'll build two tools:
    """
    )

    fig_class, fig_detect = _plot_softmax_vs_sigmoid()

    # --- Classification: chart left, questions right ---
    st.markdown("---")
    cls_chart, cls_qs = st.columns([1, 1])
    with cls_chart:
        st.plotly_chart(fig_class, use_container_width=True)
    with cls_qs:
        st.markdown("**Classification** -- pick actions per day")
        st.radio(
            "How many classes can you pick?",
            ["\u2014", "Exactly one", "Any combination"],
            key="ov_cls1",
            horizontal=True,
        )
        st.radio(
            "What should the probabilities sum to?",
            ["\u2014", "1", "Anything"],
            key="ov_cls2",
            horizontal=True,
        )
        st.radio(
            "How do you turn logits into these probabilities?",
            ["\u2014", "Softmax", "Sigmoid"],
            key="ov_cls3",
            horizontal=True,
        )
        st.radio(
            "Which loss to train with?",
            ["\u2014", "Cross-entropy", "Binary cross-entropy"],
            key="ov_cls4",
            horizontal=True,
        )

        if st.button("Check my answers", key="ov_cls_check"):
            fb = []
            c = 0
            a1 = st.session_state.get("ov_cls1", "\u2014")
            a2 = st.session_state.get("ov_cls2", "\u2014")
            a3 = st.session_state.get("ov_cls3", "\u2014")
            a4 = st.session_state.get("ov_cls4", "\u2014")
            if a1 == "Exactly one":
                c += 1
            elif a1 != "\u2014":
                fb.append(
                    (
                        "error",
                        "How many: must pick **exactly one** action (Sell, Hold, or Buy).",
                    )
                )
            if a2 == "1":
                c += 1
            elif a2 != "\u2014":
                fb.append(
                    (
                        "error",
                        "Sum: when you pick exactly one, probabilities must **sum to 1**.",
                    )
                )
            if a3 == "Softmax":
                c += 1
            elif a3 != "\u2014":
                fb.append(
                    (
                        "error",
                        "Function: **softmax** enforces sum-to-1. Sigmoid gives independent outputs.",
                    )
                )
            if a4 == "Cross-entropy":
                c += 1
            elif a4 != "\u2014":
                fb.append(
                    (
                        "error",
                        "Loss: **cross-entropy** pairs with softmax for a single correct class.",
                    )
                )
            answered = sum(1 for v in [a1, a2, a3, a4] if v != "\u2014")
            if c == 4:
                fb.insert(
                    0,
                    (
                        "success",
                        "4/4 \u2014 Classification nailed! Softmax + cross-entropy for mutually exclusive choices.",
                    ),
                )
            elif answered > 0:
                fb.insert(0, ("info", f"{c}/{answered} correct so far."))
            st.session_state["ov_cls_feedback"] = fb

        if "ov_cls_feedback" in st.session_state:
            for kind, msg in st.session_state["ov_cls_feedback"]:
                if kind == "success":
                    st.success(msg)
                elif kind == "error":
                    st.error(msg)
                elif kind == "info":
                    st.info(msg)

    # --- Temperature scaling: enclosed within classification ---
    with st.expander("Temperature scaling \u2014 controlling confidence"):
        temp_chart, temp_qs = st.columns([1, 1])
        with temp_chart:
            _ov_logits = np.array([0.0, 1.0, 2.0])
            fig_temp = _plot_temperature_scaling(_ov_logits, [0.25, 0.5, 1.0, 2.0, 5.0])
            st.plotly_chart(fig_temp, use_container_width=True, key="ov_temp")
        with temp_qs:
            st.markdown(
                "The chart shows softmax on the same logits [0, 1, 2] at different "
                "temperatures. Look at how the bars change."
            )
            st.radio(
                "As temperature T increases, what happens to the distribution?",
                [
                    "\u2014",
                    "Gets sharper (more confident)",
                    "Gets flatter (less confident)",
                ],
                key="ov_temp1",
                horizontal=True,
            )

            if st.button("Check my answers", key="ov_temp_check"):
                fb = []
                c = 0
                t1 = st.session_state.get("ov_temp1", "\u2014")
                if t1 == "Gets flatter (less confident)":
                    c += 1
                elif t1 != "\u2014":
                    fb.append(
                        (
                            "error",
                            "Look at the chart: T=0.25 is sharp, T=5 is nearly flat. **Higher T = less confident**.",
                        )
                    )
                answered = sum(1 for v in [t1] if v != "\u2014")
                if c == 1:
                    fb.insert(
                        0,
                        (
                            "success",
                            "Correct \u2014 lower temperature = more confident, higher temperature = more spread out.",
                        ),
                    )
                elif answered > 0:
                    fb.insert(0, ("info", f"{c}/{answered} correct so far."))
                st.session_state["ov_temp_feedback"] = fb

            if "ov_temp_feedback" in st.session_state:
                for kind, msg in st.session_state["ov_temp_feedback"]:
                    if kind == "success":
                        st.success(msg)
                    elif kind == "error":
                        st.error(msg)
                    elif kind == "info":
                        st.info(msg)

    # --- Detection: chart left, questions right ---
    st.markdown("---")
    det_chart, det_qs = st.columns([1, 1])
    with det_chart:
        st.plotly_chart(fig_detect, use_container_width=True, key="ov_detect")
    with det_qs:
        st.markdown("**Detection** -- flag market signals")
        st.radio(
            "How many signals can be active at once?",
            ["\u2014", "Exactly one", "Any combination"],
            key="ov_det1",
            horizontal=True,
        )
        st.radio(
            "What should each probability be?",
            ["\u2014", "Independent, each in [0, 1]", "Must sum to 1"],
            key="ov_det2",
            horizontal=True,
        )
        st.radio(
            "How do you turn logits into these probabilities?",
            ["\u2014", "Softmax", "Sigmoid"],
            key="ov_det3",
            horizontal=True,
        )
        st.radio(
            "Which loss to train with (especially with noisy labels)?",
            ["\u2014", "Cross-entropy", "Weighted binary cross-entropy"],
            key="ov_det4",
            horizontal=True,
        )
        if st.button("Check my answers", key="ov_det_check"):
            fb = []
            c = 0
            a1 = st.session_state.get("ov_det1", "\u2014")
            a2 = st.session_state.get("ov_det2", "\u2014")
            a3 = st.session_state.get("ov_det3", "\u2014")
            a4 = st.session_state.get("ov_det4", "\u2014")
            if a1 == "Any combination":
                c += 1
            elif a1 != "\u2014":
                fb.append(
                    (
                        "error",
                        "How many: **any combination** of signals can be active at once.",
                    )
                )
            if a2 == "Independent, each in [0, 1]":
                c += 1
            elif a2 != "\u2014":
                fb.append(
                    (
                        "error",
                        "Probabilities: each signal is **independent, in [0, 1]**. They don't need to sum to 1.",
                    )
                )
            if a3 == "Sigmoid":
                c += 1
            elif a3 != "\u2014":
                fb.append(
                    (
                        "error",
                        "Function: **sigmoid** gives each signal its own independent probability.",
                    )
                )
            if a4 == "Weighted binary cross-entropy":
                c += 1
            elif a4 != "\u2014":
                fb.append(
                    (
                        "error",
                        "Loss: **weighted BCE** handles independent labels and lets you down-weight noisy negatives.",
                    )
                )
            answered = sum(1 for v in [a1, a2, a3, a4] if v != "\u2014")
            if c == 4:
                fb.insert(
                    0,
                    (
                        "success",
                        "4/4 \u2014 Detection nailed! Sigmoid + weighted BCE for independent signals.",
                    ),
                )
            elif answered > 0:
                fb.insert(0, ("info", f"{c}/{answered} correct so far."))
            st.session_state["ov_det_feedback"] = fb

        if "ov_det_feedback" in st.session_state:
            for kind, msg in st.session_state["ov_det_feedback"]:
                if kind == "success":
                    st.success(msg)
                elif kind == "error":
                    st.error(msg)
                elif kind == "info":
                    st.info(msg)

    # --- Weighted BCE: enclosed within detection ---
    with st.expander("Why down-weight negatives in BCE?"):
        st.markdown(
            "Your team labeled 5 market signals on 1000 days, but annotators are "
            "**busy** \u2014 they reliably tag signals they **see**, but often skip ones "
            "they're unsure about. So a label of y=0 could mean:"
        )
        bce_col1, bce_col2 = st.columns(2)
        with bce_col1:
            st.info("**y=1** \u2192 signal is definitely present (annotator tagged it)")
        with bce_col2:
            st.warning(
                "**y=0** \u2192 signal is absent **OR** annotator just didn't tag it"
            )

        st.radio(
            "An annotator marks 2 of 5 signals. The other 3 are y=0. Can you fully trust those zeros?",
            [
                "\u2014",
                "Yes, they're reliably absent",
                "No, some might just be untagged",
            ],
            key="ov_bce1",
            horizontal=True,
        )
        st.radio(
            "Your model predicts 'signal present' for an untagged signal (y=0). Should this be penalized as heavily as missing a tagged signal (y=1)?",
            [
                "\u2014",
                "Yes, penalize equally",
                "No, reduce the penalty for untagged negatives",
            ],
            key="ov_bce2",
            horizontal=True,
        )
        st.radio(
            "In weighted BCE, setting neg_weight=0.1 means:",
            [
                "\u2014",
                "Negative labels contribute 10% as much loss as positive labels",
                "Positive labels contribute 10% as much loss as negative labels",
            ],
            key="ov_bce3",
            horizontal=True,
        )

        if st.button("Check my answers", key="ov_bce_check"):
            fb = []
            c = 0
            b1 = st.session_state.get("ov_bce1", "\u2014")
            b2 = st.session_state.get("ov_bce2", "\u2014")
            b3 = st.session_state.get("ov_bce3", "\u2014")
            if b1 == "No, some might just be untagged":
                c += 1
            elif b1 != "\u2014":
                fb.append(
                    (
                        "error",
                        "Trust: y=0 might mean the annotator **missed it**, not that the signal is truly absent.",
                    )
                )
            if b2 == "No, reduce the penalty for untagged negatives":
                c += 1
            elif b2 != "\u2014":
                fb.append(
                    (
                        "error",
                        "Penalty: since we don't trust y=0 fully, we should **reduce the penalty** when the model disagrees with a negative label.",
                    )
                )
            if b3 == "Negative labels contribute 10% as much loss as positive labels":
                c += 1
            elif b3 != "\u2014":
                fb.append(
                    (
                        "error",
                        "neg_weight multiplies the **(1-y) log(1-p)** term. At 0.1, negatives contribute **10%** as much loss as positives.",
                    )
                )
            answered = sum(1 for v in [b1, b2, b3] if v != "\u2014")
            if c == 3:
                fb.insert(
                    0,
                    (
                        "success",
                        "3/3 \u2014 You understand why down-weighting matters for noisy detection labels!",
                    ),
                )
            elif answered > 0:
                fb.insert(0, ("info", f"{c}/{answered} correct so far."))
            st.session_state["ov_bce_feedback"] = fb

        if "ov_bce_feedback" in st.session_state:
            for kind, msg in st.session_state["ov_bce_feedback"]:
                if kind == "success":
                    st.success(msg)
                elif kind == "error":
                    st.error(msg)
                elif kind == "info":
                    st.info(msg)

    # Balloons only when current radio values are all correct AND all
    # sections have been checked (feedback exists with success).
    _cls_now_ok = (
        st.session_state.get("ov_cls1") == "Exactly one"
        and st.session_state.get("ov_cls2") == "1"
        and st.session_state.get("ov_cls3") == "Softmax"
        and st.session_state.get("ov_cls4") == "Cross-entropy"
    )
    _temp_now_ok = st.session_state.get("ov_temp1") == "Gets flatter (less confident)"
    _det_now_ok = (
        st.session_state.get("ov_det1") == "Any combination"
        and st.session_state.get("ov_det2") == "Independent, each in [0, 1]"
        and st.session_state.get("ov_det3") == "Sigmoid"
        and st.session_state.get("ov_det4") == "Weighted binary cross-entropy"
    )
    _bce_now_ok = (
        st.session_state.get("ov_bce1") == "No, some might just be untagged"
        and st.session_state.get("ov_bce2")
        == "No, reduce the penalty for untagged negatives"
        and st.session_state.get("ov_bce3")
        == "Negative labels contribute 10% as much loss as positive labels"
    )
    _cls_checked = any(
        k == "success" for k, _ in st.session_state.get("ov_cls_feedback", [])
    )
    _temp_checked = any(
        k == "success" for k, _ in st.session_state.get("ov_temp_feedback", [])
    )
    _det_checked = any(
        k == "success" for k, _ in st.session_state.get("ov_det_feedback", [])
    )
    _bce_checked = any(
        k == "success" for k, _ in st.session_state.get("ov_bce_feedback", [])
    )
    _all_ok = _cls_now_ok and _temp_now_ok and _det_now_ok and _bce_now_ok
    _all_checked = _cls_checked and _temp_checked and _det_checked and _bce_checked
    if _all_ok and _all_checked:
        if not st.session_state.get("ov_balloons_shown"):
            st.balloons()
            st.session_state["ov_balloons_shown"] = True
        st.success(
            "Both sections perfect! You've got the full picture \u2014 now build each piece from scratch in the steps below."
        )

    # -- Block 3: Visual roadmap ---------------------------------------
    st.markdown("---")
    st.markdown("### Lab Roadmap")

    _PART_A_COLOR = "#E3F2FD"
    _PART_B_COLOR = "#F3E5F5"

    def _roadmap_card(title, description, tool, bg_color):
        st.markdown(
            f'<div style="background:{bg_color};border-radius:8px;padding:12px 14px;'
            f'height:130px;display:flex;flex-direction:column;justify-content:space-between;">'
            f'<div style="font-weight:700;font-size:0.95rem;">{title}</div>'
            f'<div style="font-size:0.82rem;color:#444;">{description}</div>'
            f'<div style="font-size:0.78rem;color:#777;margin-top:4px;">{tool}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.caption("**Part A** (blue) \u2014 Classification: pick one action per day")
    a1, a2, a3 = st.columns(3)
    with a1:
        _roadmap_card(
            "Step 1: Softmax",
            "Logits \u2192 mutually exclusive probabilities",
            "softmax()",
            _PART_A_COLOR,
        )
    with a2:
        _roadmap_card(
            "Step 2: Cross-Entropy",
            "Measure how wrong the prediction is",
            "\u2212log p(y*)",
            _PART_A_COLOR,
        )
    with a3:
        _roadmap_card(
            "Step 6: Trading Competition",
            "Train a classifier on real data & compete",
            "softmax + CE + SGD",
            _PART_A_COLOR,
        )

    st.caption("**Part B** (purple) \u2014 Detection: flag independent market signals")
    b1, b2, b3 = st.columns(3)
    with b1:
        _roadmap_card(
            "Step 3: Sigmoid",
            "One probability per signal (independent)",
            "\u03c3(x) = 1/(1+e\u207b\u02e3)",
            _PART_B_COLOR,
        )
    with b2:
        _roadmap_card(
            "Step 4: Weighted BCE",
            "Handle noisy / incomplete labels",
            "BCE + neg_weight",
            _PART_B_COLOR,
        )
    with b3:
        _roadmap_card(
            "Step 5: Anomaly Detection",
            "Detect earnings shocks & factor rotations",
            "sigmoid + BCE + SGD",
            _PART_B_COLOR,
        )

    st.markdown("")

    with st.expander("Quick reference \u2014 Classification vs Detection"):
        st.markdown(
            """
| | **Classification (softmax)** | **Detection (sigmoid)** |
|--|---|---|
| **Pick how many?** | Exactly one | Any subset |
| **Output constraint** | Sum to 1 | Each in [0, 1] |
| **Loss** | Cross-entropy | Binary cross-entropy |
| **Steps** | 1, 2, 6 | 3, 4, 5 |
"""
        )

    st.markdown("---")

    # -- Tabs (unchanged) ----------------------------------------------

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Step 1: Softmax",
            "Step 2: Cross-Entropy Loss",
            "Step 3: Sigmoid & Detection",
            "Step 4: Weighted BCE Loss",
            "Step 5: Anomaly Detection",
            "Step 6: Trading Competition",
        ]
    )

    with tab1:
        render_step_1_softmax()

    with tab2:
        render_step_2_cross_entropy()

    with tab3:
        render_step_3_sigmoid()

    with tab4:
        render_step_4_bce_loss()

    with tab5:
        render_step_5_anomaly_detection()

    with tab6:
        render_step_6_trading_competition()


# ── Step 1: Softmax ──────────────────────────────────────────────


def render_step_1_softmax():
    st.subheader("Step 1: Implement Softmax")

    st.markdown(
        r"""
    ### From Logits to Probabilities

    Your neural network outputs **logits** $u \in \mathbb{R}^K$ -- raw, unconstrained scores.
    To make a decision, we need **probabilities** that:
    - Are all positive
    - Sum to 1 (mutually exclusive choice)

    The **softmax** function does exactly this:

    $$p(y = k \mid a) = \frac{\exp(u_k)}{\sum_{j=1}^{K} \exp(u_j)}$$

    **Numerical stability trick:** Subtract the max logit before exponentiating to avoid overflow:
    $\exp(u_k - \max(u)) / \sum_j \exp(u_j - \max(u))$. This gives the same answer
    because the constant cancels in the fraction.

    ---

    ### Your Task

    1. Implement a `softmax(logits)` function.
    2. Apply it to `logits = [0, 1, 2]` (Sell / Hold / Buy).
    3. Store the result in `probs`.
    """
    )

    default_code = """# Logits for Sell / Hold / Buy
logits = np.array([0.0, 1.0, 2.0])

def softmax(u):
    \"\"\"Convert logits to probabilities.
    Subtract max for numerical stability.\"\"\"
    # TODO: Implement softmax.
    # 1. Subtract the max for numerical stability
    # 2. Exponentiate each element
    # 3. Normalize so the result sums to 1
    pass

probs = softmax(logits)

actions = ["Sell", "Hold", "Buy"]
for action, p in zip(actions, probs):
    print(f"  {action}: {p:.4f}")
print(f"  Sum: {probs.sum():.4f}")
print(f"  Recommendation: {actions[np.argmax(probs)]} (highest probability)")"""

    st.caption("**Inputs:** `logits` (a 1D array of K raw scores for Sell/Hold/Buy). ")
    st.markdown("**Your Code:**")
    code = st_monaco(
        value=default_code, height="320px", language="python", theme="vs-dark"
    )

    if st.button("Run Softmax", key="run_softmax"):
        ctx = {"np": np}
        ctx, success = _run_student_code(code, ctx, "lab7_console_step1")

        if success:
            passed, msg = check_step_1_softmax(ctx)
            if passed:
                st.success(msg)
                st.session_state["lab7_step_1_done"] = True

                # Visualization
                probs = ctx["probs"]
                col1, col2 = st.columns(2)
                with col1:
                    fig = _plot_softmax_bars(probs, ["Sell", "Hold", "Buy"])
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.markdown("### Key Properties")
                    st.markdown(
                        f"""
                    - All probabilities are **positive**: {np.all(probs > 0)}
                    - They **sum to 1**: {probs.sum():.6f}
                    - **Largest logit** (Buy, u=2) gets **highest probability** ({probs[2]:.4f})
                    - Softmax preserves the **ranking** of logits
                    """
                    )

                # Temperature scaling demo
                st.markdown("---")
                st.markdown("### Temperature Scaling")
                st.markdown(
                    r"""
                    Temperature $T$ controls the **confidence** of the distribution:
                    $$p(y=k) = \frac{\exp(u_k / T)}{\sum_j \exp(u_j / T)}$$

                    - $T \to 0^+$: one-hot (maximum confidence)
                    - $T = 1$: standard softmax
                    - $T \to \infty$: uniform (minimum confidence)
                    """
                )

                logits = np.array([0.0, 1.0, 2.0])
                temps = [0.25, 0.5, 1.0, 2.0, 5.0]
                fig = _plot_temperature_scaling(logits, temps)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(msg)


# ── Step 2: Cross-Entropy Loss ───────────────────────────────────


def render_step_2_cross_entropy():
    st.subheader("Step 2: Cross-Entropy Loss")

    st.markdown(
        r"""
    ### Measuring Prediction Quality

    If the correct action was $y^\star$, the **cross-entropy loss** is:

    $$\mathcal{L} = -\log\, p(y^\star \mid a) = -u_{y^\star} + \log \sum_j \exp(u_j)$$

    **Intuition:**
    - If $p(y^\star) = 1$ (perfect prediction): loss $= 0$
    - If $p(y^\star) \approx 0$ (terrible prediction): loss $\to \infty$
    - Cross-entropy rewards putting **more probability** on the **correct** answer.

    ---

    ### Your Task

    1. Implement `cross_entropy_loss(logits, y_star)` where `y_star` is the index
       of the correct class.
    2. Compute the loss for our model ($u = [0, 1, 2]$) with $y^\star = \text{Buy}$ (index 2).
       Store in `loss_ours`.
    3. Compute the loss for a "know-nothing" model ($u = [0, 0, 0]$) with the same target.
       Store in `loss_uniform`.
    """
    )

    default_code = """def softmax(u):
    shifted = u - np.max(u)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)

def cross_entropy_loss(logits, y_star):
    \"\"\"Compute cross-entropy loss.
    logits: array of K logits
    y_star: index of the correct class
    Returns: scalar loss value
    \"\"\"
    # TODO: Compute the cross-entropy loss.
    # 1. Get probabilities from logits using softmax
    # 2. Return the negative log of the correct class's probability
    pass

# Our model: logits favor Buy
logits_ours = np.array([0.0, 1.0, 2.0])
y_star = 2  # correct answer is Buy

# Know-nothing model: all logits equal
logits_uniform = np.array([0.0, 0.0, 0.0])

loss_ours = cross_entropy_loss(logits_ours, y_star)
loss_uniform = cross_entropy_loss(logits_uniform, y_star)

print(f"Our model loss:      {loss_ours:.4f}")
print(f"Uniform model loss:  {loss_uniform:.4f}")
print(f"Improvement:         {loss_uniform - loss_ours:.4f} nats")"""

    st.caption(
        "**Inputs:** `logits` (model scores per class), `y_star` (index of correct class, e.g. 2 = Buy). "
        "You'll reuse `softmax` from Step 1;"
    )
    st.markdown("**Your Code:**")
    code = st_monaco(
        value=default_code, height="380px", language="python", theme="vs-dark"
    )

    if st.button("Run Cross-Entropy", key="run_cross_entropy"):
        ctx = {"np": np}
        ctx, success = _run_student_code(code, ctx, "lab7_console_step2")

        if success:
            passed, msg = check_step_2_cross_entropy(ctx)
            if passed:
                st.success(msg)
                st.session_state["lab7_step_2_done"] = True
                st.session_state["lab7_step2_loss_ours"] = ctx["loss_ours"]
                st.session_state["lab7_step2_loss_uniform"] = ctx["loss_uniform"]
            else:
                st.error(msg)

    if st.session_state.get("lab7_step_2_done"):
        loss_ours = st.session_state["lab7_step2_loss_ours"]
        loss_uniform = st.session_state["lab7_step2_loss_uniform"]

        col1, col2 = st.columns(2)
        with col1:
            fig = _plot_loss_comparison(loss_ours, loss_uniform)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### Why Cross-Entropy?")
            st.markdown(
                r"""
            Minimizing cross-entropy is equivalent to minimizing
            **KL divergence** between the true distribution and our model:

            $$\mathrm{KL}(p^\star \| p_\theta) = \underbrace{-\sum_k p^\star(k) \log p_\theta(k)}_{\text{cross-entropy}} + \underbrace{\text{const.}}_{\text{entropy of } p^\star}$$

            Since the entropy of $p^\star$ doesn't depend on $\theta$, minimizing
            cross-entropy = minimizing KL divergence.
            """
            )

        # Interactive: try different logits (runs every rerun so sliders update the plot)
        st.markdown("---")
        st.markdown("### Experiment: Try Different Logits")
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            u_sell = st.slider("u_Sell", -5.0, 5.0, 0.0, 0.5, key="exp_sell")
        with exp_col2:
            u_hold = st.slider("u_Hold", -5.0, 5.0, 1.0, 0.5, key="exp_hold")
        with exp_col3:
            u_buy = st.slider("u_Buy", -5.0, 5.0, 2.0, 0.5, key="exp_buy")

        exp_logits = np.array([u_sell, u_hold, u_buy])
        exp_shifted = exp_logits - exp_logits.max()
        exp_probs = np.exp(exp_shifted) / np.exp(exp_shifted).sum()

        target_action = st.radio(
            "Correct action:",
            ["Sell (0)", "Hold (1)", "Buy (2)"],
            index=2,
            horizontal=True,
            key="exp_target",
        )
        target_idx = int(target_action.split("(")[1][0])
        exp_loss = -np.log(exp_probs[target_idx])

        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            fig = _plot_softmax_bars(exp_probs, ["Sell", "Hold", "Buy"], "Your Softmax")
            st.plotly_chart(fig, use_container_width=True)
        with exp_c2:
            st.metric("Cross-Entropy Loss", f"{exp_loss:.4f}")
            st.metric("P(correct action)", f"{exp_probs[target_idx]:.4f}")
            actions = ["Sell", "Hold", "Buy"]
            st.caption(
                f"Target: **{actions[target_idx]}** | "
                f"Predicted: **{actions[np.argmax(exp_probs)]}**"
            )


# ── Step 3: Sigmoid & Detection ──────────────────────────────────


def render_step_3_sigmoid():
    st.subheader("Step 3: Sigmoid & Multi-Label Detection")

    st.markdown(
        r"""
    ### Classification vs Detection

    **Classification** (softmax): choose **exactly one** action -- Sell, Hold, or Buy.
    Probabilities must sum to 1.

    **Detection** (sigmoid): flag **any subset** of signals -- price rising, high volume,
    in the news. Each signal is **independent**.
    """
    )

    fig_class, fig_detect = _plot_softmax_vs_sigmoid()
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(fig_class, use_container_width=True, key="step3_class")
    with col_right:
        st.plotly_chart(fig_detect, use_container_width=True, key="step3_detect")

    st.markdown(
        r"""
    ---

    ### The Sigmoid Function

    Each signal gets an independent sigmoid:

    $$\sigma(a) = \frac{1}{1 + \exp(-a)}$$

    **Key insight:** sigmoid is a special case of softmax with $K=2$ and one logit
    fixed to 0. Given logits $(a, 0)$:

    $$p(y=1) = \frac{\exp(a)}{\exp(a) + \exp(0)} = \frac{1}{1 + \exp(-a)} = \sigma(a)$$

    ---

    ### Your Task

    1. Implement a `sigmoid(x)` function.
    2. Given signal logits `[2.0, -1.5, 0.8]` for [Price Rising, High Volume, In the News],
       compute the detection probabilities.
    3. Store the result in `signal_probs`.
    """
    )

    default_code = """# Signal logits: Price Rising, High Volume, In the News
signal_logits = np.array([2.0, -1.5, 0.8])
signal_names = ["Price Rising", "High Volume", "In the News"]

def sigmoid(x):
    \"\"\"Numerically stable sigmoid function.\"\"\"
    # TODO: Implement sigmoid: σ(x) = 1 / (1 + exp(-x))
    # For numerical stability, handle positive and negative x separately.
    # Hint: np.where can apply different formulas based on a condition.
    pass

signal_probs = sigmoid(signal_logits)

for name, logit, prob in zip(signal_names, signal_logits, signal_probs):
    flag = "YES" if prob >= 0.5 else "no"
    print(f"  {name}: logit={logit:+.1f}, prob={prob:.4f} -> {flag}")
print(f"\\nSum of probabilities: {signal_probs.sum():.4f}")
print("(Does NOT need to sum to 1 -- signals are independent!)")"""

    st.caption(
        "**Inputs:** `signal_logits` — one logit per signal (e.g. Price Rising, High Volume, In the News). "
    )
    st.markdown("**Your Code:**")
    code = st_monaco(
        value=default_code, height="320px", language="python", theme="vs-dark"
    )

    if st.button("Run Sigmoid", key="run_sigmoid"):
        ctx = {"np": np}
        ctx, success = _run_student_code(code, ctx, "lab7_console_step3")

        if success:
            passed, msg = check_step_3_sigmoid_detection(ctx)
            if passed:
                st.success(msg)
                st.session_state["lab7_step_3_done"] = True

                signal_probs = ctx["signal_probs"]
                signal_names = ctx.get(
                    "signal_names", ["Signal 1", "Signal 2", "Signal 3"]
                )

                col1, col2 = st.columns(2)
                with col1:
                    fig = _plot_sigmoid_bars(signal_probs, signal_names)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.markdown("### Classification vs Detection")
                    st.markdown(
                        """
                    | | Classification (Softmax) | Detection (Sigmoid) |
                    |---|---|---|
                    | Pick how many? | Exactly one | Any subset |
                    | Output constraint | Sum to 1 | Each in [0, 1] |
                    | Example | Buy/Hold/Sell | Rising + In the News |
                    | Loss | Cross-entropy | Binary cross-entropy |
                    """
                    )

                # Interactive sigmoid explorer
                st.markdown("---")
                st.markdown("### Explore: Sigmoid Curve")

                x_range = np.linspace(-6, 6, 200)
                sigmoid_fn = ctx["sigmoid"]
                y_vals = sigmoid_fn(x_range)

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_vals,
                        mode="lines",
                        name="sigmoid(x)",
                        line=dict(color="#1f77b4", width=3),
                    )
                )
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    title="The Sigmoid Function",
                    xaxis_title="logit (x)",
                    yaxis_title="probability",
                    height=350,
                    width=600,
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(msg)


# ── Step 4: Weighted BCE Loss ────────────────────────────────────


def render_step_4_bce_loss():
    st.subheader("Step 4: Binary Cross-Entropy with Negative Weighting")

    st.markdown(
        r"""
    ### The Problem with Detection Labels

    In real-world detection, annotators reliably spot **positives** but often **miss things**.
    So $y^\star_k = 0$ doesn't necessarily mean the signal is absent -- it might just be untagged.

    ### Standard BCE Loss

    For each signal $k$ with predicted probability $p_k$ and label $y^\star_k$:

    $$\mathcal{L} = -\sum_{k=1}^{K}\left[ y^\star_k \log(p_k) + (1 - y^\star_k) \log(1 - p_k) \right]$$

    ### Weighted BCE: Down-weight Negatives

    Since we don't fully trust $y^\star_k = 0$, we multiply the negative term by $\omega < 1$:

    $$\mathcal{L} = -\sum_{k=1}^{K}\left[ y^\star_k \log(p_k) + \omega \cdot (1 - y^\star_k) \log(1 - p_k) \right]$$

    This **down-weights** the contribution of negatives, making the model less aggressive
    about predicting "no signal."

    ---

    ### Your Task

    1. Implement `bce_loss(probs, targets, neg_weight=1.0)`.
    2. Compute `loss_standard` with `neg_weight=1.0`.
    3. Compute `loss_weighted` with `neg_weight=0.1`.
    """
    )

    default_code = """def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def bce_loss(probs, targets, neg_weight=1.0):
    \"\"\"Binary cross-entropy loss with optional negative weighting.
    probs:      predicted probabilities (K,)
    targets:    ground truth labels 0 or 1 (K,)
    neg_weight: weight for negative (y=0) terms
    \"\"\"
    eps = 1e-7  # avoid log(0)
    probs = np.clip(probs, eps, 1 - eps)

    # TODO: Compute the weighted BCE loss using the formula above.
    # The positive term involves targets and log(probs).
    # The negative term involves (1-targets) and log(1-probs), scaled by neg_weight.
    # Return the negative sum of both terms.
    pass

# Scenario: 5 market signals, model predictions vs. sparse labels
signal_logits = np.array([2.0, -1.5, 0.8, -0.3, 1.5])
signal_names = ["Price Rising", "High Volume", "In the News", "Earnings Soon", "Sector Hot"]
probs = sigmoid(signal_logits)

# Ground truth: only 2 signals labeled as present (the rest might be missing, not wrong!)
targets = np.array([1.0, 0.0, 1.0, 0.0, 0.0])

loss_standard = bce_loss(probs, targets, neg_weight=1.0)
loss_weighted = bce_loss(probs, targets, neg_weight=0.1)

print(f"Standard BCE loss (neg_weight=1.0): {loss_standard:.4f}")
print(f"Weighted BCE loss (neg_weight=0.1): {loss_weighted:.4f}")
print(f"\\nDown-weighting negatives reduced loss by {loss_standard - loss_weighted:.4f}")
print("\\nThis makes sense: we're less penalized for predicting 'signal present'")
print("when the ground truth says 'absent' -- because absence might just be untagged.")"""

    st.caption(
        "**Inputs:** `probs` (predicted probabilities per signal), `targets` (0/1 labels), "
        "`neg_weight` (weight for negative terms)."
    )
    st.markdown("**Your Code:**")
    code = st_monaco(
        value=default_code, height="420px", language="python", theme="vs-dark"
    )

    if st.button("Run BCE Loss", key="run_bce"):
        ctx = {"np": np}
        ctx, success = _run_student_code(code, ctx, "lab7_console_step4")

        if success:
            passed, msg = check_step_4_bce_loss(ctx)
            if passed:
                st.success(msg)
                st.session_state["lab7_step_4_done"] = True
                st.session_state["lab7_step4_bce_fn"] = ctx["bce_loss"]
                st.session_state["lab7_step4_probs"] = ctx.get(
                    "probs", np.array([0.88, 0.18, 0.69, 0.43, 0.82])
                )
                st.session_state["lab7_step4_targets"] = ctx.get(
                    "targets", np.array([1.0, 0.0, 1.0, 0.0, 0.0])
                )
                st.session_state["lab7_step4_loss_standard"] = ctx["loss_standard"]
                st.session_state["lab7_step4_loss_weighted"] = ctx["loss_weighted"]
            else:
                st.error(msg)

    if st.session_state.get("lab7_step_4_done"):
        loss_standard = st.session_state["lab7_step4_loss_standard"]
        loss_weighted = st.session_state["lab7_step4_loss_weighted"]
        bce_fn = st.session_state["lab7_step4_bce_fn"]
        probs = st.session_state["lab7_step4_probs"]
        targets = st.session_state["lab7_step4_targets"]

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=["Standard (w=1.0)", "Weighted (w=0.1)"],
                        y=[loss_standard, loss_weighted],
                        marker_color=["#d62728", "#2ca02c"],
                        text=[f"{loss_standard:.4f}", f"{loss_weighted:.4f}"],
                        textposition="auto",
                    )
                ]
            )
            fig.update_layout(
                title="BCE Loss: Standard vs Weighted",
                yaxis_title="Loss",
                height=350,
                width=500,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Why Down-weight Negatives?")
            st.markdown(
                """
            In real-world detection (movie genres, market signals, medical symptoms):

            - **Positive labels are reliable:** if tagged, it's probably true
            - **Negative labels are noisy:** absence of tag != absence of signal
            - A model predicting all zeros would have high accuracy but zero recall

            Down-weighting negatives tells the model:
            *"Don't be too confident that something is absent
            just because it's not labeled."*
            """
            )

        # Interactive: explore neg_weight effect (runs every rerun so slider updates the plot)
        st.markdown("---")
        st.markdown("### Experiment: Effect of Negative Weight")

        neg_w = st.slider(
            "Negative weight (omega)", 0.0, 1.0, 0.1, 0.05, key="exp_neg_weight"
        )

        sweep_weights = np.linspace(0.0, 1.0, 50)
        sweep_losses = [bce_fn(probs, targets, neg_weight=w) for w in sweep_weights]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sweep_weights,
                y=sweep_losses,
                mode="lines",
                name="BCE Loss",
                line=dict(color="#1f77b4", width=2),
            )
        )
        current_loss = bce_fn(probs, targets, neg_weight=neg_w)
        fig.add_trace(
            go.Scatter(
                x=[neg_w],
                y=[current_loss],
                mode="markers",
                name=f"w={neg_w:.2f}",
                marker=dict(size=12, color="red"),
            )
        )
        fig.update_layout(
            title="BCE Loss vs Negative Weight",
            xaxis_title="Negative Weight (omega)",
            yaxis_title="Loss",
            height=350,
            width=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"At omega={neg_w:.2f}, loss = {current_loss:.4f}. "
            "Lower omega = less penalty for predicting 'present' when label says 'absent'."
        )


# ── Step 5: Trading Competition ──────────────────────────────────

# ---------------------------------------------------------------------------
# Leaderboard backend: Google Sheets (synced across class) with local fallback
#
# To enable Google Sheets + NYU sign-in, add to .streamlit/secrets.toml:
#
#   [google_oauth]
#   client_id = "YOUR_OAUTH_CLIENT_ID"
#   client_secret = "YOUR_OAUTH_CLIENT_SECRET"
#   redirect_uri = "http://localhost:8501"   # or your deployed URL
#
#   [gcp_service_account]
#   type = "service_account"
#   project_id = "..."
#   private_key_id = "..."
#   private_key = "..."
#   client_email = "..."
#   ... (paste full service account JSON fields)
#
#   [leaderboard]
#   sheet_id = "YOUR_GOOGLE_SHEET_ID"
#
# Sheet setup: create a Google Sheet with headers in row 1:
#   name | email | final_value | return_pct | accuracy
# Share the sheet with the service account email (Editor).
# ---------------------------------------------------------------------------


FF_DATA_PATH = os.path.join(os.path.dirname(__file__), "ff_daily.npz")

# Stocks that were in these factor portfolios (1960-1992 era)
ERA_STOCKS = [
    ("IBM", "International Business Machines"),
    ("GE", "General Electric"),
    ("T", "AT&T Corporation"),
    ("XOM", "Exxon Corporation"),
    ("GM", "General Motors"),
    ("KO", "Coca-Cola Company"),
    ("PG", "Procter & Gamble"),
    ("JNJ", "Johnson & Johnson"),
    ("MRK", "Merck & Co."),
    ("DD", "DuPont"),
    ("BA", "Boeing Company"),
    ("F", "Ford Motor Company"),
    ("CVX", "Chevron Corporation"),
    ("MMM", "3M Company"),
    ("DIS", "Walt Disney Company"),
]

# Era descriptions for different time windows within 1960-1992
ERA_DESCRIPTIONS = [
    (
        "The Go-Go Years (1960-1965)",
        "1960s",
        "Growth stocks and conglomerates dominate",
    ),
    (
        "Nifty Fifty Era (1965-1970)",
        "Early 1970s",
        "Blue-chip fever then the '73-'74 crash",
    ),
    (
        "Stagflation Period (1970-1975)",
        "Late 1970s",
        "Oil shocks, inflation, and value rebound",
    ),
    (
        "The Bull Awakens (1975-1980)",
        "Early 1980s",
        "Volcker tames inflation, bull market begins",
    ),
    ("Roaring '80s (1980-1985)", "Late 1980s", "Leveraged buyouts and the '87 crash"),
    (
        "Japan Bubble Burst (1985-1992)",
        "Early 1990s",
        "Tech emerges, value vs growth debate",
    ),
]

# Factor descriptions
FACTOR_INFO = {
    "MktRF": (
        "Market",
        "excess return of the overall stock market over risk-free rate",
    ),
    "SMB": (
        "Size",
        "Small Minus Big -- small-cap stocks tend to outperform large-caps",
    ),
    "HML": (
        "Value",
        "High Minus Low -- cheap (value) stocks tend to outperform expensive (growth)",
    ),
    "MOM": ("Momentum", "Winners Minus Losers -- stocks that went up keep going up"),
}


def _load_factor_data():
    """Load daily Fama-French factor data (pre-publication era 1960-1992)."""
    data = np.load(FF_DATA_PATH)
    return {
        "dates": data["dates"],
        "MktRF": data["mktrf"],
        "SMB": data["smb"],
        "HML": data["hml"],
        "MOM": data["mom"],
        "RF": data["rf"],
    }


# Feature names: 6 per factor (4 factors) + 3 cross-factor = 27 features
FACTOR_NAMES = ["MktRF", "SMB", "HML", "MOM"]
FEATURE_NAMES = []
for _fn in FACTOR_NAMES:
    FEATURE_NAMES.extend(
        [
            f"{_fn}_lag1d",
            f"{_fn}_mom5d",
            f"{_fn}_mom20d",
            f"{_fn}_mom60d",
            f"{_fn}_vol20d",
            f"{_fn}_sharpe20d",
        ]
    )
FEATURE_NAMES.extend(["HML_minus_MOM", "SMB_minus_MktRF", "vol_regime_change"])


def _compute_factor_features(factor_data, warmup=60):
    """Compute 27 features from daily factor returns.

    For each of the 4 factors: lag-1 return, 5/20/60-day momentum,
    20-day volatility, 20-day Sharpe ratio.
    Plus 3 cross-factor features.

    Returns (features, next_day_combined_returns, dates) starting after warmup.
    """
    factors = np.column_stack(
        [
            factor_data["MktRF"],
            factor_data["SMB"],
            factor_data["HML"],
            factor_data["MOM"],
        ]
    )
    n = len(factor_data["MktRF"])

    features = []
    next_rets = []
    valid_dates = []

    for i in range(warmup, n - 1):
        row = []
        for j in range(4):
            f = factors[:, j]
            row.append(f[i - 1])  # lag-1 return
            row.append(np.sum(f[i - 5 : i]))  # 5-day momentum
            row.append(np.sum(f[i - 20 : i]))  # 20-day momentum
            row.append(np.sum(f[i - 60 : i]))  # 60-day momentum
            row.append(np.std(f[i - 20 : i]))  # 20-day volatility
            vol = np.std(f[i - 20 : i])
            row.append(np.mean(f[i - 20 : i]) / (vol + 1e-8))  # 20-day Sharpe

        # Cross-factor features
        row.append(factor_data["HML"][i - 1] - factor_data["MOM"][i - 1])
        row.append(factor_data["SMB"][i - 1] - factor_data["MktRF"][i - 1])
        mkt_vol_recent = np.std(factor_data["MktRF"][i - 20 : i])
        mkt_vol_older = np.std(factor_data["MktRF"][i - 60 : i - 20])
        row.append(mkt_vol_recent - mkt_vol_older)

        features.append(row)

        # Next-day combined portfolio return (equal-weight all 4 factors)
        next_ret = (
            factor_data["MktRF"][i]
            + factor_data["SMB"][i]
            + factor_data["HML"][i]
            + factor_data["MOM"][i]
        ) / 4.0
        next_rets.append(next_ret)
        valid_dates.append(factor_data["dates"][i])

    return np.array(features), np.array(next_rets), np.array(valid_dates)


def _generate_stock_data(n_days=500, seed=42):
    """Generate a roguelike market scenario from real pre-publication factor data.

    Uses real daily Fama-French factor returns (1960-1992) -- before the key
    alpha papers were published. The seed controls which time window is used,
    creating a different market era each run.

    No signal injection: the alpha is real.
    """
    rng = np.random.RandomState(seed)
    factor_data = _load_factor_data()

    all_features, all_returns, all_dates = _compute_factor_features(factor_data)
    total_available = len(all_features)

    # Pick a random starting offset (different era per seed)
    max_start = total_available - n_days - 1
    if max_start < 0:
        n_days = total_available - 1
        max_start = 0
    start = rng.randint(0, max(max_start, 1))

    features = all_features[start : start + n_days].copy()
    returns = all_returns[start : start + n_days].copy()
    dates = all_dates[start : start + n_days]

    # Determine era name from date range
    start_year = dates[0] // 10000
    end_year = dates[-1] // 10000
    era_idx = min((start_year - 1960) // 5, len(ERA_DESCRIPTIONS) - 1)
    era_name, era_period, era_desc = ERA_DESCRIPTIONS[max(era_idx, 0)]

    _generate_stock_data._last_era_name = era_name
    _generate_stock_data._last_era_desc = era_desc
    _generate_stock_data._last_start_year = start_year
    _generate_stock_data._last_end_year = end_year

    return features, returns


def _returns_to_labels(returns, buy_thresh=None, sell_thresh=None):
    """Convert returns to Buy/Sell/Hold labels using tercile thresholds.

    Uses percentile-based thresholds for balanced classes:
    Buy (2): top tercile of returns
    Hold (1): middle tercile
    Sell (0): bottom tercile
    """
    if buy_thresh is None:
        buy_thresh = np.percentile(returns, 67)
    if sell_thresh is None:
        sell_thresh = np.percentile(returns, 33)
    labels = np.ones(len(returns), dtype=int)  # default Hold
    labels[returns > buy_thresh] = 2  # Buy
    labels[returns < sell_thresh] = 0  # Sell
    return labels


def _simulate_trading(predictions, actual_returns, initial_capital=10000.0):
    """Simulate a trading strategy based on model predictions.

    predictions: array of 0 (Sell), 1 (Hold), 2 (Buy)
    actual_returns: array of next-day returns in %

    Returns: portfolio_values array, final_return
    """
    capital = initial_capital
    portfolio = [capital]

    for pred, ret in zip(predictions, actual_returns):
        daily_ret_frac = ret / 100.0
        if pred == 2:  # Buy: go long
            capital *= 1 + daily_ret_frac
        elif pred == 0:  # Sell: go short
            capital *= 1 - daily_ret_frac
        # Hold: no change
        portfolio.append(capital)

    total_return = (capital - initial_capital) / initial_capital * 100
    return np.array(portfolio), total_return


# ── Step 5: Market Anomaly Detection ────────────────────────────

ANOMALY_SIGNALS = [
    (
        "Earnings Surprise Wave",
        "Market excess return spikes (many stocks beating expectations)",
    ),
    ("Value Rotation", "Value stocks sharply outperform growth (HML spike)"),
    ("Small-Cap Discovery", "Small stocks surge past large caps (SMB spike)"),
    ("Momentum Unwind", "Recent winners crash, losers rally (MOM reversal)"),
    ("Volatility Storm", "Realized volatility jumps above historical baseline"),
]


def _build_anomaly_labels(factor_data, warmup=60):
    """Build 5 binary anomaly labels from real factor data.

    Each label is independent (multiple can fire on the same day).
    Uses rolling z-score thresholds calibrated for ~10-15% event rate.
    """
    mktrf = factor_data["MktRF"]
    smb = factor_data["SMB"]
    hml = factor_data["HML"]
    mom = factor_data["MOM"]
    n = len(mktrf)
    idx_range = list(range(warmup, n - 1))  # align with _compute_factor_features

    labels = []
    for i in idx_range:
        mkt_std = np.std(mktrf[i - 60 : i])
        hml_std = np.std(hml[i - 60 : i])
        smb_std = np.std(smb[i - 60 : i])
        mom_std = np.std(mom[i - 60 : i])
        vol_5d = np.std(mktrf[i - 4 : i + 1]) if i >= 4 else mkt_std

        row = [
            1.0 if mktrf[i] > 1.2 * mkt_std else 0.0,
            1.0 if hml[i] > 1.0 * hml_std else 0.0,
            1.0 if smb[i] > 1.0 * smb_std else 0.0,
            1.0 if mom[i] < -1.0 * mom_std else 0.0,
            1.0 if vol_5d > 1.5 * mkt_std else 0.0,
        ]
        labels.append(row)

    return np.array(labels)


def render_step_5_anomaly_detection():
    st.subheader("Step 5: Market Anomaly Detection")

    st.markdown(
        """
    ### From Classification to Detection

    In Steps 1-2 you built a **classifier** (softmax + cross-entropy) that picks
    **exactly one** action: Buy, Hold, or Sell. But the **risk desk** at your fund
    cares about something more fine-grained: they want to **monitor several types of risk
    events simultaneously** and be alerted when *any* of them might be happening.

    These risk events are driven by **earnings** and **factor rotations**. The table below lists the 5 signals you
    will detect, what each one means, and what drives it.
    """
    )

    signal_df = pd.DataFrame(
        {
            "Signal": [name for name, _ in ANOMALY_SIGNALS],
            "What It Detects": [desc for _, desc in ANOMALY_SIGNALS],
            "Driven By": [
                "Broad earnings beats across sectors",
                "Cheap stocks report better-than-expected earnings",
                "Under-covered small-caps get analyst attention",
                "Overhyped winners miss earnings, losers surprise",
                "Uncertainty spikes (often around earnings season)",
            ],
        }
    )
    st.dataframe(signal_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
    **Key insight:** These signals are **independent** -- you can have an earnings
    surprise wave AND a momentum unwind on the same day (it happens!). This is why
    we need **sigmoid** (per-signal) instead of softmax (mutually exclusive).

    The labels are also **noisy** -- a day might be borderline but labeled as "no event."
    This is exactly when **weighted BCE** (Step 4) helps: down-weight negatives because
    absence of a label doesn't mean absence of the event.

    ---

    ### Your Task

    Build a **multi-label detector** using sigmoid + BCE loss:
    1. Train 5 independent sigmoid classifiers (one per signal)
    2. Produce `detection_probs` -- a `(N_test, 5)` array of probabilities
    3. Events are rare (~10-15% each), so think about class imbalance!
    """
    )

    # Load data
    factor_data = _load_factor_data()
    all_features, _, all_dates = _compute_factor_features(factor_data)
    all_labels = _build_anomaly_labels(factor_data)

    # Time split: first 70% train, last 30% test
    split = int(len(all_features) * 0.7)
    X_train = all_features[:split]
    X_test = all_features[split:]
    Y_train = all_labels[:split]
    Y_test = all_labels[split:]
    test_dates = all_dates[split:]

    signal_names = [name for name, _ in ANOMALY_SIGNALS]

    st.markdown(
        """
    **Data source:**
    The features and labels in this step come from historical **factor data**
    (1960–1992) derived from the
    [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
    For this lab, you can treat them as **precomputed numeric features** and
    **binary event labels** — you do *not* need to work with the raw dataset
    or understand the portfolio construction details.
    """
    )

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(
            f"""
        **Training set:** {len(X_train)} days, {X_train.shape[1]} features
        """
        )
        for i, name in enumerate(signal_names):
            rate = Y_train[:, i].mean()
            st.caption(f"  {name}: {int(Y_train[:, i].sum())} events ({rate:.1%})")
    with col_info2:
        st.markdown(
            f"""
        **Test set:** {len(X_test)} days (predict anomaly probabilities!)
        """
        )
        st.caption(
            f"Events can co-occur: {int((Y_train.sum(axis=1) >= 2).sum())} days have 2+ signals"
        )
        st.caption(
            f"All quiet: {(Y_train.sum(axis=1) == 0).mean():.0%} of days have no events"
        )

    st.markdown("**Inputs for your code:**")
    st.markdown("You are given precomputed features and labels:")
    st.markdown(
        "- **X_train:** (N, 27) — features from **prior days** (lag returns, momentum, volatility, etc.)."
    )
    st.markdown(
        "- **Y_train:** (N, 5) — binary labels (1 = event that day, 0 = no event) for the 5 anomaly signals."
    )
    st.markdown(
        "- **X_test:** (N_test, 27) — same features for the test days to predict."
    )

    st.markdown("**Outputs for your code:**")
    st.markdown(
        "- **detection_probs:** `(N_test, 5)` — for each day and each signal, the model’s estimated probability that the event is happening. These probabilities will be evaluated using **F1**, **precision**, and **recall** in the results section below."
    )

    default_code = """# X_train: ({n_train}, 27) features, Y_train: ({n_train}, 5) binary labels
# X_test: ({n_test}, 27) features -- detect anomalies for these!

def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

# Normalize features (do not modify)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8
X_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std

n_features = X_norm.shape[1]  # 27
n_signals = Y_train.shape[1]   # 5

# Initialize weights and bias
W = np.random.randn(n_features, n_signals) * 0.01
b = np.zeros(n_signals)

lr = 0.005
neg_weight = 0.3  # down-weight negatives (events are rare!)

for epoch in range(300):
    # TODO: Forward pass -- compute logits and probabilities
    # logits = ...          # (N, 5)
    # probs = ...           # (N, 5)  apply sigmoid

    # TODO: Compute weighted BCE gradient through sigmoid
    # Recall from Step 4: BCE has a positive term and a negative term.
    # The gradient through sigmoid combines both with probs * (1 - probs).
    # grad = ...            # (N, 5)

    # TODO: Update weights and bias using the gradient
    # dW = ...              # (n_features, 5)
    # db = ...              # (5,)
    # W -= lr * dW
    # b -= lr * db

    if epoch % 100 == 0:
        loss = -np.mean(
            Y_train * np.log(probs + 1e-8)
            + neg_weight * (1 - Y_train) * np.log(1 - probs + 1e-8)
        )
        print(f"Epoch {{epoch}}: loss={{loss:.4f}}")

# TODO: Predict on test set -- store result in 'detection_probs'
# detection_probs = ...    # (N_test, 5)  apply sigmoid to test logits

signal_names = ["Earnings Surprise", "Value Rotation", "Small-Cap Discovery",
                "Momentum Unwind", "Volatility Storm"]
for i, name in enumerate(signal_names):
    detected = (detection_probs[:, i] > 0.5).sum()
    print(f"  {{name}}: {{detected}} days flagged ({{detected/len(detection_probs):.1%}})")""".format(
        n_train=len(X_train), n_test=len(X_test)
    )

    st.markdown("**Your Code:**")
    code = st_monaco(
        value=default_code, height="520px", language="python", theme="vs-dark"
    )

    if st.button("Run Detector", key="run_detection", type="primary"):
        ctx = {
            "np": np,
            "X_train": X_train.copy(),
            "Y_train": Y_train.copy(),
            "X_test": X_test.copy(),
        }
        ctx, success = _run_student_code(code, ctx, "lab7_console_step5")

        if not success:
            return

        if "detection_probs" not in ctx:
            st.error(
                "Variable `detection_probs` not found. "
                "Your code must produce a (N_test, 5) array of detection probabilities."
            )
            return

        det_probs = ctx["detection_probs"]
        if not isinstance(det_probs, np.ndarray):
            det_probs = np.array(det_probs)

        if det_probs.shape != Y_test.shape:
            st.error(
                f"`detection_probs` has shape {det_probs.shape} but expected "
                f"{Y_test.shape} (N_test x 5 signals)."
            )
            return

        st.session_state["lab7_step5_results_ready"] = True
        st.session_state["lab7_step5_det_probs"] = det_probs
        st.session_state["lab7_step5_Y_test"] = Y_test
        st.session_state["lab7_step5_test_dates"] = test_dates
        st.session_state["lab7_step5_signal_names"] = signal_names

    if st.session_state.get("lab7_step5_results_ready"):
        det_probs = st.session_state["lab7_step5_det_probs"]
        Y_test = st.session_state["lab7_step5_Y_test"]
        test_dates = st.session_state["lab7_step5_test_dates"]
        signal_names = st.session_state["lab7_step5_signal_names"]

        st.markdown("---")
        st.subheader("Detection Results")

        # Per-signal metrics
        results = []
        for i, name in enumerate(signal_names):
            preds_i = (det_probs[:, i] > 0.5).astype(float)
            f1 = _sklearn_f1(Y_test[:, i], preds_i, zero_division=0)
            prec = precision_score(Y_test[:, i], preds_i, zero_division=0)
            rec = recall_score(Y_test[:, i], preds_i, zero_division=0)
            n_events = int(Y_test[:, i].sum())
            n_detected = int(preds_i.sum())
            results.append(
                {
                    "Signal": name,
                    "F1 Score": f"{f1:.3f}",
                    "Precision": f"{prec:.2f}",
                    "Recall": f"{rec:.2f}",
                    "True Events": n_events,
                    "Flagged": n_detected,
                }
            )

        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        avg_f1 = np.mean(
            [
                _sklearn_f1(
                    Y_test[:, i], (det_probs[:, i] > 0.5).astype(float), zero_division=0
                )
                for i in range(5)
            ]
        )
        st.metric("Average F1 Score", f"{avg_f1:.3f}")

        if avg_f1 > 0.15:
            st.success(
                f"Your detector achieves F1 = {avg_f1:.3f} -- "
                f"it's catching real market anomalies from 1960s-1990s data!"
            )
        else:
            st.warning(
                "F1 is low. Try adjusting neg_weight, learning rate, or "
                "training for more epochs. Anomaly detection is hard!"
            )

        # Timeline visualization: zoomed window with slider (runs every rerun so slider updates)
        st.markdown("---")
        st.markdown("### Detection Timeline")
        sel_col1, sel_col2 = st.columns([2, 1])
        with sel_col1:
            selected_signal = st.selectbox(
                "Signal to visualize:", signal_names, key="det_signal_viz"
            )
        with sel_col2:
            window_size = 100
            max_start = len(Y_test) - window_size
            window_start = st.slider(
                "Scroll through time",
                0,
                max_start,
                0,
                key="det_timeline_scroll",
                help="Slide to see different 100-day windows",
            )

        sig_idx = signal_names.index(selected_signal)
        w_end = window_start + window_size

        # Slice the window
        actual_w = Y_test[window_start:w_end, sig_idx]
        probs_w = det_probs[window_start:w_end, sig_idx]
        preds_w = (probs_w > 0.5).astype(float)
        dates_w = test_dates[window_start:w_end]

        date_labels = [
            f"{int(d) // 10000}-{int(d) % 10000 // 100:02d}-{int(d) % 100:02d}"
            for d in dates_w
        ]

        fig = go.Figure()

        # Background shading for actual events (green vertical bands)
        for j in range(window_size):
            if actual_w[j] == 1.0:
                fig.add_vrect(
                    x0=j - 0.4,
                    x1=j + 0.4,
                    fillcolor="rgba(44, 160, 44, 0.25)",
                    line_width=0,
                    layer="below",
                )

        # Detection probability line
        fig.add_trace(
            go.Scatter(
                x=list(range(window_size)),
                y=probs_w,
                mode="lines",
                name="Detection Probability",
                line=dict(color="#1f77b4", width=2),
                hovertext=date_labels,
            )
        )

        # Mark true positives (green dots), false positives (red dots),
        # false negatives (orange triangles)
        tp_x = [j for j in range(window_size) if actual_w[j] == 1 and preds_w[j] == 1]
        tp_y = [probs_w[j] for j in tp_x]
        fp_x = [j for j in range(window_size) if actual_w[j] == 0 and preds_w[j] == 1]
        fp_y = [probs_w[j] for j in fp_x]
        fn_x = [j for j in range(window_size) if actual_w[j] == 1 and preds_w[j] == 0]
        fn_y = [probs_w[j] for j in fn_x]

        if tp_x:
            fig.add_trace(
                go.Scatter(
                    x=tp_x,
                    y=tp_y,
                    mode="markers",
                    name=f"Hit ({len(tp_x)})",
                    marker=dict(color="#2ca02c", size=10, symbol="circle"),
                )
            )
        if fp_x:
            fig.add_trace(
                go.Scatter(
                    x=fp_x,
                    y=fp_y,
                    mode="markers",
                    name=f"False Alarm ({len(fp_x)})",
                    marker=dict(color="#d62728", size=8, symbol="x"),
                )
            )
        if fn_x:
            fig.add_trace(
                go.Scatter(
                    x=fn_x,
                    y=fn_y,
                    mode="markers",
                    name=f"Missed ({len(fn_x)})",
                    marker=dict(color="#ff7f0e", size=10, symbol="triangle-up"),
                )
            )

        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="red",
            opacity=0.6,
            annotation_text="threshold",
            annotation_position="top right",
        )

        # Date range for title
        d0, d1 = date_labels[0], date_labels[-1]
        fig.update_layout(
            title=f"{selected_signal} ({d0} to {d1})",
            xaxis_title="Day in Window",
            yaxis_title="Detection Probability",
            yaxis=dict(range=[0, 1.05]),
            height=420,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Green shading = actual event day | "
            "Green dot = correctly detected (hit) | "
            "Red X = false alarm | "
            "Orange triangle = missed event"
        )

        # Co-occurrence heatmap
        st.markdown("### Signal Co-occurrence")
        st.caption("How often do multiple anomalies fire on the same day?")

        det_binary = (det_probs > 0.5).astype(float)
        co_matrix = det_binary.T @ det_binary / len(det_binary)
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=co_matrix,
                x=signal_names,
                y=signal_names,
                colorscale="Blues",
                text=[[f"{v:.3f}" for v in row] for row in co_matrix],
                texttemplate="%{text}",
                zmin=0,
            )
        )
        fig_heat.update_layout(
            title="Co-detection Rate (fraction of days both flagged)",
            height=400,
            width=600,
        )
        st.plotly_chart(fig_heat, use_container_width=True)


def render_step_6_trading_competition():
    st.subheader("Step 6: Trading Competition")

    st.markdown(
        """
    ### Put Your Classifier to Work

    You've joined a quant fund. It's the 1980s, and the research team has uncovered
    **alpha signals** from stock market data -- patterns that predict future returns.
    These signals are built from real portfolios of stocks like:
    """
    )

    # Show era-appropriate stocks
    stock_rows = [f"**{ticker}** ({name})" for ticker, name in ERA_STOCKS[:8]]
    st.caption(" | ".join(stock_rows[:4]))
    st.caption(" | ".join(stock_rows[4:]))
    st.caption(
        "...and thousands more NYSE, AMEX, and NASDAQ stocks sorted into factor portfolios."
    )

    st.markdown(
        """
    The data uses **real daily returns** from the [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
    -- the same data used in the original research papers. The time period is
    **1960-1992**, *before* the key alpha papers were published:

    | Factor | What It Captures | Key Paper |
    |--------|------------------|-----------|
    | **MktRF** (Market) | Excess market return | Sharpe (1964) |
    | **SMB** (Size) | Small stocks beat large stocks | Banz (1981), Fama & French (1993) |
    | **HML** (Value) | Cheap stocks beat expensive stocks | Fama & French (1993) |
    | **MOM** (Momentum) | Recent winners keep winning | Jegadeesh & Titman (1993) |

    Your **27 features** combine all four alpha signals:
    - For each factor: lag-1 return, 5/20/60-day momentum, 20-day volatility, 20-day Sharpe
    - Cross-factor: value-momentum spread, size-market spread, volatility regime change

    Each day is labeled **Sell** (0), **Hold** (1), or **Buy** (2) based on the
    next day's combined factor portfolio return (tercile split for balanced classes).

    The **leaderboard** tracks everyone's results -- can you beat your classmates?
    """
    )

    with st.expander("Why does this work? (The alpha story)"):
        st.markdown(
            r"""
        **What is alpha?** In quantitative finance, *alpha* refers to signals that
        predict future returns beyond what's explained by risk.

        **Why pre-publication data?** Once alpha signals are published, everyone
        trades on them and the edge shrinks ("alpha decay"). By using data from
        **before** publication (1960-1992), the signals still have real predictive
        power -- this is the golden era of factor investing.

        **How quant funds use this:** Given alpha signals, the ML pipeline is:
        1. **Features:** compute rolling statistics of factor returns (what you have)
        2. **Classifier:** predict next-day market direction (Buy/Hold/Sell)
        3. **Portfolio:** go long (Buy), stay in cash (Hold), or go short (Sell)
        4. **Aggregate:** tiny edges × thousands of trades = profit

        **Real-world accuracy:** Even a few percentage points above random is
        valuable when trading large amounts daily. In this lab, logistic regression
        gets ~42% on 3 classes (vs ~33% random). That small edge compounds to
        nearly **double** the buy-and-hold return over the test period.

        **Key papers:**
        - Fama & French (1993), *"Common Risk Factors in the Returns of Stocks and Bonds"* -- value + size
        - Jegadeesh & Titman (1993), *"Returns to Buying Winners and Selling Losers"* -- momentum
        - Banz (1981), *"The Relationship Between Return and Market Value of Common Stocks"* -- size

        **No signal injection.** Unlike simplified lab datasets, this data has
        **zero artificial amplification**. The alpha is 100% real.
        """
        )

    st.markdown(
        """
    ---

    ### Your Task

    Write code that:
    1. Trains a model on `X_train` / `y_train` (any approach you like -- logistic regression, MLP, decision trees...)
    2. Produces `predictions` -- an array of 0/1/2 for each test day
    3. With 27 features, there's room for nonlinear models to shine!
    """
    )

    # --- Roguelike seed system ---
    seed_col1, seed_col2 = st.columns([2, 1])
    with seed_col1:
        if "lab7_market_seed" not in st.session_state:
            st.session_state["lab7_market_seed"] = 42
        market_seed = st.number_input(
            "Market Seed",
            value=st.session_state["lab7_market_seed"],
            min_value=0,
            max_value=9999,
            step=1,
            key="lab7_seed_input",
            help="Each seed generates a different market with different patterns to discover.",
        )
        st.session_state["lab7_market_seed"] = market_seed
    with seed_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Reroll Market", key="reroll_market"):
            new_seed = int(np.random.randint(0, 10000))
            st.session_state["lab7_market_seed"] = new_seed
            st.rerun()

    # Generate data for this seed
    features, returns = _generate_stock_data(n_days=2000, seed=market_seed)
    labels = _returns_to_labels(returns)
    era_name = getattr(_generate_stock_data, "_last_era_name", "Unknown Era")
    era_desc = getattr(_generate_stock_data, "_last_era_desc", "")
    start_yr = getattr(_generate_stock_data, "_last_start_year", 1960)
    end_yr = getattr(_generate_stock_data, "_last_end_year", 1992)

    # Train/test split: first 1400 days train, last 600 test (~2.4 years)
    split_idx = 1400
    X_train, X_test = features[:split_idx], features[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]
    returns_test = returns[split_idx:]

    # Show data summary
    st.info(
        f"**Market #{market_seed}** -- _{era_name}_ ({start_yr}-{end_yr}): "
        f"{era_desc}. "
        f"Real Fama-French factor data (no artificial signal injection)."
    )
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(
            f"""
        **Training set:** {len(X_train)} trading days
        - Buy days: {(y_train == 2).sum()}
        - Hold days: {(y_train == 1).sum()}
        - Sell days: {(y_train == 0).sum()}
        """
        )
    with col_info2:
        st.markdown(
            f"""
        **Test set:** {len(X_test)} trading days (hidden returns!)
        - Features: {X_train.shape[1]} per day (27 combined alpha signals)
        - Starting capital: $10,000
        """
        )

    st.markdown("**Inputs for your code:**")
    st.markdown(
        "- **X_train:** (1400, 27) — features from **prior days** (lag, momentum, vol, Sharpe, cross-factor)."
    )
    st.markdown(
        "- **y_train:** (1400,) — labels 0 (Sell), 1 (Hold), 2 (Buy) from next-day return terciles."
    )
    st.markdown(
        "- **X_test:** (600, 27) — same features for test days (predict 0/1/2 for each)."
    )

    # Show feature distributions
    with st.expander("Explore Training Data"):
        feat_names = FEATURE_NAMES
        train_df = pd.DataFrame(X_train, columns=feat_names)
        train_df["label"] = np.where(
            y_train == 2, "Buy", np.where(y_train == 0, "Sell", "Hold")
        )

        feat_to_plot = st.selectbox("Feature to visualize:", feat_names, key="feat_viz")
        fig = px.histogram(
            train_df,
            x=feat_to_plot,
            color="label",
            color_discrete_map={"Buy": "#2ca02c", "Hold": "#7f7f7f", "Sell": "#d62728"},
            barmode="overlay",
            opacity=0.6,
            title=f"Distribution of '{feat_to_plot}' by Label",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    default_code = """# X_train: (1400, 27) features, y_train: (1400,) labels 0/1/2
# X_test: (600, 27) features -- predict actions for these!
#
# Features (27 total): For each factor (MktRF, SMB, HML, MOM):
#   lag-1 return, 5/20/60-day momentum, 20-day volatility, 20-day Sharpe
# Plus: value-momentum spread, size-market spread, volatility regime change
#
# Labels: Sell(0), Hold(1), Buy(2) based on next-day combined factor return

# --- Scaffolding (do not modify) ---

def softmax(u):
    shifted = u - np.max(u, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

# One-hot encode targets
n_classes = 3
y_onehot = np.zeros((len(y_train), n_classes))
y_onehot[np.arange(len(y_train)), y_train] = 1.0

# Normalize features (important with 27 features at different scales!)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8
X_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std

# Initialize weights and bias
n_features = X_norm.shape[1]  # 27
W = np.random.randn(n_features, n_classes) * 0.01
b = np.zeros(n_classes)

# Training loop (gradient descent)
lr = 0.1
for epoch in range(300):
    # TODO: Forward pass -- compute logits and probabilities
    # logits = ...          # (N, 3)
    # probs = ...           # (N, 3)  apply softmax

    # TODO: Compute cross-entropy loss (for printing)
    # loss = ...

    # TODO: Compute gradients and update weights
    # For softmax + cross-entropy, the gradient is: probs - y_onehot
    # grad = ...            # (N, 3)
    # dW = ...              # (n_features, 3)
    # db = ...              # (3,)
    # W -= lr * dW
    # b -= lr * db

    if epoch % 100 == 0:
        acc = (np.argmax(probs, axis=1) == y_train).mean()
        print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={acc:.3f}")

# TODO: Predict on test set
# predictions = ...  # shape (600,), values in {0, 1, 2}  use np.argmax

# Summary
actions = ["Sell", "Hold", "Buy"]
print(f"\\nTest predictions: {len(predictions)} days")
for i, a in enumerate(actions):
    print(f"  {a}: {(predictions == i).sum()} days")
print("\\nTip: Try an MLP! With 27 features, nonlinear models can find")
print("interactions between factors that linear models miss.")"""

    st.markdown("**Your Code:**")
    code = st_monaco(
        value=default_code, height="500px", language="python", theme="vs-dark"
    )

    trader_name = st.text_input(
        "Your name (for the leaderboard):",
        value="",
        key="trader_name",
    )

    if st.button("Execute Trades!", key="run_trading", type="primary"):
        import re

        # Sanitize name: keep safe chars, collapse whitespace, limit length
        trader_name = re.sub(r"[^a-zA-Z0-9 _\-.']+", "", trader_name).strip()[:50]
        if not trader_name:
            st.warning("Enter your name for the leaderboard!")
            return

        ctx = {
            "np": np,
            "X_train": X_train.copy(),
            "y_train": y_train.copy(),
            "X_test": X_test.copy(),
        }
        ctx, success = _run_student_code(code, ctx, "lab7_console_step6")

        if not success:
            return

        if "predictions" not in ctx:
            st.error(
                "Variable `predictions` not found. "
                "Your code must produce an array of 0/1/2 predictions for each test day."
            )
            return

        predictions = ctx["predictions"]

        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)

        if len(predictions) != len(X_test):
            st.error(
                f"`predictions` has {len(predictions)} values but test set has "
                f"{len(X_test)} days. They must match."
            )
            return

        if not np.all(np.isin(predictions, [0, 1, 2])):
            st.error("All predictions must be 0 (Sell), 1 (Hold), or 2 (Buy).")
            return

        # Simulate trading
        portfolio, total_return = _simulate_trading(
            predictions, returns_test, initial_capital=10000.0
        )

        # Baselines
        buy_hold_port, buy_hold_ret = _simulate_trading(
            np.full(len(returns_test), 2), returns_test
        )
        sell_all_port, sell_all_ret = _simulate_trading(
            np.full(len(returns_test), 0), returns_test
        )
        hold_port, hold_ret = _simulate_trading(
            np.full(len(returns_test), 1), returns_test
        )
        # "Perfect" oracle
        oracle_preds = _returns_to_labels(returns_test, 0.0, 0.0)
        oracle_port, oracle_ret = _simulate_trading(oracle_preds, returns_test)

        # Classification accuracy on test set
        accuracy = (predictions == y_test).mean()

        st.markdown("---")
        st.subheader("Results")

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                "Final Portfolio", f"${portfolio[-1]:,.2f}", f"{total_return:+.2f}%"
            )
        with m2:
            st.metric("Accuracy", f"{accuracy:.1%}")
        with m3:
            st.metric(
                "vs. Buy & Hold",
                f"{total_return - buy_hold_ret:+.2f}%",
            )
        with m4:
            pct_of_oracle = (total_return / oracle_ret * 100) if oracle_ret != 0 else 0
            st.metric(
                "% of Best Possible",
                f"{pct_of_oracle:.1f}%",
                help=f"Your {total_return:+.2f}% vs perfect oracle's {oracle_ret:+.2f}%",
            )

        # Portfolio chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=portfolio,
                mode="lines",
                name=f"Your Strategy (${portfolio[-1]:,.0f})",
                line=dict(color="#2ca02c", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                y=buy_hold_port,
                mode="lines",
                name=f"Buy & Hold (${buy_hold_port[-1]:,.0f})",
                line=dict(color="#1f77b4", width=1.5, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                y=hold_port,
                mode="lines",
                name=f"Always Hold (${hold_port[-1]:,.0f})",
                line=dict(color="#7f7f7f", width=1.5, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                y=oracle_port,
                mode="lines",
                name=f"Perfect Oracle (${oracle_port[-1]:,.0f})",
                line=dict(color="#ff7f0e", width=1.5, dash="dash"),
            )
        )
        fig.add_hline(y=10000, line_dash="dot", line_color="gray", opacity=0.5)
        fig.update_layout(
            title="Portfolio Value Over Test Period",
            xaxis_title="Trading Day",
            yaxis_title="Portfolio Value ($)",
            height=450,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Action breakdown
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            actions = ["Sell", "Hold", "Buy"]
            pred_counts = [(predictions == i).sum() for i in range(3)]
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=actions,
                        y=pred_counts,
                        marker_color=["#d62728", "#7f7f7f", "#2ca02c"],
                        text=pred_counts,
                        textposition="auto",
                    )
                ]
            )
            fig.update_layout(title="Your Action Distribution", height=300)
            st.plotly_chart(fig, use_container_width=True)

        with act_col2:
            # Confusion: how often were you right per action?
            action_results = []
            for i, a in enumerate(actions):
                mask = predictions == i
                if mask.sum() > 0:
                    correct = (y_test[mask] == i).sum()
                    action_results.append(
                        {
                            "Action": a,
                            "Times Used": int(mask.sum()),
                            "Correct": int(correct),
                            "Accuracy": f"{correct / mask.sum():.1%}",
                        }
                    )
            st.markdown("### Per-Action Accuracy")
            st.dataframe(pd.DataFrame(action_results), use_container_width=True)

        # --- Leaderboard submission ---
        st.markdown("---")
        st.markdown("### Submit to Leaderboard")
        st.markdown(
            f"**Your results:** ${portfolio[-1]:,.2f} ({total_return:+.2f}%), "
            f"accuracy {accuracy:.1%}, seed #{market_seed} ({era_name})"
        )
        st.caption(
            "These values are computed from your predictions and cannot be modified."
        )

        # Build pre-filled URL to the deployed leaderboard
        from urllib.parse import urlencode

        _LB_URL = "https://web-production-0dca3.up.railway.app/"
        params = "?" + urlencode(
            {
                "name": trader_name.strip(),
                "seed": market_seed,
                "era": era_name,
                "final_value": f"{portfolio[-1]:.2f}",
                "return_pct": f"{total_return:.2f}",
                "accuracy": f"{accuracy * 100:.1f}",
            }
        )
        st.link_button(
            "Submit to Leaderboard",
            _LB_URL + params,
            type="primary",
        )
        st.caption(f"[View the full leaderboard]({_LB_URL})")
