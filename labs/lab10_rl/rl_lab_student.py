"""
Lab 10: Reinforcement Learning — Q-Learning and Deep Q-Networks

Three parts:
  1. GridWorld (5×5): value iteration and Q-learning side-by-side
  2. CartPole: naive DQN, then DQN with stabilization tricks
  3. Discussion: how RL formalisms apply to language models
"""

import collections
import io
import json
import math
import random
import sys
import traceback
from string import Template

import gymnasium as gym
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

from labs.lab10_rl.level_checks import (
    check_step_0_value_iteration,
    check_step_1_q_learning,
    check_step_2_qnetwork,
    check_step_3_dqn,
)

# ======================================================================
# GridWorld constants & helpers (shared with student code via _PRE_INJECTED)
# ======================================================================

GRID_SIZE = 5
NUM_ACTIONS = 4  # 0=up, 1=down, 2=left, 3=right
GAMMA = 0.9
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = {(1, 0), (2, 3)}

# State rewards: R(s) is collected while the agent occupies state s.
# This means obstacle cells have negative V*, visible directly in the heatmap.
REWARD_GRID = np.zeros((GRID_SIZE, GRID_SIZE))
REWARD_GRID[4, 4] = 1.0
REWARD_GRID[1, 0] = -1.0
REWARD_GRID[2, 3] = -1.0


def grid_step(state, action):
    """Deterministic gridworld transition.

    Returns (next_state, done).
    Actions: 0=up, 1=down, 2=left, 3=right.
    Moving into a wall keeps the agent in place.
    Rewards are in REWARD_GRID and are collected for the *current* state.
    """
    r, c = state
    if action == 0:
        r = max(r - 1, 0)
    elif action == 1:
        r = min(r + 1, GRID_SIZE - 1)
    elif action == 2:
        c = max(c - 1, 0)
    elif action == 3:
        c = min(c + 1, GRID_SIZE - 1)
    next_state = (r, c)
    done = next_state == GOAL
    return next_state, done


# ======================================================================
# Replay buffer (pre-provided for DQN step)
# ======================================================================


class ReplayBuffer:
    """Fixed-capacity experience replay buffer."""

    def __init__(self, capacity=10_000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ======================================================================
# Shared helpers
# ======================================================================

_PRE_INJECTED = {
    "np": np,
    "torch": torch,
    "nn": nn,
    "F": F,
    "optim": optim,
    "math": math,
    "random": random,
    # GridWorld
    "GRID_SIZE": GRID_SIZE,
    "NUM_ACTIONS": NUM_ACTIONS,
    "GAMMA": GAMMA,
    "START": START,
    "GOAL": GOAL,
    "OBSTACLES": OBSTACLES,
    "REWARD_GRID": REWARD_GRID,
    "grid_step": grid_step,
    # DQN
    "ReplayBuffer": ReplayBuffer,
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
    """Execute code, run check, persist result in session state. Returns result dict."""
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
    """Render persisted output / check result. Returns (passed, shown)."""
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


# ======================================================================
# Visualisation helpers
# ======================================================================

ACTION_ARROW = {0: "↑", 1: "↓", 2: "←", 3: "→"}

_GRID_LAYOUT = dict(
    xaxis=dict(
        showticklabels=False,
        showgrid=True,
        gridcolor="white",
        gridwidth=2,
        zeroline=False,
        range=[-0.5, GRID_SIZE - 0.5],
        dtick=1,
    ),
    yaxis=dict(
        showticklabels=False,
        showgrid=True,
        gridcolor="white",
        gridwidth=2,
        zeroline=False,
        range=[-0.5, GRID_SIZE - 0.5],
        dtick=1,
    ),
    height=370,
    margin=dict(l=10, r=10, t=40, b=10),
)


def _trace_path(policy, max_steps=50):
    """Trace the greedy path from START to GOAL. Returns list of (r, c) tuples."""
    path = [START]
    state = START
    visited = set()
    for _ in range(max_steps):
        if state == GOAL:
            break
        if state in visited:
            break  # loop — policy doesn't reach goal
        visited.add(state)
        r, c = state
        action = int(policy[r, c])
        state, done = grid_step(state, action)
        path.append(state)
        if done:
            break
    return path


def _make_env_figure():
    """Static heatmap showing the GridWorld layout before any computation."""
    # Color key: 0 = normal, 1 = start, 2 = obstacle, 3 = goal
    display = np.zeros((GRID_SIZE, GRID_SIZE))
    display[GOAL] = 3
    for r, c in OBSTACLES:
        display[r, c] = 2
    display[START] = 1

    colorscale = [
        [0.00, "#e8e8e8"],
        [0.24, "#e8e8e8"],  # 0 — neutral gray
        [0.25, "#5599dd"],
        [0.49, "#5599dd"],  # 1 — start blue
        [0.50, "#cc4444"],
        [0.74, "#cc4444"],  # 2 — obstacle red
        [0.75, "#33aa55"],
        [1.00, "#33aa55"],  # 3 — goal green
    ]

    cell_labels = {
        START: ("→", "#ffffff"),  # agent arrow (initial dir = right)
        GOAL: ("🎯", "#ffffff"),
        **{obs: ("✗", "#ffffff") for obs in OBSTACLES},
    }
    reward_labels = {
        GOAL: "+1",
        **{obs: "−1" for obs in OBSTACLES},
    }

    z = display[::-1]
    fig = go.Figure(
        data=go.Heatmap(z=z, colorscale=colorscale, showscale=False, zmin=0, zmax=3)
    )

    for r in range(GRID_SIZE):
        dr = GRID_SIZE - 1 - r
        for c in range(GRID_SIZE):
            if (r, c) in cell_labels:
                sym, col = cell_labels[(r, c)]
                fig.add_annotation(
                    x=c,
                    y=dr,
                    text=f"<b>{sym}</b>",
                    showarrow=False,
                    font=dict(size=22, color=col),
                    yshift=6,
                )
                if (r, c) in reward_labels:
                    fig.add_annotation(
                        x=c,
                        y=dr,
                        text=reward_labels[(r, c)],
                        showarrow=False,
                        font=dict(size=11, color=col),
                        yshift=-12,
                    )
            else:
                fig.add_annotation(
                    x=c,
                    y=dr,
                    text="0",
                    showarrow=False,
                    font=dict(size=11, color="#aaaaaa"),
                )

    fig.update_layout(title="GridWorld Environment", **_GRID_LAYOUT)
    return fig


def _make_grid_figure(grid_vals, title, policy=None, show_path=False):
    """Heatmap of grid_vals with 🎯 goal, policy arrows, and optional path trace."""
    # For display: GOAL stays 0 by algorithm design (terminal), but show it bright
    V_display = grid_vals.astype(float).copy()
    vmax = grid_vals.max()
    V_display[GOAL] = vmax  # visual only — doesn't affect checks

    z = V_display[::-1]

    fig = go.Figure(
        data=go.Heatmap(
            z=z, colorscale="Blues", showscale=True, zmin=z.min(), zmax=z.max()
        )
    )

    for r in range(GRID_SIZE):
        dr = GRID_SIZE - 1 - r
        for c in range(GRID_SIZE):
            if (r, c) == GOAL:
                fig.add_annotation(
                    x=c,
                    y=dr,
                    text="<b>🎯</b>",
                    showarrow=False,
                    font=dict(size=22, color="white"),
                )
            elif (r, c) in OBSTACLES:
                fig.add_annotation(
                    x=c,
                    y=dr,
                    text="<b>✗</b>",
                    showarrow=False,
                    font=dict(size=20, color="#ff6666"),
                )
            elif policy is not None:
                arrow = ACTION_ARROW[int(policy[r, c])]
                color = "#ffe066" if (r, c) == START else "white"
                fig.add_annotation(
                    x=c,
                    y=dr,
                    text=f"<b>{arrow}</b>",
                    showarrow=False,
                    font=dict(size=22, color=color),
                )

    if show_path and policy is not None:
        path = _trace_path(policy)
        if len(path) > 1:
            xs = [c for _, c in path]
            ys = [GRID_SIZE - 1 - r for r, _ in path]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="rgba(255, 220, 0, 0.65)", width=6),
                    showlegend=False,
                )
            )

    fig.update_layout(title=title, **_GRID_LAYOUT)
    return fig


def _greedy_policy_from_V(V):
    # With state rewards, R(s) is constant across actions for a given s,
    # so the greedy action is simply argmax_a V[next_state].
    policy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            best_a, best_v = 0, -1e9
            for a in range(NUM_ACTIONS):
                ns, _ = grid_step((r, c), a)
                nr, nc = ns
                if V[nr, nc] > best_v:
                    best_v = V[nr, nc]
                    best_a = a
            policy[r, c] = best_a
    return policy


# ======================================================================
# Step 0: Value Iteration
# ======================================================================


def _render_step_0(show_solutions=False):
    st.subheader("Step 0: Value Iteration")
    st.info(
        "**Goal**: implement the Bellman optimality backup to compute the optimal "
        "value function V* for the 5×5 gridworld."
    )

    # Always show the environment layout
    col_env, col_legend = st.columns([1, 1])
    with col_env:
        st.plotly_chart(_make_env_figure(), use_container_width=True)
    with col_legend:
        st.markdown(
            f"""
**Environment**

| Cell | Reward | Description |
|------|--------|-------------|
| → (blue) | 0 | Start `{START}` — agent begins here |
| 🎯 (green) | +1 | Goal `{GOAL}` — episode ends |
| ✗ (red) | −1 | Obstacles `(1,0)` and `(2,3)` |
| gray | 0 | Empty cells |

**Actions**: 0 = up · 1 = down · 2 = left · 3 = right

Hitting a wall keeps the agent in place.
Discount factor `GAMMA = {GAMMA}`.
"""
        )

    with st.expander("Value iteration — the Bellman backup"):
        st.markdown(
            r"""
**Value iteration** applies the Bellman optimality operator repeatedly until convergence:

$$V^*(s) = R(s) + \gamma \max_a V^*(s')$$

where $R(s)$ is the immediate reward for *being* in state $s$, and $s'$ is the
(deterministic) next state when taking action $a$.

Because the reward belongs to the current state:
- **Obstacle cells** have $R(s)=-1$, so their $V^*$ values are negative — visible in the heatmap.
- **Goal cell** is terminal: $V^*(\text{GOAL}) = R(\text{GOAL}) = +1$, set at init and skipped.

The algorithm sweeps over all non-terminal states, updating $V$ in-place, and stops when
the maximum change $\delta = \max_s |V_{\text{new}}(s) - V_{\text{old}}(s)|$ falls below $\theta$.
"""
        )

    st.markdown(
        """
**Your task**: fill in the two `...` lines inside `value_iteration`.

1. Compute `next_vals` — a list of `V[next_state]` for each of the 4 actions.
   Use `grid_step(state, a)` to get the next state (ignore the returned `done`).
2. Set `V[r, c] = REWARD_GRID[r, c] + gamma * max(next_vals)`.

> **Hint**: `V[GOAL]` is initialised to `REWARD_GRID[GOAL] = 1.0` before the loop
> and is kept fixed (terminal state). This value propagates backward through `next_vals`.
"""
    )

    student_code = """\
def value_iteration(gamma=GAMMA, theta=1e-4):
    V = np.zeros((GRID_SIZE, GRID_SIZE))
    V[GOAL] = REWARD_GRID[GOAL]  # terminal: value = immediate reward (+1)

    while True:
        delta = 0
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                state = (r, c)
                if state == GOAL:
                    continue  # terminal — keep fixed
                v = V[r, c]

                # TODO: collect V[next_state] for each of the NUM_ACTIONS actions.
                # Use grid_step(state, a) which returns (next_state, done).
                next_vals = ...

                # TODO: V[r, c] = R(s) + gamma * best next-state value
                V[r, c] = REWARD_GRID[r, c] + gamma * ...

                delta = max(delta, abs(v - V[r, c]))

        if delta < theta:
            break

    return V

V = value_iteration()
print(f"V at goal  {GOAL}:               {V[GOAL]:.3f}")
print(f"V at (3,4) — one step from goal: {V[3, 4]:.3f}")
print(f"V at start {START}:              {V[START]:.3f}")
print(f"V at (1,0) obstacle:             {V[1, 0]:.3f}")"""

    solution_code = """\
def value_iteration(gamma=GAMMA, theta=1e-4):
    V = np.zeros((GRID_SIZE, GRID_SIZE))
    V[GOAL] = REWARD_GRID[GOAL]  # terminal: value = immediate reward (+1)

    while True:
        delta = 0
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                state = (r, c)
                if state == GOAL:
                    continue
                v = V[r, c]

                next_vals = []
                for a in range(NUM_ACTIONS):
                    next_state, _ = grid_step(state, a)
                    nr, nc = next_state
                    next_vals.append(V[nr, nc])

                V[r, c] = REWARD_GRID[r, c] + gamma * max(next_vals)
                delta = max(delta, abs(v - V[r, c]))

        if delta < theta:
            break

    return V

V = value_iteration()
print(f"V at goal  {GOAL}:               {V[GOAL]:.3f}")
print(f"V at (3,4) — one step from goal: {V[3, 4]:.3f}")
print(f"V at start {START}:              {V[START]:.3f}")
print(f"V at (1,0) obstacle:             {V[1, 0]:.3f}")"""

    default_code = solution_code if show_solutions else student_code
    code = st_monaco(
        value=default_code, height="400px", language="python", theme="vs-dark"
    )

    if st.button("Run Step 0", key="lab10_run_0"):
        st.session_state["lab10_vars"] = dict(_PRE_INJECTED)
        result = _run_and_save(
            "lab10_step_0_result",
            code,
            st.session_state["lab10_vars"],
            check_step_0_value_iteration,
            "Running value iteration...",
        )
        if result["passed"]:
            st.session_state["lab10_step_0_done"] = True

    passed, _ = _show_result("lab10_step_0_result")
    if passed:
        lv = st.session_state["lab10_vars"]
        V = lv["V"]
        policy = _greedy_policy_from_V(V)

        col1, col2 = st.columns(2)
        with col1:
            fig = _make_grid_figure(V, "V* — Optimal Value Function")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = _make_grid_figure(
                V,
                "Greedy Policy — agent arrows, gold path",
                policy=policy,
                show_path=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
**Notice**: values increase as you approach 🎯, and obstacle cells depress nearby values.
The arrows show each cell's optimal action; the gold line traces the agent's path from
start to goal.
"""
        )


# ======================================================================
# Step 1: Q-Learning
# ======================================================================


def _render_step_1(show_solutions=False):
    st.divider()
    st.subheader("Step 1: Q-Learning")
    st.info(
        "**Goal**: implement the Q-learning update rule. The agent will learn an "
        "action-value table Q(s, a) purely from experience — no model of the environment."
    )

    with st.expander("Value iteration vs Q-learning — what's different?"):
        st.markdown(
            r"""
**Value iteration** (Step 0) required knowing the reward and next state for every
$(s, a)$ pair — it used a *model* of the environment.

**Q-learning** learns from *sampled transitions* collected by running the agent in the
environment. Using the same state-reward convention as value iteration:

$$Q(s, a) \;\leftarrow\; Q(s, a) \;+\; \alpha \Bigl[R(s) + v_\text{next} - Q(s, a)\Bigr]$$

where $v_\text{next} = R(s')$ if $s'$ is terminal (GOAL), otherwise $\gamma \max_{a'} Q(s', a')$.

The bracketed term is the **TD error**. With $\alpha \in (0,1)$ and an $\varepsilon$-greedy
policy the table converges to $Q^*$ (given enough visits to every $(s,a)$ pair).

**$\varepsilon$-greedy exploration**: with probability $\varepsilon$ take a random action;
otherwise take $\arg\max_a Q(s, a)$.
"""
        )

    st.markdown(
        """
**Your task**: fill in the three `...` blocks inside `q_learning`.

1. **$\\varepsilon$-greedy action selection** — with probability `epsilon` pick
   `np.random.randint(NUM_ACTIONS)`; otherwise pick `np.argmax(Q[r, c])`.
2. **`next_val`** — if `done`, use `REWARD_GRID[nr, nc]` (terminal bonus);
   otherwise use `gamma * Q[nr, nc].max()`.
3. **Q-learning update** — `Q[r, c, action] += alpha * (REWARD_GRID[r, c] + next_val - Q[r, c, action])`.
"""
    )

    student_code = """\
def q_learning(gamma=GAMMA, alpha=0.5, epsilon=0.1, n_episodes=3000):
    Q = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS))
    episode_returns = []

    for episode in range(n_episodes):
        state = START
        done = False
        total_return = 0.0

        while not done:
            r, c = state

            # TODO: epsilon-greedy action selection
            if ...:
                action = ...
            else:
                action = ...

            next_state, done = grid_step(state, action)
            nr, nc = next_state

            # TODO: next_val = terminal reward if done, else discounted max Q
            next_val = ...

            # TODO: Q-learning update using REWARD_GRID[r, c] as the current reward
            Q[r, c, action] = ...

            total_return += REWARD_GRID[r, c]
            state = next_state

        total_return += REWARD_GRID[state]
        episode_returns.append(total_return)

    return Q, episode_returns

Q, episode_returns = q_learning()
print(f"max Q at goal  {GOAL}:  {Q[GOAL].max():.3f}")
print(f"max Q at start {START}: {Q[START].max():.3f}")
print(f"Mean return (last 500 eps): {np.mean(episode_returns[-500:]):.2f}")"""

    solution_code = """\
def q_learning(gamma=GAMMA, alpha=0.5, epsilon=0.1, n_episodes=3000):
    Q = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS))
    episode_returns = []

    for episode in range(n_episodes):
        state = START
        done = False
        total_return = 0.0

        while not done:
            r, c = state

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(NUM_ACTIONS)
            else:
                action = np.argmax(Q[r, c])

            next_state, done = grid_step(state, action)
            nr, nc = next_state

            # Use terminal reward when done, otherwise discounted max Q
            next_val = REWARD_GRID[nr, nc] if done else gamma * Q[nr, nc].max()

            Q[r, c, action] = Q[r, c, action] + alpha * (
                REWARD_GRID[r, c] + next_val - Q[r, c, action]
            )

            total_return += REWARD_GRID[r, c]
            state = next_state

        total_return += REWARD_GRID[state]
        episode_returns.append(total_return)

    return Q, episode_returns

Q, episode_returns = q_learning()
print(f"max Q at goal  {GOAL}:  {Q[GOAL].max():.3f}")
print(f"max Q at start {START}: {Q[START].max():.3f}")
print(f"Mean return (last 500 eps): {np.mean(episode_returns[-500:]):.2f}")"""

    default_code = solution_code if show_solutions else student_code
    code = st_monaco(
        value=default_code, height="460px", language="python", theme="vs-dark"
    )

    if st.button("Run Step 1", key="lab10_run_1"):
        result = _run_and_save(
            "lab10_step_1_result",
            code,
            st.session_state["lab10_vars"],
            check_step_1_q_learning,
            "Running Q-learning (3000 episodes)...",
        )
        if result["passed"]:
            st.session_state["lab10_step_1_done"] = True

    passed, _ = _show_result("lab10_step_1_result")
    if passed:
        lv = st.session_state["lab10_vars"]
        Q = lv["Q"]
        V_from_Q = Q.max(axis=2)
        policy_q = np.argmax(Q, axis=2)
        episode_returns = lv.get("episode_returns", [])

        col1, col2 = st.columns(2)
        with col1:
            fig = _make_grid_figure(
                V_from_Q, "max_a Q(s,a) — Q-Learning Value Estimate"
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = _make_grid_figure(
                V_from_Q,
                "Greedy Policy from Q — gold path",
                policy=policy_q,
                show_path=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        if episode_returns:
            window = 100
            smoothed = [
                np.mean(episode_returns[max(0, i - window) : i + 1])
                for i in range(len(episode_returns))
            ]
            ret_df = {
                "Episode": list(range(1, len(episode_returns) + 1)),
                "Return": episode_returns,
                "Smoothed (100-ep avg)": smoothed,
            }
            ret_df = pd.DataFrame(ret_df)
            fig = px.line(
                ret_df,
                x="Episode",
                y=["Return", "Smoothed (100-ep avg)"],
                title="Q-Learning: Episode Returns",
                labels={"value": "Total Return", "variable": ""},
                color_discrete_map={
                    "Return": "rgba(74,144,217,0.3)",
                    "Smoothed (100-ep avg)": "#E74C3C",
                },
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
**Compare with Step 0**: `max_a Q(s,a)` should look very similar to `V*`. Q-learning
recovers the same policy without ever querying a model — it only needed sampled transitions.
"""
        )


# ======================================================================
# Step 2: Q-Network for CartPole
# ======================================================================


def _render_step_2(show_solutions=False):
    st.divider()
    st.subheader("Step 2: Define the Q-Network (CartPole)")
    st.info(
        "**Goal**: define a neural network that takes a CartPole state vector "
        "and outputs one Q-value per action."
    )

    with st.expander("From tabular Q to deep Q-networks"):
        st.markdown(
            r"""
The gridworld had only 25 states — a table $Q \in \mathbb{R}^{25 \times 4}$ was fine.

**CartPole-v1** has a *continuous* 4-dimensional state:
$[\text{position},\; \text{velocity},\; \text{pole angle},\; \text{angular velocity}]$.
There are 2 discrete actions: push left or push right.

Instead of a table we use a neural network $Q_\theta(s) \in \mathbb{R}^2$ that outputs
a Q-value for each action.  The same TD update applies — we just use gradient descent
on the loss rather than a direct assignment.

**Architecture**: a two-layer MLP with ReLU activations is more than enough for CartPole.
"""
        )

    st.markdown(
        """
**Your task**: implement `QNetwork`.

- `__init__`: define a 2-layer MLP (`nn.Sequential`) with hidden size `hidden`.
  Input dimension = `state_dim` (4 for CartPole), output dimension = `action_dim` (2).
- `forward`: pass `x` through `self.net` and return the result.
"""
    )

    student_code = """\
class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=128):
        super().__init__()
        # TODO: Define a 2-layer MLP with ReLU activations.
        # Layer sizes: state_dim -> hidden -> hidden -> action_dim
        self.net = ...

    def forward(self, x):
        # TODO: pass x through self.net
        return ...

# Quick sanity check
net = QNetwork()
x_test = torch.randn(16, 4)
out = net(x_test)
print(f"Input shape:  {tuple(x_test.shape)}")
print(f"Output shape: {tuple(out.shape)}  (expected: (16, 2))")
print(f"Parameters:   {sum(p.numel() for p in net.parameters()):,}")"""

    solution_code = """\
class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)

# Quick sanity check
net = QNetwork()
x_test = torch.randn(16, 4)
out = net(x_test)
print(f"Input shape:  {tuple(x_test.shape)}")
print(f"Output shape: {tuple(out.shape)}  (expected: (16, 2))")
print(f"Parameters:   {sum(p.numel() for p in net.parameters()):,}")"""

    default_code = solution_code if show_solutions else student_code
    code = st_monaco(
        value=default_code, height="320px", language="python", theme="vs-dark"
    )

    if st.button("Check QNetwork", key="lab10_run_2"):
        exec_vars = dict(st.session_state["lab10_vars"])
        result = _run_and_save(
            "lab10_step_2_result",
            code,
            exec_vars,
            check_step_2_qnetwork,
            "Checking architecture...",
        )
        if result["passed"]:
            st.session_state["lab10_vars"].update(exec_vars)
            st.session_state["lab10_step_2_done"] = True

    _show_result("lab10_step_2_result")


# ======================================================================
# CartPole interactive widget (HTML / JS / Canvas)
# ======================================================================

_CARTPOLE_HTML = Template(r"""
<div style="background:#1a1a2e;border-radius:8px;padding:12px;text-align:center;font-family:monospace;">
  <canvas id="cpCanvas" width="600" height="280"
          style="display:block;margin:0 auto;border-radius:4px;cursor:pointer;"></canvas>
  <div id="cpOverlay" style="color:#aaa;margin-top:6px;font-size:13px;min-height:20px;"></div>
</div>
<script>
(function(){
  var MODE = "$mode";
  var STATES = $states;
  var canvas = document.getElementById("cpCanvas");
  var ctx = canvas.getContext("2d");
  var info = document.getElementById("cpOverlay");
  var W = canvas.width, H = canvas.height;

  /* ── physics (matches gymnasium CartPole-v1, euler integrator) ── */
  var G=9.8, MC=1.0, MP=0.1, TM=1.1, L=0.5, PML=0.05, FM=10.0, DT=0.02;
  var XMAX=2.4, TMAX=12*Math.PI/180, MAXSTEPS=500;

  var s=[0,0,0,0], steps=0, done=false, started=false, best=0;
  var action=0, leftHeld=false, rightHeld=false;

  function reset(){
    s=[(Math.random()-.5)*.1,(Math.random()-.5)*.1,
       (Math.random()-.5)*.1,(Math.random()-.5)*.1];
    steps=0; done=false;
  }
  function phys(a){
    var x=s[0],xd=s[1],th=s[2],thd=s[3];
    var f=a===1?FM:-FM, ct=Math.cos(th), st=Math.sin(th);
    var tmp=(f+PML*thd*thd*st)/TM;
    var tha=(G*st-ct*tmp)/(L*(4/3-MP*ct*ct/TM));
    var xa=tmp-PML*tha*ct/TM;
    s=[x+DT*xd, xd+DT*xa, th+DT*thd, thd+DT*tha];
    steps++;
    if(Math.abs(s[0])>XMAX||Math.abs(s[2])>TMAX||steps>=MAXSTEPS) done=true;
  }

  /* ── rendering ── */
  function draw(st){
    var x=st[0], th=st[2];
    var scale=W/5.6, cx=W/2+x*scale, cy=H*0.72;
    ctx.fillStyle="#1a1a2e"; ctx.fillRect(0,0,W,H);

    /* track */
    ctx.strokeStyle="#444"; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(0,cy+22); ctx.lineTo(W,cy+22); ctx.stroke();

    /* threshold markers */
    ctx.strokeStyle="#333"; ctx.setLineDash([4,4]);
    var lx=W/2-XMAX*scale, rx=W/2+XMAX*scale;
    ctx.beginPath();
    ctx.moveTo(lx,cy-50); ctx.lineTo(lx,cy+22);
    ctx.moveTo(rx,cy-50); ctx.lineTo(rx,cy+22);
    ctx.stroke(); ctx.setLineDash([]);

    /* cart */
    var cw=50, ch=30;
    ctx.fillStyle="#2196F3";
    ctx.fillRect(cx-cw/2, cy-ch/2, cw, ch);

    /* wheels */
    ctx.fillStyle="#555";
    ctx.beginPath(); ctx.arc(cx-15,cy+ch/2+5,7,0,6.3); ctx.fill();
    ctx.beginPath(); ctx.arc(cx+15,cy+ch/2+5,7,0,6.3); ctx.fill();

    /* pole */
    var pLen=L*2*scale;
    var px=cx+pLen*Math.sin(th), py=cy-pLen*Math.cos(th);
    ctx.strokeStyle="#E74C3C"; ctx.lineWidth=6; ctx.lineCap="round";
    ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(px,py); ctx.stroke();

    /* pivot dot */
    ctx.fillStyle="#FFD700";
    ctx.beginPath(); ctx.arc(cx,cy,4,0,6.3); ctx.fill();
  }

  function showSteps(n, extra){
    ctx.fillStyle="#ddd"; ctx.font="16px monospace"; ctx.textAlign="left";
    ctx.fillText("Steps: "+n, 10, 22);
    if(extra){ ctx.textAlign="right"; ctx.fillText(extra, W-10, 22); }
  }

  /* ── interactive mode ── */
  if(MODE==="interactive"){
    reset(); draw(s); showSteps(0);
    info.textContent="Click here, then press \u2190 or \u2192 to start!";

    document.body.tabIndex=0;
    canvas.addEventListener("click",function(){ document.body.focus(); });

    document.addEventListener("keydown",function(e){
      if(e.key==="ArrowLeft"){ leftHeld=true; action=0; e.preventDefault(); }
      if(e.key==="ArrowRight"){ rightHeld=true; action=1; e.preventDefault(); }
      if(!started && !done && (e.key==="ArrowLeft"||e.key==="ArrowRight")){
        started=true; info.textContent=""; gameLoop();
      }
      if(done && e.key===" "){ reset(); started=true; info.textContent=""; gameLoop(); e.preventDefault(); }
    });
    document.addEventListener("keyup",function(e){
      if(e.key==="ArrowLeft") leftHeld=false;
      if(e.key==="ArrowRight") rightHeld=false;
    });

    function gameLoop(){
      if(done){
        if(steps>best) best=steps;
        draw(s); showSteps(steps, "Best: "+best);
        if(steps>=MAXSTEPS) info.innerHTML="<span style='color:#2ecc71'>Perfect! 500 steps! You're a natural.</span>";
        else info.innerHTML="Pole fell after <b>"+steps+"</b> steps. Press <b>Space</b> to retry.";
        return;
      }
      if(leftHeld) action=0;
      if(rightHeld) action=1;
      phys(action);
      draw(s); showSteps(steps, "Best: "+best);
      setTimeout(gameLoop, 20);
    }
  }

  /* ── replay mode ── */
  if(MODE==="replay"){
    var idx=0;
    draw(STATES[0]); showSteps(0);
    info.textContent="Playing trained DQN policy...";

    function replay(){
      if(idx<STATES.length){
        draw(STATES[idx]); showSteps(idx, "/ "+STATES.length);
        idx++;
        setTimeout(replay, 20);
      } else {
        info.innerHTML="<span style='color:#2ecc71'>DQN balanced for <b>"+(STATES.length-1)+"</b> steps.</span>  "
          + "<a href='#' id='cpReplay' style='color:#4a90d9'>Replay</a>";
        document.getElementById("cpReplay").addEventListener("click",function(e){
          e.preventDefault(); idx=0; replay();
        });
      }
    }
    setTimeout(replay, 500);
  }
})();
</script>
""")


def _cartpole_widget(mode="interactive", recorded_states=None):
    """Embed an HTML/JS CartPole visualization via streamlit components."""
    import streamlit.components.v1 as components

    states_json = json.dumps(recorded_states) if recorded_states else "null"
    html = _CARTPOLE_HTML.safe_substitute(mode=mode, states=states_json)
    components.html(html, height=340)


# ======================================================================
# Step 3: Vanilla DQN (Naive)
# ======================================================================


def _render_step_3(show_solutions=False):
    st.divider()
    st.subheader("Step 3: Vanilla Deep Q-Learning (Naive)")
    st.info(
        "**Goal**: first try the most direct neural-network version of Q-learning. "
        "This is the vanilla idea before DQN's stabilization tricks."
    )

    with st.expander("Try it yourself — can you balance the pole?", expanded=True):
        st.markdown(
            "Use **left/right arrow keys** to push the cart. "
            "How long can you keep the pole upright? "
            "The DQN agent you'll train below will try to reach **500** steps."
        )
        _cartpole_widget("interactive")

    with st.expander("Vanilla deep Q-learning — the tempting first attempt"):
        st.markdown(
            r"""
Tabular Q-learning updated one table entry:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \bigl[r + \gamma \max_{a'}Q(s',a') - Q(s,a)\bigr]$$

The naive neural version keeps the same Bellman target, but replaces the table with
a network $Q_\theta(s)$ and uses gradient descent:

$$\mathcal{L}(\theta) = \bigl(Q_\theta(s,a) - [r + \gamma \max_{a'} Q_\theta(s',a')]\bigr)^2$$

This is the "vanilla" part of DQN:

| Piece | Vanilla implementation |
|-------|------------------------|
| Q-function | `policy_net(state)` outputs one value per action |
| Action selection | epsilon-greedy over `policy_net` |
| TD target | reward plus discounted best next-action value |
| Optimizer step | backpropagate the TD error |

But this version trains on one transition at a time and uses the same network to both
predict the current Q-value and define the target. On CartPole it is often unstable:
returns may jump around, plateau early, or collapse after briefly improving.
"""
        )

    st.markdown(
        """
**Your task**: implement `compute_vanilla_loss` (the three `...` blocks).
The training loop below is intentionally naive — do not add replay or a target network here.

1. **`q_value`**: call `policy_net(state.unsqueeze(0))` and select the Q-value for `action`.
2. **`next_q_value`**: call the same `policy_net(next_state.unsqueeze(0))` inside `torch.no_grad()`
   and take the max over actions.
3. **`loss`**: use `F.mse_loss(q_value, target)`.
"""
    )

    student_code = """\
def compute_vanilla_loss(transition, policy_net, gamma=0.99):
    state, action, reward, next_state, done = transition

    # TODO: Q_theta(s, a) for the action actually taken.
    # policy_net(state.unsqueeze(0)) has shape (1, 2).
    q_value = ...

    # TODO: Naive Bellman target using the SAME network for next-state values.
    # Detach the target with torch.no_grad(); it is still a moving target because
    # policy_net changes after every optimizer step.
    with torch.no_grad():
        next_q_value = ...
        target = reward + gamma * next_q_value * (1 - done)

    # TODO: Squared TD error
    loss = ...
    return loss


# ── Naive online training loop (provided — do not add tricks here) ─────
import gymnasium as gym

env = gym.make("CartPole-v1")
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = QNetwork(state_dim=state_dim, action_dim=action_dim)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)

EPSILON_START, EPSILON_END, EPSILON_DECAY = 1.0, 0.05, 0.98
epsilon = EPSILON_START
episode_returns = []

for ep in range(150):
    obs, _ = env.reset()
    state = torch.tensor(obs, dtype=torch.float32)
    total_reward = 0.0
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state.unsqueeze(0)).argmax().item()

        obs2, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = torch.tensor(obs2, dtype=torch.float32)

        transition = (state, action, float(reward), next_state, float(done))
        loss = compute_vanilla_loss(transition, policy_net)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    episode_returns.append(total_reward)
    if (ep + 1) % 25 == 0:
        mean_r = np.mean(episode_returns[-25:])
        print(f"Episode {ep+1:3d} | Mean return (last 25): {mean_r:6.1f} | ε={epsilon:.3f}")

env.close()
print(f"\\nNaive final mean return (last 25 eps): {np.mean(episode_returns[-25:]):.1f}")"""

    solution_code = """\
def compute_vanilla_loss(transition, policy_net, gamma=0.99):
    state, action, reward, next_state, done = transition

    q_value = policy_net(state.unsqueeze(0))[0, action]

    with torch.no_grad():
        next_q_value = policy_net(next_state.unsqueeze(0)).max()
        target = reward + gamma * next_q_value * (1 - done)

    loss = F.mse_loss(q_value, target)
    return loss


# ── Naive online training loop (provided — do not add tricks here) ─────
import gymnasium as gym

env = gym.make("CartPole-v1")
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = QNetwork(state_dim=state_dim, action_dim=action_dim)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)

EPSILON_START, EPSILON_END, EPSILON_DECAY = 1.0, 0.05, 0.98
epsilon = EPSILON_START
episode_returns = []

for ep in range(150):
    obs, _ = env.reset()
    state = torch.tensor(obs, dtype=torch.float32)
    total_reward = 0.0
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state.unsqueeze(0)).argmax().item()

        obs2, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = torch.tensor(obs2, dtype=torch.float32)

        transition = (state, action, float(reward), next_state, float(done))
        loss = compute_vanilla_loss(transition, policy_net)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    episode_returns.append(total_reward)
    if (ep + 1) % 25 == 0:
        mean_r = np.mean(episode_returns[-25:])
        print(f"Episode {ep+1:3d} | Mean return (last 25): {mean_r:6.1f} | ε={epsilon:.3f}")

env.close()
print(f"\\nNaive final mean return (last 25 eps): {np.mean(episode_returns[-25:]):.1f}")"""

    default_code = solution_code if show_solutions else student_code
    code = st_monaco(
        value=default_code, height="560px", language="python", theme="vs-dark"
    )

    if st.button("Run Naive DQN", key="lab10_run_3"):
        exec_vars = dict(st.session_state["lab10_vars"])
        with st.spinner(
            "Training naive online DQN on CartPole (150 episodes — this may take ~30 seconds)..."
        ):
            output, tb = _exec_with_capture(code, exec_vars)

        result = {"output": output, "tb": tb, "passed": False, "msg": ""}
        if tb is None and "compute_vanilla_loss" not in exec_vars:
            result["msg"] = "Function `compute_vanilla_loss` not found."
        elif tb is None and len(exec_vars.get("episode_returns", [])) < 50:
            result["msg"] = (
                "The naive training loop did not record enough episode returns. "
                "Make sure the provided loop still runs."
            )
        elif tb is None:
            st.session_state["lab10_vars"].update(exec_vars)
            st.session_state["lab10_step_3_done"] = True
            episode_returns = exec_vars.get("episode_returns", [])
            st.session_state["lab10_naive_episode_returns"] = list(episode_returns)
            mean_last = np.mean(episode_returns[-25:])
            result["passed"] = True
            if mean_last < 195:
                result["msg"] = (
                    f"Naive DQN finished. Mean return over the last 25 episodes = {mean_last:.1f}. "
                    "It has not solved CartPole, which is the instability this step is meant to expose."
                )
            else:
                result["msg"] = (
                    f"Naive DQN finished. Mean return over the last 25 episodes = {mean_last:.1f}. "
                    "CartPole is forgiving enough that naive DQN can occasionally get lucky, "
                    "but this setup is still much less reliable than the stabilized version."
                )
        st.session_state["lab10_step_3_result"] = result

    result = st.session_state.get("lab10_step_3_result")
    if result is not None:
        if result["output"]:
            st.code(result["output"], language="text")
        if result["tb"]:
            st.error("Runtime Error")
            st.code(result["tb"], language="python")
        elif not result["passed"]:
            st.error(result["msg"])
        else:
            st.warning(result["msg"])
            episode_returns = st.session_state.get("lab10_naive_episode_returns", [])
            if episode_returns:
                window = 20
                smoothed = [
                    np.mean(episode_returns[max(0, i - window) : i + 1])
                    for i in range(len(episode_returns))
                ]
                ret_df = pd.DataFrame(
                    {
                        "Episode": list(range(1, len(episode_returns) + 1)),
                        "Return": episode_returns,
                        f"Smoothed ({window}-ep avg)": smoothed,
                    }
                )
                fig = px.line(
                    ret_df,
                    x="Episode",
                    y=["Return", f"Smoothed ({window}-ep avg)"],
                    title="Naive Online DQN on CartPole-v1",
                    labels={"value": "Total Return", "variable": ""},
                    color_discrete_map={
                        "Return": "rgba(74,144,217,0.3)",
                        f"Smoothed ({window}-ep avg)": "#E74C3C",
                    },
                )
                fig.add_hline(
                    y=195,
                    line_dash="dash",
                    line_color="gold",
                    annotation_text="Solved (195)",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
**What to notice**: this code has the right Bellman idea, but the optimization problem is
poorly behaved. The next step keeps the same vanilla pieces and adds the engineering tricks
that make DQN train reliably.
"""
            )


# ======================================================================
# Step 4: DQN Tricks + Training
# ======================================================================


def _render_step_4(show_solutions=False):
    st.divider()
    st.subheader("Step 4: DQN Tricks — Replay Buffer and Target Network")
    st.info(
        "**Goal**: keep the vanilla DQN loss, then add experience replay and a "
        "target network so training becomes much more stable."
    )

    with st.expander("Why the naive version breaks"):
        st.markdown(
            r"""
Naively applying neural-network function approximation to Q-learning is unstable because:

1. **Correlated updates** — consecutive transitions are highly correlated; gradient steps
   on one region of the state space overwrite what was learned elsewhere.
2. **Moving targets** — the Bellman target $r + \gamma \max_{a'} Q_\theta(s', a')$ changes
   every time we update $\theta$, creating a feedback loop.

**DQN** (Mnih et al., 2015) fixes both with two additions:

| Trick | How it helps |
|-------|-------------|
| **Experience replay** | Store transitions in a buffer; sample random mini-batches to break correlation |
| **Target network** | Keep a *frozen* copy $Q_{\bar\theta}$ for computing targets; sync periodically |

The code also uses two small practical stabilizers: Huber loss instead of raw squared
error, and gradient clipping to avoid very large updates.

The loss for a mini-batch of transitions $(s, a, r, s', \text{done})$:

$$\mathcal{L}(\theta) = \frac{1}{B}\sum_{i=1}^{B}\Bigl(Q_\theta(s_i, a_i) - \underbrace{\bigl[r_i + \gamma \max_{a'} Q_{\bar{\theta}}(s'_i, a')\bigr]}_{\text{Bellman target (stop gradient)}}\Bigr)^2$$

Note: targets are computed with **no gradient** through $Q_{\bar\theta}$.
"""
        )

    st.markdown(
        """
**Your task**: implement `compute_loss` (the three `...` blocks).
The training loop below is provided — do not modify it.

1. **`q_values`**: call `policy_net(states)` and use `.gather` to select the Q-value
   for each action that was actually taken.
2. **`next_q_values`**: call `target_net(next_states)` inside `torch.no_grad()` and
   take the max over actions.
3. **`loss`**: `F.smooth_l1_loss(q_values, targets)` (Huber loss).
"""
    )

    student_code = """\
def compute_loss(batch, policy_net, target_net, gamma=0.99):
    states, actions, rewards, next_states, dones = batch

    # TODO: Get Q-values for the actions that were actually taken.
    # policy_net(states) has shape (B, 2); use .gather(1, ...) to select
    # the Q-value for each taken action, then .squeeze(1) to get shape (B,).
    q_values = ...  # shape: (B,)

    # TODO: Compute Bellman targets using the frozen target network.
    # target_net(next_states) has shape (B, 2); take max over dim=1.
    # Multiply by (1 - dones) to zero out targets for terminal transitions.
    with torch.no_grad():
        next_q_values = ...  # shape: (B,)

    targets = rewards + gamma * next_q_values * (1 - dones)

    # TODO: Huber loss between q_values and targets
    loss = ...
    return loss


# ── Training loop (provided — do not modify) ──────────────────────────
import gymnasium as gym

env = gym.make("CartPole-v1")
state_dim  = env.observation_space.shape[0]   # 4
action_dim = env.action_space.n               # 2

policy_net = QNetwork(state_dim=state_dim, action_dim=action_dim)
target_net = QNetwork(state_dim=state_dim, action_dim=action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
memory    = ReplayBuffer(capacity=10_000)

BATCH_SIZE   = 64
TARGET_UPDATE = 10
EPSILON_START, EPSILON_END, EPSILON_DECAY = 1.0, 0.05, 0.99
epsilon = EPSILON_START

episode_returns = []

for ep in range(300):
    obs, _ = env.reset()
    state  = torch.tensor(obs, dtype=torch.float32)
    total_reward = 0.0
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state.unsqueeze(0)).argmax().item()

        obs2, reward, terminated, truncated, _ = env.step(action)
        done       = terminated or truncated
        next_state = torch.tensor(obs2, dtype=torch.float32)

        memory.push(state, action, reward, next_state, float(done))
        state        = next_state
        total_reward += reward

        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            loss  = compute_loss(batch, policy_net, target_net)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
            optimizer.step()

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    episode_returns.append(total_reward)
    if (ep + 1) % 50 == 0:
        mean_r = np.mean(episode_returns[-50:])
        print(f"Episode {ep+1:3d} | Mean return (last 50): {mean_r:6.1f} | ε={epsilon:.3f}")

env.close()
print(f"\\nFinal mean return (last 50 eps): {np.mean(episode_returns[-50:]):.1f}")"""

    solution_code = """\
def compute_loss(batch, policy_net, target_net, gamma=0.99):
    states, actions, rewards, next_states, dones = batch

    # Q-values for the taken actions
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Bellman targets from frozen target network
    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1).values

    targets = rewards + gamma * next_q_values * (1 - dones)

    loss = F.smooth_l1_loss(q_values, targets)
    return loss


# ── Training loop (provided — do not modify) ──────────────────────────
import gymnasium as gym

env = gym.make("CartPole-v1")
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = QNetwork(state_dim=state_dim, action_dim=action_dim)
target_net = QNetwork(state_dim=state_dim, action_dim=action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
memory    = ReplayBuffer(capacity=10_000)

BATCH_SIZE   = 64
TARGET_UPDATE = 10
EPSILON_START, EPSILON_END, EPSILON_DECAY = 1.0, 0.05, 0.99
epsilon = EPSILON_START

episode_returns = []

for ep in range(300):
    obs, _ = env.reset()
    state  = torch.tensor(obs, dtype=torch.float32)
    total_reward = 0.0
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state.unsqueeze(0)).argmax().item()

        obs2, reward, terminated, truncated, _ = env.step(action)
        done       = terminated or truncated
        next_state = torch.tensor(obs2, dtype=torch.float32)

        memory.push(state, action, reward, next_state, float(done))
        state        = next_state
        total_reward += reward

        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            loss  = compute_loss(batch, policy_net, target_net)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
            optimizer.step()

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    episode_returns.append(total_reward)
    if (ep + 1) % 50 == 0:
        mean_r = np.mean(episode_returns[-50:])
        print(f"Episode {ep+1:3d} | Mean return (last 50): {mean_r:6.1f} | ε={epsilon:.3f}")

env.close()
print(f"\\nFinal mean return (last 50 eps): {np.mean(episode_returns[-50:]):.1f}")"""

    default_code = solution_code if show_solutions else student_code
    code = st_monaco(
        value=default_code, height="560px", language="python", theme="vs-dark"
    )

    if st.button("Run Stabilized DQN", key="lab10_run_4"):
        st.session_state.pop("lab10_dqn_replay_states", None)
        result = _run_and_save(
            "lab10_step_4_result",
            code,
            st.session_state["lab10_vars"],
            check_step_3_dqn,
            "Training DQN on CartPole (300 episodes — this may take ~1 minute)...",
        )
        if result["passed"]:
            st.session_state["lab10_step_4_done"] = True

    passed, _ = _show_result("lab10_step_4_result")
    if passed:
        lv = st.session_state["lab10_vars"]
        episode_returns = lv.get("episode_returns", [])
        if episode_returns:
            window = 20
            smoothed = [
                np.mean(episode_returns[max(0, i - window) : i + 1])
                for i in range(len(episode_returns))
            ]
            ret_df = pd.DataFrame(
                {
                    "Episode": list(range(1, len(episode_returns) + 1)),
                    "Return": episode_returns,
                    f"Smoothed ({window}-ep avg)": smoothed,
                }
            )
            fig = px.line(
                ret_df,
                x="Episode",
                y=["Return", f"Smoothed ({window}-ep avg)"],
                title="DQN on CartPole-v1: Episode Returns",
                labels={"value": "Total Return", "variable": ""},
                color_discrete_map={
                    "Return": "rgba(74,144,217,0.3)",
                    f"Smoothed ({window}-ep avg)": "#E74C3C",
                },
            )
            fig.add_hline(
                y=195,
                line_dash="dash",
                line_color="gold",
                annotation_text="Solved (195)",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # ── Run trained policy and show replay ──
        policy_net = lv.get("policy_net")
        if policy_net is not None:
            if "lab10_dqn_replay_states" not in st.session_state:
                env = gym.make("CartPole-v1")
                obs, _ = env.reset(seed=42)
                replay_states = [obs.tolist()]
                ep_done = False
                while not ep_done:
                    state_t = torch.tensor(obs, dtype=torch.float32)
                    with torch.no_grad():
                        act = policy_net(state_t.unsqueeze(0)).argmax().item()
                    obs, _, terminated, truncated, _ = env.step(act)
                    ep_done = terminated or truncated
                    replay_states.append(obs.tolist())
                env.close()
                st.session_state["lab10_dqn_replay_states"] = replay_states

            st.markdown("**Watch your trained DQN agent:**")
            _cartpole_widget(
                "replay", st.session_state["lab10_dqn_replay_states"]
            )


# ======================================================================
# Part 3: Discussion — RL and Language Models
# ======================================================================


def _render_part_3():
    st.divider()
    st.subheader("Part 3: RL Formalisms and Language Models")

    st.markdown(
        """
No implementation here — just a discussion worth thinking through carefully.

You just trained a DQN agent on a 4-dimensional continuous state space with 2 actions.
Modern large language models are trained with reinforcement learning too. The formalism
is identical; only the scale is different. Let's map the pieces.
"""
    )

    with st.expander("What is the state?"):
        st.markdown(
            """
In CartPole, the state was a 4-dimensional vector: position, velocity, angle, angular
velocity.

In a language model, the **state** is the full context window — every token that has
appeared so far, including the system prompt, the user message, and all tokens the model
has generated. This is a sequence that can be thousands of tokens long.

The model's internal representation (the key-value cache, the hidden states) is what
the "policy network" operates on — analogous to `policy_net(state)` in the DQN steps.
"""
        )

    with st.expander("What is the action space?"):
        st.markdown(
            """
In CartPole, the agent chose from **2 actions**: push left or push right.

In a language model, the **action** at each step is selecting the next token from a
vocabulary of ~50,000 tokens. This is an enormous discrete action space.

Things get more interesting if you define actions at a higher level of abstraction:
a *reasoning step*, a *tool call*, or a full *chain of thought* could each be an action.
The choice of action granularity affects what the Q-function must represent.
"""
        )

    with st.expander("What is the reward?"):
        st.markdown(
            """
In CartPole the reward was simple: +1 for every timestep the pole stays upright.

In language model training, reward is harder to define:

- **RLHF (Reinforcement Learning from Human Feedback)**: a separate *reward model* is
  trained on human preference comparisons. The LLM then gets a reward signal for each
  complete response.

- **GRPO / verifiable rewards** (DeepSeek-R1, OpenAI o-series): for tasks with a
  ground-truth answer (math, code), the reward is binary — correct or not.
  No human labels needed at inference time.

- **Process Reward Models (PRMs)**: instead of one reward at the end, a PRM scores
  *each reasoning step*. This is dense reward — closer to the CartPole +1/step setup —
  but expensive to collect labels for.
"""
        )

    with st.expander("What is the Q-value of a chain of thought?"):
        st.markdown(
            r"""
This is the central hard question.

In the gridworld, $Q(s, a)$ was the expected cumulative reward when taking action $a$
from state $s$ and then following the optimal policy. We could store this exactly in
a table because the state and action spaces were finite and small.

For a language model, $Q(\text{context}, \text{next token})$ would be the expected
**future reward** (e.g., answer correctness) given the current context and the next
token choice. Estimating this requires:

1. **Credit assignment over a long horizon** — a wrong turn in step 3 of a 20-step
   chain of thought may only become apparent at step 20.
2. **A very large action space** — 50,000 tokens, so you cannot enumerate all $Q(s, a)$
   values explicitly the way we did in the gridworld.
3. **A reward signal that may be binary and delayed** — making the TD error noisy.

One active research direction is using a *process reward model* as a learned
$Q$-function approximator for intermediate steps, but this remains an open problem.
"""
        )

    with st.expander("Why policy gradient, not Q-learning, for LLMs?"):
        st.markdown(
            r"""
Most LLM RL fine-tuning uses **policy gradient** methods (PPO, REINFORCE, GRPO)
rather than Q-learning. Why?

| | Q-learning (DQN) | Policy gradient (PPO) |
|---|---|---|
| **Action space** | Works well for small discrete spaces (2 in CartPole) | Scales naturally to large discrete or continuous spaces |
| **What is learned** | A value function $Q(s,a)$ | Directly the policy $\pi_\theta(a \mid s)$ |
| **Sample efficiency** | Higher (off-policy, replay buffer) | Lower (on-policy — each sample used once) |
| **Stability tricks** | Target network, replay buffer | Clipping (PPO), KL penalty |

With a 50,000-token vocabulary, computing $\max_{a'} Q(s', a')$ would require 50,000
forward passes per step — infeasible. Policy gradient methods avoid this by directly
differentiating through the log-probability of sampled actions.
"""
        )

    st.markdown(
        """
**Closing thought**

The vocabulary you built in this lab — state, action, reward, Bellman equation,
TD error, policy, value function — is exactly the vocabulary researchers use when
designing training recipes for GPT-4, Gemini, Claude, and DeepSeek-R1.

The challenges scale dramatically (sparse reward, enormous action spaces, long credit
assignment horizons), but the underlying formalism is what you just implemented
in a 5×5 grid and a CartPole environment.
"""
    )


# ======================================================================
# Main entry point
# ======================================================================


def render_rl_lab(show_solutions=False):
    st.title("Lab 10: Reinforcement Learning")
    st.markdown(
        """
**Overview**

| Part | Topic | Key concept |
|------|-------|-------------|
| Part 1 | 5×5 GridWorld | Value iteration, Q-learning |
| Part 2 | CartPole-v1 | Naive DQN, then replay + target network |
| Part 3 | Discussion | RL formalisms in language models |
"""
    )

    if "lab10_vars" not in st.session_state:
        st.session_state["lab10_vars"] = dict(_PRE_INJECTED)

    st.markdown("## Part 1 — GridWorld")
    _render_step_0(show_solutions)

    if st.session_state.get("lab10_step_0_done"):
        _render_step_1(show_solutions)

    st.markdown("## Part 2 — Deep Q-Network on CartPole")
    if st.session_state.get("lab10_step_1_done"):
        _render_step_2(show_solutions)
    else:
        st.info("Complete Part 1 (Steps 0 and 1) to unlock Part 2.")

    if st.session_state.get("lab10_step_2_done"):
        _render_step_3(show_solutions)

    if st.session_state.get("lab10_step_3_done"):
        _render_step_4(show_solutions)

    st.markdown("## Part 3 — Discussion")
    if st.session_state.get("lab10_step_4_done"):
        _render_part_3()
    else:
        st.info("Complete Parts 1 and 2 to unlock the discussion.")
