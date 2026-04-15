import numpy as np
import torch

GRID_SIZE = 5
NUM_ACTIONS = 4
GOAL = (4, 4)
OBSTACLES = {(1, 0), (2, 3)}
START = (0, 0)

_REWARD_GRID = np.zeros((GRID_SIZE, GRID_SIZE))
_REWARD_GRID[4, 4] = 1.0
_REWARD_GRID[1, 0] = -1.0
_REWARD_GRID[2, 3] = -1.0


def _grid_step(state, action):
    """Reference environment step (returns next_state, done — no reward)."""
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


def _greedy_policy(V_or_Q):
    """Derive a greedy policy array from V (5,5) or Q (5,5,4)."""
    if V_or_Q.ndim == 2:
        # State-reward model: greedy = argmax_a V[next_state] (R(s) cancels across actions)
        policy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                best_a, best_v = 0, -np.inf
                for a in range(NUM_ACTIONS):
                    ns, _ = _grid_step((r, c), a)
                    nr, nc = ns
                    if V_or_Q[nr, nc] > best_v:
                        best_v = V_or_Q[nr, nc]
                        best_a = a
                policy[r, c] = best_a
        return policy
    else:
        # Q function (5,5,4)
        return np.argmax(V_or_Q, axis=2)


def _policy_reaches_goal(policy, max_steps=50):
    """Return True if the greedy policy reaches GOAL from START within max_steps."""
    state = START
    visited = set()
    for _ in range(max_steps):
        if state == GOAL:
            return True
        if state in visited:
            return False  # loop
        visited.add(state)
        r, c = state
        action = policy[r, c]
        state, done = _grid_step(state, action)
        if done:
            return True
    return False


# ======================================================================
# Step 0: Value Iteration
# ======================================================================


def check_step_0_value_iteration(local_vars):
    """
    Expected: 'V' — a (5, 5) numpy array of state values.
    With the state-reward model: V[GOAL] = 1.0, obstacle cells < 0.
    """
    if "V" not in local_vars:
        return False, "⚠️ Variable `V` not found. Make sure you assign the result to `V`."

    V = local_vars["V"]

    if not isinstance(V, np.ndarray):
        return False, f"⚠️ `V` should be a numpy array, got {type(V).__name__}."

    if V.shape != (GRID_SIZE, GRID_SIZE):
        return False, f"⚠️ `V` should have shape (5, 5), got {V.shape}."

    if not np.isfinite(V).all():
        return False, "⚠️ `V` contains NaN or Inf. Check your Bellman update."

    # With state-reward model, V[GOAL] is initialised to REWARD_GRID[GOAL] = 1.0
    goal_val = V[GOAL]
    if abs(goal_val - 1.0) > 0.05:
        return (
            False,
            f"⚠️ V[GOAL] = {goal_val:.3f}, expected 1.0. "
            "Make sure you initialise V[GOAL] = REWARD_GRID[GOAL] before the loop "
            "and skip updating it (it's terminal).",
        )

    # Obstacle cells must be negative (their state reward is -1)
    for obs in OBSTACLES:
        if V[obs] >= 0:
            return (
                False,
                f"⚠️ V at obstacle {obs} = {V[obs]:.3f} — should be negative. "
                "Use V[r,c] = REWARD_GRID[r,c] + gamma * max(next_vals), "
                "not R(next_state) + gamma * V[next_state].",
            )

    # Start should be reachable (positive V)
    if V[START] <= 0:
        return (
            False,
            f"⚠️ V at start {START} = {V[START]:.3f} — should be positive "
            "(there is a path to the goal). Check your Bellman update.",
        )

    # Greedy policy should reach the goal
    policy = _greedy_policy(V)
    if not _policy_reaches_goal(policy):
        return (
            False,
            "⚠️ The greedy policy derived from `V` does not reach the goal. "
            "Double-check your Bellman update.",
        )

    return True, "✅ Value iteration converged! Obstacle cells are negative, goal = +1."


# ======================================================================
# Step 1: Q-Learning
# ======================================================================


def check_step_1_q_learning(local_vars):
    """
    Expected: 'Q' — shape (5, 5, 4); 'episode_returns' — list of per-episode returns.
    """
    if "Q" not in local_vars:
        return False, "⚠️ Variable `Q` not found. Assign the Q-table to `Q`."

    Q = local_vars["Q"]

    if not isinstance(Q, np.ndarray):
        return False, f"⚠️ `Q` should be a numpy array, got {type(Q).__name__}."

    if Q.shape != (GRID_SIZE, GRID_SIZE, NUM_ACTIONS):
        return (
            False,
            f"⚠️ `Q` should have shape (5, 5, 4), got {Q.shape}.",
        )

    if not np.isfinite(Q).all():
        return False, "⚠️ `Q` contains NaN or Inf. Check your update rule."

    # Q values at cells adjacent to GOAL should be ≈ REWARD_GRID[GOAL] = 1.0
    # (one-step away: Q ≈ 1, two steps ≈ 0.9*1 = 0.9, etc.)
    max_q = Q.max()
    if max_q < 0.7:
        return (
            False,
            f"⚠️ max Q across all states = {max_q:.3f}, expected ≈ 1. "
            "Did you run enough episodes, or is the terminal reward missing?",
        )

    # Greedy policy should reach the goal
    policy = _greedy_policy(Q)
    if not _policy_reaches_goal(policy):
        return (
            False,
            "⚠️ The greedy policy from `Q` does not reach the goal. "
            "Check your epsilon-greedy selection and TD update.",
        )

    if "episode_returns" not in local_vars:
        return (
            True,
            "✅ Q-table looks good! (Tip: store per-episode returns in `episode_returns` "
            "to see the learning curve.)",
        )

    episode_returns = local_vars["episode_returns"]
    if len(episode_returns) >= 400:
        first_half = np.mean(episode_returns[: len(episode_returns) // 2])
        second_half = np.mean(episode_returns[len(episode_returns) // 2 :])
        if second_half <= first_half:
            return (
                True,
                "✅ Q-table looks good! Note: episode returns didn't clearly improve — "
                "try more episodes or a lower epsilon.",
            )

    return True, "✅ Q-learning converged! The agent has learned to navigate the gridworld."


# ======================================================================
# Step 2: QNetwork
# ======================================================================


def check_step_2_qnetwork(local_vars):
    """
    Expected: 'QNetwork' — a torch.nn.Module subclass mapping (B, 4) -> (B, 2).
    """
    if "QNetwork" not in local_vars:
        return False, "⚠️ Class `QNetwork` not found."

    QNetwork = local_vars["QNetwork"]

    try:
        net = QNetwork()
    except Exception as e:
        return False, f"⚠️ Could not instantiate QNetwork(): {e}"

    if not isinstance(net, torch.nn.Module):
        return False, "⚠️ `QNetwork` must inherit from `nn.Module`."

    num_params = sum(p.numel() for p in net.parameters())
    if num_params == 0:
        return False, "⚠️ `QNetwork` has no parameters. Define your layers in `__init__`."

    try:
        x = torch.randn(16, 4)
        out = net(x)
    except Exception as e:
        return False, f"⚠️ Forward pass failed on input shape (16, 4): {e}"

    if out.shape != (16, 2):
        return (
            False,
            f"⚠️ Expected output shape (16, 2), got {tuple(out.shape)}. "
            "CartPole has 2 actions — your network should output one Q-value per action.",
        )

    return True, "✅ QNetwork architecture looks good!"


# ======================================================================
# Step 3: DQN Training
# ======================================================================


def check_step_3_dqn(local_vars):
    """
    Expected: 'episode_returns' — list of per-episode returns from CartPole training;
              'compute_loss' — the loss function; 'policy_net', 'target_net'.
    """
    if "compute_loss" not in local_vars:
        return False, "⚠️ Function `compute_loss` not found."

    if not callable(local_vars["compute_loss"]):
        return False, "⚠️ `compute_loss` must be a callable function."

    if "episode_returns" not in local_vars:
        return False, "⚠️ Variable `episode_returns` not found. Did the training loop run?"

    episode_returns = local_vars["episode_returns"]

    if len(episode_returns) < 100:
        return (
            False,
            f"⚠️ Only {len(episode_returns)} episodes recorded. "
            "Did the full training loop run?",
        )

    mean_last = np.mean(episode_returns[-50:])
    if mean_last < 60:
        return (
            False,
            f"⚠️ Mean return over last 50 episodes = {mean_last:.1f} (need ≥ 60). "
            "Check your compute_loss — especially the gather and target computation.",
        )

    # Verify compute_loss doesn't backprop through target_net
    if "policy_net" in local_vars and "target_net" in local_vars:
        try:
            QNetwork = local_vars.get("QNetwork")
            if QNetwork is not None:
                compute_loss = local_vars["compute_loss"]
                policy_net = QNetwork()
                target_net = QNetwork()
                target_net.load_state_dict(policy_net.state_dict())

                dummy_states = torch.randn(8, 4)
                dummy_actions = torch.randint(0, 2, (8,))
                dummy_rewards = torch.randn(8)
                dummy_next = torch.randn(8, 4)
                dummy_dones = torch.zeros(8)
                batch = (dummy_states, dummy_actions, dummy_rewards, dummy_next, dummy_dones)

                loss = compute_loss(batch, policy_net, target_net)

                if not isinstance(loss, torch.Tensor):
                    return False, "⚠️ `compute_loss` must return a torch.Tensor."
                if loss.ndim != 0:
                    return False, f"⚠️ Loss must be a scalar tensor, got shape {tuple(loss.shape)}."

                loss.backward()
                for p in target_net.parameters():
                    if p.grad is not None:
                        return (
                            False,
                            "⚠️ Gradients are flowing into `target_net`. "
                            "Wrap the target Q-value computation in `torch.no_grad()`.",
                        )
        except Exception as e:
            return False, f"⚠️ Error verifying compute_loss: {e}"

    mean_str = f"{mean_last:.1f}"
    return (
        True,
        f"✅ DQN training complete! Mean return (last 50 eps) = {mean_str}. "
        "The agent has learned to balance the pole.",
    )
