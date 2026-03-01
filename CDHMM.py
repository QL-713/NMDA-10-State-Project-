"""
Dwell- and context-sensitive HMM for a single NMDA receptor.

Model overview
--------------
We simulate a single 5-state NMDA receptor driven by glutamate transients:

    0. C   (closed, unbound)
    1. C1  (single ligand bound)
    2. C2  (double ligand bound)
    3. O   (open, conducting)
    4. D   (desensitized)

True dynamics:
- Continuous-time Markov process with Adam-style kinetics.
- Glutamate(t) controls the association (binding) rates.
- We simulate in discrete time with dt, sampling the state and current.

HMM decoder:
- Hidden states are the same 5 states.
- Emissions: simple Gaussian current model
    - C, C1, C2, D → mean 0 pA
    - O → mean I_open = g_open * (V_hold - E_rev)
    - All states share the same noise std.

- Transitions are context-dependent:
    - For each time step t, we build a transition matrix P_t
      from the same kinetic rates and the instantaneous glutamate[t].
    - Semi-Markov & contextual decoder:
        * Penalizes over-long dwell in any state (based on mean dwell).
        * Adds extra penalty for staying in D too long when it's been
          a long time since the last puff and last open-like event.

We then:
- Run context-sensitive semi-Markov Viterbi to infer the most likely
  hidden state sequence from the noisy current trace.
- Run Forward to get per-time-step state probabilities.
- Compute accuracy and a confusion matrix.
- Plot glutamate, true states, currents, and inferred states.
- Plot the confusion matrix as a separate figure.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

# Optional plotting
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


StateIndex = int


# ---------------------------------------------------------------------
# Kinetic parameters
# ---------------------------------------------------------------------
@dataclass
class NMDAMarkovParameters:
    """
    Transition parameters for the 5-state NMDA receptor Markov model.

    Rates are in ms^-1; association rates (k_on1, k_on2) are
    scaled by the glutamate concentration [Glu](t) (in µM).

    State index mapping:
        0: C   (closed, unbound)
        1: C1  (one glutamate bound)
        2: C2  (two glutamates bound)
        3: O   (open, conducting)
        4: D   (desensitized)
    """

    # Adam-aligned (5-state reduction)
    k_on1: float = 0.005      # C → C1 (µM^-1 ms^-1); 5 mM^-1 ms^-1 = 0.005 µM^-1 ms^-1
    k_off1: float = 0.006     # C1 → C (ms^-1)
    k_on2: float = 0.005      # C1 → C2 (µM^-1 ms^-1)
    k_off2: float = 0.006     # C2 → C1 (ms^-1)
    k_open: float = 0.0465    # β : C2 → O (ms^-1)
    k_close: float = 0.0916   # α : O → C2 (ms^-1)
    k_desensitize: float = 0.0084   # kd : C2 → D (ms^-1)
    k_resensitize: float = 0.0018   # kr : D → C2 (ms^-1)


# ---------------------------------------------------------------------
# True single-receptor simulator
# ---------------------------------------------------------------------
class NMDAMarkovSynapse:
    """
    Five-state NMDA receptor model with stochastic transitions.

    This class generates the *ground-truth* state trajectory and
    current trace for a single receptor driven by glutamate pulses.
    """

    state_labels: Tuple[str, ...] = ("C", "C1", "C2", "O", "D")

    def __init__(
        self,
        release_times: Sequence[float],
        *,
        dt: float = 0.05,
        params: NMDAMarkovParameters | None = None,
        g_open: float = 5e-5,            # 50 pS in micro-Siemens
        e_rev: float = 0.0,              # mV
        v_hold: float = -55.0,           # mV
        glutamate_peak: float = 1200.0,  # µM
        glutamate_tau: float = 1.2,      # ms
        observation_noise_std: float = 0.0,  # pA
        rng: random.Random | None = None,
    ) -> None:
        self.release_times = [float(t) for t in release_times]
        self.dt = dt
        self.params = params or NMDAMarkovParameters()
        # Only O conducts
        self.g_states = [0.0, 0.0, 0.0, g_open, 0.0]
        self.e_rev = e_rev
        self.v_hold = v_hold
        self.glutamate_peak = glutamate_peak
        self.glutamate_tau = glutamate_tau
        self.observation_noise_std = observation_noise_std
        self.rng = rng or random.Random(42)

        self.state: StateIndex = 0
        self._state_history: List[StateIndex] = []
        self._current_history: List[float] = []
        self._glutamate_history: List[float] = []

    # -------------------------------
    # Glutamate transient
    # -------------------------------
    def glutamate_concentration(self, t: float) -> float:
        """Return instantaneous [Glu](t) given release times."""
        total = 0.0
        for release in self.release_times:
            if t >= release:
                total += self.glutamate_peak * math.exp(-(t - release) / self.glutamate_tau)
        return total

    # -------------------------------
    # True Markov dynamics
    # -------------------------------
    def _transition_rates(self, state: StateIndex, glutamate: float) -> Tuple[Sequence[StateIndex], List[float]]:
        """
        Outgoing transition rates for a given state, at a given glutamate level.
        This is used both by the generator and the HMM decoder.
        """
        p = self.params

        if state == 0:  # C → C1
            targets = (1,)
            rates = [p.k_on1 *2 * glutamate]
        elif state == 1:  # C1 → {C, C2}
            targets = (0, 2)
            rates = [p.k_off1, p.k_on2 * glutamate]
        elif state == 2:  # C2 → {C1, O, D}
            targets = (1, 3, 4)
            rates = [p.k_off2, p.k_open, p.k_desensitize]
        elif state == 3:  # O → C2
            targets = (2,)
            rates = [p.k_close]
        elif state == 4:  # D → C2
            targets = (2,)
            rates = [p.k_resensitize]
        else:
            raise ValueError(f"Invalid state index: {state}")

        return targets, rates

    def _maybe_transition(self, glutamate: float) -> None:
        """Stochastically update the current state using continuous-time rates."""
        targets, rates = self._transition_rates(self.state, glutamate)
        if not targets:
            return

        total_rate = sum(rates)
        if total_rate <= 0.0:
            return

        # Probability of leaving this state in dt
        leave_prob = min(0.99, total_rate * self.dt)
        if self.rng.random() < leave_prob:
            probabilities = [rate / total_rate for rate in rates]
            r = self.rng.random()
            cumulative = 0.0
            for target, w in zip(targets, probabilities):
                cumulative += w
                if r <= cumulative:
                    self.state = target
                    break

    def step(self, t: float) -> None:
        """Advance the receptor one time step at time t."""
        glutamate = self.glutamate_concentration(t)
        self._maybe_transition(glutamate)

        conductance = self.g_states[self.state]
        # Convert µS·mV to nA, then to pA:
        # (µS * mV) = nA, so multiply by 1000 to get pA
        current = conductance * (self.v_hold - self.e_rev) * 1000.0

        self._state_history.append(self.state)
        self._current_history.append(current)
        self._glutamate_history.append(glutamate)

    # -------------------------------
    # Public API
    # -------------------------------
    def simulate(self, t_stop: float) -> Tuple[List[float], List[int], List[float]]:
        """Simulate from t = 0 to t = t_stop (ms)."""
        self._state_history.clear()
        self._current_history.clear()
        self._glutamate_history.clear()
        self.state = 0  # start in Closed

        steps = int(math.ceil(t_stop / self.dt))
        times = [self.dt * i for i in range(steps)]
        for t in times:
            self.step(t)

        return list(times), list(self._state_history), list(self._current_history)

    def glutamate_trace(self) -> List[float]:
        return list(self._glutamate_history)

    def noisy_current(self, noise_std: float | None = None) -> List[float]:
        """Add Gaussian noise to the clean current."""
        std = self.observation_noise_std if noise_std is None else noise_std
        noisy = []
        for value in self._current_history:
            noisy.append(value + self.rng.gauss(0.0, std))
        return noisy

    @property
    def state_history(self) -> List[int]:
        return list(self._state_history)


# ---------------------------------------------------------------------
# Plot helper: confusion matrix (separate figure)
# ---------------------------------------------------------------------
def plot_confusion_matrix(confusion, labels, title="Confusion Matrix (Viterbi)"):
    """
    Plot a confusion matrix as a separate figure.
    Rows = true states, columns = inferred states.
    """
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="viridis", aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Inferred state")
    ax.set_ylabel("True state")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Annotate counts
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, confusion[i][j], ha="center", va="center", color="white")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# HMM utilities
# ---------------------------------------------------------------------
def gaussian_log_likelihood(x: float, mean: float, std: float) -> float:
    var = std * std
    return -0.5 * ((x - mean) ** 2 / var + math.log(2.0 * math.pi * var))


def build_time_varying_transition_matrices(
    times: Sequence[float],
    glutamate: Sequence[float],
    params: NMDAMarkovParameters,
    dt: float,
    n_states: int = 5,
) -> List[List[List[float]]]:
    """
    Build a list of transition matrices P_t (n_states x n_states),
    one for each time step, using the same kinetics as the true model.

    For each state s:
      - compute outgoing rates at glutamate[t]
      - total_rate = sum(rates)
      - leave_prob = min(0.99, total_rate * dt)
      - P_t[s, s] = 1 - leave_prob
      - P_t[s, target] = leave_prob * (rate / total_rate)
    """
    matrices: List[List[List[float]]] = []

    def transition_rates_for_state(state: int, glu: float) -> Tuple[Sequence[int], List[float]]:
        p = params
        if state == 0:  # C → C1
            targets = (1,)
            rates = [p.k_on1 * 2* glu]
        elif state == 1:  # C1 → {C, C2}
            targets = (0, 2)
            rates = [p.k_off1, p.k_on2 * glu]
        elif state == 2:  # C2 → {C1, O, D}
            targets = (1, 3, 4)
            rates = [p.k_off2, p.k_open, p.k_desensitize]
        elif state == 3:  # O → C2
            targets = (2,)
            rates = [p.k_close]
        elif state == 4:  # D → C2
            targets = (2,)
            rates = [p.k_resensitize]
        else:
            raise ValueError(f"Invalid state index: {state}")
        return targets, rates

    for t_idx, _t in enumerate(times):
        glu = glutamate[t_idx]
        P_t = [[0.0 for _ in range(n_states)] for _ in range(n_states)]

        for s in range(n_states):
            targets, rates = transition_rates_for_state(s, glu)
            if not targets:
                P_t[s][s] = 1.0
                continue

            total_rate = sum(rates)
            if total_rate <= 0.0:
                P_t[s][s] = 1.0
                continue

            leave_prob = min(0.99, total_rate * dt)
            stay_prob = 1.0 - leave_prob
            P_t[s][s] = stay_prob

            for target, rate in zip(targets, rates):
                P_t[s][target] = leave_prob * (rate / total_rate)

        matrices.append(P_t)

    return matrices


def compute_mean_dwell_steps(states: Sequence[int], n_states: int) -> List[float]:
    """
    Compute the mean dwell length (in time steps) for each state
    from a ground-truth state sequence.

    Returns a list mean_dwell[s]. If a state is never visited,
    fallback to 1.0 step to avoid division by zero.
    """
    dwell_sums = [0] * n_states
    dwell_counts = [0] * n_states

    if not states:
        return [1.0] * n_states

    current_state = states[0]
    current_len = 1

    for prev, curr in zip(states[:-1], states[1:]):
        if curr == prev:
            current_len += 1
        else:
            dwell_sums[current_state] += current_len
            dwell_counts[current_state] += 1
            current_state = curr
            current_len = 1

    dwell_sums[current_state] += current_len
    dwell_counts[current_state] += 1

    mean_dwell = []
    for s in range(n_states):
        if dwell_counts[s] > 0:
            mean_dwell.append(dwell_sums[s] / dwell_counts[s])
        else:
            mean_dwell.append(1.0)
    return mean_dwell


def compute_time_since_puff(times: Sequence[float], release_times: Sequence[float]) -> List[float]:
    """
    For each time t in `times`, return Δt_puff = time since the last glutamate puff.
    If no puff has occurred yet, returns 0.0.
    """
    tau_puff: List[float] = []
    last_puff_time = -1e9
    idx_rel = 0
    rel_sorted = sorted(release_times)

    for t in times:
        while idx_rel < len(rel_sorted) and rel_sorted[idx_rel] <= t:
            last_puff_time = rel_sorted[idx_rel]
            idx_rel += 1

        if last_puff_time < -1e8:
            tau_puff.append(0.0)
        else:
            tau_puff.append(t - last_puff_time)

    return tau_puff


def compute_time_since_open_obs(
    times: Sequence[float],
    observations: Sequence[float],
    noise_std: float,
    threshold_factor: float = 3.0,
) -> List[float]:
    """
    For each time t, return Δt_open = time since the last *open-like* observation.

    We approximate 'open' by |I| > threshold_factor * noise_std.
    This is a decoder-side estimate grounded in the observable current.
    """
    tau_open: List[float] = []
    last_open_time = -1e9
    thr = threshold_factor * noise_std

    for t, y in zip(times, observations):
        if abs(y) > thr:
            last_open_time = t

        if last_open_time < -1e8:
            tau_open.append(0.0)
        else:
            tau_open.append(t - last_open_time)

    return tau_open


def forward_time_varying(
    observations: Sequence[float],
    start_prob: Sequence[float],
    transition_matrices: Sequence[Sequence[Sequence[float]]],
    emission_means: Sequence[float],
    emission_stds: Sequence[float],
) -> Tuple[List[List[float]], float]:
    """
    Forward algorithm for a context-dependent HMM.
    Returns:
      - alpha[t][s]: filtered state probabilities at time t
      - log_likelihood of the observation sequence
    """
    n_obs = len(observations)
    n_states = len(start_prob)

    alpha = [[0.0] * n_states for _ in range(n_obs)]
    scales = [0.0] * n_obs

    # initialization
    for s in range(n_states):
        emission = math.exp(
            gaussian_log_likelihood(observations[0], emission_means[s], emission_stds[s])
        )
        alpha[0][s] = start_prob[s] * emission

    scale0 = sum(alpha[0]) or 1e-300
    scales[0] = scale0
    for s in range(n_states):
        alpha[0][s] /= scale0

    # recursion
    for t in range(1, n_obs):
        P_prev = transition_matrices[t - 1]  # t-1 → t
        o_t = observations[t]
        for s in range(n_states):
            trans_sum = 0.0
            for prev in range(n_states):
                trans_sum += alpha[t - 1][prev] * P_prev[prev][s]
            emission = math.exp(
                gaussian_log_likelihood(o_t, emission_means[s], emission_stds[s])
            )
            alpha[t][s] = emission * trans_sum

        scale = sum(alpha[t]) or 1e-300
        scales[t] = scale
        for s in range(n_states):
            alpha[t][s] /= scale

    log_likelihood = sum(math.log(s) for s in scales)
    return alpha, log_likelihood


def viterbi_time_varying_contextual(
    observations: Sequence[float],
    start_prob: Sequence[float],
    transition_matrices: Sequence[Sequence[Sequence[float]]],
    emission_means: Sequence[float],
    emission_stds: Sequence[float],
    mean_dwell_steps: Sequence[float],
    tau_puff: Sequence[float],
    tau_open: Sequence[float],
    dwell_penalty_strength: float = 1.0,
    context_penalty_strength: float = 0.5,
    puff_grace: float = 100.0,   # ms before puff-based penalty kicks in
    open_grace: float = 100.0,   # ms before open-based penalty kicks in
    puff_scale: float = 200.0,   # ms scale for τ_puff penalty
    open_scale: float = 200.0,   # ms scale for τ_open penalty
    d_state_index: int = 4,      # index of D in state list
) -> List[int]:
    """
    Context-sensitive semi-Markov Viterbi.

    - Uses time-varying transition matrices (glutamate-driven kinetics).
    - Uses mean_dwell_steps[s] to penalize over-long dwell in any state.
    - Adds extra context penalty specifically for staying in D too long when:
        - τ_puff is large (long time since puff), AND
        - τ_open is large (long time since open-like observation).
    """
    n_obs = len(observations)
    n_states = len(start_prob)

    log_start = [math.log(max(p, 1e-12)) for p in start_prob]

    log_delta = [[float("-inf")] * n_states for _ in range(n_obs)]
    psi = [[0] * n_states for _ in range(n_obs)]
    dwell_len = [[0] * n_states for _ in range(n_obs)]

    # initialization
    for s in range(n_states):
        log_emission = gaussian_log_likelihood(observations[0], emission_means[s], emission_stds[s])
        log_delta[0][s] = log_start[s] + log_emission
        psi[0][s] = s
        dwell_len[0][s] = 1

    # recursion
    for t in range(1, n_obs):
        P_prev = transition_matrices[t - 1]
        o_t = observations[t]
        tau_p = tau_puff[t]
        tau_o = tau_open[t]

        for s in range(n_states):
            log_emission = gaussian_log_likelihood(o_t, emission_means[s], emission_stds[s])

            best_log_prob = float("-inf")
            best_prev_state = 0
            best_dwell = 1

            for prev in range(n_states):
                base_prob = log_delta[t - 1][prev]
                p_trans = max(P_prev[prev][s], 1e-12)
                candidate = base_prob + math.log(p_trans)

                # semi-Markov dwell penalty for staying in the same state
                if prev == s:
                    prev_dwell = dwell_len[t - 1][prev]
                    new_dwell = prev_dwell + 1
                    mean_dwell = max(mean_dwell_steps[s], 1e-6)

                    if new_dwell > mean_dwell:
                        excess = (new_dwell - mean_dwell) / mean_dwell
                        candidate += -dwell_penalty_strength * excess

                    cand_dwell = new_dwell
                else:
                    cand_dwell = 1

                # extra context penalty for staying in D too long
                if prev == s == d_state_index:
                    excess_puff = max(0.0, (tau_p - puff_grace) / puff_scale)
                    excess_open = max(0.0, (tau_o - open_grace) / open_scale)
                    context_excess = excess_puff + excess_open
                    if context_excess > 0.0:
                        candidate += -context_penalty_strength * context_excess

                candidate_total = candidate + log_emission

                if candidate_total > best_log_prob:
                    best_log_prob = candidate_total
                    best_prev_state = prev
                    best_dwell = cand_dwell

            log_delta[t][s] = best_log_prob
            psi[t][s] = best_prev_state
            dwell_len[t][s] = best_dwell

    # backtrack
    last_state = max(range(n_states), key=lambda s: log_delta[-1][s])
    states: List[int] = [last_state]
    for t in range(n_obs - 1, 0, -1):
        states.insert(0, psi[t][states[0]])

    return states


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
def main() -> None:
    # 1. Set up the true synapse
    dt = 0.05  # ms
    release_times = [50, 75, 100, 125, 300]    #independent variable for this investigation
    t_stop = 900.0  # ms

    synapse = NMDAMarkovSynapse(
        release_times,
        dt=dt,
        params=NMDAMarkovParameters(),
        g_open=5e-5, #Siemens
        e_rev=0.0,     #mV
        v_hold=-50.0,   #mV
        glutamate_peak=1200.0, #uM
        glutamate_tau=1.2,     #uM ms-1
        observation_noise_std=0.35, #pA
        rng=random.Random(12), #allows for repeatable data per run, changing this value changes the seed
    )

    # Simulate true trajectory
    times, true_states, clean_current = synapse.simulate(t_stop)
    glutamate = synapse.glutamate_trace()
    noisy_current = synapse.noisy_current()

    n_states = len(synapse.state_labels)

    # 2. Build context-dependent HMM pieces
    start_prob = [1e-6] * n_states
    start_prob[0] = 1.0 - (n_states - 1) * 1e-6

    emission_means = [g * (synapse.v_hold - synapse.e_rev) * 1000.0 for g in synapse.g_states]  # pA
    emission_stds = [synapse.observation_noise_std for _ in range(n_states)]

    transition_matrices = build_time_varying_transition_matrices(
        times,
        glutamate,
        synapse.params,
        synapse.dt,
        n_states=n_states,
    )

    # Additional context signals
    tau_puff = compute_time_since_puff(times, synapse.release_times)
    tau_open = compute_time_since_open_obs(times, noisy_current, synapse.observation_noise_std)
    
    # 3. Run context-sensitive semi-Markov Viterbi and Forward
    mean_dwell_steps = compute_mean_dwell_steps(true_states, n_states)
    print("Mean dwell (steps) per state:")
    print({lab: md for lab, md in zip(synapse.state_labels, mean_dwell_steps)})

    inferred_states = viterbi_time_varying_contextual(
        noisy_current,
        start_prob,
        transition_matrices,
        emission_means,
        emission_stds,
        mean_dwell_steps,
        tau_puff,
        tau_open,
        dwell_penalty_strength=1.0,
        context_penalty_strength=0.5,
        puff_grace=100.0,
        open_grace=100.0,
        puff_scale=200.0,
        open_scale=200.0,
        d_state_index=4,
    )

    alpha, log_likelihood = forward_time_varying(
        noisy_current,
        start_prob,
        transition_matrices,
        emission_means,
        emission_stds,
    )
    forward_states = [max(range(n_states), key=lambda s: alpha_t[s]) for alpha_t in alpha]

    # 4. Metrics
    matches_viterbi = sum(1 for inf, true in zip(inferred_states, true_states) if inf == true)
    accuracy_viterbi = matches_viterbi / len(true_states)

    matches_forward = sum(1 for f, true in zip(forward_states, true_states) if f == true)
    accuracy_forward = matches_forward / len(true_states)

    print(f"\nViterbi (contextual) state accuracy: {accuracy_viterbi * 100:.2f}%")
    print(f"Forward MAP state accuracy: {accuracy_forward * 100:.2f}%")
    print(f"Sequence log-likelihood (forward): {log_likelihood:.3f}")

    # Confusion matrix (Viterbi)
    confusion = [[0 for _ in range(n_states)] for _ in range(n_states)]
    for true, inf in zip(true_states, inferred_states):
        confusion[true][inf] += 1

    print("\nConfusion matrix (rows = true, cols = inferred):")
    header = "         " + " ".join(f"{lab:>5s}" for lab in synapse.state_labels)
    print(header)
    for i, row in enumerate(confusion):
        row_str = " ".join(f"{val:5d}" for val in row)
        print(f"{synapse.state_labels[i]:>5s} | {row_str}")

    # 5. Plots
    if plt is None:
        print("\nmatplotlib not installed; skipping plots.")
        return

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(times, glutamate)
    axes[0].set_ylabel("[Glu] (µM)")
    axes[0].set_title("Glutamate transients")

    axes[1].plot(times, true_states, drawstyle="steps-pre")
    axes[1].set_ylabel("State (true)")
    axes[1].set_yticks(range(n_states))
    axes[1].set_yticklabels(synapse.state_labels)
    axes[1].set_title("True NMDA receptor state")

    axes[2].plot(times, clean_current, label="Clean current")
    axes[2].plot(times, noisy_current, label="Noisy current", alpha=0.6)
    axes[2].set_ylabel("Current (pA)")
    axes[2].set_title("Simulated single-channel current")
    axes[2].legend(loc="upper right")

    axes[3].plot(times, inferred_states, drawstyle="steps-pre")
    axes[3].set_ylabel("State (Viterbi)")
    axes[3].set_yticks(range(n_states))
    axes[3].set_yticklabels(synapse.state_labels)
    axes[3].set_xlabel("Time (ms)")
    axes[3].set_title("HMM inferred state sequence (contextual semi-Markov)")

    fig.tight_layout()
    plt.show()

    # Separate figure: confusion matrix
    plot_confusion_matrix(confusion, synapse.state_labels, title="Confusion Matrix (Viterbi)")


if __name__ == "__main__":
    main()