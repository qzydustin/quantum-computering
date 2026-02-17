from typing import Dict, Tuple


def calculate_tvd(noisy_counts: Dict[str, int], real_counts: Dict[str, int]) -> Tuple[float, Dict[str, dict]]:
    """Calculate per-state distribution difference and aggregate loss."""
    total_noisy = sum((noisy_counts or {}).values())
    total_real = sum((real_counts or {}).values())

    if total_noisy == 0 or total_real == 0:
        return 0.0, {}

    all_states = set(noisy_counts.keys()) | set(real_counts.keys())
    tvd_loss = 0.0
    state_details: Dict[str, dict] = {}

    for state in sorted(all_states):
        noisy_prob = noisy_counts.get(state, 0) / total_noisy
        real_prob = real_counts.get(state, 0) / total_real
        diff = abs(noisy_prob - real_prob)
        tvd_loss += diff
        state_details[state] = {
            'noisy_prob': noisy_prob,
            'real_prob': real_prob,
            'abs_diff': diff,
            'noisy_count': noisy_counts.get(state, 0),
            'real_count': real_counts.get(state, 0),
        }

    return tvd_loss, state_details
