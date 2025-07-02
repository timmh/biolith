from operator import attrgetter

import jax.numpy as jnp
from numpyro.diagnostics import summary
from numpyro.infer import MCMC


def diagnostics(mcmc: MCMC, exclude_deterministic=True):

    sites = mcmc._states[mcmc._sample_field]
    if isinstance(sites, dict) and exclude_deterministic:

        state_sample_field = attrgetter(mcmc._sample_field)(mcmc._last_state)
        # adapted from NumPyro's summary code
        if isinstance(state_sample_field, dict):
            sites = {
                k: v
                for k, v in mcmc._states[mcmc._sample_field].items()
                if k in state_sample_field
            }

    # Compute summary statistics (includes mean, std, quantiles, R-hat, and ESS)
    summary_dict = summary(sites)

    mean_r_hat = sum([v["r_hat"].mean().item() for v in summary_dict.values()]) / len(
        summary_dict
    )
    mean_frac_eff = (
        sum([v["n_eff"].mean().item() for v in summary_dict.values()])
        / len(summary_dict)
        / (mcmc.num_samples * mcmc.num_chains)
    )

    diagnostics = mcmc.get_extra_fields()

    if diagnostics is not None and "diverging" in diagnostics:
        frac_diverging = jnp.sum(diagnostics["diverging"]).item() / (
            mcmc.num_samples * mcmc.num_chains
        )
    else:
        frac_diverging = float("nan")

    return dict(
        mean_r_hat=mean_r_hat,
        mean_frac_eff=mean_frac_eff,
        frac_diverging=frac_diverging,
    )
