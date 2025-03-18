import jax
import jax.numpy as jnp
import numpyro.distributions as dist


def RightTruncatedPoisson(rate, max_cutoff=100, factor=3):
    """
    Sample from a right-truncated Poisson distribution (up to a cutoff)
    using NumPyro's Categorical, with support for batching over 'rate'.

    :param jnp.ndarray or float rate: Poisson rate parameter(s).
    :returns: A NumPyro sample (or batch of samples) from the
              right-truncated Poisson distribution.
    """
    # Convert rate to jax array
    rate = jnp.asarray(rate)

    try:
        # Check whether max_cutoff is sensible
        sensible_cutoff = jnp.ceil(rate + factor * jnp.sqrt(rate)).astype(int).max().item()
        if sensible_cutoff > max_cutoff:
            print(f"max_cutoff={max_cutoff} might be too small for the given rate parameters. Set to at least {sensible_cutoff}.")
    except jax.errors.ConcretizationTypeError:
        # Does not work under JAX JIT
        pass

    # Build a common support range up to 'max_cutoff' (must be a Python int)
    support = jnp.arange(max_cutoff + 1)  # shape [max_cutoff+1,]

    # Poisson log_prob is broadcast over the last dimension
    # unnormalized_pmf shape => (*batch_shape, max_cutoff+1)
    unnormalized_logits = dist.Poisson(rate[..., None]).log_prob(support)

    # Return a sample from the batched Categorical distribution
    # The shape of the sample will match rate.shape
    # TODO: these should really be normalized but that somehow breaks convergence
    return dist.Categorical(logits=unnormalized_logits)