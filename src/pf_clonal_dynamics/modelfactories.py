import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from .modeloptions import DirMultinomialSeq

def _spline_model_factory(
    SeqLik=None,
):
    if SeqLik is None:
        SeqLik = DirMultinomialSeq()

    def _variant_model(parasite_density, seq_counts, N, X, X_prime):
        _, N_variant = seq_counts.shape
        T, k = X.shape

        # Need some way of making the R parameter formations a bit more usable
        # Time varying base trajectory
        gam = numpyro.sample("gam", dist.HalfCauchy(0.1))

        beta_rw = numpyro.sample(
            "beta_rw", dist.GaussianRandomWalk(scale=gam, num_steps=k - 1)
        )
        beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 10.0))
        beta = numpyro.deterministic(
            "beta", beta_0 + jnp.concatenate([jnp.array([0.0]), beta_rw])
        )

        # Time varying growth advantage as random walk
        # Regularizes changes in growth advantage of variants
        gam_delta = numpyro.sample("gam_delta", dist.Exponential(rate=50))
        with numpyro.plate("N_variant_m1", N_variant - 1):
            delta_rw = numpyro.sample(
                "delta_rw", dist.GaussianRandomWalk(scale=gam_delta, num_steps=k)
            )

        delta = delta_rw.T
        beta_mat = beta[:, None] + jnp.hstack((delta, jnp.zeros((k, 1))))

        variant_specific_P = jnp.exp(jnp.dot(X, beta_mat))  # Variant-specific parasite_density
        numpyro.deterministic("r", X_prime @ beta_mat)

        numpyro.deterministic("variant_specific_P", variant_specific_P)

        # Evaluate 
        obs_error_scale = numpyro.sample("obs_error_scale", dist.HalfNormal(1.0))
        numpyro.sample("parasite_density", dist.LogNormal(loc=variant_specific_P.sum(axis=-1)), scale=obs_error_scale)

        # Compute frequency
        freq = numpyro.deterministic(
            "freq", jnp.divide(variant_specific_P, variant_specific_P.sum(axis=1)[:, None])
        )

        # Evaluate frequency likelihood
        SeqLik.model(seq_counts, N, freq)

    return _variant_model
