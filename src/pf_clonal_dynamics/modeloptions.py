import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

class MultinomialSeq:
    def __init__(self):
        pass

    def model(self, seq_counts, N, freq):
        # Sample with Multinomial
        numpyro.sample(
            "Y",
            dist.Multinomial(total_count=jnp.nan_to_num(N), probs=freq),
            obs=jnp.nan_to_num(seq_counts),
        )


class DirMultinomialSeq:
    def __init__(self, xi_prior=99):
        self.xi_prior = xi_prior

    def model(self, seq_counts, N, freq):
        # Overdispersion in sequence counts
        xi = numpyro.sample("xi", dist.Beta(1, self.xi_prior))
        trans_xi = jnp.reciprocal(xi) - 1

        # Sample with DirichletMultinomial
        numpyro.sample(
            "Y",
            dist.DirichletMultinomial(
                total_count=jnp.nan_to_num(N), concentration=1e-8 + trans_xi * freq
            ),
            obs=jnp.nan_to_num(seq_counts),
        )
