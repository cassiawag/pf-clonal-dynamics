import jax.numpy as jnp

def is_obs_idx(v):
    return jnp.where(jnp.isnan(v), jnp.zeros_like(v), jnp.ones_like(v))


def pad_to_obs(v, obs_idx, eps=1e-12):
    return v * obs_idx + (1 - obs_idx) * eps
