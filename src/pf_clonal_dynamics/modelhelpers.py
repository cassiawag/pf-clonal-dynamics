import jax.numpy as jnp
import numpy as np

def is_obs_idx(v, dtype=bool):
    return jnp.where(jnp.isnan(v), jnp.zeros_like(v,dtype=dtype), jnp.ones_like(v,dtype=dtype))

def is_obs_idx_np(v, dtype=bool):
    return np.where(np.isnan(v), np.zeros_like(v,dtype=dtype), np.ones_like(v,dtype=dtype))


def pad_to_obs(v, obs_idx, eps=1e-12):
    return v * obs_idx + (1 - obs_idx) * eps
