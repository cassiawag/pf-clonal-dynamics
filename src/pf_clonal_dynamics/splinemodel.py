import jax.numpy as jnp

from .modelfactories import _spline_model_factory
from .Splines import Spline, SplineDeriv


class SplineModel:
    def __init__(self, k=None, SLik=None):
        if k is None:
            k = 20
        self.k = k

        self.SLik = SLik
        self.make_model()

    def make_model(self):
        self.model = _spline_model_factory(SeqLik=self.SLik)

    def augment_data(self, data, order=4):
        T = len(data["parasite_density"])
        s = jnp.linspace(0, T, self.k)
        data["X"] = Spline.matrix(jnp.arange(T), s, order=order)
        data["X_prime"] = SplineDeriv.matrix(jnp.arange(T), s, order=order)
