import pandas as pd

import numpyro
from numpyro.infer.autoguide import AutoDelta

from .posteriorhelpers import (
    get_growth_rates,
    get_parasitemia,
    get_freq,
)
from .inferencehandler import SVIHandler, MCMCHandler
from .posteriorhandler import PosteriorHandler

def fit_SVI(
    VD,
    PM,
    opt,
    guide_fn = AutoDelta,
    iters=100_000,
    num_samples=1000,
    path=".",
    name="Test",
    load=False,
    save=False,
):
    # Defining optimization
    SVIH = SVIHandler(optimizer=opt)
    guide = guide_fn(PM.model)

    # Unpacking data
    data = VD.make_numpyro_input()
    PM.augment_data(data)

    # Loading past state
    if load:
        file_name = name.replace(" ", "-")
        SVIH.load_state(f"{path}/models/{file_name}_svi.p")

    # Fitting model
    if iters > 0:
        SVIH.fit(PM.model, guide, data, iters, log_each=0)

    if save:
        file_name = name.replace(" ", "-")
        SVIH.save_state(f"{path}/models/{file_name}_svi.p")

    dataset = SVIH.predict(LM.model, guide, data, num_samples=num_samples)
    return PosteriorHandler(dataset=dataset, data=VD, name=name)

def fit_MCMC(VD, PM, kernel, num_samples=1000, num_warmup=500, path=".", name="Test", load=False, save=False):
    # Defining MCMC algorithm
    MC = MCMCHandler(kernel=kernel)
    # Unpacking data
    data = VD.make_numpyro_input()
    MC.fit(PM.model, data, num_warmup, num_samples)
    dataset = MC.predict(PM.model, data, num_samples = num_samples)
    return PosteriorHandler(dataset=dataset,data=VD, name=name)  

def save_posteriors(MP, path):
    for name, p in MP.locator.items():
        p.save_posterior(f"{path}/posteriors/{name}_samples.json")
    return None

def sample_loaded_posterior(VD, LM, guide_fn = AutoDelta,num_samples=1000, path=".", name="Test"):
    # Defining optimization
    SVIH = SVIHandler(optimizer=numpyro.optim.Adam(step_size=1e-2))
    guide = guide_fn(LM.model)

    # Upacking data
    data = VD.make_numpyro_input()
    LM.augment_data(data)

    # Loading past state
    file_name = name.replace(" ", "-")
    SVIH.load_state(f"{path}/models/{file_name}_svi.p")

    dataset = SVIH.predict(LM.model, guide, data, num_samples=num_samples)
    return PosteriorHandler(dataset=dataset, data=VD, name=name)


def unpack_model(MP, name):
    posterior = MP.get(name)
    return posterior.dataset, posterior.data

def gather_growth_rates(MP, ps, forecast=False):
    r_dfs = []
    for name, p in MP.locator.items():
        r_dfs.append(
            pd.DataFrame(get_growth_rates(p.dataset, p.data, ps, name, forecast=forecast))
        )
    return pd.concat(r_dfs)

def gather_parasitemia(MP, ps, forecast=False):
    P_dfs = []
    for name, p in MP.locator.items():
        P_dfs.append(
            pd.DataFrame(get_parasitemia(p.dataset, p.data, ps, name, forecast=forecast))
        )
    return pd.concat(P_dfs)


def gather_freq(MP, ps, forecast=False):
    freq_dfs = []
    for name, p in MP.locator.items():
        freq_dfs.append(
            pd.DataFrame(get_freq(p.dataset, p.data, ps, name, forecast=forecast))
        )
    return pd.concat(freq_dfs)
