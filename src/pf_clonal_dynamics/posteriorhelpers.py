import jax.numpy as jnp

def get_site_by_variant(dataset, LD, ps, name, site):

    # Unpack variant info
    seq_names = LD.seq_names
    dates = LD.dates

    # Unpack posterior
    var_name = site 
    var = dataset[var_name]
    N_variant = var.shape[-1]
    T = var.shape[-2]

    # Compute medians and hdis for ps
    var_median = jnp.median(var, axis=0)

    var_hdis = [
        jnp.quantile(var, q=jnp.array([0.5 * (1 - p), 0.5 * (1 + p)]), axis=0)
        for i, p in enumerate(ps)
    ]

    var_dict = dict()
    var_dict["date"] = []
    var_dict["location"] = []
    var_dict["variant"] = []
    var_dict[f"median_{var_name}"] = []
    for p in ps:
        var_dict[f"{var_name}_upper_{round(p * 100)}"] = []
        var_dict[f"{var_name}_lower_{round(p * 100)}"] = []

    for variant in range(N_variant):
        var_dict["date"] += list(dates)
        var_dict["location"] += [name] * T
        var_dict["variant"] += [seq_names[variant]] * T
        var_dict[f"median_{var_name}"] += list(var_median[:, variant])
        for i, p in enumerate(ps):
            var_dict[f"{var_name}_upper_{round(ps[i] * 100)}"] += list(
                var_hdis[i][1, :, variant]
            )
            var_dict[f"{var_name}_lower_{round(ps[i] * 100)}"] += list(
                var_hdis[i][0, :, variant]
            )
    return var_dict


def add_median_freq(var_dict, dataset, excluded=None):
    f_name = "freq"
    f_name = f_name 

    freq = dataset[f_name]
    freq_median = jnp.median(freq, axis=0)
    var_dict["median_freq"] = []
    for variant in range(freq_median.shape[-1]):
        if variant != excluded:
            var_dict["median_freq"] += list(freq_median[:, variant])
    return var_dict

def get_growth_rates(dataset, LD, ps, name, forecast=False):
    var_dict = get_site_by_variant(dataset, LD, ps, name, "r", forecast=forecast)
    return add_median_freq(var_dict, dataset, forecast)

def get_parasitemia(dataset, LD, ps, name, forecast=False):
    var_dict = get_site_by_variant(dataset, LD, ps, name, "variant_specific_P", forecast=forecast)
    return add_median_freq(var_dict, dataset, forecast)

def get_freq(dataset, LD, ps, name, forecast=False):
    return get_site_by_variant(dataset, LD, ps, name, "freq", forecast=forecast)
