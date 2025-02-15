import datetime
import numpy as np
import pandas as pd

def prep_dates(raw_dates: pd.Series):
    _dates = pd.to_datetime(raw_dates)
    dmn = _dates.min()
    dmx = _dates.max()
    dates = [dmn + datetime.timedelta(days=d) for d in range(0, 1 + (dmx - dmn).days)]
    date_to_index = {d: i for (i, d) in enumerate(dates)}
    return dates, date_to_index


def prep_density(raw_density: pd.DataFrame, date_to_index=None, density_col="parasite_density"):
    raw_density["date"] = pd.to_datetime(raw_density["date"])
    if date_to_index is None:
        _, date_to_index = prep_dates(raw_density)

    T = len(date_to_index)
    C = np.full(T, np.nan)
    for _, row in raw_density.iterrows():
        C[date_to_index[row.date]] = row[density_col]
    return C


def format_seq_names(raw_names):
    if "other" in raw_names:
        names = []
        for s in raw_names:
            if s != "other":
                names.append(s)
        names.append("other")
        return names
    return raw_names


def counts_to_matrix(raw_seqs, seq_names, date_to_index=None, var_col="haplotype", seq_col="reads"):
    raw_seqs["date"] = pd.to_datetime(raw_seqs["date"])
    if date_to_index is None:
        _, date_to_index = prep_dates(raw_seqs)
    T = len(date_to_index)
    C = np.full((T, len(seq_names)), np.nan)
    for i, s in enumerate(seq_names):
        for _, row in raw_seqs[raw_seqs[var_col] == s].iterrows():
            C[date_to_index[row.date], i] = row[seq_col]

    # Loop over rows to correct for zero counts
    for d in range(T):
        if not np.isnan(C[d, :]).all():
            # If not all days sample are nan, replace nan with zeros
            np.nan_to_num(C[d, :], copy=False)
    return C


def prep_sequence_counts(raw_seqs: pd.DataFrame, date_to_index=None, var_col="haplotype"):
    raw_seq_names = pd.unique(raw_seqs[var_col])
    raw_seq_names.sort()
    seq_names = format_seq_names(raw_seq_names)
    C = counts_to_matrix(raw_seqs, seq_names, date_to_index=date_to_index, var_col=var_col)
    return seq_names, C


def prep_vaccination(raw_vacc: pd.DataFrame):
    dates, date_to_index = prep_dates(raw_vacc)
    C = np.zeros(len(dates))
    for _, row in raw_vacc.iterrows():
        C[date_to_index[row.date]] += row.percent_complete
    return dates, C
