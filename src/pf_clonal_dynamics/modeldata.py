import pandas as pd
from .datahelpers import prep_dates, prep_density, prep_sequence_counts


class DensityData:
    def __init__(self, raw_density):
        self.dates, self.raw_cases = prep_density(raw_density)


class VariantData:
    def __init__(self, raw_density, raw_seqs):
        self.dates, date_to_index = prep_dates(
            pd.concat((raw_density.date, raw_seqs.date))
        )
        self.density = prep_density(raw_density, date_to_index=date_to_index)
        self.seq_names, self.seq_counts = prep_sequence_counts(
            raw_seqs, date_to_index=date_to_index
        )

    def make_numpyro_input(self):
        data = dict()
        data["parasite_density"] = self.density
        data["seq_counts"] = self.seq_counts
        data["N"] = self.seq_counts.sum(axis=1)
        data["seq_names"] = self.seq_names
        return data
