using StanSample, Distributions, Dates
using DataFrames, CSV

ENV["JULIA_CMDSTAN_HOME"] = expanduser("~/cmdstan") # Path for cmdstan

include("../src/rt-from-frequency-dynamics.jl") # Loading project module
using .rt_from_frequency_dynamics

# 1. Get data
seq_df = DataFrame(CSV.File("../data/location-variant-sequence-counts.tsv"))
cases_df = DataFrame(CSV.File("../data/location-case-counts.tsv"))

# 2. Get lifetimes
g, onset = get_standard_delays()

# 3. Create the model for Rt
n_days = length(unique(cases_df.date))
rt_model = SplineTrend(20, n_days, 4)
priors = [["LAS"]]
seed_L = 7
forecast_L = 0

# 4. Choose lineage model type
LM =  FixedLineageModel(g, onset, rt_model, priors, seed_L, forecast_L)

# 5. Make stan models using shared model type
function make_state_models(seq_df::DataFrame, cases_df::DataFrame, LM::LineageModel, model_dir::String)
  loc_names = unique(seq_df.location)

  # Get state specific data structures
  LDs = LineageData[]
  for state in loc_names
    push!(LDs, 
      LineageData(filter(x -> x.location == state, seq_df), filter(x -> x.location == state, cases_df))
    )
  end

  # Making model objects to be run
  MSs = ModelStan[]
  for (state, LD) in zip(loc_names, LDs)
    # Changing name to be read by stan
    _state = replace(state, " " => "_")
    push!(MSs, make_stan(LM, LD, "rt-lineages-$(_state)", "$model_dir/$(_state)"))
  end
  return LDs, MSs
end

model_name = "all-states-preprint"
model_dir = "../data/sims/$(model_name)"
LDs, MSs = make_state_models(seq_df, cases_df, LM, model_dir)

# 6. Run stan models
for MS in MSs
  run!(MS, num_warmup = 1000, num_samples=1000, num_chains=4, get_samples=false) # Make sure to save samples
end
