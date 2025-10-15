using Ipopt
using PowerModels, PGLib
using PowerModelsWildfire

network_data = pglib("case14_")

# this will just fomrulate and solve the redispatch model
solve_model(network_data, SOCWRPowerModel, Ipopt.Optimizer, PowerModelsWildfire._build_redispatch)

# THIS will return a JuMP model of the problem, ready to be solved
pm = instantiate_model(network_data, SOCWRPowerModel, PowerModelsWildfire._build_redispatch)
pm.model

# now, instead, we need to do this for the OPS problem...
pm = instantiate_model(network_data, _PM.SOCWRPowerModel, _build_ops) # ??
# once you have this model, add a constraint that links the NN line predictions with the 
# line status variables. then, we optimize over the full thing!