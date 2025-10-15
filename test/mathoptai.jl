import Ipopt
import Pkg
using PGLib
Pkg.build("MathOptAI")
Pkg.precompile()
import MathOptAI 
import MathOptInterface as MOI
ENV["JULIA_PYTHONCALL_EXE"] = "/pyenv-june/bin/python3"
import PythonCall
using JuMP
import PowerModels
import PowerModelsWildfire as pmw

data = pglib("case14_")

optimizer = Ipopt.Optimizer
model = Model(optimizer)
set_silent(model)
@variable(model, x[i=1:43])

predictor = MathOptAI.PytorchModel(joinpath(@__DIR__, "trained_model.pt"))
y, _ = MathOptAI.add_predictor(model, predictor, x)
σ(x) = 1 / (1 + exp(-x))

qd_upper_bounds = Float32[]
qd_lower_bounds = Float32[]
pd_upper_bounds = Float32[]
pd_lower_bounds = Float32[]
for (key, value) in data["load"]
    if (value["pd"] > 0)
        push!(pd_upper_bounds, value["pd"] * 2)
        push!(pd_lower_bounds, value["pd"] * 0.25)
    else
        push!(pd_upper_bounds, value["pd"] * 0.25)
        push!(pd_lower_bounds, value["pd"] * 2)
    end

    if (value["qd"] > 0)
        push!(qd_upper_bounds, value["qd"] * 2)
        push!(qd_lower_bounds, value["qd"] * 0.25)
    else
        push!(qd_upper_bounds, value["qd"] * 0.25)
        push!(qd_lower_bounds, value["qd"] * 2)
    end    
end
alpha_upper_bound = 0.9
alpha_lower_bound = 0
power_risk_upper_bound = 1
power_risk_lower_bound = 0

for (e, i) in enumerate(x)
    if (1 <= e <= length(data["branch"]))
        @constraint(model, x[e] >= power_risk_lower_bound)
        @constraint(model, x[e] <= power_risk_upper_bound)
    elseif (length(data["branch"]) + 1 <= e <= (length(data["branch"]) + 1 + length(qd_upper_bounds)))
        @constraint(model, x[e] >= qd_lower_bounds[e - length(data["branch"])])
        @constraint(model, x[e] <= qd_upper_bounds[e - length(data["branch"])])
    elseif ((length(data["branch"]) + 1 + length(qd_upper_bounds) + 1) <= e <= (length(data["branch"]) + 1 + length(qd_upper_bounds) + 1 + length(pd_upper_bounds)))
        @constraint(model, x[e] >= pd_lower_bounds[e - (length(data["branch"]) + length(qd_upper_bounds))])
        @constraint(model, x[e] <= pd_upper_bounds[e - (length(data["branch"]) + length(qd_upper_bounds))])
    else
        @constraint(model, x[e] >= alpha_lower_bound)
        @constraint(model, x[e] <= alpha_upper_bound)
    end
end

#for (e, i) in enumerate(y)
#    @constraint(model, σ(y[e]) <= 0.5)
#end

@objective(model, Min, sum(σ.(y)))

#set_optimizer_attribute(model, "max_iter", 6000)

optimize!(model)

model_decisions = convert(Vector{Int64}, σ.(value.(y)) .>= 0.5)

re_data = deepcopy(data)
for (e, (id, branch)) in enumerate(re_data["branch"])
   re_data["branch"][id]["br_status"] = model_decisions[e]
end

result = pmw._run_redispatch(re_data, PowerModels.SOCWRPowerModel, optimizer)