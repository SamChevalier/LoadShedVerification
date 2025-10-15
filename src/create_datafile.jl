#----------------------------------------------------------------------
using LinearSOC
using PGLib, Random, HDF5
using JuMP

# PGLib model
model_name = "case118_ieee" 

# Filename (WILL OVERWRITE)
h5write_filename = "data_file_118bus.h5"

# Number of datasets
n_data = 100000

# Hyperparams
alpha_min = 0.25
alpha_max = 0.85
perturb_percent = 0.50

# Debug?
debug = true

#----------------------------------------------------------------------
data = pglib(model_name)
data_copy = pglib(model_name)

Random.seed!(1234)

function generate_pd_qd!(data, data_copy)
    for (id, comp) in data["load"]
        comp["qd"] = data_copy["load"][id]["qd"]*(1-perturb_percent) + (2 * perturb_percent)*data_copy["load"][id]["qd"] * rand()
        comp["pd"] = data_copy["load"][id]["pd"]*(1-perturb_percent) + (2 * perturb_percent)*data_copy["load"][id]["pd"] * rand()
    end
    return data
end

file = h5open(h5write_filename, "w")
write_dataset(file, "alpha_max", alpha_max)
write_dataset(file, "alpha_min", alpha_min)
write_dataset(file, "total_samples", n_data)
write_dataset(file, "perturb_percent", perturb_percent)

g_data = create_group(file, "sample_data")
write_dataset(g_data, "index", [1])
write_dataset(g_data, "num_samples", [n_data])
for i in 1:n_data
    alpha = alpha_min + (alpha_max - alpha_min) * rand()
    generate_risk!(data, alpha)
    generate_pd_qd!(data, data_copy)

    group = create_group(g_data, string(i))

    # Load data
    load = create_group(group, "load")
    load_size = length(keys(data["bus"]))
    (qd_vals, pd_vals) = (zeros(Float32, load_size), zeros(Float32, load_size))
    for (key, value) in data["load"]
        qd_vals[value["load_bus"]] = value["qd"]
        pd_vals[value["load_bus"]] = value["pd"]
    end
    write_dataset(load, "qd", qd_vals)
    write_dataset(load, "pd", pd_vals)


    # Branch data
    branch = create_group(group, "branch")
    branch_size = length(keys(data["branch"]))
    prisk = Array{Float32}(undef, branch_size)
    for (key, value) in data["branch"]
        prisk[parse(Int, key)] = data["branch"][key]["power_risk"]
    end
    write_dataset(branch, "power_risk", prisk)
        
    # Alpha
    write_dataset(group, "alpha", [data["risk_weight"]])
end

close(file)