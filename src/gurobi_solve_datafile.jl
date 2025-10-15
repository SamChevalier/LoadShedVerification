using LinearSOC
using PGLib, Random, HDF5
using Gurobi
using JuMP

##################################################################
# STOP      STOP       STOP        STOP        STOP        STOP
# This file can take HOURS to run, it is built to hopefully
# save between sample solves but if it is force stopped at the right moment
# you will corrupt your HDF5 file. Run with caution and BACKUP YOUR FILE!
##################################################################

# PGLib model
model_name = "case118_ieee"
# Name of the file generated using create_datafile.jl
output_file = "data_file_118bus.h5"
# MIPGap percent to use in Gurobi
mip_gap = 0.05

# File will keep track of how many of the total samples have been solved
# automatically by default (-1).
# Change this below to essentially solve a certain amount and stop 
# (useful if you can only run the program for a few hours and are 
# afraid of abrupt program exit corrupting your file).
# i.e. number_to_solve = 1000 will solve 1000 unsolved samples using gurobi_optimizer
# in the file, and save. It will pickup at the next unsolved sample when you rerun this file.
global number_to_solve = -1

##################################################################
h5open(output_file, "r+") do file
    global sample_index = file["sample_data"]["index"][1]
    global sample_total = file["sample_data"]["num_samples"][1]
end

data = pglib(model_name)
generate_risk!(data, 1)
gurobi_optimizer = optimizer_with_attributes(Gurobi.Optimizer, "MIPGap" => mip_gap, "NodeFileStart" => 2.0)

while (sample_index != sample_total + 1 && number_to_solve != 0)
    h5open(output_file, "r+") do file
        for (id, comp) in data["load"]
            comp["qd"] = file["sample_data"][string(sample_index)]["load"]["qd"][parse(Int, id)]
            comp["pd"] = file["sample_data"][string(sample_index)]["load"]["pd"][parse(Int, id)]
        end
        data["risk_weight"] = file["sample_data"][string(sample_index)]["alpha"][1]
        for (id, comp) in data["branch"]
            comp["power_risk"] = file["sample_data"][string(sample_index)]["branch"]["power_risk"][parse(Int, id)]
        end
    end
    
    solution = solve_ops_perp(data, gurobi_optimizer)

    size = length(solution["solution"]["branch"])
    b_status = Array{Float32}(undef, size)
    for (key, value) in solution["solution"]["branch"]
        b_status[parse(Int, key)] = value["br_status"]
    end
    h5open(output_file, "r+") do file
        write_dataset(file["sample_data"][string(sample_index)]["branch"], "status", b_status)
        global sample_index += 1
        file["sample_data"]["index"][1] = sample_index
    end

    if (number_to_solve != -1 && number_to_solve != 0)
        global number_to_solve -= 1
    end
end