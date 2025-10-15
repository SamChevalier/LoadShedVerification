using PGLib
using PowerModels
using JuMP, Ipopt
using Plots
using Gurobi
using Zygote, LinearAlgebra, SparseArrays
using HDF5

using HSL
import HSL_jll
bool = LIBHSL_isfunctional()
@info "HSL solvers are working: "*string(bool)

# see here: https://juliapy.github.io/PythonCall.jl/stable/pythoncall/#pythoncall-config
    # => these env definitions must go BEFORE "using PythonCall"
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "path/to/python3"

using PythonCall
using MathOptAI

sys = pyimport("sys")
println("Python version: ", sys.version)
println("Virtual Env Location: ", sys.prefix)

# ================
include("./functions.jl")
include("./structs.jl")

# %% call the test case
nl_list   = [20; 38; 186]
bus_list  = [14; 24; 118]
node_list = [32; 128; 512; 2048]
pg_list   = ["case14_"; "case24_ieee"; "pglib_opf_case118_ieee.m"]

#  bounds over which to verify
bounds = Dict(:load_scale_lb => 0.75,   :load_scale_ub  => 1.25,
              :risk_lb       => 0.25,   :risk_ub        => 0.75,
              :alpha_lb      => 0.25,   :alpha_ub       => 0.75)

# Print the number of NN parameters
print_NN_params(nl_list, bus_list, node_list)

n_samples = 100
verify_vs_sample_loadshed(bus_list, node_list, pg_list, n_samples)
