# %% read and test
using HDF5

nl_list   = [20; 38; 186]
bus_list  = [14; 24; 118]
node_list = [32; 128; 512; 2048]
pg_list   = ["case14_"; "case24_ieee"; "pglib_opf_case118_ieee.m"]

# %% Create plot!
c1 = 165/256
c2 = 42/256
c3 = 42/256
redd = RGB(c1,c2,c3)

plots = []
for bus in bus_list
    for node in node_list
        data_file_ai = "data/"*string(bus)*"bus_"*string(node)*"node_MathOptAI_test.h5"
        data_file_sample = "data/"*string(bus)*"bus_"*string(node)*"node_sampling_test.h5"

        fid   = h5open(data_file_ai, "r")
        obj_soc_zrelax_ai   = read(fid, "obj_soc_zrelax")
        obj_soc_zsnap_ai    = read(fid, "obj_soc_zsnap")
        obj_acopf_zsnap_ai  = read(fid, "obj_acopf_zsnap")
        close(fid)

        fid                     = h5open(data_file_sample, "r")
        obj_soc_zrelax_sample   = read(fid, "obj_soc_zrelax")
        obj_soc_zsnap_sample    = read(fid, "obj_soc_zsnap")
        obj_acopf_zsnap_sample  = read(fid, "obj_acopf_zsnap")
        close(fid)

        title = string(bus)*"-bus "*string(node)*"NN"

        if node == 32
            if bus == 118
                p = scatter(obj_soc_zrelax_sample, label="random sample",legendfont = 8, ylabel="Load Shed (pu)", xlabel="Sample index", markersize  = 3, color = :steelblue, markerstrokewidth = 0.01)
                plot!(p, obj_soc_zrelax_ai*ones(100), label="MathOptAI bound",legend = :topright, foreground_color_legend = nothing, width = 2.5, color = redd, title=title, titlefont=11,linestyle = :dash)
            else
                p = scatter(obj_soc_zrelax_sample, label="", ylabel="Load Shed (pu)", markersize  = 3, color = :steelblue, markerstrokewidth = 0.01)
                plot!(p, obj_soc_zrelax_ai*ones(100), label="", width = 2.5, color = redd, title=title, titlefont=11,linestyle = :dash)
            end

        elseif bus == 118
            p = scatter(obj_soc_zrelax_sample, label="random sample",legendfont = 8, xlabel="Sample index", markersize  = 3, color = :steelblue, markerstrokewidth = 0.01)
            plot!(p, obj_soc_zrelax_ai*ones(100), label="MathOptAI bound",legend = :topright, foreground_color_legend = nothing, width = 2.5, color = redd, title=title, titlefont=11,linestyle = :dash) 
        else
            p = scatter(obj_soc_zrelax_sample, label="", markersize  = 3, color = :steelblue, markerstrokewidth = 0.01)
            plot!(p, obj_soc_zrelax_ai*ones(100), label="", width = 2.5, color = redd, title=title, titlefont=11,linestyle = :dash)
        end

        push!(plots,p)
    end
end

plot(plots..., layout=(3,4), size=(1000, 600), left_margin=3Plots.mm, bottom_margin=3.5Plots.mm)
# => savefig("bound.pdf")

# %% get Table data
TmI   = zeros(4,6)
TmII  = zeros(4,6)
TmIII = zeros(4,6)
row = 1
for node in node_list
    col = 1
    for bus in bus_list
        data_file_ai = "data/"*string(bus)*"bus_"*string(node)*"node_MathOptAI_test.h5"
        data_file_sample = "data/"*string(bus)*"bus_"*string(node)*"node_sampling_test.h5"

        fid   = h5open(data_file_ai, "r")
        obj_soc_zrelax_ai   = read(fid, "obj_soc_zrelax")
        obj_soc_zsnap_ai    = read(fid, "obj_soc_zsnap")
        obj_acopf_zsnap_ai  = read(fid, "obj_acopf_zsnap")
        # dt                  = read(fid, "solve_time_mathoptai")
        # println(dt)
        close(fid)

        println()
        fid                     = h5open(data_file_sample, "r")
        obj_soc_zrelax_sample   = read(fid, "obj_soc_zrelax")
        obj_soc_zsnap_sample    = read(fid, "obj_soc_zsnap")
        obj_acopf_zsnap_sample  = read(fid, "obj_acopf_zsnap")
        close(fid)

        TmI[row,2*col-1] = round(obj_soc_zrelax_ai; digits=2)
        TmI[row,2*col  ] = round(maximum(obj_soc_zrelax_sample); digits=2)

        TmII[row,2*col-1] = round(obj_soc_zsnap_ai; digits=2)
        TmII[row,2*col  ] = round(maximum(obj_soc_zsnap_sample); digits=2)

        TmIII[row,2*col-1] = round(obj_acopf_zsnap_ai; digits=2)
        TmIII[row,2*col  ] = round(maximum(obj_acopf_zsnap_sample); digits=2)

        col+=1
    end
    row+=1
end

# %% get timing data
T1   = zeros(4,3)
T2   = zeros(4,3)
T3   = zeros(4,3)
T4   = zeros(4,3)
T5   = zeros(4,6)
row = 1
for node in node_list
    col = 1
    for bus in bus_list
        data_file_ai = "data/"*string(bus)*"bus_"*string(node)*"node_MathOptAI_test.h5"
        data_file_sample = "data/"*string(bus)*"bus_"*string(node)*"node_sampling_test.h5"

        fid   = h5open(data_file_ai, "r")
        time_mathoptai       = read(fid, "time_mathoptai")
        time_soc_zrelax_ai   = read(fid, "time_soc_zrelax")
        time_soc_zsnap_ai    = read(fid, "time_soc_zsnap")
        time_acopf_zsnap_ai  = read(fid, "time_acopf_zsnap")
        close(fid)

        T1[row,col] = time_mathoptai
        T2[row,col] = time_soc_zrelax_ai 
        T3[row,col] = time_soc_zsnap_ai  
        T4[row,col] = time_acopf_zsnap_ai

        T5[row,2*col-1] = round(time_mathoptai; digits=2)
        T5[row,2*col  ] = round(time_mathoptai+time_acopf_zsnap_ai; digits=2)

        col+=1
    end
    row+=1
end

# %% PowerPlots tests
using PowerPlots

# parse
network_data = pglib(pg_list[1])
gm = parse_PM_to_SOCGridModel(network_data; perturb=false)
pd_max = gm.pd*1.25
qd_max = gm.qd*1.25
pd0    = gm.pd
qd0    = gm.qd

bus  = 14
node = 2048
data_file_ai = "data/"*string(bus)*"bus_"*string(node)*"node_MathOptAI_test.h5"
fid   = h5open(data_file_ai, "r")
pd_mathoptai       = read(fid, "pd_mathoptai")
qd_mathoptai       = read(fid, "qd_mathoptai")
zl0                = read(fid, "zl0")
close(fid)

# create two networks: one with the relative pd, and one with the relative pd
network_data_pd = deepcopy(network_data)
network_data_qd = deepcopy(network_data)

# we will use the active power as the surrogate for plotting qd
for (load,val) in network_data_pd["load"]
    if val["status"] == 1
        bus = val["load_bus"]
        if pd0[bus] > 1e-6
            network_data_pd["load"][load]["pd"] = 100*(pd_mathoptai[bus]-pd0[bus])/pd0[bus]
            println(pd_mathoptai[bus]/pd0[bus])
        else
            network_data_qd["load"][load]["pd"] = 0.0
        end
    end
end

println()


for (load,val) in network_data_qd["load"]
    if val["status"] == 1
        bus = val["load_bus"]
        if qd0[bus] > 1e-6
            network_data_qd["load"][load]["pd"] = 100(qd_mathoptai[bus] - qd0[bus])/qd0[bus]
            println(qd_mathoptai[bus]/qd0[bus])
        else
            network_data_qd["load"][load]["pd"] = 0.0
        end
    end
end

network_data_qd["shunt"] = Dict{String, Any}()
network_data_qd["gen"]   = Dict{String, Any}()

network_data_pd["shunt"] = Dict{String, Any}()
network_data_pd["gen"]   = Dict{String, Any}()

# %%
using Setfield

p = powerplot(network_data_pd, load   = (:data=>"pd",  :data_type=>"quantitative"); width=400, height=300)
p.layer[4]["encoding"]["color"]["title"] = "ΔPd (%)"
p.layer[4]["encoding"]["color"]["scale"]["domain"]=[-25,-1,1,25]
p.layer[4]["encoding"]["color"]["scale"]["range"]=["#4A6741","#FFFFFF","#FFFFFF","#"*string(hex(redd))] #["#A6550D", string(hex(redd)), "#EB7433", "#FFFFFF", "#F8B17C"]
p.layer[4]["mark"] = Dict(
    "opacity" => 1.0,
    "type" => :circle,
    "stroke" => "black",        # outline color
    "strokeWidth" => 1          # optional: outline thickness
)
# => p.layer[3]["encoding"]["color"]["legend"]=false
p.layer[2]["encoding"]["color"]["legend"]=false
# => p.layer[1]["layer"][1]["encoding"]["color"]["legend"]=false
p.layer[3]["encoding"]["size"]["value"] = 200

p.layer[1]["layer"][1]["encoding"]["color"]["scale"]["range"] = ["#000000"]
p.layer[1]["layer"][1]["encoding"]["size"]["value"] = 2
p

# %%

save("powerplot_pd.pdf", p)
# %% ==============
using Setfield

p = powerplot(network_data_qd, load   = (:data=>"pd",  :data_type=>"quantitative"); width=400, height=300)
p.layer[4]["encoding"]["color"]["title"] = "ΔQd (%)"
p.layer[4]["encoding"]["color"]["scale"]["domain"]=[-25,-1,1,25]
p.layer[4]["encoding"]["color"]["scale"]["range"]=["#4A6741","#FFFFFF","#FFFFFF","#"*string(hex(redd))] #["#A6550D", string(hex(redd)), "#EB7433", "#FFFFFF", "#F8B17C"]
p.layer[4]["mark"] = Dict(
    "opacity" => 1.0,
    "type" => :circle,
    "stroke" => "black",        # outline color
    "strokeWidth" => 1          # optional: outline thickness
)

# => p.layer[3]["encoding"]["color"]["legend"]=false
p.layer[2]["encoding"]["color"]["legend"]=false
# => p.layer[1]["layer"][1]["encoding"]["color"]["legend"]=false
p.layer[3]["encoding"]["size"]["value"] = 200

p.layer[1]["layer"][1]["encoding"]["color"]["scale"]["range"] = ["#000000"]
p.layer[1]["layer"][1]["encoding"]["size"]["value"] = 2
p

# %%

save("powerplot_qd.pdf", p)

# %%
gm_shed        = deepcopy(gm)

zl0, logit_zl0 = line_status(gm, bounds, nn_model; high_load=true)

# %%
# model to call the predictor
model = Model(Ipopt.Optimizer)
set_silent(model)
bus = 14

# define predictor inputs as scalars
x     = [0.5*ones(gm.nl); 1.25*qd0; 1.25*pd0; 0.5]
# now, we need to normalize
nn_model = "outputs/"*string(bus)*"_bus/"*string(bus)*"_bus_"*string(node)*"node.pt"

normalization_data = nn_model[1:findlast(==('_'), nn_model)]*"normalization_values.h5"
fid   = h5open(normalization_data, "r")
mean = read(fid, "mean")
std  = read(fid, "std")
close(fid)

xn   = (x .- mean)./(std)
predictor = MathOptAI.PytorchModel(joinpath(pwd(), nn_model))
logit_zl, _ = MathOptAI.add_predictor(model, predictor, xn; gray_box = true)

@objective(model, Min, 0)
optimize!(model)

#plot(sig.(value.(logit_zl)))