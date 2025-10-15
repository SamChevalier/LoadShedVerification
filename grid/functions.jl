using LinearAlgebra, JuMP
include("./structs.jl")

# sigmoid!
sig(x) = 1.0 / (1.0 + exp(-x))

function pfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return  @. (g+g_fr)*(vm_fr/tm)^2 - g*vm_fr/tm*vm_to*cos(va_fr-va_to-ta) + -b*vm_fr/tm*vm_to*sin(va_fr-va_to-ta)
end

function pfr_dvmfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return  @. 2*(g+g_fr)*(vm_fr/(tm^2)) - g/tm*vm_to*cos(va_fr-va_to-ta) + -b/tm*vm_to*sin(va_fr-va_to-ta)
end

function pfr_dvmto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return  @. - g*vm_fr/tm*cos(va_fr-va_to-ta) + -b*vm_fr/tm*sin(va_fr-va_to-ta)
end

function pfr_dvafr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return  @. g*vm_fr/tm*vm_to*sin(va_fr-va_to-ta) + -b*vm_fr/tm*vm_to*cos(va_fr-va_to-ta)
end

function pfr_dvato(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return  @. - g*vm_fr/tm*vm_to*sin(va_fr-va_to-ta) + b*vm_fr/tm*vm_to*cos(va_fr-va_to-ta)
end

function qfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. -(b+b_fr)*(vm_fr/tm)^2 + b*vm_fr/tm*vm_to*cos(va_fr-va_to-ta) + -g*vm_fr/tm*vm_to*sin(va_fr-va_to-ta)
end

function qfr_dvmfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. -2*(b+b_fr)*(vm_fr/tm^2) + b/tm*vm_to*cos(va_fr-va_to-ta) + -g/tm*vm_to*sin(va_fr-va_to-ta)
end

function qfr_dvmto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. b*vm_fr/tm*cos(va_fr-va_to-ta) + -g*vm_fr/tm*sin(va_fr-va_to-ta)
end

function qfr_dvafr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. - b*vm_fr/tm*vm_to*sin(va_fr-va_to-ta) + -g*vm_fr/tm*vm_to*cos(va_fr-va_to-ta)
end

function qfr_dvato(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. b*vm_fr/tm*vm_to*sin(va_fr-va_to-ta) + g*vm_fr/tm*vm_to*cos(va_fr-va_to-ta)
end

function pto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. (g+g_to)*vm_to^2 - g*vm_to*vm_fr/tm*cos(va_to-va_fr+ta) + -b*vm_to*vm_fr/tm*sin(va_to-va_fr+ta)
end

function pto_dvmfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. - g*vm_to/tm*cos(va_to-va_fr+ta) + -b*vm_to/tm*sin(va_to-va_fr+ta)
end

function pto_dvmto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. 2*(g+g_to)*vm_to - g*vm_fr/tm*cos(va_to-va_fr+ta) + -b*vm_fr/tm*sin(va_to-va_fr+ta)
end

function pto_dvafr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. - g*vm_to*vm_fr/tm*sin(va_to-va_fr+ta) + b*vm_to*vm_fr/tm*cos(va_to-va_fr+ta)
end

function pto_dvato(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. g*vm_to*vm_fr/tm*sin(va_to-va_fr+ta) + -b*vm_to*vm_fr/tm*cos(va_to-va_fr+ta)
end

function qto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. -(b+b_to)*vm_to^2 + b*vm_to*vm_fr/tm*cos(va_to-va_fr+ta) + -g*vm_to*vm_fr/tm*sin(va_to-va_fr+ta)
end

function qto_dvmfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. b*vm_to/tm*cos(va_to-va_fr+ta) + -g*vm_to/tm*sin(va_to-va_fr+ta)
end

function qto_dvmto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. -2*(b+b_to)*vm_to + b*vm_fr/tm*cos(va_to-va_fr+ta) + -g*vm_fr/tm*sin(va_to-va_fr+ta)
end

function qto_dvafr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. b*vm_to*vm_fr/tm*sin(va_to-va_fr+ta) + g*vm_to*vm_fr/tm*cos(va_to-va_fr+ta)
end

function qto_dvato(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    vm_fr = @view vm[fr_buses]
    vm_to = @view vm[to_buses]
    va_fr = @view va[fr_buses]
    va_to = @view va[to_buses]
    return @. -b*vm_to*vm_fr/tm*sin(va_to-va_fr+ta) + -g*vm_to*vm_fr/tm*cos(va_to-va_fr+ta)
end

Jsp_two(nl, nb, fr_buses, to_buses, dfr, dto) = sparse(1:nl, fr_buses, dfr, nl, nb) + sparse(1:nl, to_buses, dto, nl, nb)

# update for JuMP: summing sparse seemed to return dense!
Jsp(nl, nb, fr_buses, to_buses, dfr, dto) = sparse([1:nl; 1:nl], [fr_buses; to_buses], [dfr; dto], nl, nb)

function build_Jacobian(fnct::String, vr::String, vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    # fnct = function
    # vr   = differentiating variable
    if fnct == "pfr" && vr == "vm"
        dfr = pfr_dvmfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
        dto = pfr_dvmto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    elseif fnct == "pfr" && vr == "va"
        dfr = pfr_dvafr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
        dto = pfr_dvato(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    elseif fnct == "pto" && vr == "vm" 
        dfr = pto_dvmfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
        dto = pto_dvmto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    elseif fnct == "pto" && vr == "va" 
        dfr = pto_dvafr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
        dto = pto_dvato(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    elseif fnct == "qfr" && vr == "vm" 
        dfr = qfr_dvmfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
        dto = qfr_dvmto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    elseif fnct == "qfr" && vr == "va" 
        dfr = qfr_dvafr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
        dto = qfr_dvato(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    elseif fnct == "qto" && vr == "vm" 
        dfr = qto_dvmfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
        dto = qto_dvmto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    else
        if !(fnct == "qto" && vr == "va")
            @warn("combination not recognized!")
        end
        dfr = qto_dvafr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
        dto = qto_dvato(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    end

    M = Jsp(nl, nb, fr_buses, to_buses, dfr, dto)

    return M
end

# call the ref
function parse_PM_to_SOCGridModel(network_data; perturb=false)
    # build the ref
    ref = PowerModels.build_ref(network_data)[:it][pm_it_sym][:nw][nw_id_default]

    #  build a custom OPF objective function -- the key here is "_build_opf_cl"
    pm = instantiate_model(network_data, ACPPowerModel, PowerModels.build_opf)#, PowerModels._build_opf_cl)
    OPF_soln = optimize_model!(pm, optimizer=Ipopt.Optimizer)
    println(OPF_soln["objective"])

    OPF_soln["solution"]
    nb = length(OPF_soln["solution"]["bus"])
    vm_pm = zeros(nb)
    va_pm = zeros(nb)
    for ii in 1:nb
        vm_pm[ii] = OPF_soln["solution"]["bus"][string(ii)]["vm"]
        va_pm[ii] = OPF_soln["solution"]["bus"][string(ii)]["va"]
    end

    nl = length(OPF_soln["solution"]["branch"])
    pf_pm = zeros(nl)
    qf_pm = zeros(nl)
    pt_pm = zeros(nl)
    qt_pm = zeros(nl)

    for ii in 1:nl
        pf_pm[ii] = OPF_soln["solution"]["branch"][string(ii)]["pf"]
        qf_pm[ii] = OPF_soln["solution"]["branch"][string(ii)]["qf"]
        pt_pm[ii] = OPF_soln["solution"]["branch"][string(ii)]["pt"]
        qt_pm[ii] = OPF_soln["solution"]["branch"][string(ii)]["qt"]
    end

    # ============== Try to replicate
    model = Model(Ipopt.Optimizer)
    nb = length(network_data["bus"])
    ng = length(network_data["gen"])
    nl = length(network_data["branch"])

    bus_list  = collect(1:nb)
    line_list = collect(1:nl)
    gen_list  = collect(1:ng)

    # are all lines on?
    if nl != sum([network_data["branch"][string(line)]["br_status"] for line in line_list])
        @warn("some lines are off")
    end

    # are all gens on?
    if ng != sum([network_data["gen"][string(gen)]["gen_status"] for gen in gen_list])
        @warn("some gens are off")
    end

    @variable(model, vm[1:nb])
    @variable(model, va[1:nb])
    @variable(model, pg[1:ng])
    @variable(model, qg[1:ng])
    @variable(model, pto[1:nl])
    @variable(model, pfr[1:nl])
    @variable(model, qto[1:nl])
    @variable(model, qfr[1:nl])

    # set starts
    if perturb == false
        for ii in 1:nb
            set_start_value(vm[ii], vm_pm[ii])
            set_start_value(va[ii], va_pm[ii])
        end

        for ii in 1:nl
            set_start_value(pto[ii], pf_pm[ii])
            set_start_value(pfr[ii], qf_pm[ii])
            set_start_value(qto[ii], pt_pm[ii])
            set_start_value(qfr[ii], qt_pm[ii])
        end
    else
        # perturb!!
        for ii in 1:nb
            set_start_value(vm[ii], 1.0 + 0.1*randn())
            set_start_value(va[ii], 0.0 + 0.1*randn())
        end

        for ii in 1:nl
            set_start_value(pto[ii], randn())
            set_start_value(pfr[ii], randn())
            set_start_value(qto[ii], randn())
            set_start_value(qfr[ii], randn())
        end
        for ii in 1:ng
            set_start_value(pg[ii], randn())
            set_start_value(qg[ii], randn())
        end
    end

    ref_bus = 1
    for (bus,val) in network_data["bus"]
        if val["bus_type"] == 3
            ref_bus = val["bus_i"]
        end
    end

    # generator parameters
    pg_max = [network_data["gen"][string(gen)]["pmax"] for gen in gen_list]
    pg_min = [network_data["gen"][string(gen)]["pmin"] for gen in gen_list]
    qg_max = [network_data["gen"][string(gen)]["qmax"] for gen in gen_list]
    qg_min = [network_data["gen"][string(gen)]["qmin"] for gen in gen_list]

    # map gens to buses
    Eg = zeros(nb,ng)
    clin = zeros(ng)
    c0 = zeros(ng)
    ii = 1
    for (gen,val) in network_data["gen"]
        Eg[val["gen_bus"],val["index"]] = 1
        if val["cost"] == []
            clin[val["index"]] = 0.0
            c0[val["index"]] = 0.0
        elseif length(val["cost"]) == 2
            clin[val["index"]] = val["cost"][1]
            c0[val["index"]] = val["cost"][2]
        else # => length(val["cost"]) == 3
            @warn("this one has quadratic terms!")
            clin[val["index"]] = val["cost"][2]
        end

    end
    # => push!(cl, val["cost"][1])
    # => push!(c0, val["cost"][2])
    # => push!(cg_ind, val["index"])

    # network parameters
    fr_buses = [network_data["branch"][string(line)]["f_bus"] for line in line_list]
    to_buses = [network_data["branch"][string(line)]["t_bus"] for line in line_list] 
    r        = [network_data["branch"][string(line)]["br_r"]  for line in line_list] 
    x        = [network_data["branch"][string(line)]["br_x"]  for line in line_list]
    g        = real(1 ./ (r+im*x))
    b        = imag(1 ./ (r+im*x))
    ta       = [network_data["branch"][string(line)]["shift"] for line in line_list] 
    tm       = [network_data["branch"][string(line)]["tap"]   for line in line_list] 
    g_to     = [network_data["branch"][string(line)]["g_to"]  for line in line_list] 
    g_fr     = [network_data["branch"][string(line)]["g_fr"]  for line in line_list] 
    b_to     = [network_data["branch"][string(line)]["b_to"]  for line in line_list] 
    b_fr     = [network_data["branch"][string(line)]["b_fr"]  for line in line_list] 
    amax     = [network_data["branch"][string(line)]["angmax"]  for line in line_list] 
    amin     = [network_data["branch"][string(line)]["angmin"]  for line in line_list] 

    # loads
    pd = zeros(nb)
    qd = zeros(nb)
    for (load,val) in network_data["load"]
        if val["status"] == 1
            bus = val["load_bus"]
            pd[bus] += val["pd"]
            qd[bus] += val["qd"]
        end
    end

    # shunts
    gs = zeros(nb)
    bs = zeros(nb)
    for (shunt,val) in network_data["shunt"]
        if val["status"] == 1
            bus = val["shunt_bus"]
            gs[bus] += val["gs"]
            bs[bus] += val["bs"]
        end
    end

    # build the incidence matrix
    E = zeros(nl,nb)
    for ii in 1:nl
        E[ii,fr_buses[ii]] = 1.0
        E[ii,to_buses[ii]] = -1.0
    end
    Efr = (E + abs.(E))/2
    Eto = (abs.(E) - E)/2

    # constraint 1: voltage magnitudes
    vmax = [network_data["bus"][string(bus)]["vmax"] for bus in bus_list]
    vmin = [network_data["bus"][string(bus)]["vmin"] for bus in bus_list]

    # constraint 2: flow limits
    smax = [minimum([network_data["branch"][string(line)]["rate_a"];
                    network_data["branch"][string(line)]["rate_b"];
                    network_data["branch"][string(line)]["rate_b"]]) for line in line_list]

    # flows
    vm_fr = Efr*vm
    vm_to = Eto*vm
    va_fr = Efr*va
    va_to = Eto*va

    @constraint(model, pfr .== @.  (g+g_fr)*(vm_fr/tm)^2 - g*vm_fr/tm*vm_to*cos(va_fr-va_to-ta) + -b*vm_fr/tm*vm_to*sin(va_fr-va_to-ta) )
    @constraint(model, qfr .== @. -(b+b_fr)*(vm_fr/tm)^2 + b*vm_fr/tm*vm_to*cos(va_fr-va_to-ta) + -g*vm_fr/tm*vm_to*sin(va_fr-va_to-ta) )
    @constraint(model, pto .== @.  (g+g_to)*vm_to^2      - g*vm_to*vm_fr/tm*cos(va_to-va_fr+ta) + -b*vm_to*vm_fr/tm*sin(va_to-va_fr+ta) )
    @constraint(model, qto .== @. -(b+b_to)*vm_to^2      + b*vm_to*vm_fr/tm*cos(va_to-va_fr+ta) + -g*vm_to*vm_fr/tm*sin(va_to-va_fr+ta) )

    # add constraints -- ignore angle limits
    @constraint(model, va[ref_bus] == 0.0)
    @constraint(model, vmin   .<= vm .<= vmax)
    @constraint(model, pg_min .<= pg .<= pg_max)
    @constraint(model, qg_min .<= qg .<= qg_max)

    @constraint(model, pfr.^2 + qfr.^2 .<= smax.^2 )
    @constraint(model, pto.^2 + qto.^2 .<= smax.^2 )

    @constraint(model, Eg*pg .== pd - gs.*vm.^2 + Efr'*pfr + Eto'*pto)
    @constraint(model, Eg*qg .== qd - bs.*vm.^2 + Efr'*qfr + Eto'*qto)

    @objective(model, Min, clin'*pg)
    optimize!(model)

    if abs(objective_value(model) - OPF_soln["objective"]) < 1e-3
        println("Reformulated model matches the PM \u2705")
    else
        @warn("Reformulated model objective doesn't match PM objective (missing quadratic cost terms?).")
        println(objective_value(model))
        println(OPF_soln["objective"])
    end

    xtr = tm .* cos.(ta)
    xti = tm .* sin.(ta)

    # sparsify
    Efr  = sparse(Efr)
    Eto  = sparse(Eto)

    # build the power flow matrices
    Tpfr  = sparse(diagm(@. (g+g_fr)/tm^2)*Efr)
    TpRfr = sparse(diagm(@. (-g*xtr+b*xti)/tm^2))
    TpIfr = sparse(diagm(@. (-b*xtr-g*xti)/tm^2))
    Tqfr  = sparse(diagm(@. (-(b+b_fr)/tm^2))*Efr)
    TqRfr = sparse(diagm(@. -(-b*xtr-g*xti)/tm^2))
    TqIfr = sparse(diagm(@. (-g*xtr+b*xti)/tm^2))
    Tpto  = sparse(diagm(@. (g+g_to))*Eto)
    TpRto = sparse(diagm(@. (-g*xtr-b*xti)/tm^2))
    TpIto = sparse(diagm(@. -(-b*xtr+g*xti)/tm^2))
    Tqto  = sparse(diagm(@. -(b+b_to))*Eto)
    TqRto = sparse(diagm(@. -(-b*xtr+g*xti)/tm^2))
    TqIto = sparse(diagm(@. -(-g*xtr-b*xti)/tm^2))

    # for use with w_fr and w_to
    Tp_wfr  = sparse(diagm(@. (g+g_fr)/tm^2))
    Tq_wfr  = sparse(diagm(@. (-(b+b_fr)/tm^2)))
    Tp_wto  = sparse(diagm(@. (g+g_to)))
    Tq_wto  = sparse(diagm(@. -(b+b_to)))

    # grab the SOC bounds
    wr_min, wr_max, wi_min, wi_max = soc_voltage_bound_vectors(network_data, nl, fr_buses, to_buses)

    # build diagonal shunt matrices
    Bs_neg = diagm(min.(bs,0.0))
    Bs_pos = diagm(max.(bs,0.0))
    Gs     = diagm(gs)

    # now, build it
    gm = SOCGridModel(nb,nl,ng,value.(vm),value.(va),value.(pg),value.(qg),g,g_fr,g_to,b,b_fr,b_to,tm,ta,
                      xtr,xti,pd,qd,gs,bs,vmax,vmin,wr_min,wr_max,wi_min,wi_max,pg_max,pg_min,qg_max,
                      qg_min,smax,clin,fr_buses,to_buses,Eg,Efr,Eto,Tpfr,TpRfr,TpIfr,Tqfr,TqRfr,TqIfr,
                      Tpto,TpRto,TpIto,Tqto,TqRto,TqIto,Tp_wfr,Tq_wfr,Tp_wto,Tq_wto,Bs_neg,Bs_pos,Gs) 

    # output useful stuff
    return gm
end

# call the ref
function sovle_acopf_directly(network_data, perturb)
    # build the ref
    ref = PowerModels.build_ref(network_data)[:it][pm_it_sym][:nw][nw_id_default]

    #  build a custom OPF objective function -- the key here is "_build_opf_cl"
    pm = instantiate_model(network_data, ACPPowerModel, PowerModels.build_opf)#, PowerModels._build_opf_cl)
    OPF_soln = optimize_model!(pm, optimizer=Ipopt.Optimizer)
    println(OPF_soln["objective"])

    OPF_soln["solution"]
    nb = length(OPF_soln["solution"]["bus"])
    vm_pm = zeros(nb)
    va_pm = zeros(nb)
    for ii in 1:nb
        vm_pm[ii] = OPF_soln["solution"]["bus"][string(ii)]["vm"]
        va_pm[ii] = OPF_soln["solution"]["bus"][string(ii)]["va"]
    end

    nl = length(OPF_soln["solution"]["branch"])
    pf_pm = zeros(nl)
    qf_pm = zeros(nl)
    pt_pm = zeros(nl)
    qt_pm = zeros(nl)

    for ii in 1:nl
        pf_pm[ii] = OPF_soln["solution"]["branch"][string(ii)]["pf"]
        qf_pm[ii] = OPF_soln["solution"]["branch"][string(ii)]["qf"]
        pt_pm[ii] = OPF_soln["solution"]["branch"][string(ii)]["pt"]
        qt_pm[ii] = OPF_soln["solution"]["branch"][string(ii)]["qt"]
    end

    # ============== Try to replicate
    model = Model(Ipopt.Optimizer)
    nb = length(network_data["bus"])
    ng = length(network_data["gen"])
    nl = length(network_data["branch"])

    bus_list  = collect(1:nb)
    line_list = collect(1:nl)
    gen_list  = collect(1:ng)

    # are all lines on?
    if nl != sum([network_data["branch"][string(line)]["br_status"] for line in line_list])
        @warn("some lines are off")
    end

    # are all gens on?
    if ng != sum([network_data["gen"][string(gen)]["gen_status"] for gen in gen_list])
        @warn("some gens are off")
    end

    @variable(model, vm[1:nb])
    @variable(model, va[1:nb])
    @variable(model, pg[1:ng])
    @variable(model, qg[1:ng])
    @variable(model, pto[1:nl])
    @variable(model, pfr[1:nl])
    @variable(model, qto[1:nl])
    @variable(model, qfr[1:nl])

    # set starts
    if perturb == false
        for ii in 1:nb
            set_start_value(vm[ii], vm_pm[ii])
            set_start_value(va[ii], va_pm[ii])
        end

        for ii in 1:nl
            set_start_value(pto[ii], pf_pm[ii])
            set_start_value(pfr[ii], qf_pm[ii])
            set_start_value(qto[ii], pt_pm[ii])
            set_start_value(qfr[ii], qt_pm[ii])
        end
    else
        # perturb!!
        for ii in 1:nb
            set_start_value(vm[ii], 1.0 + 0.1*randn())
            set_start_value(va[ii], 0.0 + 0.1*randn())
        end

        for ii in 1:nl
            set_start_value(pto[ii], randn())
            set_start_value(pfr[ii], randn())
            set_start_value(qto[ii], randn())
            set_start_value(qfr[ii], randn())
        end
        for ii in 1:ng
            set_start_value(pg[ii], randn())
            set_start_value(qg[ii], randn())
        end
    end

    ref_bus = 1
    for (bus,val) in network_data["bus"]
        if val["bus_type"] == 3
            ref_bus = val["bus_i"]
        end
    end

    # generator parameters
    pg_max = [network_data["gen"][string(gen)]["pmax"] for gen in gen_list]
    pg_min = [network_data["gen"][string(gen)]["pmin"] for gen in gen_list]
    qg_max = [network_data["gen"][string(gen)]["qmax"] for gen in gen_list]
    qg_min = [network_data["gen"][string(gen)]["qmin"] for gen in gen_list]

    # map gens to buses
    Eg = zeros(nb,ng)
    clin = zeros(ng)
    c0 = zeros(ng)
    ii = 1
    for (gen,val) in network_data["gen"]
        Eg[val["gen_bus"],val["index"]] = 1
        if val["cost"] == []
            clin[val["index"]] = 0.0
            c0[val["index"]] = 0.0
        else
            clin[val["index"]] = val["cost"][1]
            c0[val["index"]] = val["cost"][2]
        end
    end
    # => push!(cl, val["cost"][1])
    # => push!(c0, val["cost"][2])
    # => push!(cg_ind, val["index"])
    # => if length(val["cost"]) == 3
    # =>     @warn("this one has quadratic terms!")
    # => end

    # network parameters
    fr_buses = [network_data["branch"][string(line)]["f_bus"] for line in line_list]
    to_buses = [network_data["branch"][string(line)]["t_bus"] for line in line_list] 
    r        = [network_data["branch"][string(line)]["br_r"]  for line in line_list] 
    x        = [network_data["branch"][string(line)]["br_x"]  for line in line_list]
    g        = real(1 ./ (r+im*x))
    b        = imag(1 ./ (r+im*x))
    ta       = [network_data["branch"][string(line)]["shift"] for line in line_list] 
    tm       = [network_data["branch"][string(line)]["tap"]   for line in line_list] 
    g_to     = [network_data["branch"][string(line)]["g_to"]  for line in line_list] 
    g_fr     = [network_data["branch"][string(line)]["g_fr"]  for line in line_list] 
    b_to     = [network_data["branch"][string(line)]["b_to"]  for line in line_list] 
    b_fr     = [network_data["branch"][string(line)]["b_fr"]  for line in line_list] 
    amax     = [network_data["branch"][string(line)]["angmax"]  for line in line_list] 
    amin     = [network_data["branch"][string(line)]["angmin"]  for line in line_list] 

    # loads
    pd = zeros(nb)
    qd = zeros(nb)
    for (load,val) in network_data["load"]
        if val["status"] == 1
            bus = val["load_bus"]
            pd[bus] += val["pd"]
            qd[bus] += val["qd"]
        end
    end

    # shunts
    gs = zeros(nb)
    bs = zeros(nb)
    for (shunt,val) in network_data["shunt"]
        if val["status"] == 1
            bus = val["shunt_bus"]
            gs[bus] += val["gs"]
            bs[bus] += val["bs"]
        end
    end

    # build the incidence matrix
    E = zeros(nl,nb)
    for ii in 1:nl
        E[ii,fr_buses[ii]] = 1.0
        E[ii,to_buses[ii]] = -1.0
    end
    Efr = (E + abs.(E))/2
    Eto = (abs.(E) - E)/2

    # constraint 1: voltage magnitudes
    vmax = [network_data["bus"][string(bus)]["vmax"] for bus in bus_list]
    vmin = [network_data["bus"][string(bus)]["vmin"] for bus in bus_list]

    # constraint 2: flow limits
    smax = [minimum([network_data["branch"][string(line)]["rate_a"];
                    network_data["branch"][string(line)]["rate_b"];
                    network_data["branch"][string(line)]["rate_b"]]) for line in line_list]

    # flows
    vm_fr = Efr*vm
    vm_to = Eto*vm
    va_fr = Efr*va
    va_to = Eto*va

    @constraint(model, pfr .== @.  (g+g_fr)*(vm_fr/tm)^2 - g*vm_fr/tm*vm_to*cos(va_fr-va_to-ta) + -b*vm_fr/tm*vm_to*sin(va_fr-va_to-ta) )
    @constraint(model, qfr .== @. -(b+b_fr)*(vm_fr/tm)^2 + b*vm_fr/tm*vm_to*cos(va_fr-va_to-ta) + -g*vm_fr/tm*vm_to*sin(va_fr-va_to-ta) )
    @constraint(model, pto .== @.  (g+g_to)*vm_to^2      - g*vm_to*vm_fr/tm*cos(va_to-va_fr+ta) + -b*vm_to*vm_fr/tm*sin(va_to-va_fr+ta) )
    @constraint(model, qto .== @. -(b+b_to)*vm_to^2      + b*vm_to*vm_fr/tm*cos(va_to-va_fr+ta) + -g*vm_to*vm_fr/tm*sin(va_to-va_fr+ta) )

    # add constraints -- ignore angle limits
    @constraint(model, va[ref_bus] == 0.0)
    @constraint(model, vmin   .<= vm .<= vmax)
    @constraint(model, pg_min .<= pg .<= pg_max)
    @constraint(model, qg_min .<= qg .<= qg_max)

    @constraint(model, pfr.^2 + qfr.^2 .<= smax.^2 )
    @constraint(model, pto.^2 + qto.^2 .<= smax.^2 )

    @constraint(model, Eg*pg .== pd - gs.*vm.^2 + Efr'*pfr + Eto'*pto)
    @constraint(model, Eg*qg .== qd - bs.*vm.^2 + Efr'*qfr + Eto'*qto)

    @objective(model, Min, clin'*pg)
    optimize!(model)

    println(objective_value(model))

    # output useful stuff
    return value.(vm), value.(va), value.(pg), value.(qg), Eg, Efr, Eto, g, g_fr, g_to, b, b_fr, b_to, tm, ta, pd, qd, gs, bs, nb, nl, ng, vmax, vmin, pg_max, pg_min, qg_max, qg_min, smax, clin, fr_buses, to_buses
end

function test_jacobians(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_pfr = Zygote.jacobian(vm0 -> pfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta), vm0)[1]
    Jvm_pto = Zygote.jacobian(vm0 -> pto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta), vm0)[1]
    Jvm_qfr = Zygote.jacobian(vm0 -> qfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta), vm0)[1]
    Jvm_qto = Zygote.jacobian(vm0 -> qto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta), vm0)[1]
    Jva_pfr = Zygote.jacobian(va0 -> pfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta), va0)[1]
    Jva_pto = Zygote.jacobian(va0 -> pto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta), va0)[1]
    Jva_qfr = Zygote.jacobian(va0 -> qfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta), va0)[1]
    Jva_qto = Zygote.jacobian(va0 -> qto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta), va0)[1]

    Jvm_pfr_n = build_Jacobian("pfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_pto_n = build_Jacobian("pto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qfr_n = build_Jacobian("qfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qto_n = build_Jacobian("qto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pfr_n = build_Jacobian("pfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pto_n = build_Jacobian("pto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qfr_n = build_Jacobian("qfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qto_n = build_Jacobian("qto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    # compare
    println(norm(Matrix(Jvm_pfr_n) - Jvm_pfr))
    println(norm(Matrix(Jvm_pto_n) - Jvm_pto))
    println(norm(Matrix(Jvm_qfr_n) - Jvm_qfr))
    println(norm(Matrix(Jvm_qto_n) - Jvm_qto))
    println(norm(Matrix(Jva_pfr_n) - Jva_pfr))
    println(norm(Matrix(Jva_pto_n) - Jva_pto))
    println(norm(Matrix(Jva_qfr_n) - Jva_qfr))
    println(norm(Matrix(Jva_qto_n) - Jva_qto))
end

""" Solve ACOPF, using IPOPT, with a linearized ACOPF model"""
function linear_min_ipopt(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax,
    lambda_p, lambda_q, mu_sfr, mu_sto, vm0, va0, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin)
    # vm0 = bias point
    # va0 = bias point

    model = Model(Ipopt.Optimizer)
    nb = length(vm0)
    nl = length(g)
    ng = length(clin)
    @variable(model, dvm[1:nb])
    @variable(model, dva[1:nb])
    @variable(model, pg[1:ng])
    @variable(model, qg[1:ng])

    @constraint(model,    vmin   .<= dvm .<= vmax    )
    @constraint(model, -ones(nb) .<= dva .<= ones(nb))
    @constraint(model, pg_min    .<= pg  .<= pg_max  )
    @constraint(model, qg_min    .<= qg  .<= qg_max  )

    @constraint(model, dva[69] == 0.0)

    pfr_0 = pfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    pto_0 = pto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qfr_0 = qfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qto_0 = qto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    pfr_sq_0 = pfr_0.^2
    pto_sq_0 = pto_0.^2
    qfr_sq_0 = qfr_0.^2
    qto_sq_0 = qto_0.^2

    Jvm_pfr = build_Jacobian("pfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_pto = build_Jacobian("pto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qfr = build_Jacobian("qfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qto = build_Jacobian("qto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pfr = build_Jacobian("pfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pto = build_Jacobian("pto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qfr = build_Jacobian("qfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qto = build_Jacobian("qto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    Jvm_pfr_sq = 2.0.*pfr_0.*Jvm_pfr
    Jvm_pto_sq = 2.0.*pto_0.*Jvm_pto
    Jvm_qfr_sq = 2.0.*qfr_0.*Jvm_qfr
    Jvm_qto_sq = 2.0.*qto_0.*Jvm_qto
    Jva_pfr_sq = 2.0.*pfr_0.*Jva_pfr
    Jva_pto_sq = 2.0.*pto_0.*Jva_pto
    Jva_qfr_sq = 2.0.*qfr_0.*Jva_qfr
    Jva_qto_sq = 2.0.*qto_0.*Jva_qto

    pfr_approx = pfr_0 + Jvm_pfr*(dvm - vm0) + Jva_pfr*(dva - va0)
    pto_approx = pto_0 + Jvm_pto*(dvm - vm0) + Jva_pto*(dva - va0)
    qfr_approx = qfr_0 + Jvm_qfr*(dvm - vm0) + Jva_qfr*(dva - va0)
    qto_approx = qto_0 + Jvm_qto*(dvm - vm0) + Jva_qto*(dva - va0)

    pfr_sq_approx = pfr_sq_0 + Jvm_pfr_sq*(dvm - vm0) + Jva_pfr_sq*(dva - va0)
    pto_sq_approx = pto_sq_0 + Jvm_pto_sq*(dvm - vm0) + Jva_pto_sq*(dva - va0)
    qfr_sq_approx = qfr_sq_0 + Jvm_qfr_sq*(dvm - vm0) + Jva_qfr_sq*(dva - va0)
    qto_sq_approx = qto_sq_0 + Jvm_qto_sq*(dvm - vm0) + Jva_qto_sq*(dva - va0)

    # vm.^2 = 2*vm0
    L = clin'*pg + 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*pfr_approx + Eto'*pto_approx - (Eg*pg)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*qfr_approx + Eto'*qto_approx - (Eg*qg)) + 
        mu_sfr'*(pfr_sq_approx + qfr_sq_approx - (smax.^2)) + 
        mu_sto'*(pto_sq_approx + qto_sq_approx - (smax.^2)) 

    @objective(model, Min, L)
    optimize!(model)
    println(objective_value(model))

    return value.(dvm), value.(dva)
end

""" This function does this same thing as linear_min_ipopt(), but the Lagrangian is written
so that decision are written with their linear coefficients"""
function linear_min_ipopt_decomposed(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax,
    lambda_p, lambda_q, mu_sfr, mu_sto, vm0, va0, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin)
    # vm0 = bias point
    # va0 = bias point

    model = Model(Ipopt.Optimizer)
    nb = length(vm0)
    nl = length(g)
    ng = length(clin)
    @variable(model, dvm[1:nb])
    @variable(model, dva[1:nb])
    @variable(model, pg[1:ng])
    @variable(model, qg[1:ng])

    @constraint(model,    vmin   .<= dvm .<= vmax    )
    @constraint(model, -ones(nb) .<= dva .<= ones(nb))
    @constraint(model, pg_min    .<= pg  .<= pg_max  )
    @constraint(model, qg_min    .<= qg  .<= qg_max  )

    @constraint(model, dva[69] == 0.0)

    pfr_0 = pfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    pto_0 = pto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qfr_0 = qfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qto_0 = qto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    pfr_sq_0 = pfr_0.^2
    pto_sq_0 = pto_0.^2
    qfr_sq_0 = qfr_0.^2
    qto_sq_0 = qto_0.^2

    Jvm_pfr = build_Jacobian("pfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_pto = build_Jacobian("pto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qfr = build_Jacobian("qfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qto = build_Jacobian("qto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pfr = build_Jacobian("pfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pto = build_Jacobian("pto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qfr = build_Jacobian("qfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qto = build_Jacobian("qto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    Jvm_pfr_sq = 2.0.*pfr_0.*Jvm_pfr
    Jvm_pto_sq = 2.0.*pto_0.*Jvm_pto
    Jvm_qfr_sq = 2.0.*qfr_0.*Jvm_qfr
    Jvm_qto_sq = 2.0.*qto_0.*Jvm_qto
    Jva_pfr_sq = 2.0.*pfr_0.*Jva_pfr
    Jva_pto_sq = 2.0.*pto_0.*Jva_pto
    Jva_qfr_sq = 2.0.*qfr_0.*Jva_qfr
    Jva_qto_sq = 2.0.*qto_0.*Jva_qto

    # vm.^2 = 2*vm0
    """ 
    Original Lagrangian!
    L = clin'*pg + 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*pfr_approx + Eto'*pto_approx - (Eg*pg)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*qfr_approx + Eto'*qto_approx - (Eg*qg)) + 
        mu_sfr'*(pfr_sq_approx + qfr_sq_approx - (smax.^2)) + 
        mu_sto'*(pto_sq_approx + qto_sq_approx - (smax.^2)) 
    """

    # linear components
    L_pg = clin' - lambda_p'*Eg
    L_qg = - lambda_q'*Eg

    L_vm = 
        lambda_p'*(-diagm(gs.*2.0 .*vm0) + Efr'*Jvm_pfr + Eto'*Jvm_pto) + 
        lambda_q'*(-diagm(bs.*2.0 .*vm0) + Efr'*Jvm_qfr + Eto'*Jvm_qto) + 
        mu_sfr'*(Jvm_pfr_sq + Jvm_qfr_sq) + 
        mu_sto'*(Jvm_pto_sq + Jvm_qto_sq) 

    L_va = 
        lambda_p'*(Efr'*Jva_pfr + Eto'*Jva_pto) + 
        lambda_q'*(Efr'*Jva_qfr + Eto'*Jva_qto) + 
        mu_sfr'*(Jva_pfr_sq + Jva_qfr_sq) + 
        mu_sto'*(Jva_pto_sq + Jva_qto_sq) 

    L_0 = 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*-vm0) + Efr'*(pfr_0 + Jvm_pfr*-vm0 + Jva_pfr*-va0) + Eto'*(pto_0 + Jvm_pto*-vm0 + Jva_pto*-va0)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*-vm0) + Efr'*(qfr_0 + Jvm_qfr*-vm0 + Jva_qfr*-va0) + Eto'*(qto_0 + Jvm_qto*-vm0 + Jva_qto*-va0)) + 
        mu_sfr'*(pfr_sq_0 + Jvm_pfr_sq*-vm0 + Jva_pfr_sq*-va0 + qfr_sq_0 + Jvm_qfr_sq*-vm0 + Jva_qfr_sq*-va0 - (smax.^2)) + 
        mu_sto'*(pto_sq_0 + Jvm_pto_sq*-vm0 + Jva_pto_sq*-va0 + qto_sq_0 + Jvm_qto_sq*-vm0 + Jva_qto_sq*-va0 - (smax.^2)) 

    @objective(model, Min, L_pg*pg + L_qg*qg + L_vm*dvm + L_va*dva + L_0)

    optimize!(model)
    println(objective_value(model))

    return value.(dvm), value.(dva)
end

""" solve the dual variables with IPOPT directly :) """
function linear_min_dual_norm_ipopt(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax,
    vm0, va0, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
    # vm0 = bias point
    # va0 = bias point

    nb = length(vm0)
    nl = length(g)
    ng = length(clin)

    # normalization tools
    sig_pg = 0.5*(pg_max-pg_min)
    mu_pg  = 0.5*(pg_max+pg_min)
    sig_qg = 0.5*(qg_max-qg_min)
    mu_qg  = 0.5*(qg_max+qg_min)
    sig_vm = 0.5*(vmax-vmin)
    mu_vm  = 0.5*(vmax+vmin)
    amax   =  ones(nb)
    amin   = -ones(nb)
    sig_va = 0.5*(amax-amin)
    mu_va  = 0.5*(amax+amin)

    pfr_0 = pfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    pto_0 = pto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qfr_0 = qfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qto_0 = qto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    pfr_sq_0 = pfr_0.^2
    pto_sq_0 = pto_0.^2
    qfr_sq_0 = qfr_0.^2
    qto_sq_0 = qto_0.^2

    Jvm_pfr = build_Jacobian("pfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_pto = build_Jacobian("pto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qfr = build_Jacobian("qfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qto = build_Jacobian("qto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pfr = build_Jacobian("pfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pto = build_Jacobian("pto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qfr = build_Jacobian("qfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qto = build_Jacobian("qto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    Jvm_pfr_sq = 2.0.*pfr_0.*Jvm_pfr
    Jvm_pto_sq = 2.0.*pto_0.*Jvm_pto
    Jvm_qfr_sq = 2.0.*qfr_0.*Jvm_qfr
    Jvm_qto_sq = 2.0.*qto_0.*Jvm_qto
    Jva_pfr_sq = 2.0.*pfr_0.*Jva_pfr
    Jva_pto_sq = 2.0.*pto_0.*Jva_pto
    Jva_qfr_sq = 2.0.*qfr_0.*Jva_qfr
    Jva_qto_sq = 2.0.*qto_0.*Jva_qto

    # vm.^2 = 2*vm0
    """ 
    Original Lagrangian!
    L = clin'*pg + 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*pfr_approx + Eto'*pto_approx - (Eg*pg)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*qfr_approx + Eto'*qto_approx - (Eg*qg)) + 
        mu_sfr'*(pfr_sq_approx + qfr_sq_approx - (smax.^2)) + 
        mu_sto'*(pto_sq_approx + qto_sq_approx - (smax.^2)) 
    """

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "max_iter", 250)
    @variable(model, lambda_p[1:nb])
    @variable(model, lambda_q[1:nb])
    @variable(model, mu_sfr[1:nl])
    @variable(model, mu_sto[1:nl])

    @constraint(model, 0.0 .<= mu_sfr)
    @constraint(model, 0.0 .<= mu_sto)
    # linear components
    L_pg = clin' - lambda_p'*Eg
    L_qg = - lambda_q'*Eg

    L_vm = 
        lambda_p'*(-diagm(gs.*2.0 .*vm0) + Efr'*Jvm_pfr + Eto'*Jvm_pto) + 
        lambda_q'*(-diagm(bs.*2.0 .*vm0) + Efr'*Jvm_qfr + Eto'*Jvm_qto) + 
        mu_sfr'*(Jvm_pfr_sq + Jvm_qfr_sq) + 
        mu_sto'*(Jvm_pto_sq + Jvm_qto_sq) 

    L_va = 
        lambda_p'*(Efr'*Jva_pfr + Eto'*Jva_pto) + 
        lambda_q'*(Efr'*Jva_qfr + Eto'*Jva_qto) + 
        mu_sfr'*(Jva_pfr_sq + Jva_qfr_sq) + 
        mu_sto'*(Jva_pto_sq + Jva_qto_sq) 

    L_0 = 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*-vm0) + Efr'*(pfr_0 + Jvm_pfr*-vm0 + Jva_pfr*-va0) + Eto'*(pto_0 + Jvm_pto*-vm0 + Jva_pto*-va0)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*-vm0) + Efr'*(qfr_0 + Jvm_qfr*-vm0 + Jva_qfr*-va0) + Eto'*(qto_0 + Jvm_qto*-vm0 + Jva_qto*-va0)) + 
        mu_sfr'*(pfr_sq_0 + Jvm_pfr_sq*-vm0 + Jva_pfr_sq*-va0 + qfr_sq_0 + Jvm_qfr_sq*-vm0 + Jva_qfr_sq*-va0 - (smax.^2)) + 
        mu_sto'*(pto_sq_0 + Jvm_pto_sq*-vm0 + Jva_pto_sq*-va0 + qto_sq_0 + Jvm_qto_sq*-vm0 + Jva_qto_sq*-va0 - (smax.^2)) 

    # add a normalization step
    """
    L  =  L_pg*pg + L_qg*qg + L_vm*dvm + L_va*dva + L_0
       =  L_pg*(sig_pg*pg_norm+mu_pg) + L_qg*(sig_qg*qg_norm+mu_qg) + L_vm*(sig_dvm*dvm_norm+mu_dvm) + L_va*(sig_dva*dva_norm+mu_dva) + L_0
       =  L_pg*sig_pg*pg_norm+L_pg*mu_pg + L_qg*sig_qg*qg_norm+L_qg*mu_qg + L_vm*sig_dvm*dvm_norm+L_vm*mu_dvm + L_va*sig_dva*dva_norm+L_va*mu_dva + L_0
       =  (L_pg*sig_pg)*pg_norm + (L_qg*sig_qg)*qg_norm + (L_vm*sig_dvm)*dvm_norm + (L_va*sig_dva)*dva_norm + L_0+L_pg*mu_pg+L_qg*mu_qg+L_vm*mu_dvm+L_va*mu_dva
    """
    L_pg_norm = L_pg.*(sig_pg')
    L_qg_norm = L_qg.*(sig_qg')
    L_vm_norm = L_vm.*(sig_vm')
    L_va_norm = L_va.*(sig_va')
    L0_norm   = L_0 + L_pg*mu_pg + L_qg*mu_qg + L_vm*mu_vm + L_vm*mu_va

    # remove the reference bus
    #L_va_norm[69] = 0.0*L_va_norm[69]
    # deleteat!(L_va_norm_v, 69)
    # L_va_norm = L_va_norm_v'

    # solve!
    # => t = norm([L_pg_norm L_qg_norm L_vm_norm scale_va.*L_va_norm], 1)
    x = vec([L_pg_norm L_qg_norm L_vm_norm scale_va.*L_va_norm])
    @variable(model, t)
    @constraint(model, [t; x] in MOI.NormOneCone(1 + length(x)))


    @objective(model, Max, -t + L0_norm)

    optimize!(model)
    println(objective_value(model))

    return objective_value(model), value.(lambda_p), value.(lambda_q), value.(mu_sfr), value.(mu_sto)
end


""" Thus function takes the primal, linearizes, and solves the dual norm"""
function linear_min_dual_norm(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax,lambda_p, 
    lambda_q, mu_sfr, mu_sto, vm0, va0, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, 
    b, b_fr, b_to, tm, ta, clin, scale_va)
    # vm0 = bias point
    # va0 = bias point

    nb = length(vm0)
    nl = length(g)
    ng = length(clin)

    # normalization tools
    sig_pg = 0.5*(pg_max-pg_min)
    mu_pg  = 0.5*(pg_max+pg_min)
    sig_qg = 0.5*(qg_max-qg_min)
    mu_qg  = 0.5*(qg_max+qg_min)
    sig_vm = 0.5*(vmax-vmin)
    mu_vm  = 0.5*(vmax+vmin)
    amax   =  ones(nb)
    amin   = -ones(nb)
    sig_va = 0.5*(amax-amin)
    mu_va  = 0.5*(amax+amin)

    pfr_0 = pfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    pto_0 = pto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qfr_0 = qfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qto_0 = qto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    pfr_sq_0 = pfr_0.^2
    pto_sq_0 = pto_0.^2
    qfr_sq_0 = qfr_0.^2
    qto_sq_0 = qto_0.^2

    Jvm_pfr = build_Jacobian("pfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_pto = build_Jacobian("pto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qfr = build_Jacobian("qfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qto = build_Jacobian("qto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pfr = build_Jacobian("pfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pto = build_Jacobian("pto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qfr = build_Jacobian("qfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qto = build_Jacobian("qto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    Jvm_pfr_sq = 2.0.*pfr_0.*Jvm_pfr
    Jvm_pto_sq = 2.0.*pto_0.*Jvm_pto
    Jvm_qfr_sq = 2.0.*qfr_0.*Jvm_qfr
    Jvm_qto_sq = 2.0.*qto_0.*Jvm_qto
    Jva_pfr_sq = 2.0.*pfr_0.*Jva_pfr
    Jva_pto_sq = 2.0.*pto_0.*Jva_pto
    Jva_qfr_sq = 2.0.*qfr_0.*Jva_qfr
    Jva_qto_sq = 2.0.*qto_0.*Jva_qto

    # vm.^2 = 2*vm0
    """ 
    Original Lagrangian!
    L = clin'*pg + 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*pfr_approx + Eto'*pto_approx - (Eg*pg)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*qfr_approx + Eto'*qto_approx - (Eg*qg)) + 
        mu_sfr'*(pfr_sq_approx + qfr_sq_approx - (smax.^2)) + 
        mu_sto'*(pto_sq_approx + qto_sq_approx - (smax.^2)) 
    """

    # linear components
    L_pg = clin' - lambda_p'*Eg
    L_qg = - lambda_q'*Eg

    L_vm = 
        lambda_p'*(-diagm(gs.*2.0 .*vm0) + Efr'*Jvm_pfr + Eto'*Jvm_pto) + 
        lambda_q'*(-diagm(bs.*2.0 .*vm0) + Efr'*Jvm_qfr + Eto'*Jvm_qto) + 
        mu_sfr'*(Jvm_pfr_sq + Jvm_qfr_sq) + 
        mu_sto'*(Jvm_pto_sq + Jvm_qto_sq) 

    L_va = 
        lambda_p'*(Efr'*Jva_pfr + Eto'*Jva_pto) + 
        lambda_q'*(Efr'*Jva_qfr + Eto'*Jva_qto) + 
        mu_sfr'*(Jva_pfr_sq + Jva_qfr_sq) + 
        mu_sto'*(Jva_pto_sq + Jva_qto_sq) 

    L_0 = 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*-vm0) + Efr'*(pfr_0 + Jvm_pfr*-vm0 + Jva_pfr*-va0) + Eto'*(pto_0 + Jvm_pto*-vm0 + Jva_pto*-va0)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*-vm0) + Efr'*(qfr_0 + Jvm_qfr*-vm0 + Jva_qfr*-va0) + Eto'*(qto_0 + Jvm_qto*-vm0 + Jva_qto*-va0)) + 
        mu_sfr'*(pfr_sq_0 + Jvm_pfr_sq*-vm0 + Jva_pfr_sq*-va0 + qfr_sq_0 + Jvm_qfr_sq*-vm0 + Jva_qfr_sq*-va0 - (smax.^2)) + 
        mu_sto'*(pto_sq_0 + Jvm_pto_sq*-vm0 + Jva_pto_sq*-va0 + qto_sq_0 + Jvm_qto_sq*-vm0 + Jva_qto_sq*-va0 - (smax.^2)) 

    # add a normalization step
    """
    L  =  L_pg*pg + L_qg*qg + L_vm*dvm + L_va*dva + L_0
       =  L_pg*(sig_pg*pg_norm+mu_pg) + L_qg*(sig_qg*qg_norm+mu_qg) + L_vm*(sig_dvm*dvm_norm+mu_dvm) + L_va*(sig_dva*dva_norm+mu_dva) + L_0
       =  L_pg*sig_pg*pg_norm+L_pg*mu_pg + L_qg*sig_qg*qg_norm+L_qg*mu_qg + L_vm*sig_dvm*dvm_norm+L_vm*mu_dvm + L_va*sig_dva*dva_norm+L_va*mu_dva + L_0
       =  (L_pg*sig_pg)*pg_norm + (L_qg*sig_qg)*qg_norm + (L_vm*sig_dvm)*dvm_norm + (L_va*sig_dva)*dva_norm + L_0+L_pg*mu_pg+L_qg*mu_qg+L_vm*mu_dvm+L_va*mu_dva
    """
    L_pg_norm = L_pg.*(sig_pg')
    L_qg_norm = L_qg.*(sig_qg')
    L_vm_norm = L_vm.*(sig_vm')
    L_va_norm = L_va.*(sig_va')
    L0_norm   = L_0 + L_pg*mu_pg + L_qg*mu_qg + L_vm*mu_vm + L_vm*mu_va

    # remove the reference bus
    #L_va_norm[69] = 0.0*L_va_norm[69]
    # deleteat!(L_va_norm_v, 69)
    # L_va_norm = L_va_norm_v'

    # solve!
    opt_val = -norm([L_pg_norm L_qg_norm L_vm_norm scale_va.*L_va_norm], 1) + L0_norm

    return opt_val
end

""" Compute the Lagrangian"""
function Lagrangian(pg, qg, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Eg, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin)

    # compute flows
    pfr_0 = pfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    pto_0 = pto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qfr_0 = qfr(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qto_0 = qto(vm, va, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    # square flows
    pfr_sq = pfr_0.^2
    pto_sq = pto_0.^2
    qfr_sq = qfr_0.^2
    qto_sq = qto_0.^2

    L = clin'*pg + 
        lambda_p'*(pd - gs.*vm + Efr'*pfr_0 + Eto'*pto_0 - (Eg*pg)) + 
        lambda_q'*(qd - bs.*vm + Efr'*qfr_0 + Eto'*qto_0 - (Eg*qg)) + 
        mu_sfr'*(pfr_sq + qfr_sq - (smax.^2)) + 
        mu_sto'*(pto_sq + qto_sq - (smax.^2)) 

    return L
end

""" incorporate constraints on vm and va (not standard)"""
function linear_min_dual_norm_constrained(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax,
    lambda_p, lambda_q, mu_sfr, mu_sto, lam_vm, lam_va, vm0, va0, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
    # vm0 = bias point
    # va0 = bias point

    nb = length(vm0)
    nl = length(g)
    ng = length(clin)

    # normalization tools
    sig_pg = 0.5*(pg_max-pg_min)
    mu_pg  = 0.5*(pg_max+pg_min)
    sig_qg = 0.5*(qg_max-qg_min)
    mu_qg  = 0.5*(qg_max+qg_min)
    sig_vm = 0.5*(vmax-vmin)
    mu_vm  = 0.5*(vmax+vmin)
    amax   =  ones(nb)
    amin   = -ones(nb)
    sig_va = 0.5*(amax-amin)
    mu_va  = 0.5*(amax+amin)

    pfr_0 = pfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    pto_0 = pto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qfr_0 = qfr(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    qto_0 = qto(vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    pfr_sq_0 = pfr_0.^2
    pto_sq_0 = pto_0.^2
    qfr_sq_0 = qfr_0.^2
    qto_sq_0 = qto_0.^2

    Jvm_pfr = build_Jacobian("pfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_pto = build_Jacobian("pto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qfr = build_Jacobian("qfr", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jvm_qto = build_Jacobian("qto", "vm", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pfr = build_Jacobian("pfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_pto = build_Jacobian("pto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qfr = build_Jacobian("qfr", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)
    Jva_qto = build_Jacobian("qto", "va", vm0, va0, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta)

    Jvm_pfr_sq = 2.0.*pfr_0.*Jvm_pfr
    Jvm_pto_sq = 2.0.*pto_0.*Jvm_pto
    Jvm_qfr_sq = 2.0.*qfr_0.*Jvm_qfr
    Jvm_qto_sq = 2.0.*qto_0.*Jvm_qto
    Jva_pfr_sq = 2.0.*pfr_0.*Jva_pfr
    Jva_pto_sq = 2.0.*pto_0.*Jva_pto
    Jva_qfr_sq = 2.0.*qfr_0.*Jva_qfr
    Jva_qto_sq = 2.0.*qto_0.*Jva_qto

    # vm.^2 = 2*vm0
    """ 
    Original Lagrangian!
    L = clin'*pg + 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*pfr_approx + Eto'*pto_approx - (Eg*pg)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*(dvm - vm0)) + Efr'*qfr_approx + Eto'*qto_approx - (Eg*qg)) + 
        mu_sfr'*(pfr_sq_approx + qfr_sq_approx - (smax.^2)) + 
        mu_sto'*(pto_sq_approx + qto_sq_approx - (smax.^2)) 
    """

    # linear components
    L_pg = clin' - lambda_p'*Eg
    L_qg = - lambda_q'*Eg

    L_vm = 
        lambda_p'*(-diagm(gs.*2.0 .*vm0) + Efr'*Jvm_pfr + Eto'*Jvm_pto) + 
        lambda_q'*(-diagm(bs.*2.0 .*vm0) + Efr'*Jvm_qfr + Eto'*Jvm_qto) + 
        mu_sfr'*(Jvm_pfr_sq + Jvm_qfr_sq) + 
        mu_sto'*(Jvm_pto_sq + Jvm_qto_sq) 

    L_va = 
        lambda_p'*(Efr'*Jva_pfr + Eto'*Jva_pto) + 
        lambda_q'*(Efr'*Jva_qfr + Eto'*Jva_qto) + 
        mu_sfr'*(Jva_pfr_sq + Jva_qfr_sq) + 
        mu_sto'*(Jva_pto_sq + Jva_qto_sq) 

    L_0 = 
        lambda_p'*(pd - gs.*(vm0.^2 + 2.0 .*vm0.*-vm0) + Efr'*(pfr_0 + Jvm_pfr*-vm0 + Jva_pfr*-va0) + Eto'*(pto_0 + Jvm_pto*-vm0 + Jva_pto*-va0)) + 
        lambda_q'*(qd - bs.*(vm0.^2 + 2.0 .*vm0.*-vm0) + Efr'*(qfr_0 + Jvm_qfr*-vm0 + Jva_qfr*-va0) + Eto'*(qto_0 + Jvm_qto*-vm0 + Jva_qto*-va0)) + 
        mu_sfr'*(pfr_sq_0 + Jvm_pfr_sq*-vm0 + Jva_pfr_sq*-va0 + qfr_sq_0 + Jvm_qfr_sq*-vm0 + Jva_qfr_sq*-va0 - (smax.^2)) + 
        mu_sto'*(pto_sq_0 + Jvm_pto_sq*-vm0 + Jva_pto_sq*-va0 + qto_sq_0 + Jvm_qto_sq*-vm0 + Jva_qto_sq*-va0 - (smax.^2)) 

    # add a normalization step
    """
    L  =  L_pg*pg + L_qg*qg + L_vm*dvm + L_va*dva + L_0 + lam_vm'*(vm0-dvm) + lam_va'*(va0-dva)
       =
       =  L_pg*pg + L_qg*qg + (L_vm-lam_vm')*dvm + (L_va-lam_va')*dva + (L_0 + lam_vm'*vm0 + lam_va'*va0)
       =
       =  L_pg*(sig_pg*pg_norm+mu_pg) + L_qg*(sig_qg*qg_norm+mu_qg) + L_vm*(sig_dvm*dvm_norm+mu_dvm) + L_va*(sig_dva*dva_norm+mu_dva) + L_0
       =  L_pg*sig_pg*pg_norm+L_pg*mu_pg + L_qg*sig_qg*qg_norm+L_qg*mu_qg + L_vm*sig_dvm*dvm_norm+L_vm*mu_dvm + L_va*sig_dva*dva_norm+L_va*mu_dva + L_0
       =  (L_pg*sig_pg)*pg_norm + (L_qg*sig_qg)*qg_norm + (L_vm*sig_dvm)*dvm_norm + (L_va*sig_dva)*dva_norm + L_0+L_pg*mu_pg+L_qg*mu_qg+L_vm*mu_dvm+L_va*mu_dva
    """
    L_pg_norm = L_pg.*(sig_pg')
    L_qg_norm = L_qg.*(sig_qg')
    L_vm_norm = (L_vm-lam_vm').*(sig_vm')
    L_va_norm = (L_va-lam_va').*(sig_va')
    L0_norm   = L_0 + L_pg*mu_pg + L_qg*mu_qg + L_vm*mu_vm + L_vm*mu_va + (lam_vm'*vm0 + lam_va'*va0)

    # solve!
    opt_val = -norm([L_pg_norm L_qg_norm L_vm_norm scale_va.*L_va_norm], 1) + L0_norm

    return opt_val
end

function finiteDiff(epsilon, vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
    # get f0
    f0 = linear_min_dual_norm(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)

    grad_lambda_p = 0*similar(lambda_p)
    grad_lambda_q = 0*similar(lambda_q)
    grad_mu_sfr   = 0*similar(mu_sfr)
    grad_mu_sto   = 0*similar(mu_sto)
    grad_vm       = 0*similar(vm)
    grad_va       = 0*similar(va)

    # lambda_p
    for ii in 1:length(lambda_p)
        lambda_p[ii] += epsilon
        fp = linear_min_dual_norm(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
        grad_lambda_p[ii] = (fp - f0)/epsilon
        lambda_p[ii] -= epsilon
    end

    # lambda_q
    for ii in 1:length(lambda_q)
        lambda_q[ii] += epsilon
        fp = linear_min_dual_norm(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
        grad_lambda_q[ii] = (fp - f0)/epsilon
        lambda_q[ii] -= epsilon
    end

    # grad_mu_sfr
    for ii in 1:length(mu_sfr)
        mu_sfr[ii] += epsilon
        fp = linear_min_dual_norm(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
        grad_mu_sfr[ii] = (fp - f0)/epsilon
        mu_sfr[ii] -= epsilon
    end

    # grad_mu_sto
    for ii in 1:length(mu_sto)
        mu_sto[ii] += epsilon
        fp = linear_min_dual_norm(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
        grad_mu_sto[ii] = (fp - f0)/epsilon
        mu_sto[ii] -= epsilon
    end

    # vm
    for ii in 1:length(vm)
        vm[ii] += epsilon
        fp = linear_min_dual_norm(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
        grad_vm[ii] = (fp - f0)/epsilon
        vm[ii] -= epsilon
    end

    # va
    for ii in 1:length(va)
        va[ii] += epsilon
        fp = linear_min_dual_norm(vmin, vmax, pg_min, pg_max, qg_min, qg_max, smax, lambda_p, lambda_q, mu_sfr, mu_sto, vm, va, Efr, Eto, fr_buses, to_buses, g, g_fr, g_to, b, b_fr, b_to, tm, ta, clin, scale_va)
        grad_va[ii] = (fp - f0)/epsilon
        va[ii] -= epsilon
    end

    # output
    return grad_lambda_p, grad_lambda_q, grad_mu_sfr, grad_mu_sto, grad_vm, grad_va
end

function soc_voltage_bound_vectors(data, nl, fr_buses, to_buses)
    # build the reg
    pm_ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    wr_min_pm, wr_max_pm, wi_min_pm, wi_max_pm = ref_calc_voltage_product_bounds(pm_ref[:buspairs])

    # now, loop over the line list, and populate the updated vectors
    wr_min = zeros(nl)
    wr_max = zeros(nl)
    wi_min = zeros(nl)
    wi_max = zeros(nl)
    for ii in 1:nl
        pair = (fr_buses[ii], to_buses[ii])
        wr_min[ii] = wr_min_pm[pair]
        wr_max[ii] = wr_max_pm[pair]
        wi_min[ii] = wi_min_pm[pair]
        wi_max[ii] = wi_max_pm[pair]
    end

    return wr_min, wr_max, wi_min, wi_max
end

function canonicalize(zl, 
                      pd0, 
                      qd0, 
                      gm::SOCGridModel; 
                      use_float64::Bool=false,
                      include_cost::Bool=false,
                      normalize_shed::Bool=false)
    "zl, pd0, qd0 can be Float64 vectors, or they can be Affine (variable) vectors"
    nl = gm.nl
    nb = gm.nb
    ng = gm.ng

    w_idx       =                     (1:nb)
    wr_idx      = w_idx[end]       .+ (1:nl)
    wi_idx      = wr_idx[end]      .+ (1:nl)
    zd_idx      = wi_idx[end]      .+ (1:nb)
    pg_idx      = zd_idx[end]      .+ (1:ng)  
    qg_idx      = pg_idx[end]      .+ (1:ng)  
    pgs_idx     = qg_idx[end]      .+ (1:nb)
    qbs_pos_idx = pgs_idx[end]     .+ (1:nb)
    qbs_neg_idx = qbs_pos_idx[end] .+ (1:nb)
    p_fr_idx    = qbs_neg_idx[end] .+ (1:nl)
    p_to_idx    = p_fr_idx[end]    .+ (1:nl)   
    q_fr_idx    = p_to_idx[end]    .+ (1:nl)   
    q_to_idx    = q_fr_idx[end]    .+ (1:nl)
    t_idx       = q_to_idx[end]     + 1

    nvar  = t_idx[end]
    neq   = 2*nb + 4*nl
    nineq = 2*nb + 2*nb + 4*nl + 4*ng + 6*nb
    if use_float64 == true
        A = spzeros(neq, nvar)
    else
        A = SparseMatrixCSC{AffExpr, Int64}(undef, neq, nvar)
    end
    # => A = Matrix{AffExpr}(undef, neq, nvar)
    # => A .= 0.0
    # => A = zeros(neq, nvar)
    # if zl is embedded as a nonlinear constraint:
        # => A = Matrix{NonlinearExpr}(undef, neq, nvar)
            # =>    A = SparseMatrixCSC{NonlinearExpr, Int64}(undef, neq, nvar)
        # => A .= 0.0
    b = zeros(neq) # this stays 0

    # flow constraints
    A[1:nl, p_fr_idx]          = sparse(I,nl,nl)
    A[(nl+1):2*nl, p_to_idx]   = sparse(I,nl,nl)
    A[(2*nl+1):3*nl, q_fr_idx] = sparse(I,nl,nl)
    A[(3*nl+1):4*nl, q_to_idx] = sparse(I,nl,nl)
    Mflow = [-zl.*[gm.Tpfr gm.TpRfr gm.TpIfr];
             -zl.*[gm.Tpto gm.TpRto gm.TpIto];
             -zl.*[gm.Tqfr gm.TqRfr gm.TqIfr];
             -zl.*[gm.Tqto gm.TqRto gm.TqIto]]
    A[1:4*nl,w_idx[1]:wi_idx[end]] = Mflow

    # injection constraints
    pjidx = 4*nl    .+ (1:nb)
    qjidx = 4*nl+nb .+ (1:nb)

    A[pjidx,pg_idx]   = -gm.Eg
    A[pjidx,zd_idx]   = spdiagm(pd0)
    A[pjidx,p_fr_idx] = gm.Efr'
    A[pjidx,p_to_idx] = gm.Eto'
    A[pjidx,pgs_idx]  = sparse(I,nb,nb)

    A[qjidx,qg_idx]   = -gm.Eg
    A[qjidx,zd_idx]   = spdiagm(qd0)
    A[qjidx,q_fr_idx] = gm.Efr'
    A[qjidx,q_to_idx] = gm.Eto'
    A[qjidx,qbs_pos_idx]  = -sparse(I,nb,nb)
    A[qjidx,qbs_neg_idx]  = -sparse(I,nb,nb)

    C = spzeros(nineq, nvar)
    d = zeros(nineq)

    idx_nd = 0
    C[idx_nd .+ (1:nb), zd_idx] = -sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    C[idx_nd .+ (1:nb), zd_idx]= +sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= -1.0
    idx_nd += nb

    C[idx_nd .+ (1:nb), w_idx] = -sparse(I,nb,nb) 
    d[idx_nd .+ (1:nb)] .= gm.vmin.^2
    idx_nd += nb

    C[idx_nd .+ (1:nb), w_idx] = +sparse(I,nb,nb) 
    d[idx_nd .+ (1:nb)] .= -gm.vmax.^2
    idx_nd += nb

    C[idx_nd .+ (1:nl), wr_idx] = -sparse(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= gm.wr_min
    idx_nd += nl

    C[idx_nd .+ (1:nl), wr_idx] = sparse(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= -gm.wr_max
    idx_nd += nl

    C[idx_nd .+ (1:nl), wi_idx] = -sparse(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= gm.wi_min
    idx_nd += nl

    C[idx_nd .+ (1:nl), wi_idx] = sparse(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= -gm.wi_max
    idx_nd += nl

    C[idx_nd .+ (1:ng), pg_idx] = -sparse(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= 0.0
    idx_nd += ng

    C[idx_nd .+ (1:ng), pg_idx] = sparse(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= -gm.pg_max
    idx_nd += ng

    C[idx_nd .+ (1:ng), qg_idx] = -sparse(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= gm.qg_min
    idx_nd += ng

    C[idx_nd .+ (1:ng), qg_idx] = sparse(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= -gm.qg_max
    idx_nd += ng

    C[idx_nd .+ (1:nb), pgs_idx] = -sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # double
    C[idx_nd .+ (1:nb), pgs_idx] = sparse(I,nb,nb)
    C[idx_nd .+ (1:nb), w_idx]   = -gm.Gs
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    C[idx_nd .+ (1:nb), qbs_pos_idx] = -sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # double
    C[idx_nd .+ (1:nb), qbs_pos_idx] = sparse(I,nb,nb)
    C[idx_nd .+ (1:nb), w_idx]       = -gm.Bs_pos
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # double
    C[idx_nd .+ (1:nb), qbs_neg_idx] = -sparse(I,nb,nb)
    C[idx_nd .+ (1:nb), w_idx]       = gm.Bs_neg
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    C[idx_nd .+ (1:nb), qbs_neg_idx] = sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # linear terms in the objective
    if use_float64 == true
        H  = zeros(nvar)
    else
        H  = Vector{AffExpr}(undef, nvar)
        H .= 0.0
    end
    h = sum(pd0)
    H[zd_idx] = - pd0

    # normalize?
    if (normalize_shed == true) && (typeof(pd0) == Vector{Float64})
        println("Normalizing load shed objective by total load.")
        H = H./sum(pd0)
        h = h/sum(pd0)
    elseif (normalize_shed == true) && (typeof(pd0) == Vector{VariableRef})
        println("Use epigraph trick, in this case, to enforce normalization.")
    end

    if include_cost == true
        # this is mainly for testing and troubleshooting
        H[pg_idx] = gm.clin
    end

    # apply all RSOCs
    nrsoc = 3*nl

    m1 = Vector{Any}(undef,nrsoc)#[Vector{Any}(undef,10)for ii in 1:nrsoc]
    m2 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]
    m3 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]

    b1 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]
    b2 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]
    b3 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]

    # flow limit f -> t
    for ii in 1:nl
        m1[ii] = zeros(1,nvar)
        b1[ii] = gm.smax[ii]^2

        m2[ii] = zeros(1,nvar)
        b2[ii] = 0.5

        m3[ii] = zeros(2,nvar)
        m3[ii][1,p_fr_idx[ii]] = 1
        m3[ii][2,q_fr_idx[ii]] = 1
        b3[ii] = zeros(2)
    end

    # flow limit t -> f
    for ii in 1:nl
        m1[ii+nl] = zeros(1,nvar)
        b1[ii+nl] = gm.smax[ii]^2

        m2[ii+nl] = zeros(1,nvar)
        b2[ii+nl] = 0.5

        m3[ii+nl] = zeros(2,nvar)
        m3[ii+nl][1,p_to_idx[ii]] = 1
        m3[ii+nl][2,q_to_idx[ii]] = 1
        b3[ii+nl] = zeros(2)
    end

    # RSOC on voltage
    for ii in 1:nl
        m1[ii+2*nl] = zeros(1,nvar)
        m1[ii+2*nl][w_idx] = gm.Efr[ii,:]
        b1[ii+2*nl] = 0

        m2[ii+2*nl] = zeros(1,nvar)
        m2[ii+2*nl][w_idx] = 0.5*gm.Eto[ii,:]
        b2[ii+2*nl] = 0

        m3[ii+2*nl] = zeros(2,nvar)
        m3[ii+2*nl][1,wr_idx[ii]] = 1
        m3[ii+2*nl][2,wi_idx[ii]] = 1
        b3[ii+2*nl] = zeros(2)
    end

    # throw mi and bi into a common struct
    lp = Dict(:A => A,
              :b => b,
              :C => C,
              :d => d,
              :H => H,
              :h => h)
    soc = Dict(:m1 => m1,
               :m2 => m2,
               :m3 => m3,
               :b1 => b1,
               :b2 => b2,
               :b3 => b3)

    return lp, soc
end

function canonicalize_flowcuts(zl, 
                               pd0, 
                               qd0, 
                               gm::SOCGridModel; 
                               use_float64::Bool=false,
                               include_cost::Bool=false,
                               normalize_shed::Bool=false)
    "zl, pd0, qd0 can be Float64 vectors, or they can be Affine (variable) vectors"
    nl = gm.nl
    nb = gm.nb
    ng = gm.ng

    w_idx       =                     (1:nb)
    w_fr_idx    = w_idx[end]       .+ (1:nl)
    w_to_idx    = w_fr_idx[end]    .+ (1:nl)
    wr_idx      = w_to_idx[end]    .+ (1:nl)
    wi_idx      = wr_idx[end]      .+ (1:nl)
    zd_idx      = wi_idx[end]      .+ (1:nb)
    pg_idx      = zd_idx[end]      .+ (1:ng)  
    qg_idx      = pg_idx[end]      .+ (1:ng)  
    pgs_idx     = qg_idx[end]      .+ (1:nb)
    qbs_pos_idx = pgs_idx[end]     .+ (1:nb)
    qbs_neg_idx = qbs_pos_idx[end] .+ (1:nb)
    p_fr_idx    = qbs_neg_idx[end] .+ (1:nl)
    p_to_idx    = p_fr_idx[end]    .+ (1:nl)   
    q_fr_idx    = p_to_idx[end]    .+ (1:nl)   
    q_to_idx    = q_fr_idx[end]    .+ (1:nl)
    t_idx       = q_to_idx[end]     + 1

    # injection constraints
    p_fr_flowidx = 1:nl
    p_to_flowidx = (nl+1):2*nl
    q_fr_flowidx = (2*nl+1):3*nl
    q_to_flowidx = (3*nl+1):4*nl
    pjidx        = 4*nl    .+ (1:nb)
    qjidx        = 4*nl+nb .+ (1:nb)

    nvar  = t_idx[end]
    neq   = 2*nb + 4*nl
    nineq = 2*nl + 2*nl + 2*nb + 2*nb + 8*nl + 4*ng + 6*nb
    if use_float64 == true
        A = spzeros(neq, nvar)
    else
        A = SparseMatrixCSC{AffExpr, Int64}(undef, neq, nvar)
    end
    # => A = SparseMatrixCSC{AffExpr, Int64}(undef, neq, nvar)
    # => A = Matrix{AffExpr}(undef, neq, nvar)
    # => A .= 0.0
    # => A = zeros(neq, nvar)
    # if zl is embedded as a nonlinear constraint:
        # => A = Matrix{NonlinearExpr}(undef, neq, nvar)
            # =>    A = SparseMatrixCSC{NonlinearExpr, Int64}(undef, neq, nvar)
        # => A .= 0.0
    b = zeros(neq) # this stays 0

    # flow constraints
    A[p_fr_flowidx, p_fr_idx] = sparse(I,nl,nl)
    A[p_to_flowidx, p_to_idx] = sparse(I,nl,nl)
    A[q_fr_flowidx, q_fr_idx] = sparse(I,nl,nl)
    A[q_to_flowidx, q_to_idx] = sparse(I,nl,nl)
    zm                        = spzeros(nl,nl)
    Mflow = [-[gm.Tp_wfr   zm         gm.TpRfr gm.TpIfr];
             -[zm          gm.Tp_wto  gm.TpRto gm.TpIto];
             -[gm.Tq_wfr   zm         gm.TqRfr gm.TqIfr];
             -[zm          gm.Tq_wto  gm.TqRto gm.TqIto]]
    A[1:4*nl,w_fr_idx[1]:wi_idx[end]] = Mflow

    # injection constraints
    A[pjidx,pg_idx]   = -gm.Eg
    A[pjidx,zd_idx]   = spdiagm(pd0)
    A[pjidx,p_fr_idx] = gm.Efr'
    A[pjidx,p_to_idx] = gm.Eto'
    A[pjidx,pgs_idx]  = sparse(I,nb,nb)

    A[qjidx,qg_idx]   = -gm.Eg
    A[qjidx,zd_idx]   = spdiagm(qd0)
    A[qjidx,q_fr_idx] = gm.Efr'
    A[qjidx,q_to_idx] = gm.Eto'
    A[qjidx,qbs_pos_idx]  = -sparse(I,nb,nb)
    A[qjidx,qbs_neg_idx]  = -sparse(I,nb,nb)

    # inequality constraints!
    C  = spzeros(nineq, nvar)
    d  = Vector{AffExpr}(undef, nineq)
    d .= 0.0

    #C1
    idx_nd = 0
    C[idx_nd .+ (1:nl), w_fr_idx] = +sparse(I,nl,nl)
    C[idx_nd .+ (1:nl), w_idx]    = -gm.Efr
    d[idx_nd .+ (1:nl)] .= 0.0
    idx_nd += nl

    #C2
    C[idx_nd .+ (1:nl), w_to_idx] = +sparse(I,nl,nl)
    C[idx_nd .+ (1:nl), w_idx]    = -gm.Eto
    d[idx_nd .+ (1:nl)] .= 0.0
    idx_nd += nl

    #C3
    C[idx_nd .+ (1:nl), w_idx]    = gm.Efr
    C[idx_nd .+ (1:nl), w_fr_idx] = -sparse(I,nl,nl)
    d[idx_nd .+ (1:nl)] .= -gm.vmax[gm.fr_buses].^2 .* (1 .- zl)
    idx_nd += nl

    #C4
    C[idx_nd .+ (1:nl), w_idx]    = gm.Eto
    C[idx_nd .+ (1:nl), w_to_idx] = -sparse(I,nl,nl)
    d[idx_nd .+ (1:nl)] .= -gm.vmax[gm.to_buses].^2 .* (1 .- zl)
    idx_nd += nl

    #C5
    C[idx_nd .+ (1:nb), zd_idx] = -sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    #C6
    C[idx_nd .+ (1:nb), zd_idx]= +sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= -1.0
    idx_nd += nb

    #C7
    C[idx_nd .+ (1:nb), w_idx] = -sparse(I,nb,nb) 
    d[idx_nd .+ (1:nb)] .= gm.vmin.^2
    idx_nd += nb

    #C8
    C[idx_nd .+ (1:nb), w_idx] = +sparse(I,nb,nb) 
    d[idx_nd .+ (1:nb)] .= -gm.vmax.^2
    idx_nd += nb

    #C9
    C[idx_nd .+ (1:nl), w_fr_idx] = -sparse(I,nl,nl)
    d[idx_nd .+ (1:nl)] .= zl.*gm.vmin[gm.fr_buses].^2
    idx_nd += nl

    #C10
    C[idx_nd .+ (1:nl), w_fr_idx] = +sparse(I,nl,nl)
    d[idx_nd .+ (1:nl)] .= -zl.*gm.vmax[gm.fr_buses].^2
    idx_nd += nl

    #C11
    C[idx_nd .+ (1:nl), w_to_idx] = -sparse(I,nl,nl)
    d[idx_nd .+ (1:nl)] .= zl.*gm.vmin[gm.fr_buses].^2
    idx_nd += nl

    #C12
    C[idx_nd .+ (1:nl), w_to_idx] = +sparse(I,nl,nl)
    d[idx_nd .+ (1:nl)] .= -zl.*gm.vmax[gm.fr_buses].^2
    idx_nd += nl

    #C13
    C[idx_nd .+ (1:nl), wr_idx] = -sparse(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= zl.*gm.wr_min
    idx_nd += nl

    #C14
    C[idx_nd .+ (1:nl), wr_idx] = sparse(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= -zl.*gm.wr_max
    idx_nd += nl

    #C15
    C[idx_nd .+ (1:nl), wi_idx] = -sparse(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= zl.*gm.wi_min
    idx_nd += nl

    #C16
    C[idx_nd .+ (1:nl), wi_idx] = sparse(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= -zl.*gm.wi_max
    idx_nd += nl

    #C etc...
    C[idx_nd .+ (1:ng), pg_idx] = -sparse(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= 0.0
    idx_nd += ng

    C[idx_nd .+ (1:ng), pg_idx] = sparse(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= -gm.pg_max
    idx_nd += ng

    C[idx_nd .+ (1:ng), qg_idx] = -sparse(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= gm.qg_min
    idx_nd += ng

    C[idx_nd .+ (1:ng), qg_idx] = sparse(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= -gm.qg_max
    idx_nd += ng

    C[idx_nd .+ (1:nb), pgs_idx] = -sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # double
    C[idx_nd .+ (1:nb), pgs_idx] = sparse(I,nb,nb)
    C[idx_nd .+ (1:nb), w_idx]   = -gm.Gs
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    C[idx_nd .+ (1:nb), qbs_pos_idx] = -sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # double
    C[idx_nd .+ (1:nb), qbs_pos_idx] = sparse(I,nb,nb)
    C[idx_nd .+ (1:nb), w_idx]       = -gm.Bs_pos
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # double
    C[idx_nd .+ (1:nb), qbs_neg_idx] = -sparse(I,nb,nb)
    C[idx_nd .+ (1:nb), w_idx]       = gm.Bs_neg
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    C[idx_nd .+ (1:nb), qbs_neg_idx] = sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # linear terms in the objective
    if use_float64 == true
        H  = zeros(nvar)
    else
        H  = Vector{AffExpr}(undef, nvar)
        H .= 0.0
    end
    h = sum(pd0)
    H[zd_idx] = - pd0

    # normalize?
    if (normalize_shed == true) && (typeof(pd0) == Vector{Float64})
        println("Normalizing load shed objective by total load.")
        H = H./sum(pd0)
        h = h/sum(pd0)
    elseif (normalize_shed == true) && (typeof(pd0) == Vector{VariableRef})
        println("Use epigraph trick, in this case, to enforce normalization.")
    end

    if include_cost == true
        # this is mainly for testing and troubleshooting
        H[pg_idx] = gm.clin
    end

    # apply all RSOCs
    nrsoc = 3*nl

    m1 = Vector{Any}(undef,nrsoc)#[Vector{Any}(undef,10)for ii in 1:nrsoc]
    m2 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]
    m3 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]

    b1 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]
    b2 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]
    b3 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]

    # flow limit f -> t
    for ii in 1:nl
        m1[ii] = zeros(1,nvar)
        b1[ii] = gm.smax[ii]^2 # zl[ii]*

        m2[ii] = zeros(1,nvar)
        b2[ii] = 0.5

        m3[ii] = zeros(2,nvar)
        m3[ii][1,p_fr_idx[ii]] = 1
        m3[ii][2,q_fr_idx[ii]] = 1
        b3[ii] = zeros(2)
    end

    # flow limit t -> f
    for ii in 1:nl
        m1[ii+nl] = zeros(1,nvar)
        b1[ii+nl] = gm.smax[ii]^2 # zl[ii]*

        m2[ii+nl] = zeros(1,nvar)
        b2[ii+nl] = 0.5

        m3[ii+nl] = zeros(2,nvar)
        m3[ii+nl][1,p_to_idx[ii]] = 1
        m3[ii+nl][2,q_to_idx[ii]] = 1
        b3[ii+nl] = zeros(2)
    end

    # RSOC on voltage
    for ii in 1:nl
        m1[ii+2*nl] = zeros(1,nvar)
        m1[ii+2*nl][w_fr_idx[ii]] = 1
        b1[ii+2*nl] = 0

        m2[ii+2*nl] = zeros(1,nvar)
        m2[ii+2*nl][w_to_idx[ii]] = 0.5
        b2[ii+2*nl] = 0

        m3[ii+2*nl] = zeros(2,nvar)
        m3[ii+2*nl][1,wr_idx[ii]] = 1
        m3[ii+2*nl][2,wi_idx[ii]] = 1
        b3[ii+2*nl] = zeros(2)
    end

    # throw into a common dict
    lp = Dict(:A => A,
              :b => b,
              :C => C,
              :d => d,
              :H => H,
              :h => h)
    soc = Dict(:m1 => m1,
               :m2 => m2,
               :m3 => m3,
               :b1 => b1,
               :b2 => b2,
               :b3 => b3)

    return lp, soc
end

function canonicalize_with_SOC_load(zl, pd0, qd0, gm::SOCGridModel; include_cost=false, include_QGB_shedding=false, use_float64=false)
    "zl, pd0, qd0 can be Float64 vectors, or they can be variable vectors"
    nl = gm.nl
    nb = gm.nb
    ng = gm.ng

    w_idx       =                     (1:nb)
    wr_idx      = w_idx[end]       .+ (1:nl)
    wi_idx      = wr_idx[end]      .+ (1:nl)
    zd_idx      = wi_idx[end]      .+ (1:nb)
    pg_idx      = zd_idx[end]      .+ (1:ng)  
    qg_idx      = pg_idx[end]      .+ (1:ng)  
    pgs_idx     = qg_idx[end]      .+ (1:nb)
    qbs_pos_idx = pgs_idx[end]     .+ (1:nb)
    qbs_neg_idx = qbs_pos_idx[end] .+ (1:nb)
    p_fr_idx    = qbs_neg_idx[end] .+ (1:nl)
    p_to_idx    = p_fr_idx[end]    .+ (1:nl)   
    q_fr_idx    = p_to_idx[end]    .+ (1:nl)   
    q_to_idx    = q_fr_idx[end]    .+ (1:nl)
    t_idx       = q_to_idx[end]     + 1

    nvar  = t_idx[end]
    neq   = 2*nb + 4*nl
    nineq = 2*nb + 2*nb + 4*nl + 4*ng + 6*nb
    if use_float64 == true
        A = spzeros(neq, nvar)
    else
        A = SparseMatrixCSC{AffExpr, Int64}(undef, neq, nvar)
    end
    # => A = Matrix{AffExpr}(undef, neq, nvar)
    # => A .= 0.0
    # => A = zeros(neq, nvar)
    # if zl is embedded as a nonlinear constraint:
        # => A = Matrix{NonlinearExpr}(undef, neq, nvar)
            # =>    A = SparseMatrixCSC{NonlinearExpr, Int64}(undef, neq, nvar)
        # => A .= 0.0
    b = zeros(neq) # this stays 0

    # flow constraints
    A[1:nl, p_fr_idx]          = sparse(I,nl,nl)
    A[(nl+1):2*nl, p_to_idx]   = sparse(I,nl,nl)
    A[(2*nl+1):3*nl, q_fr_idx] = sparse(I,nl,nl)
    A[(3*nl+1):4*nl, q_to_idx] = sparse(I,nl,nl)
    Mflow = [-zl.*[gm.Tpfr gm.TpRfr gm.TpIfr];
             -zl.*[gm.Tpto gm.TpRto gm.TpIto];
             -zl.*[gm.Tqfr gm.TqRfr gm.TqIfr];
             -zl.*[gm.Tqto gm.TqRto gm.TqIto]]
    A[1:4*nl,w_idx[1]:wi_idx[end]] = Mflow

    # injection constraints
    pjidx = 4*nl    .+ (1:nb)
    qjidx = 4*nl+nb .+ (1:nb)

    A[pjidx,pg_idx]   = -gm.Eg
    A[pjidx,zd_idx]   = spdiagm(pd0)
    A[pjidx,p_fr_idx] = gm.Efr'
    A[pjidx,p_to_idx] = gm.Eto'
    A[pjidx,pgs_idx]  = sparse(I,nb,nb)

    A[qjidx,qg_idx]   = -gm.Eg
    A[qjidx,zd_idx]   = spdiagm(qd0)
    A[qjidx,q_fr_idx] = gm.Efr'
    A[qjidx,q_to_idx] = gm.Eto'
    A[qjidx,qbs_pos_idx]  = -sparse(I,nb,nb)
    A[qjidx,qbs_neg_idx]  = -sparse(I,nb,nb)

    C = spzeros(nineq, nvar)
    d = zeros(nineq)

    idx_nd = 0
    C[idx_nd .+ (1:nb), zd_idx] = -sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    C[idx_nd .+ (1:nb), zd_idx]= +sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= -1.0
    idx_nd += nb

    C[idx_nd .+ (1:nb), w_idx] = -sparse(I,nb,nb) 
    d[idx_nd .+ (1:nb)] .= gm.vmin.^2
    idx_nd += nb

    C[idx_nd .+ (1:nb), w_idx] = +sparse(I,nb,nb) 
    d[idx_nd .+ (1:nb)] .= -gm.vmax.^2
    idx_nd += nb

    C[idx_nd .+ (1:nl), wr_idx] = -sparse(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= gm.wr_min
    idx_nd += nl

    C[idx_nd .+ (1:nl), wr_idx] = sparse(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= -gm.wr_max
    idx_nd += nl

    C[idx_nd .+ (1:nl), wi_idx] = -sparse(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= gm.wi_min
    idx_nd += nl

    C[idx_nd .+ (1:nl), wi_idx] = sparse(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= -gm.wi_max
    idx_nd += nl

    C[idx_nd .+ (1:ng), pg_idx] = -sparse(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= 0.0
    idx_nd += ng

    C[idx_nd .+ (1:ng), pg_idx] = sparse(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= -gm.pg_max
    idx_nd += ng

    C[idx_nd .+ (1:ng), qg_idx] = -sparse(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= gm.qg_min
    idx_nd += ng

    C[idx_nd .+ (1:ng), qg_idx] = sparse(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= -gm.qg_max
    idx_nd += ng

    C[idx_nd .+ (1:nb), pgs_idx] = -sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # double
    C[idx_nd .+ (1:nb), pgs_idx] = sparse(I,nb,nb)
    C[idx_nd .+ (1:nb), w_idx]   = -gm.Gs
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    C[idx_nd .+ (1:nb), qbs_pos_idx] = -sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # double
    C[idx_nd .+ (1:nb), qbs_pos_idx] = sparse(I,nb,nb)
    C[idx_nd .+ (1:nb), w_idx]       = -gm.Bs_pos
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # double
    C[idx_nd .+ (1:nb), qbs_neg_idx] = -sparse(I,nb,nb)
    C[idx_nd .+ (1:nb), w_idx]       = gm.Bs_neg
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    C[idx_nd .+ (1:nb), qbs_neg_idx] = sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    nobj = nb + nb + nb + nb
    # => F = zeros(nobj,nvar)
    if use_float64 == true
        F = spzeros(nobj,nvar)
        f = zeros(nobj)
    else
        F = SparseMatrixCSC{AffExpr, Int64}(undef, nobj,nvar)
        f = Vector{AffExpr}(undef, nobj)
        f.= 0.0
    end
    # => F = Matrix{NonlinearExpr}(undef, nobj,nvar)
    # => F .= 0.0
    # => f = zeros(nobj)

    # d1
    F[1:nb,zd_idx] = diagm(pd0)
    f[1:nb] = -pd0

    if include_QGB_shedding == true
        # d2
        F[nb .+ (1:nb),pgs_idx] = Matrix(I,nb,nb)
        F[nb .+ (1:nb),w_idx]   = -gm.Gs
        f[nb .+ (1:nb)] .= 0.0

        # d3
        F[2*nb .+ (1:nb),qbs_pos_idx] = Matrix(I,nb,nb)
        F[2*nb .+ (1:nb),w_idx]       = -gm.Bs_pos
        f[2*nb .+ (1:nb)] .= 0.0

        # d4
        F[3*nb .+ (1:nb),qbs_neg_idx] = -Matrix(I,nb,nb)
        F[3*nb .+ (1:nb),w_idx]       = gm.Bs_neg
        f[3*nb .+ (1:nb)] .= 0.0
    end

    # linear terms in the objective
    h = zeros(nvar)
    if include_cost == true
        # this is mainly for testing and troubleshooting
        h[pg_idx] = gm.clin
    end
    h[end]    = 1

    # apply all RSOCs
    nrsoc = 3*nl + 1

    m1 = Vector{Any}(undef,nrsoc)#[Vector{Any}(undef,10)for ii in 1:nrsoc]
    m2 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]
    m3 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]

    b1 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]
    b2 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]
    b3 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]

    # flow limit f -> t
    for ii in 1:nl
        m1[ii] = zeros(1,nvar)
        b1[ii] = gm.smax[ii]^2

        m2[ii] = zeros(1,nvar)
        b2[ii] = 0.5

        m3[ii] = zeros(2,nvar)
        m3[ii][1,p_fr_idx[ii]] = 1
        m3[ii][2,q_fr_idx[ii]] = 1
        b3[ii] = zeros(2)
    end

    # flow limit t -> f
    for ii in 1:nl
        m1[ii+nl] = zeros(1,nvar)
        b1[ii+nl] = gm.smax[ii]^2

        m2[ii+nl] = zeros(1,nvar)
        b2[ii+nl] = 0.5

        m3[ii+nl] = zeros(2,nvar)
        m3[ii+nl][1,p_to_idx[ii]] = 1
        m3[ii+nl][2,q_to_idx[ii]] = 1
        b3[ii+nl] = zeros(2)
    end

    # RSOC on voltage
    for ii in 1:nl
        m1[ii+2*nl] = zeros(1,nvar)
        m1[ii+2*nl][w_idx] = gm.Efr[ii,:]
        b1[ii+2*nl] = 0

        m2[ii+2*nl] = zeros(1,nvar)
        m2[ii+2*nl][w_idx] = 0.5*gm.Eto[ii,:]
        b2[ii+2*nl] = 0

        m3[ii+2*nl] = zeros(2,nvar)
        m3[ii+2*nl][1,wr_idx[ii]] = 1
        m3[ii+2*nl][2,wi_idx[ii]] = 1
        b3[ii+2*nl] = zeros(2)
    end

    # cost term!
    m1[end] = zeros(1,nvar)
    m1[end][t_idx] = 1
    b1[end] = 0

    m2[end] = zeros(1,nvar)
    b2[end] = 0.5

    m3[end] = F
    b3[end] = f

    # throw mi and bi into a common struct
    lp = Dict(:A => A,
              :b => b,
              :C => C,
              :d => d,
              :h => h)
    soc = Dict(:m1 => m1,
               :m2 => m2,
               :m3 => m3,
               :b1 => b1,
               :b2 => b2,
               :b3 => b3)

    return lp, soc
end

function scale_load(load, alpha_lb, alpha_ub)
    nload = length(load)
    lb = zeros(nload)
    ub = zeros(nload)

    for ii in 1:nload
        if load[ii] > 0
            ub[ii] = load[ii]*alpha_ub
            lb[ii] = load[ii]*alpha_lb
        else
            ub[ii] = load[ii]*alpha_lb
            lb[ii] = load[ii]*alpha_ub 
        end
    end
    return lb, ub
end

function min_loadshed_soc_primal_explicit(gm::SOCGridModel, 
                                          zl::Vector{Float64};
                                          soc::Bool=true, 
                                          host_start::Bool=false,
                                          include_cost::Bool=false, 
                                          normalize_shed::Bool=false,
                                          host_start_model::Model=Model(),
                                          include_QGB_shedding::Bool=false)
    # loads
    pd0 = copy(gm.pd)
    qd0 = copy(gm.qd)

    # build soc_relaxation
    nl    = gm.nl
    nb    = gm.nb
    ng    = gm.ng
    if soc == true
        model = Model(Gurobi.Optimizer)
    else
        model = Model(Ipopt.Optimizer)
    end

    # add lifted voltages
    @variable(model, w[1:nb])
    @variable(model, wr[1:nl])
    @variable(model, wi[1:nl])
    @variable(model, zd[1:nb])
    @variable(model, pg[1:ng])
    @variable(model, qg[1:ng])
    @variable(model, pgs[1:nb])
    @variable(model, qbs_pos[1:nb])
    @variable(model, qbs_neg[1:nb])
    @variable(model, p_fr[1:nl])
    @variable(model, p_to[1:nl])
    @variable(model, q_fr[1:nl])
    @variable(model, q_to[1:nl])

    # ================================================ #
    # define the equality constraints
    @constraint(model, p_fr .- zl.*(gm.Tpfr*w + gm.TpRfr*wr + gm.TpIfr*wi) .== 0.0)
    @constraint(model, p_to .- zl.*(gm.Tpto*w + gm.TpRto*wr + gm.TpIto*wi) .== 0.0)
    @constraint(model, q_fr .- zl.*(gm.Tqfr*w + gm.TqRfr*wr + gm.TqIfr*wi) .== 0.0)
    @constraint(model, q_to .- zl.*(gm.Tqto*w + gm.TqRto*wr + gm.TqIto*wi) .== 0.0)

    @constraint(model,  -gm.Eg*pg + diagm(pd0)*zd + gm.Efr'*p_fr + gm.Eto'*p_to + pgs               .== 0.0)
    @constraint(model,  -gm.Eg*qg + diagm(qd0)*zd + gm.Efr'*q_fr + gm.Eto'*q_to - qbs_pos - qbs_neg .== 0.0)

    # ================================================ #
    # Define the inequality constraints
    @constraint(model, 0.0 .- zd .<= 0.0)
    @constraint(model, zd .- 1.0 .<= 0.0)

    @constraint(model, gm.vmin.^2 .- w .<= 0.0)
    @constraint(model, w .- gm.vmax.^2 .<= 0.0)

    @constraint(model, gm.wr_min .- wr .<= 0.0)
    @constraint(model, wr .- gm.wr_max .<= 0.0)

    @constraint(model, gm.wi_min .- wi .<= 0.0)
    @constraint(model, wi .- gm.wi_max .<= 0.0)

    @constraint(model, 0.0 .- pg    .<= 0.0)
    @constraint(model, pg .- gm.pg_max .<= 0.0)

    @constraint(model, gm.qg_min .- qg .<= 0.0)
    @constraint(model, qg .- gm.qg_max .<= 0.0)

    @constraint(model, -pgs                  .<= 0.0)
    @constraint(model, pgs - gm.Gs*w         .<= 0.0)
    @constraint(model, -qbs_pos              .<= 0.0)
    @constraint(model, qbs_pos - gm.Bs_pos*w .<= 0.0)
    @constraint(model, gm.Bs_neg*w - qbs_neg .<= 0.0)
    @constraint(model, qbs_neg               .<= 0.0)

    # ================================================ #
    # Define the RSOC constraints
    @constraint(model, p_fr.^2 .+ q_fr.^2 .<= gm.smax.^2)
    @constraint(model, p_to.^2 .+ q_to.^2 .<= gm.smax.^2)
    if soc == true
        v1 = gm.Efr*w
        v2 = gm.Eto*w
        for ij in 1:gm.nl
            @constraint(model, [gm.smax[ij].^2; 0.5; p_fr[ij]; q_fr[ij]] in RotatedSecondOrderCone())
            @constraint(model, [gm.smax[ij].^2; 0.5; p_to[ij]; q_to[ij]] in RotatedSecondOrderCone())
            @constraint(model, [v1[ij]; 0.5*v2[ij]; wr[ij]; wi[ij]]      in RotatedSecondOrderCone())
        end
    else
        # soc line flow constraints
        @constraint(model, p_fr.^2 .+ q_fr.^2 .<= gm.smax.^2)
        @constraint(model, p_to.^2 .+ q_to.^2 .<= gm.smax.^2)

        # nonconvex model!
        @constraint(model, (wr).^2 .+ (wi).^2 .== (gm.Efr*w).*(gm.Eto*w))
        
        # also, add the phase angle cycle constraint
        @variable(model, theta[1:gm.nb])
        @constraint(model, theta[1] .== 0)
        E = gm.Efr - gm.Eto
        @constraint(model, wi .== tan.(E*theta).*wr)
    end

    # ================================================ #
    # objective!
    d1 = pd0 - diagm(pd0)*zd
    # alternative: d1'*d1
    obj = sum(d1)
    if normalize_shed == true
        println("Normalizing load shed objective by total load.")
        obj = obj/sum(pd0)
    end

    # include a cost regularizier? This is mainly for troubleshooting
    if include_cost == true
        obj += gm.clin'*pg
    end

    # include shedding of Q and G/B in the objective?
    if include_QGB_shedding
        d2 = pgs -  gm.Gs*w
        d3 = qbs_pos - gm.Bs_pos*w
        d4 = gm.Bs_neg*w - qbs_neg
        obj += d2'*d2 + d3'*d3 + d4'*d4
    end

    # hot start?
    if host_start == true
        set_start_value.(w, value.(host_start_model[:w]))
        set_start_value.(wr, value.(host_start_model[:wr]))
        set_start_value.(wi, value.(host_start_model[:wi]))
        set_start_value.(zd, value.(host_start_model[:zd]))
        set_start_value.(pg, value.(host_start_model[:pg]))
        set_start_value.(qg, value.(host_start_model[:qg]))
        set_start_value.(pgs, value.(host_start_model[:pgs]))
        set_start_value.(qbs_pos, value.(host_start_model[:qbs_pos]))
        set_start_value.(qbs_neg, value.(host_start_model[:qbs_neg]))
        set_start_value.(p_fr, value.(host_start_model[:p_fr]))
        set_start_value.(p_to, value.(host_start_model[:p_to]))
        set_start_value.(q_fr, value.(host_start_model[:q_fr]))
        set_start_value.(q_to, value.(host_start_model[:q_to]))

        # infer theta values
        E = gm.Efr - gm.Eto
        theta_hs = E\atan.(value.(host_start_model[:wi])./value.(host_start_model[:wr]))
        set_start_value.(theta, theta_hs)
    end

    # add the lines and loads
    model[:pd0] = copy(pd0)
    model[:qd0] = copy(qd0)
    model[:zl]  = copy(zl)

    @objective(model, Min, obj)
    optimize!(model)
    println(objective_value(model))

    return model
end

function ac_redispatch_snapped(gm::SOCGridModel, 
                               zl::Vector{Float64};
                               soft_flow::Bool=false,
                               host_start::Bool=false,
                               include_cost::Bool=false, 
                               normalize_shed::Bool=false,
                               host_start_model::Model=Model(),
                               include_QGB_shedding::Bool=false)
    # loads
    pd0 = copy(gm.pd)
    qd0 = copy(gm.qd)

    # build soc_relaxation
    nl    = gm.nl
    nb    = gm.nb
    ng    = gm.ng
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "max_wall_time",     3600.0)
    # => tol   = 1e-4
    # => set_optimizer_attribute(model, "tol",                  tol) # overall convergence tolerance
    # => set_optimizer_attribute(model, "acceptable_tol",       tol) # "Acceptable" convergence tolerance (relative).
    set_optimizer_attribute(model, "max_iter",           10000)
    set_optimizer_attribute(model, "mu_init",             1e-8)

    # add lifted voltages
    @variable(model, vm[1:nb])
    @variable(model, va[1:nb])
    @variable(model, zd[1:nb])
    @variable(model, pg[1:ng])
    @variable(model, qg[1:ng])
    @variable(model, pgs[1:nb])
    @variable(model, qbs_pos[1:nb])
    @variable(model, qbs_neg[1:nb])
    @variable(model, p_fr[1:nl])
    @variable(model, p_to[1:nl])
    @variable(model, q_fr[1:nl])
    @variable(model, q_to[1:nl])

    # define pseudo-SOC variables
    E  = gm.Efr - gm.Eto
    w  = vm.^2
    wr = (gm.Efr*vm).*(gm.Eto*vm).*cos.(E*va)
    wi = (gm.Efr*vm).*(gm.Eto*vm).*sin.(E*va)

    # ================================================ #
    # define the equality constraints
    obj_penalty = 0
    if soft_flow == false
        @constraint(model, p_fr .- zl.*(gm.Tpfr*w + gm.TpRfr*wr + gm.TpIfr*wi) .== 0.0)
        @constraint(model, p_to .- zl.*(gm.Tpto*w + gm.TpRto*wr + gm.TpIto*wi) .== 0.0)
        @constraint(model, q_fr .- zl.*(gm.Tqfr*w + gm.TqRfr*wr + gm.TqIfr*wi) .== 0.0)
        @constraint(model, q_to .- zl.*(gm.Tqto*w + gm.TqRto*wr + gm.TqIto*wi) .== 0.0)
    else
        @variable(model, tp_fr[1:gm.nl], lower_bound = 0.0)
        @variable(model, tp_to[1:gm.nl], lower_bound = 0.0)
        @variable(model, tq_fr[1:gm.nl], lower_bound = 0.0)
        @variable(model, tq_to[1:gm.nl], lower_bound = 0.0)

        @constraint(model, p_fr .- zl.*(gm.Tpfr*w + gm.TpRfr*wr + gm.TpIfr*wi) .<= tp_fr)
        @constraint(model, p_to .- zl.*(gm.Tpto*w + gm.TpRto*wr + gm.TpIto*wi) .<= tp_to)
        @constraint(model, q_fr .- zl.*(gm.Tqfr*w + gm.TqRfr*wr + gm.TqIfr*wi) .<= tq_fr)
        @constraint(model, q_to .- zl.*(gm.Tqto*w + gm.TqRto*wr + gm.TqIto*wi) .<= tq_to)

        @constraint(model, -p_fr .+ zl.*(gm.Tpfr*w + gm.TpRfr*wr + gm.TpIfr*wi) .<= tp_fr)
        @constraint(model, -p_to .+ zl.*(gm.Tpto*w + gm.TpRto*wr + gm.TpIto*wi) .<= tp_to)
        @constraint(model, -q_fr .+ zl.*(gm.Tqfr*w + gm.TqRfr*wr + gm.TqIfr*wi) .<= tq_fr)
        @constraint(model, -q_to .+ zl.*(gm.Tqto*w + gm.TqRto*wr + gm.TqIto*wi) .<= tq_to)

        obj_penalty = 1e4*(sum(tp_fr) + sum(tp_to) + sum(tq_fr) + sum(tq_to))
    end


    @constraint(model,  -gm.Eg*pg + diagm(pd0)*zd + gm.Efr'*p_fr + gm.Eto'*p_to + pgs               .== 0.0)
    @constraint(model,  -gm.Eg*qg + diagm(qd0)*zd + gm.Efr'*q_fr + gm.Eto'*q_to - qbs_pos - qbs_neg .== 0.0)

    # ================================================ #
    # Define the inequality constraints
    @constraint(model, 0.0 .- zd .<= 0.0)
    @constraint(model, zd .- 1.0 .<= 0.0)

    @constraint(model, gm.vmin .- vm .<= 0.0)
    @constraint(model, vm .- gm.vmax .<= 0.0)

    @constraint(model, 0.0 .- pg    .<= 0.0)
    @constraint(model, pg .- gm.pg_max .<= 0.0)

    @constraint(model, gm.qg_min .- qg .<= 0.0)
    @constraint(model, qg .- gm.qg_max .<= 0.0)

    @constraint(model, -pgs                  .<= 0.0)
    @constraint(model, pgs - gm.Gs*w         .<= 0.0)
    @constraint(model, -qbs_pos              .<= 0.0)
    @constraint(model, qbs_pos - gm.Bs_pos*w .<= 0.0)
    @constraint(model, gm.Bs_neg*w - qbs_neg .<= 0.0)
    @constraint(model, qbs_neg               .<= 0.0)

    # ================================================ #
    # Define the RSOC constraints
    @constraint(model, p_fr.^2 .+ q_fr.^2 .<= gm.smax.^2)
    @constraint(model, p_to.^2 .+ q_to.^2 .<= gm.smax.^2)

    # ================================================ #
    # objective!
    d1 = pd0 - diagm(pd0)*zd
    # alternative: d1'*d1
    obj = sum(d1)
    if normalize_shed == true
        println("Normalizing load shed objective by total load.")
        obj = obj/sum(pd0)
    end

    # include a cost regularizier? This is mainly for troubleshooting
    if include_cost == true
        obj += gm.clin'*pg
    end

    # include shedding of Q and G/B in the objective?
    if include_QGB_shedding
        d2 = pgs -  gm.Gs*w
        d3 = qbs_pos - gm.Bs_pos*w
        d4 = gm.Bs_neg*w - qbs_neg
        obj += d2'*d2 + d3'*d3 + d4'*d4
    end

    # hot start?
    if host_start == true
        set_start_value.(vm, value.(host_start_model[:w]).^2)
        set_start_value.(va, 0.0)
        set_start_value.(zd, value.(host_start_model[:zd]))
        set_start_value.(pg, value.(host_start_model[:pg]))
        set_start_value.(qg, value.(host_start_model[:qg]))
        set_start_value.(pgs, value.(host_start_model[:pgs]))
        set_start_value.(qbs_pos, value.(host_start_model[:qbs_pos]))
        set_start_value.(qbs_neg, value.(host_start_model[:qbs_neg]))
        set_start_value.(p_fr, value.(host_start_model[:p_fr]))
        set_start_value.(p_to, value.(host_start_model[:p_to]))
        set_start_value.(q_fr, value.(host_start_model[:q_fr]))
        set_start_value.(q_to, value.(host_start_model[:q_to]))
        if soft_flow == true
            set_start_value.(tp_fr, 0.0)
            set_start_value.(tp_to, 0.0)
            set_start_value.(tq_fr, 0.0)
            set_start_value.(tq_to, 0.0)
        end
    end

    obj = obj + obj_penalty

    # add the lines and loads
    model[:obj_penalty] = obj_penalty
    model[:pd0]         = copy(pd0)
    model[:qd0]         = copy(qd0)
    model[:zl]          = copy(zl)

    @objective(model, Min, obj)
    optimize!(model)

    return model
end

function min_loadshed_soc_primal_explicit_flowcuts(gm::SOCGridModel, 
                                                   zl::Vector{Float64};
                                                   soc::Bool=true, 
                                                   host_start::Bool=false,
                                                   include_cost::Bool=false, 
                                                   normalize_shed::Bool=false,
                                                   host_start_model::Model=Model(),
                                                   include_QGB_shedding::Bool=false)
    # loads
    pd0 = copy(gm.pd)
    qd0 = copy(gm.qd)

    # build soc_relaxation
    nl    = gm.nl
    nb    = gm.nb
    ng    = gm.ng
    if soc == true
        model = Model(Gurobi.Optimizer)
        set_silent(model)
    else
        model = Model(Ipopt.Optimizer)
        set_silent(model)
        tol   = 1e-4
        set_optimizer_attribute(model, "max_wall_time",     1800.0)
        set_optimizer_attribute(model, "tol",                  tol) # overall convergence tolerance
        set_optimizer_attribute(model, "acceptable_tol",       tol) # "Acceptable" convergence tolerance (relative).
        set_optimizer_attribute(model, "max_iter",           10000)
        set_optimizer_attribute(model, "mu_init",             1e-8)
    end

    # add lifted voltages
    @variable(model, w[1:nb])

    @variable(model, w_fr[1:nl])
    @variable(model, w_to[1:nl])

    @variable(model, wr[1:nl])
    @variable(model, wi[1:nl])
    @variable(model, zd[1:nb])
    @variable(model, pg[1:ng])
    @variable(model, qg[1:ng])
    @variable(model, pgs[1:nb])
    @variable(model, qbs_pos[1:nb])
    @variable(model, qbs_neg[1:nb])
    @variable(model, p_fr[1:nl])
    @variable(model, p_to[1:nl])
    @variable(model, q_fr[1:nl])
    @variable(model, q_to[1:nl])

    # ================================================ #
    # define the equality constraints
    @constraint(model, p_fr .- (gm.Tp_wfr*w_fr + gm.TpRfr*wr + gm.TpIfr*wi) .== 0.0)
    @constraint(model, p_to .- (gm.Tp_wto*w_to + gm.TpRto*wr + gm.TpIto*wi) .== 0.0)
    @constraint(model, q_fr .- (gm.Tq_wfr*w_fr + gm.TqRfr*wr + gm.TqIfr*wi) .== 0.0)
    @constraint(model, q_to .- (gm.Tq_wto*w_to + gm.TqRto*wr + gm.TqIto*wi) .== 0.0)

    @constraint(model,  -gm.Eg*pg + diagm(pd0)*zd + gm.Efr'*p_fr + gm.Eto'*p_to + pgs               .== 0.0)
    @constraint(model,  -gm.Eg*qg + diagm(qd0)*zd + gm.Efr'*q_fr + gm.Eto'*q_to - qbs_pos - qbs_neg .== 0.0)

    # ================================================ #
    
    # add voltage constraints
    @constraint(model, w_fr .- gm.Efr*w .<= 0.0) #C1
    @constraint(model, w_to .- gm.Eto*w .<= 0.0) #C2

    @constraint(model, gm.Efr*w .- gm.vmax[gm.fr_buses].^2 .* (1 .- zl) .- w_fr .<= 0.0) #C3
    @constraint(model, gm.Eto*w .- gm.vmax[gm.to_buses].^2 .* (1 .- zl) .- w_to .<= 0.0) #C4

    # Define the inequality constraints
    @constraint(model, 0.0 .- zd .<= 0.0) #C5
    @constraint(model, zd .- 1.0 .<= 0.0) #C6

    @constraint(model, gm.vmin.^2 .- w .<= 0.0) #C7
    @constraint(model, w .- gm.vmax.^2 .<= 0.0) #C8

    @constraint(model, zl.*gm.vmin[gm.fr_buses].^2 .- w_fr .<= 0.0) #C9
    @constraint(model, w_fr .- zl.*gm.vmax[gm.fr_buses].^2 .<= 0.0) #C10

    @constraint(model, zl.*gm.vmin[gm.to_buses].^2 .- w_to .<= 0.0) #C11
    @constraint(model, w_to .- zl.*gm.vmax[gm.to_buses].^2 .<= 0.0) #C12

    @constraint(model, zl.*gm.wr_min .- wr .<= 0.0) #C13
    @constraint(model, wr .- zl.*gm.wr_max .<= 0.0) #C14

    @constraint(model, zl.*gm.wi_min .- wi .<= 0.0) #C15
    @constraint(model, wi .- zl.*gm.wi_max .<= 0.0) #C16

    @constraint(model, 0.0 .- pg    .<= 0.0) #C etc...
    @constraint(model, pg .- gm.pg_max .<= 0.0)

    @constraint(model, gm.qg_min .- qg .<= 0.0)
    @constraint(model, qg .- gm.qg_max .<= 0.0)

    @constraint(model, -pgs                  .<= 0.0)
    @constraint(model, pgs - gm.Gs*w         .<= 0.0)
    @constraint(model, -qbs_pos              .<= 0.0)
    @constraint(model, qbs_pos - gm.Bs_pos*w .<= 0.0)
    @constraint(model, gm.Bs_neg*w - qbs_neg .<= 0.0)
    @constraint(model, qbs_neg               .<= 0.0)

    # ================================================ #
    # Define the RSOC constraints
    if soc == true
        for ij in 1:gm.nl
            @constraint(model, [zl[ij]*gm.smax[ij].^2; 0.5; p_fr[ij]; q_fr[ij]] in RotatedSecondOrderCone())
            @constraint(model, [zl[ij]*gm.smax[ij].^2; 0.5; p_to[ij]; q_to[ij]] in RotatedSecondOrderCone())
            @constraint(model, [w_fr[ij]; 0.5*w_to[ij]; wr[ij]; wi[ij]]  in RotatedSecondOrderCone())
        end
    else
        # soc line flow constraints
        @constraint(model, p_fr.^2 .+ q_fr.^2 .<= zl.*gm.smax.^2)
        @constraint(model, p_to.^2 .+ q_to.^2 .<= zl.*gm.smax.^2)

        # nonconvex model!
        @constraint(model, (wr).^2 .+ (wi).^2 .== (w_fr).*(w_to))
        
        # also, add the phase angle cycle constraint
        @variable(model, theta[1:gm.nb])
        @constraint(model, theta[1] .== 0)
        E = gm.Efr - gm.Eto
        @constraint(model, wi .== tan.(E*theta).*wr)
    end

    # ================================================ #
    # objective!
    d1 = pd0 - diagm(pd0)*zd
    # alternative: d1'*d1
    obj = sum(d1)
    if normalize_shed == true
        println("Normalizing load shed objective by total load.")
        obj = obj/sum(pd0)
    end

    # include a cost regularizier? This is mainly for troubleshooting
    if include_cost == true
        obj += gm.clin'*pg
    end

    # include shedding of Q and G/B in the objective?
    if include_QGB_shedding
        d2 = pgs -  gm.Gs*w
        d3 = qbs_pos - gm.Bs_pos*w
        d4 = gm.Bs_neg*w - qbs_neg
        obj += d2'*d2 + d3'*d3 + d4'*d4
    end

    # hot start?
    if host_start == true
        set_start_value.(w, value.(host_start_model[:w]))
        set_start_value.(w_fr, value.(host_start_model[:w_fr]))
        set_start_value.(w_to, value.(host_start_model[:w_to]))
        set_start_value.(wr, value.(host_start_model[:wr]))
        set_start_value.(wi, value.(host_start_model[:wi]))
        set_start_value.(zd, value.(host_start_model[:zd]))
        set_start_value.(pg, value.(host_start_model[:pg]))
        set_start_value.(qg, value.(host_start_model[:qg]))
        set_start_value.(pgs, value.(host_start_model[:pgs]))
        set_start_value.(qbs_pos, value.(host_start_model[:qbs_pos]))
        set_start_value.(qbs_neg, value.(host_start_model[:qbs_neg]))
        set_start_value.(p_fr, value.(host_start_model[:p_fr]))
        set_start_value.(p_to, value.(host_start_model[:p_to]))
        set_start_value.(q_fr, value.(host_start_model[:q_fr]))
        set_start_value.(q_to, value.(host_start_model[:q_to]))

        # infer theta values
        # => E = gm.Efr - gm.Eto
        # => theta_hs = E\atan.(value.(host_start_model[:wi])./value.(host_start_model[:wr]))
        set_start_value.(theta, 0.0)
    end

    # add the lines and loads
    model[:pd0] = copy(pd0)
    model[:qd0] = copy(qd0)
    model[:zl]  = copy(zl)

    @objective(model, Min, obj)
    optimize!(model)
    # => println(objective_value(model))

    return model
end

"
min h'x
st. Ax + b  = 0
    Cx + d <= 0
    s in K
    x = [w; wr; wi; zd; pg; qg; pgs; qbs_pos; qbs_neg; p_fr; p_to; p_fr; q_to]"
function min_loadshed_soc_primal(gm::SOCGridModel, 
                                 zl::Vector{Float64}; 
                                 include_cost::Bool=false, 
                                 normalize_shed::Bool=false,
                                 flowcuts::Bool=true)
    # canonicalize 
    pd0 = copy(gm.pd)
    qd0 = copy(gm.qd)
    if flowcuts == true
        lp, soc = canonicalize_flowcuts(zl, pd0, qd0, gm)
    else
        lp, soc = canonicalize(zl, pd0, qd0, gm)
    end

    # =================
    neq   = size(lp[:A],1)
    nineq = size(lp[:C],1)
    nvar  = size(lp[:A],2)
    nrsoc = length(soc[:m1])

    # solve the model
    model = Model(Gurobi.Optimizer)
    @variable(model, x[1:nvar])
    @constraint(model, lp[:A]*x + lp[:b] .== 0.0)
    @constraint(model, lp[:C]*x + lp[:d] .<= 0.0)

    # RSOC
    @constraint(model, [ii in 1:nrsoc], [dot(soc[:m1][ii],x) + soc[:b1][ii]; dot(soc[:m2][ii],x) + soc[:b2][ii]; soc[:m3][ii]*x + soc[:b3][ii]] in RotatedSecondOrderCone())

    @objective(model, Min, lp[:H]'*x + lp[:h])
    optimize!(model)
    println(objective_value(model))

    return model
end

function min_loadshed_soc_dual(gm::SOCGridModel, 
                               zl::Vector{Float64};
                               hot_start::Bool=false,
                               include_cost::Bool=false, 
                               flowcuts::Bool=true,
                               hot_start_model::Model=Model(),
                               include_QGB_shedding::Bool=false,
                               solver::Symbol=:Gurobi,)
    # canonicalize 
    pd0 = copy(gm.pd)
    qd0 = copy(gm.qd)
    if flowcuts == true
        lp, soc = canonicalize_flowcuts(zl, pd0, qd0, gm)
    else
        lp, soc = canonicalize(zl, pd0, qd0, gm)
    end

    # =================
    neq   = size(lp[:A],1)
    nineq = size(lp[:C],1)
    nvar  = size(lp[:A],2)
    nrsoc = length(soc[:m1])

    if solver == :Gurobi
        model = Model(Gurobi.Optimizer)
    else
        model = Model(Ipopt.Optimizer)
        set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        set_attribute(model, "linear_solver", "ma57")
    end
    @variable(model, lambda[1:neq])
    @variable(model, mu[1:nineq], lower_bound = 0.0)
    @variable(model, s1[1:nrsoc], lower_bound = 0.0)
    @variable(model, s2[1:nrsoc], lower_bound = 0.0)
    s = Dict(ii => @variable(model, [1:size(soc[:m3][ii],1)]) for ii in 1:nrsoc)
    model[:s] = s

    if solver == :Gurobi
        @constraint(model, [ii in 1:nrsoc], [s1[ii]; s2[ii]; s[ii]] in RotatedSecondOrderCone())
    else
        @constraint(model, [ii in 1:nrsoc],  dot(s[ii],s[ii]) <= 2*s1[ii]*s2[ii])
    end

    @constraint(model,     lp[:H] + lp[:A]'*lambda + lp[:C]'*mu - sum(s1[ii]*soc[:m1][ii]' + s2[ii]*soc[:m2][ii]' + soc[:m3][ii]'*s[ii] for ii in 1:nrsoc) .== 0.0)
    @objective(model, Max, lp[:h] + lambda'*lp[:b] + mu'*lp[:d] - sum(s1[ii]*soc[:b1][ii]  + s2[ii]*soc[:b2][ii]  + s[ii]'*soc[:b3][ii] for ii in 1:nrsoc))
    
    if hot_start == true
        set_start_value.(lambda, value.(hot_start_model[:lambda]))
        set_start_value.(mu, value.(hot_start_model[:mu]))
        set_start_value.(s1, value.(hot_start_model[:s1]))
        set_start_value.(s2, value.(hot_start_model[:s2]))
        for ii in 1:nrsoc
            set_start_value.(s[ii], value.(hot_start_model[:s][ii]))
        end
    end

    optimize!(model)
    println(objective_value(model))

    return model
end

function maxmin_loadshed(gm::SOCGridModel, 
                         bounds::Dict{Symbol, Float64}, 
                         nn_model::String;
                         tol::Float64=1e-5,
                         tmax::Float64=100.0,
                         hot_start::Bool=true,
                         include_cost::Bool=false,
                         flowcuts::Bool=true,
                         include_QGB_shedding::Bool=false)

    if hot_start == true
        # first, get the nominal line status
        zl0, logit_zl0 = line_status(gm, bounds, nn_model; high_load=true)
        gm_shed        = deepcopy(gm)
        gm_shed.pd    .= bounds[:load_scale_ub]*copy(gm.pd)
        gm_shed.qd    .= bounds[:load_scale_ub]*copy(gm.qd)
        dual_soln      = min_loadshed_soc_dual(gm_shed, zl0; flowcuts=flowcuts)
    end

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_wall_time",      tmax)
    set_optimizer_attribute(model, "tol",                 tol) # overall convergence tolerance
    set_optimizer_attribute(model, "acceptable_tol",      tol) # "Acceptable" convergence tolerance (relative).
    set_optimizer_attribute(model, "max_iter",          10000)
    set_optimizer_attribute(model, "mu_init",            1e-8)

    set_attribute(model, "hsllib", HSL_jll.libhsl_path)
    set_attribute(model, "linear_solver", "ma57")

    @variable(model, pd0_var[1:gm.nb])
    @variable(model, qd0_var[1:gm.nb])
    @variable(model, risk[1:gm.nl])
    @variable(model, alpha)

    # this only works if pd0 and qd0 are positive
    pd0 = copy(gm.pd)
    qd0 = copy(gm.qd)

    p_lb, p_ub = scale_load(pd0, bounds[:load_scale_lb], bounds[:load_scale_ub])
    q_lb, q_ub = scale_load(qd0, bounds[:load_scale_lb], bounds[:load_scale_ub])

    @constraint(model, p_lb              .<= pd0_var .<= p_ub)
    @constraint(model, q_lb              .<= qd0_var .<= q_ub)
    @constraint(model, bounds[:risk_lb]  .<=  risk   .<= bounds[:risk_ub])
    @constraint(model, bounds[:alpha_lb]  <=  alpha   <= bounds[:alpha_ub])

    # call the NN
    x = [risk; qd0_var; pd0_var; alpha]
    model[:x] = x

    # now, we need to normalize the nn input
    normalization_data = nn_model[1:findlast(==('_'), nn_model)]*"normalization_values.h5"
    fid   = h5open(normalization_data, "r")
    mean = read(fid, "mean")
    std  = read(fid, "std")
    close(fid)

    xn   = (x .- mean)./(std)
    predictor = MathOptAI.PytorchModel(joinpath(pwd(), nn_model))
    # config = Dict(:ReLU => MOAI.ReLUQuadratic(relaxation_parameter = 1e-6))
    logit_zl, _ = MathOptAI.add_predictor(model, predictor, xn; hessian=true, vector_nonlinear_oracle = true)#, gray_box = true)#; reduced_space = true)#gray_box = true)#; gray_box=true) #; gray_box = true)#; reduced_space = true)#; gray_box = true)

    @variable(model, zl[1:gm.nl])
    @constraint(model, zl .== sig.(logit_zl) )

    # now, canonicalize
    if flowcuts == true
        lp, soc = canonicalize_flowcuts(zl, pd0_var, qd0_var, gm) 
    else
        lp, soc = canonicalize(zl, pd0_var, qd0_var, gm) 
    end
    neq   = size(lp[:A],1)
    nineq = size(lp[:C],1)
    nvar  = size(lp[:A],2)
    nrsoc = length(soc[:m1])

    @variable(model, lambda[1:neq])
    @variable(model, mu[1:nineq], lower_bound = 0.0)
    @variable(model, s1[1:nrsoc], lower_bound = 0.0)
    @variable(model, s2[1:nrsoc], lower_bound = 0.0)
    s = Dict(ii => @variable(model, [1:size(soc[:m3][ii],1)]) for ii in 1:nrsoc)
    
    #s2 = Vector{NonlinearExpr}(undef, nvar)
    #s2 .= 0.0
    #for ii in 1:nrsoc
    #    s2[ii] = dot(s[ii],s[ii])/(2*s1[ii] + 0.0001)
    #end

    # epigraph trick!
    # =? Rather than H/p, use constrain H = p*G, and use G
    # =? Rather than h/p, use constrain h = p*g, and use g, but g = 1, since p=sum(p0), and h=sum(p0)
    #nl = gm.nl
    #nb = gm.nb
    #ng = gm.ng
    #w_idx       =                     (1:nb)
    #wr_idx      = w_idx[end]       .+ (1:nl)
    #wi_idx      = wr_idx[end]      .+ (1:nl)
    #zd_idx      = wi_idx[end]      .+ (1:nb)
    #G  = Vector{AffExpr}(undef, nvar)
    #G .= 0.0
    #@variable(model, Gvar[1:length(zd_idx)])
    #G[zd_idx] = Gvar

    # ======= WITH normalization of the load
        # => @variable(model, G[1:length(lp[:H])])
        # => @constraint(model, G*sum(pd0_var) .== lp[:H])
        # => g = 1.0
        # => 
        # => @constraint(model, [ii in 1:nrsoc],  dot(s[ii],s[ii]) <= 2*s1[ii]*s2[ii])
        # => @constraint(model, G + lp[:A]'*lambda + lp[:C]'*mu - sum(s1[ii]*soc[:m1][ii]' + s2[ii]*soc[:m2][ii]' + soc[:m3][ii]'*s[ii] for ii in 1:nrsoc) .== 0.0)
        # => obj = g + lambda'*lp[:b] + mu'*lp[:d] - sum(s1[ii]*soc[:b1][ii]  + s2[ii]*soc[:b2][ii]  + s[ii]'*soc[:b3][ii] for ii in 1:nrsoc)

    # ======= WITHOUT normalization of the load
    @constraint(model, [ii in 1:nrsoc],  dot(s[ii],s[ii]) <= 2*s1[ii]*s2[ii])
    @constraint(model, lp[:H] + lp[:A]'*lambda + lp[:C]'*mu - sum(s1[ii]*soc[:m1][ii]' + s2[ii]*soc[:m2][ii]' + soc[:m3][ii]'*s[ii] for ii in 1:nrsoc) .== 0.0)
    
    # regularize?
     # => s_vec = []
     # => s0_vec = []
     # => for ii in 1:nrsoc
     # =>     s_vec  = vcat(s_vec, s[ii])
     # =>     s0_vec = vcat(s0_vec, value.(dual_soln[:s][ii]))
     # => end
     # => lp0, _ = canonicalize(zl0, gm.pd, gm.qd, gm) 
     # => xx  = [pd0_var; qd0_var; risk; alpha; zl; logit_zl; lambda; mu; s1; s2; s_vec; G]
     # => xx0 = [gm.pd; gm.qd; 0.5*(bounds[:risk_lb] + bounds[:risk_ub])*ones(gm.nl);
     # =>       0.5*(bounds[:alpha_lb] + bounds[:alpha_ub]); zl0; logit_zl0; value.(dual_soln[:lambda]);
     # =>       value.(dual_soln[:mu]); value.(dual_soln[:s1]); value.(dual_soln[:s2]);
     # =>       s0_vec; value.(lp0[:H])]
     # => #regularization = -0.1*dot(xx-xx0,xx-xx0)
    obj = lp[:h] + lambda'*lp[:b] + mu'*lp[:d] - sum(s1[ii]*soc[:b1][ii]  + s2[ii]*soc[:b2][ii]  + s[ii]'*soc[:b3][ii] for ii in 1:nrsoc)
    model[:obj] = obj
    @objective(model, Max, obj)

    if hot_start == true
        set_start_value.(pd0_var, gm_shed.pd)
        set_start_value.(qd0_var, gm_shed.qd)
        set_start_value.(risk, bounds[:risk_ub]*ones(gm.nl))
        set_start_value(alpha, bounds[:alpha_ub])
        set_start_value.(zl, zl0)
        set_start_value.(logit_zl, logit_zl0)
        set_start_value.(lambda, value.(dual_soln[:lambda]))
        set_start_value.(mu, value.(dual_soln[:mu]))
        set_start_value.(s1, value.(dual_soln[:s1]))
        set_start_value.(s2, value.(dual_soln[:s2]))

        # loop and set the soc variables
        for ii in 1:nrsoc
            set_start_value.(s[ii], value.(dual_soln[:s][ii]))
        end

        # to get H, we need to re-run the canonicalization with constant inputs
        # => lp0, _ = canonicalize(zl0, gm.pd, gm.qd, gm; normalize_shed=false) 
        # => set_start_value.(G, value.(lp0[:H])) # no need to normalize!
    end

    println()
    println("=============================")
    t1 = time()
    optimize!(model) #solve_time(model)
    dt = time() - t1

    # did the model solve ok?
    ss(model)

    model[:solve_time] = dt
    objective_value(model)
    println(objective_value(model))

    return model
end

function compute_loadsheds(gm::SOCGridModel, pd0::Vector{Float64}, qd0::Vector{Float64}, zl0::Vector{Float64})
    # compute 4 loadsheds
    # 1. SOC load shed with RELAXED line statuses
    # 2. SOC load shed with FIXED line statuses
    # 3. Nonconvex load shed with FIXED line statuses
    gm_shed     = deepcopy(gm)
    gm_shed.pd .= copy(pd0)
    gm_shed.qd .= copy(qd0)
    zl          = copy(zl0)
    zl[ zl .<= 1e-6] .= 0.0 # for numerical stability

    # prepare empty models
    m_soc_zrelax   = Model() 
    m_soc_zsnap    = Model() 
    m_acopf_zsnap  = Model() 

    @info "Running Gurobi on the SOC (z relaxed)"
    m_soc_zrelax     = min_loadshed_soc_primal_explicit_flowcuts(gm_shed, zl)
    solution_valid_1 = ss(m_soc_zrelax)   # test for optimization failure
    if solution_valid_1 == true

        zl[ zl .< 0.5]  .= 0.0
        zl[ zl .>= 0.5] .= 1.0
        @info "Running Gurobi on the SOC (z snapped)"
        m_soc_zsnap      = min_loadshed_soc_primal_explicit_flowcuts(gm_shed, zl)
        solution_valid_2 = ss(m_soc_zsnap)    # test for optimization failure
        if solution_valid_2 == true

            @info "Running Ipopt on the nonconvex model (z snapped)"
            m_acopf_zsnap    = ac_redispatch_snapped(gm_shed, zl; soft_flow=false, host_start=true, host_start_model=m_soc_zsnap)
            solution_valid_3 = ss(m_acopf_zsnap)  # test for optimization failure
            if solution_valid_3 == false
                # re-run with soft flow constraints
                @info "Running Ipopt on the nonconvex model (z snapped)"
                m_acopf_zsnap    = ac_redispatch_snapped(gm_shed, zl; soft_flow=true, host_start=true, host_start_model=m_soc_zsnap)
                solution_valid_3 = ss(m_acopf_zsnap)  # test for optimization failure
                if solution_valid_3 == true
                    solutions_valid = true
                else
                    solutions_valid = false
                end
            else
                solutions_valid = true
            end
        else
            solutions_valid = false
        end
    else
        solutions_valid = false
    end

    if solutions_valid
        o1 = objective_value(m_soc_zrelax)
        o2 = objective_value(m_soc_zsnap)
        o3 = objective_value(m_acopf_zsnap) - value(m_acopf_zsnap[:obj_penalty])

        println()

        println("After solving the load shedding verification problem:")
        
        println("\u2705 Worst-case load shed (AC-SOC and relaxed lines): ", o1, " pu")
        println("\u2705 Worst-case load shed (AC-SOC and snapped lines): ", o2, " pu")
        println("\u2705 Worst-case load shed (AC-OPF and snapped lines): ", o3, " pu")
    else
        @warn "One of the restorations failed."
        @info objective_value(m_soc_zrelax)
        @info objective_value(m_soc_zsnap)
        @info objective_value(m_acopf_zsnap)
    end

    return solutions_valid, m_soc_zrelax, m_soc_zsnap, m_acopf_zsnap
end

function line_status(gm::SOCGridModel, 
                     bounds::Dict{Symbol, Float64}, 
                     nn_model::String;
                     x_input=Nothing,
                     high_load=true)
    # model to call the predictor
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # define predictor inputs as scalars
    if x_input == Nothing
        if high_load == true
            pd0   = bounds[:load_scale_ub]*copy(gm.pd)
            qd0   = bounds[:load_scale_ub]*copy(gm.qd)
            risk  = bounds[:risk_ub]*ones(gm.nl)
            alpha = bounds[:alpha_ub]
            x     = [risk; qd0; pd0; alpha]
        else
            pd0   = copy(gm.pd)
            qd0   = copy(gm.qd)
            risk  = 0.5*(bounds[:risk_lb] + bounds[:risk_ub])*ones(gm.nl)
            alpha = 0.5*(bounds[:alpha_lb] + bounds[:alpha_ub])
            x     = [risk; qd0; pd0; alpha]
        end
    else
        # provided an input!
        x = copy(x_input)
    end

    # now, we need to normalize
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

    return sig.(value.(logit_zl)), value.(logit_zl)
end

function maxmin_via_sampling(gm::SOCGridModel, 
                             bounds::Dict{Symbol, Float64}, 
                             nn_model::String,
                             n_fails::Int64)
    resample       = true
    m_soc_zrelax   = Model()  
    m_soc_zsnap    = Model() 
    m_acopf_zsnap  = Model() 

    while resample == true
        # step 1: sample from the bounds
        pd0 = copy(gm.pd)
        qd0 = copy(gm.qd)

        p_lb, p_ub = scale_load(pd0, bounds[:load_scale_lb], bounds[:load_scale_ub])
        q_lb, q_ub = scale_load(qd0, bounds[:load_scale_lb], bounds[:load_scale_ub])

        bounds = Dict(:load_scale_lb => 0.75,  :load_scale_ub  => 1.25,
                    :risk_lb       => 0.4,   :risk_ub          => 0.5,
                    :alpha_lb      => 0.4,   :alpha_ub         => 0.5)

        p_rand     = rand(gm.nb)
        q_rand     = rand(gm.nb)
        risk_rand  = rand(gm.nl)
        alpha_rand = rand()

        # instantiate
        pd_sample    = p_rand.*p_lb                  .+ (1.0 .- p_rand).*p_ub
        qd_sample    = q_rand.*q_lb                  .+ (1.0 .- q_rand).*q_ub
        risk_sample  = risk_rand.*bounds[:risk_lb]   .+ (1.0 .- risk_rand).*bounds[:risk_ub]
        alpha_sample = alpha_rand.*bounds[:alpha_lb] .+ (1.0 .- alpha_rand).*bounds[:alpha_ub]
        x_input      = [risk_sample; qd_sample; pd_sample; alpha_sample]

        # step 2: compute the line statuses
        zl0, logit_zl0 = line_status(gm, bounds, nn_model; x_input=x_input)

        # step 3: minimize load shed with Gurobi
        solutions_valid, m_soc_zrelax, m_soc_zsnap, m_acopf_zsnap = compute_loadsheds(gm, pd_sample, qd_sample, zl0)
        
        if solutions_valid == true
            resample = false
        else
            n_fails += 1
            resample = true
        end
    end

    o1 = objective_value(m_soc_zrelax)
    o2 = objective_value(m_soc_zsnap)
    o3 = objective_value(m_acopf_zsnap) - value(m_acopf_zsnap[:obj_penalty])

    return o1, o2, o3, n_fails
end

function loop_and_sample(gm::SOCGridModel, 
                         bounds::Dict{Symbol, Float64}, 
                         nn_model::String,
                         data_file::String;
                         n_samples::Int64=100)

    # initialize
    obj_soc_zrelax   = zeros(n_samples)
    obj_soc_zsnap    = zeros(n_samples)
    obj_acopf_zsnap  = zeros(n_samples)
    n_fails          = 0

    for ii in 1:n_samples
        obj_soc_zrelax[ii], obj_soc_zsnap[ii], obj_acopf_zsnap[ii], n_fails = maxmin_via_sampling(gm, bounds, nn_model, n_fails) 
        println(n_fails)     
    end

    # save to memory
    fid = h5open(data_file, "w")
    write(fid, "obj_soc_zrelax",   obj_soc_zrelax)
    write(fid, "obj_soc_zsnap",    obj_soc_zsnap)
    write(fid, "obj_acopf_zsnap",  obj_acopf_zsnap)

    close(fid)

    return obj_soc_zrelax, obj_soc_zsnap, obj_acopf_zsnap, n_fails
end    

# test solution status!
function ss(model::Model)
    # to get all potential statuses, call: typeof(termination_status(model))
    # 
    #   OPTIMIZE_NOT_CALLED = 0
    #   OPTIMAL = 1
    #   INFEASIBLE = 2
    #   DUAL_INFEASIBLE = 3
    #   LOCALLY_SOLVED = 4
    #   LOCALLY_INFEASIBLE = 5
    #   INFEASIBLE_OR_UNBOUNDED = 6
    #   ALMOST_OPTIMAL = 7
    #   ALMOST_INFEASIBLE = 8
    #   ALMOST_DUAL_INFEASIBLE = 9
    #   ALMOST_LOCALLY_SOLVED = 10
    #   ITERATION_LIMIT = 11
    #   TIME_LIMIT = 12
    #   NODE_LIMIT = 13
    #   SOLUTION_LIMIT = 14
    #   MEMORY_LIMIT = 15
    #   OBJECTIVE_LIMIT = 16
    #   NORM_LIMIT = 17
    #   OTHER_LIMIT = 18
    #   SLOW_PROGRESS = 19
    #   NUMERICAL_ERROR = 20
    #   INVALID_MODEL = 21
    #   INVALID_OPTION = 22
    #   INTERRUPTED = 23
    #   OTHER_ERROR = 24
    # => println(termination_status(model))
    # => println(Int(termination_status(model)))
    soln_status = Int(termination_status(model))
    if soln_status in [1, 4, 7, 10] # optimal, locally solved, or almost optimal
        soln_valid = true
    else
        println(termination_status(model))
        println(Int(termination_status(model)))
        # => @assert "Optimization failed."
        soln_valid = false
    end
    return soln_valid
end

function process_maxmin_loadshed(gm::SOCGridModel,
                                 bounds::Dict{Symbol, Float64},
                                 nn_model::String,
                                 data_file::String,
                                 maxmin_model::Model)

    # grab the loads and find the line statuses
    x_input = value.(maxmin_model[:x])
    zl0, logit_zl0 = line_status(gm, bounds, nn_model; x_input=x_input)
    pd = value.(maxmin_model[:pd0_var])
    qd = value.(maxmin_model[:qd0_var])

    # run the sequence of load sheddings
    solutions_valid, m_soc_zrelax, m_soc_zsnap, m_acopf_zsnap = compute_loadsheds(gm, pd, qd, zl0)

    # write to file
    fid = h5open(data_file, "w")
    # save the optimal network loading
    write(fid, "pd_mathoptai",        pd)
    write(fid, "qd_mathoptai",        qd)
    write(fid, "zl0",                zl0)

    # save the sovle times
    write(fid, "time_mathoptai",        solve_time(maxmin_model))
    write(fid, "time_soc_zrelax",       solve_time(m_soc_zrelax))
    write(fid, "time_soc_zsnap",        solve_time(m_soc_zsnap))
    write(fid, "time_acopf_zsnap",      solve_time(m_acopf_zsnap))

    write(fid, "obj_mathoptai",        objective_value(maxmin_model))
    write(fid, "obj_soc_zrelax",       objective_value(m_soc_zrelax))
    write(fid, "obj_soc_zsnap",        objective_value(m_soc_zsnap))
    write(fid, "obj_acopf_zsnap",      objective_value(m_acopf_zsnap) - value(m_acopf_zsnap[:obj_penalty]))

    close(fid)

    return m_soc_zrelax, m_soc_zsnap, m_acopf_zsnap
end

function print_NN_params(nl_list, bus_list, node_list)
    ii = 1
    for bus in bus_list
        for node in node_list
            nl = nl_list[ii]
            num_out = nl
            num_in  = nl + bus*2 + 1
            nl = [32; 128; 512; 2048]
            h = nl[4]
            np = ((num_in*node + node) + (node^2 + node) + (node^2 + node) + (node*num_out + num_out))/1000
            println("bus: ", bus, ", node: ", node, ", params: ", np)
        end
        ii += 1
    end
end

function verify_vs_sample_loadshed(bus_list, node_list, pg_list, n_samples)
    ii = 1
    for bus in bus_list
        # parse the network data and build a grid model
        network_data = pglib(pg_list[ii])
        gm = parse_PM_to_SOCGridModel(network_data; perturb=false)

        # warmup: test the model reformulations
        zl = ones(gm.nl)
        m1 = min_loadshed_soc_primal_explicit_flowcuts(gm, zl);
        m2 = min_loadshed_soc_primal(gm, zl; flowcuts=true);
        m3 = min_loadshed_soc_dual(gm, zl; solver=:Gurobi, flowcuts=true);

        # loop over NN architectures
        for node in node_list
            nn_model = "outputs/"*string(bus)*"_bus/"*string(bus)*"_bus_"*string(node)*"node.pt"
            maxmin_model = maxmin_loadshed(gm, bounds, nn_model; tol = 5e-3, hot_start=true, tmax = 3600.0, flowcuts=true)

            data_file = "data/"*string(gm.nb)*"bus_"*string(node)*"node_MathOptAI_test.h5"
            _,_,_ = process_maxmin_loadshed(gm, bounds, nn_model, data_file, maxmin_model)

            # now, use sampling-based approach to find load sheds
            data_file = "data/"*string(gm.nb)*"bus_"*string(node)*"node_sampling_test.h5"
            _,_,_ = loop_and_sample(gm, bounds, nn_model, data_file; n_samples = n_samples)
        end
        ii += 1
    end
end