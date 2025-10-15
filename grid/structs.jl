using LinearAlgebra, JuMP
using SparseArrays

struct SOCGridModel
    nb::Int64
    nl::Int64
    ng::Int64
    vm::Vector{Float64}
    va::Vector{Float64}
    pg::Vector{Float64}
    qg::Vector{Float64}
    g::Vector{Float64}
    g_fr::Vector{Float64}
    g_to::Vector{Float64}
    b::Vector{Float64}
    b_fr::Vector{Float64}
    b_to::Vector{Float64}
    tm::Vector{Float64}
    ta::Vector{Float64}
    xtr::Vector{Float64}
    xti::Vector{Float64}
    pd::Vector{Float64}
    qd::Vector{Float64}
    gs::Vector{Float64}
    bs::Vector{Float64}
    vmax::Vector{Float64}
    vmin::Vector{Float64}
    wr_min::Vector{Float64} 
    wr_max::Vector{Float64} 
    wi_min::Vector{Float64} 
    wi_max::Vector{Float64} 
    pg_max::Vector{Float64}
    pg_min::Vector{Float64}
    qg_max::Vector{Float64}
    qg_min::Vector{Float64}
    smax::Vector{Float64}
    clin::Vector{Float64}
    fr_buses::Vector{Int64}
    to_buses::Vector{Int64}
    Eg::SparseMatrixCSC{Float64, Int64}
    Efr::SparseMatrixCSC{Float64, Int64}
    Eto::SparseMatrixCSC{Float64, Int64}
    Tpfr::SparseMatrixCSC{Float64, Int64} 
    TpRfr::SparseMatrixCSC{Float64, Int64}
    TpIfr::SparseMatrixCSC{Float64, Int64}
    Tqfr::SparseMatrixCSC{Float64, Int64} 
    TqRfr::SparseMatrixCSC{Float64, Int64}
    TqIfr::SparseMatrixCSC{Float64, Int64}
    Tpto::SparseMatrixCSC{Float64, Int64} 
    TpRto::SparseMatrixCSC{Float64, Int64}
    TpIto::SparseMatrixCSC{Float64, Int64}
    Tqto::SparseMatrixCSC{Float64, Int64} 
    TqRto::SparseMatrixCSC{Float64, Int64}
    TqIto::SparseMatrixCSC{Float64, Int64}
    Tp_wfr::SparseMatrixCSC{Float64, Int64}
    Tq_wfr::SparseMatrixCSC{Float64, Int64}
    Tp_wto::SparseMatrixCSC{Float64, Int64}
    Tq_wto::SparseMatrixCSC{Float64, Int64}
    Bs_neg::SparseMatrixCSC{Float64, Int64}
    Bs_pos::SparseMatrixCSC{Float64, Int64}
    Gs::SparseMatrixCSC{Float64, Int64}   
end
