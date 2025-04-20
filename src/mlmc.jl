# Optimal MLMC
abstract type OptimalMLMC{N} end

function optimal_mlmc(basis, MLMC::OptimalMLMC; Ecut_init=50, kws...)
    basis_init = PlaneWaveBasis(basis, Ecut_init)
    scfres_init = self_consistent_field(basis_init)

    optimal_mlmc(basis, scfres_init.εF, MLMC; scfres_ref=scfres_init, kws...)
end

struct OptimalPD{N} <: OptimalMLMC{N}
    ML::Integer
    nsl::NTuple{N,Integer}
    d::Distribution
end
OptimalPD(ML::T, nsl::Vector{T}, d) where {T<:Integer} = OptimalPD(ML, tuple(nsl...), d)
OptimalPD(ML, nsl; d=DEFAULT_DISTR) = OptimalPD(ML, nsl, d)

function optimal_mlmc(basis, εF, PD::OptimalPD;
                      pmax=10, ph=0.1, Q0=0, 
                      ρ=guess_density(basis), 
                      tot_tol=1e-1, kws...)           
    Ml = Int.(optimal_hierarchy(pmax, ph, Q0, PD.ML, basis, PD; kws...))

    vars, ψ = estimate_var(basis, εF, PDegreeML(Ml, PD.nsl, PD.d); ρ, kws...)
    opt_nsl = optimal_ns(vars, Ml, tot_tol)
    
    PDegreeML(Ml, opt_nsl, PD.d), vars, ψ
end

# Var[ϕ̂_χ^ℓ - ϕ] ≤ c1exp(-2*c2*M_ℓ)
function mlmc_cost(pdl::Function, basis::PlaneWaveBasis, 
                   c1, c2, PD::OptimalPD{N}) where {N}    
    Vl(l) = c1*exp(-2 * c2 * pdl(l))
    Cl(l) = pdl(l)

    ne = basis.model.n_electrons
    cost = sqrt((ne^2 - Vl(0)) * Cl(0))
    for l = 1:N-1
        cost += sqrt((Vl(l - 1) - Vl(l)) * Cl(l))
    end

    return cost
end

function eval_conv_const(basis::PlaneWaveBasis, ::OptimalPD; kws...)
    ne = basis.model.n_electrons
    dof = length(basis.kpoints[1].mapping)
    c1 = 4 * dof * ne
    beta = inv(basis.model.temperature)
    c2 = log(pi / beta + sqrt((pi / beta)^2 + 1))

    return c1, c2
end

function algebraic_hierarchy(ps, Q0, QL, ::OptimalPD{N}) where {N}
    f(l, p) = ceil((QL - Q0) * ((l + 1) / N)^p + Q0)
    Qlfun = [l -> f(l, p) for p in ps]
end

struct OptimalEC{N} <: OptimalMLMC{N}
    EcL::Integer
    nsl::NTuple{N,Integer}
    d::Distribution
end
OptimalEC(EcL::T, nsl::Vector{T}, d) where {T<:Integer} = OptimalEC(EcL, tuple(nsl...), d)
OptimalEC(EcL, nsl; d=DEFAULT_DISTR) = OptimalEC(EcL, nsl, d)

function optimal_mlmc(basis, εF, EC::OptimalEC{N};
                      pmax=10, ph=0.1, Q0=10, 
                      tot_tol=1e-1, slope=0.1, 
                      scfres_ref=nothing,
                      ρ=guess_density(basis), 
                      kws...) where {N}
    model = basis.model
    dim = model.n_dim

    Ecl = optimal_hierarchy(pmax, ph, Q0, EC.EcL, basis, EC; 
                            slope, scfres_ref, kws...)
    vars, ψ = estimate_var(basis, εF, ECutoffML(basis, Ecl, EC.nsl, EC.d); ρ, kws...)
    fc(l) = isone(l) ? Ecl[l]^(dim/2) : (Ecl[l]^(dim/2) + Ecl[l-1]^(dim/2))
    opt_nsl = optimal_ns(vars, fc.(1:N), tot_tol)

    ECutoffML(basis, Ecl, opt_nsl, EC.d), vars, ψ
end

# Var[ϕ̂_χ^ℓ - ϕ] ≤ c1exp(-2*c2*√E_{c,ℓ})
function mlmc_cost(ecl::Function, basis::PlaneWaveBasis, 
                   c1, c2, ::OptimalEC{N}) where {N}
    dim = basis.model.n_dim
    Vl(l) = c1 * exp(-2 * c2 * sqrt(ecl(l)))
    Cl(l) = ecl(l)^(dim / 2)

    ne = basis.model.n_electrons
    cost = sqrt((ne^2 - Vl(0)) * Cl(0))
    for l = 1:N-1
        cost += sqrt((Vl(l - 1) - Vl(l)) * (Cl(l - 1) + Cl(l)))
    end

    return cost
end

function eval_conv_const(basis::PlaneWaveBasis, ::OptimalEC;
                         scfres_ref=nothing, Ecut_ref=50, 
                         Ecuts=20:2:30, kws...)
    if isnothing(scfres_ref)
        basis_ref = PlaneWaveBasis(basis, Ecut_ref)
        scfres_ref = self_consistent_field(basis_ref)
    end
    smearf = FermiDirac(scfres_ref.εF, inv(basis.model.temperature))
    f2(x) = sqrt(evalf(x, smearf))

    c1, c2 = eval_ec_const(Ecuts, scfres_ref.basis, scfres_ref.eigenvalues[1],
                           scfres_ref.ψ[1], f2, scfres_ref.ρ)

    return 4 * c1 * basis.model.n_electrons, c2
end

function eval_ec_const(Ecuts, basis, eigref, ψref, smearf, ρ)
    T = eltype(ψref)
    dof = size(ψref, 1)
    fHref = zeros(T, dof, dof)
    n_bands = length(eigref)
    @views for i in 1:n_bands
        outersum!(fHref, ψref[:, i], smearf(eigref[i]))
    end

    err = fill(0.0, length(Ecuts))
    FHl = zero(fHref)
    for (l, ecl) in enumerate(Ecuts)
        basisl = PlaneWaveBasis(basis, ecl)
        idcsk_in, idcsk_out = DFTK.transfer_mapping(basis, basis.kpoints[1], 
                                                    basisl, basisl.kpoints[1])
        ψl = [ψref[idcsk_in, :]]
        ρl = transfer_density(ρ, basis, basisl)
        haml = Hamiltonian(basisl; ρ=ρl)

        eigresl = diagonalize_all_kblocks(lobpcg_hyper, haml, n_bands; ψguess=ψl)
        dofl = size(eigresl.X[1], 1)
        fHl = zeros(T, dofl, dofl)
        @views for i in 1:n_bands
            outersum!(fHl, eigresl.X[1][:, i], smearf(eigresl.λ[1][i]))
        end
        fill!(FHl, 0)
        FHl[idcsk_in, idcsk_in] = fHl
        err[l] = norm(FHl - fHref)
    end

    X = [ones(length(Ecuts)) sqrt.(Ecuts)]
    a, b = X \ log.(err)

    return exp(a), -b
end

function outersum!(result::AbstractMatrix, x::AbstractVector, a::Number)
    is, js = axes(result)
    if (is != js) || (is != axes(x, 1))
        error("mismatched array sizes")
    end
    for j in js
        for i in is
            @inbounds ci, cj = x[i], x[j]
            @inbounds result[i, j] = muladd(ci * conj(cj), a, result[i, j])
        end
    end
    result
end

function algebraic_hierarchy(ps, Q0, QL, EC::OptimalEC{N}) where {N}
    f(l, p) = (QL - Q0) * ((l + 1) / N)^p + Q0
    Qlfun = [l -> f(l, p) for p in ps]
end

function optimal_hierarchy(pmax, ph, Q0, QL, 
                           basis::PlaneWaveBasis{T},
                           MLMC::OptimalMLMC{N}; 
                           slope=nothing, 
                           kws...) where {T,N}
    @assert pmax > ph
    ps = ph:ph:pmax
    Ql = algebraic_hierarchy(ps, Q0, QL, MLMC)

    c1, c2 = eval_conv_const(basis, MLMC; kws...)
    cs = zeros(T,length(Ql))
    for (j, Qlj) in enumerate(Ql)
        try
            cs[j] = mlmc_cost(Qlj, basis, c1, c2, MLMC)
        catch e
            if isa(e, DomainError)
                cs[j] = Inf
            else
                throw(e)
            end
        end
    end

    if isnothing(slope)
        opt = findmin(cs)[2]
    else
        @assert slope >= 0
        vn = 10^estimate_digits(maximum(cs))
        vs = cs[findfirst(x->x<slope*vn, abs.(cs[2:end].-cs[1:end-1])./ph)]
        opt = findfirst(x -> x < vs, cs)
    end

    Ql[opt].(0:N-1)
end

function estimate_digits(x::Real)
    @assert x > 0
    dm = floor(log10(x))
    dr = round(x / 10^(dm + 1))
    return Int(dm + dr)
end

optimal_ns(vars, costs, tot_tol) = ceil.(Int, sqrt.(vars ./ costs) .* inv(tot_tol)^2 * sum(sqrt.(vars .* costs)))
