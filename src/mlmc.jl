# Optimal MLMC
abstract type OptimalMLMC{N} end

function optimal_mlmc(basis, εF::Real, ST::OptimalMLMC; M=Int(1e5), tol_cheb=1e-6, kws...)
    smearf = FermiDirac(εF, inv(basis.model.temperature))
    if length(basis.kpoints) == 1
        ham = Hamiltonian(basis; kws...)
        Cheb = chebyshev_info(ham.blocks[1], smearf, M, cal_way; tol_cheb, kws...)
    else
        Cheb = chebyshev_info(basis, smearf, M, cal_way; tol_cheb, kws...)
    end

    optimal_mlmc(basis, Cheb, ST; kws...)
end

function optimal_mlmc(basis, Cheb::ChebInfo, ST::OptimalMLMC{N}; 
                      tot_tol=1e-1, kws...) where {N}
    if N > 1
        return build_optimal_mlmc(basis, Cheb, ST; tot_tol, kws...)
    elseif N == 1
        ST1 = MC(ST.nsl)
        hambl = all_level_ham_blocks(basis, ST1; kws...)
        vars, ψ, hambl = estimate_var(basis, Cheb, ST1; ρ=guess_density(basis), kws...)
        nsl = optimal_ns([vars[1, 1]], [1.0], tot_tol, basis)

        return MC(nsl), 0.0, vars, ψ, hambl
    end
end

struct OptimalPD{N} <: OptimalMLMC{N}
    ML::Integer
    nsl::NTuple{N,Integer}
    d::Distribution
end
OptimalPD(ML::T, nsl::Vector{T}, d) where {T<:Integer} = OptimalPD(ML, tuple(nsl...), d)
OptimalPD(ML, nsl; d=DEFAULT_DISTR) = OptimalPD(ML, nsl, d)

function build_optimal_mlmc(basis, Cheb::ChebInfo, PD::OptimalPD{N};
                            tot_tol=1e-1, Q0=nothing, kws...) where {N}
    if isnothing(Q0)
        Q0 = findlast(x->abs(x)>1e-2, Cheb.coef)[2]
    end
    Ml, p_opt, vars, ψ, hambl = _optimal_mlmc(basis, Cheb, PD; Q0, kws...)
    opt_nsl = optimal_ns(vars[1,:], Ml, tot_tol, basis)
    
    PDegreeML(Ml, opt_nsl, PD.d), p_opt, vars, ψ, hambl
end

function _optimal_mlmc(basis, Cheb::ChebInfo, 
                       PD::OptimalPD;
                       pmin=0.5, pmax=10, ph=0.1, 
                       Q0=100, Qc=0, 
                       ρ=guess_density(basis), kws...)           
    Ml, p_opt = optimal_hierarchy(pmin, pmax, ph, Q0, PD.ML, Qc, basis, Cheb, PD; ρ, kws...)
    Ml = Int.(Ml)
    vars, ψ, hambl = estimate_var(basis, Cheb, PDegreeML(Ml, PD.nsl, PD.d); ρ, kws...)
    
    (; Ml, p_opt, vars, ψ, hambl)
end

# Var[ϕ̂_χ^ℓ - ϕ] ≤ c1exp(-2*c2*M_ℓ)
function mlmc_cost(pdl::Function, basis::PlaneWaveBasis, 
                   c1, c2, PD::OptimalPD{N}) where {N}    
    Vl(l) = c1*exp(-2 * c2 * pdl(l))
    Cl(l) = pdl(l)

    cost = basis.model.n_electrons * sqrt(Cl(0))
    for l = 1:N-1
        #cost += (sqrt(Vl(l)) + sqrt(Vl(l-1))) * sqrt(Cl(l))
        cost += sqrt((Vl(l - 1) - Vl(l)) * Cl(l))
    end
   
    return cost
end

function eval_conv_const(basis::PlaneWaveBasis, Cheb::ChebInfo, ::OptimalPD; kws...)
    ne = basis.model.n_electrons
    dof = length(basis.kpoints[1].mapping)
    c1 = 4 * dof * ne
    #x0 = abs(pi/inv(basis.model.temperature))
    x0 = abs(pi/(Cheb.E2*inv(basis.model.temperature)))
    c2 = log(x0 + sqrt(x0^2 + 1))

    return c1, c2
end

function algebraic_hierarchy(ps, Q0, QL, Qc, ::OptimalPD{N}) where {N}
    L = N - 1
    f(l, p) = ceil((QL - Q0) * ((l + Qc) / (L + Qc))^p + Q0)
    Qlfun = [l -> f(l, p) for p in ps]
end

struct OptimalEC{N} <: OptimalMLMC{N}
    EcL::Real
    nsl::NTuple{N,Integer}
    d::Distribution
end
OptimalEC(EcL::Real, nsl::Vector{T}, d) where {T<:Integer} = OptimalEC(EcL, tuple(nsl...), d)
OptimalEC(EcL, nsl; d=DEFAULT_DISTR) = OptimalEC(EcL, nsl, d)

function build_optimal_mlmc(basis, Cheb::ChebInfo, EC::OptimalEC{N};
                            tot_tol=1e-1, kws...) where {N}
    Ecl, p_opt, vars, ψ, hambl = _optimal_mlmc(basis, Cheb, EC; kws...)
    dim = basis.model.n_dim
    fc(l) = isone(l) ? Ecl[l]^(dim/2) : (Ecl[l]^(dim/2) + Ecl[l-1]^(dim/2))
    opt_nsl = optimal_ns(vars[1,:], fc.(1:N), tot_tol, basis)

    ECutoffML(basis, Ecl, opt_nsl, EC.d), p_opt, vars, ψ, hambl
end

function _optimal_mlmc(basis, Cheb::ChebInfo,
                       EC::OptimalEC{N};
                       pmin=0.1, pmax=10, ph=0.1, 
                       Q0=6, Qc=0.1,
                       ρ=guess_density(basis), 
                       kws...) where {N}
    Ecl, p_opt = optimal_hierarchy(pmin, pmax, ph, Q0, EC.EcL, Qc, basis, Cheb, EC; ρ, kws...)
    vars, ψ, hambl = estimate_var(basis, Cheb, ECutoffML(basis, Ecl, EC.nsl, EC.d); ρ, kws...)
    
    (; Ecl, p_opt, vars, ψ, hambl)
end

# Var[ϕ̂_χ^ℓ - ϕ] ≤ c1exp(-2*c2*√E_{c,ℓ})
function mlmc_cost(ecl::Function, basis::PlaneWaveBasis, 
                   c1, c2, ::OptimalEC{N}) where {N}
    dim = basis.model.n_dim
    Vl(l) = c1 * exp(-2 * c2 * sqrt(ecl(l)))
    Cl(l) = ecl(l)^(dim / 2) * log(1 + ecl(l)^(dim / 2))
      
    cost = basis.model.n_electrons * sqrt(Cl(0))
    for l = 1:N-1
        cost += (sqrt(Vl(l-1)) + sqrt(Vl(l))) * sqrt(Cl(l-1) + Cl(l))
    end

    return cost
end


function estimate_fermi(basis, ρ; Ecut_fermi=10, extra_bands=100)
    basis_f = PlaneWaveBasis(basis, Ecut_fermi)
    ρf = DFTK.transfer_density(ρ, basis, basis_f)

    if basis.model.temperature ≤ 0.2
        try
            nbands_f = AdaptiveBands(basis_f.model).n_bands_compute + extra_bands
            max_nb = div(minimum(ik -> take_dof(basis_f, ik), 1:length(basis_f.kpoints)),3)
            nbands_f = min(max_nb, nbands_f)
            ham_f = Hamiltonian(basis_f; ρ=ρf)
            eigres_f = diagonalize_all_kblocks(lobpcg_hyper, ham_f, nbands_f; ψguess=nothing)

            _, εF = DFTK.compute_occupation(basis_f, eigres_f.λ)
        catch e
            εF = compute_fermi_level(basis_f; ρ=ρf, tol_cheb=1e-4, tol_n_elec=1e-4)
        end
    else
        εF = compute_fermi_level(basis_f; ρ=ρf, tol_cheb=1e-4, tol_n_elec=1e-4)
    end

    return εF
end

function eval_conv_const(basis::PlaneWaveBasis, Cheb::ChebInfo, 
                         EC::OptimalEC; c1=nothing, c2=nothing, kws...)
    if isnothing(c1) && isnothing(c2)
        c1, c2 = _eval_ec_const(basis, Cheb, EC; kws...)
    end

    return 4 * c1 * basis.model.n_electrons, c2
end

function _eval_ec_const(basis::PlaneWaveBasis, Cheb::ChebInfo, 
                        ::OptimalEC; ρ=nothing, εF=nothing, 
                        Ecut_ref=20, Ecuts=8:2:16, kws...)
    if isnothing(ρ) 
        ρ = guess_density(basis)
    end

    basis_coarse = PlaneWaveBasis(basis, 5)
    ρ_coarse = guess_density(basis_coarse)
    nbands = if basis.model.temperature ≥ 0.01
        εF_coarse = estimate_fermi(basis, ρ; Ecut_fermi=5)
        determine_n_bands_ks(diag_full, basis_coarse, εF_coarse; ρ=ρ_coarse, occupation_threshold=1e-2)
    else
        AdaptiveBands(basis.model).n_bands_compute
    end
    eigensolver = (nbands / take_dof(basis_coarse) ≤ 0.33) ? lobpcg_hyper : lapack_partial
    eigres_coarse = diagonalize(eigensolver, basis_coarse, nbands; ρ=ρ_coarse, ψguess=nothing)

    basis_ref = PlaneWaveBasis(basis, Ecut_ref)
    ψguess_ref = transfer_blochwave(eigres_coarse.X, basis_coarse, basis_ref)
    ρref = DFTK.transfer_density(ρ, basis, basis_ref)
    eigensolver = (nbands / take_dof(basis_ref) ≤ 0.33) ? lobpcg_hyper : lapack_partial
    eigres = diagonalize(eigensolver, basis_ref, nbands; ρ=ρref, ψguess=ψguess_ref)

    if isnothing(εF)
        occupation, εF = DFTK.compute_occupation(basis_ref, eigres.λ)
    else
        occupation = DFTK.compute_occupation(basis_ref, eigres.λ, εF).occupation
    end

    smearf = FermiDirac(εF, inv(basis.model.temperature))
    c1, c2 = _eval_ec_const(Ecuts, basis_ref, eigres.λ, eigres.X, x -> sqrt(evalf(x, smearf)), ρref)
    
    return c1, c2
end

function _eval_ec_const(Ecuts, basis, eigref, ψref, smearf, ρ)
    nk = length(basis.kpoints)
    T = eltype(ψref[1])
    f_ref = [smearf.(λ) for λ in eigref]
    norm_ref_sq = norm.(f_ref) .^ 2
    err = fill(0.0, length(Ecuts))
    n_bands = length(eigref[1])
    for (l, ecl) in enumerate(Ecuts)
        basisl = PlaneWaveBasis(basis, ecl)
        ρl = transfer_density(ρ, basis, basisl)
        eigensolver = (n_bands / take_dof(basisl) ≤ 0.33) ? lobpcg_hyper : lapack_partial
        eigresl = diagonalize(eigensolver, basisl, n_bands; ρ=ρl, ψguess=nothing)
        n_bandsl = length(eigresl[1])
        for ik = 1:nk
            ψl = eigresl.X[ik]
            f_l = smearf.(eigresl.λ[ik])
            norm_l_sq = norm(f_l)^2

            ψref_cut = transfer_blochwave_kpt(ψref[ik], basis, basis.kpoints[ik],
                basisl, basisl.kpoints[ik])
            S = ψref_cut' * ψl

            cross_term = 0.0
            for j in 1:n_bandsl, i in 1:n_bands
                cross_term += f_ref[ik][i] * f_l[j] * real(abs2(S[i, j]))
            end

            current_err_sq = max(0.0, norm_ref_sq[ik] + norm_l_sq - 2 * cross_term)

            err[l] += basis.kweights[ik] * sqrt(current_err_sq)
        end
    end
    X = [ones(length(Ecuts)) sqrt.(Ecuts)]
    a, b = X \ log.(err)

    return exp(a), -b
end

function algebraic_hierarchy(ps, Q0, QL, Qc, EC::OptimalEC{N}) where {N}
    L = N - 1
    f(l, p) = (QL - Q0) * ((l + Qc) / (L + Qc))^p + Q0
    Qlfun = [l -> f(l, p) for p in ps]
end

function optimal_hierarchy(pmin, pmax, ph, Q0, QL, Qc,
                           basis::PlaneWaveBasis{T},
                           Cheb::ChebInfo,
                           MLMC::OptimalMLMC{N}; 
                           opt_ratio=0.01, 
                           pcustom::Union{Nothing,Real}=nothing, 
                           kws...) where {T,N}
    if !isnothing(pcustom)
        Ql = algebraic_hierarchy([pcustom], Q0, QL, Qc, MLMC)
        return Ql[1].(0:N-1), pcustom
    end

    @assert pmax > ph
    ps = pmin:ph:pmax
    Ql = algebraic_hierarchy(ps, Q0, QL, Qc, MLMC)
    pind = collect(1:length(ps))

    c1, c2 = eval_conv_const(basis, Cheb, MLMC; kws...)
    cs = zeros(T,length(Ql))
    for (j, Qlj) in enumerate(Ql)
        try
            cs[j] = mlmc_cost(Qlj, basis, c1, c2, MLMC)
        catch e
            if isa(e, DomainError)
                cs[j] = NaN
                setdiff!(pind, j)
            else
                throw(e)
            end
        end
    end
    filter!(!isnan, cs)
    
    opt = findmin(cs)[2]
    if !isnothing(opt_ratio)
        @assert opt_ratio >= 0

        pks = findmaxima(cs) |> peakproms!() |> peakwidths()
        maxima = filterpeaks!(pks, :widths; min=0.1/ph).indices
        maxind = findlast(x -> x < opt, maxima)
        lmax = isnothing(maxind) ? 1 : maxima[maxind]

        #dist = estimate_digits(cs[lmax] - cs[opt])
        dist = cs[lmax] - cs[opt]
        ind = findlast(x -> x > cs[opt] + dist * opt_ratio, cs[lmax:opt])
        isnothing(ind) && (ind = 1)
        opt = (lmax:opt)[ind]
    end

    return Ql[pind[opt]].(0:N-1), ps[pind[opt]]
end

function estimate_digits(x::Real)
    iszero(x) && return zero(x)
    dm = floor(log10(x))
    dr = round(x / 10^(dm + 1))
    return 10^Int(dm + dr)
end

function optimal_ns(vars::AbstractVector, costs::AbstractVector, tot_tol, n_elec::Int)
    @assert length(vars) == length(costs)

    vars_pos = max.(vars, 0)
    S = sum(sqrt.(vars_pos .* costs))

    return ceil.(Int, inv(tot_tol)^2 .* sqrt.(vars_pos ./ costs) .* S ./ n_elec)
end
function optimal_ns(vars::AbstractVector, costs::AbstractVector, tot_tol, basis::PlaneWaveBasis)
    optimal_ns(vars, costs, tot_tol, basis.model.n_electrons)
end
