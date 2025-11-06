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

function optimal_mlmc(basis, FD::Union{Real,ChebInfo}, 
                      PD::OptimalPD; tot_tol=1e-1, kws...)
    Ml, vars, ψ, hambl = _optimal_mlmc(basis, FD, PD; kws...)
    opt_nsl = optimal_ns(vars[1], Ml, tot_tol)
    
    PDegreeML(Ml, opt_nsl, PD.d), vars, ψ, hambl
end

function _optimal_mlmc(basis, FD::Union{Real,ChebInfo}, 
                       PD::OptimalPD;
                       pmax=10, ph=0.1, 
                       Q0=500, Qc=0, 
                       ρ=guess_density(basis), kws...)           
    Ml = Int.(optimal_hierarchy(pmax, ph, Q0, PD.ML, Qc, basis, PD; ρ, kws...))
    vars, ψ, hambl = estimate_var(basis, FD, PDegreeML(Ml, PD.nsl, PD.d); ρ, kws...)
    
    (; Ml, vars, ψ, hambl)
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

function eval_conv_const(basis::PlaneWaveBasis, ::OptimalPD; kws...)
    ne = basis.model.n_electrons
    dof = length(basis.kpoints[1].mapping)
    c1 = 4 * dof * ne
    beta = inv(basis.model.temperature)
    c2 = log(pi / beta + sqrt((pi / beta)^2 + 1))

    return c1, c2
end

#=
function eval_conv_const(basis::PlaneWaveBasis, ::OptimalPD; 
                         Mref=Int(2e4), Ms=1000 .* collect(1:10),
                         εF=0.0, kws...)
    xs = collect(-1:0.01:1)[2:end-1]
    smearf = FermiDirac(εF, inv(basis.model.temperature))
    ham = Hamiltonian(basis; kws...);
    Cheb = chebyshev_info(ham.blocks[1], smearf; tol_cheb=nothing, M=Mref)
    Tm(x;m) = cos(m*acos(x))
    fM(x) = sum(1:Cheb.order+1) do i
        Cheb.coef[i] .* Tm.(x;m=i-1)
    end

    err = []
    for Ml in Ms
        Npt = round(Int,1.1Ml)
        pt = cos.(range(0, 2pi - pi / Npt, length=2Npt))
        _, coefl = SDFT.genCheb(smearf, true, Ml, pt, Cheb.E1, Cheb.E2, SDFT.KPM(); tol_cheb=nothing)
        fMl(x) = sum(1:Ml+1) do i
            coefl[i] .* Tm.(x;m=i-1)
        end
        push!(err, norm(fMl(xs) - fM(xs),Inf))
    end
    X = [ones(length(Ms)) Ms]
    a, b = X \ log.(err)
    c1, c2 = length(basis.kpoints[1].mapping)*exp(a), -b 

    return 4 * c1 * basis.model.n_electrons, c2
end
=#

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

function optimal_mlmc(basis, FD::Union{Real,ChebInfo},
                      EC::OptimalEC{N}; tot_tol=1e-1, 
                      kws...) where {N}
    Ecl, vars, ψ, hambl = _optimal_mlmc(basis, FD, EC; kws...)
    dim = basis.model.n_dim
    fc(l) = isone(l) ? Ecl[l]^(dim/2) : (Ecl[l]^(dim/2) + Ecl[l-1]^(dim/2))
    opt_nsl = optimal_ns(vars[1], fc.(1:N), tot_tol)

    ECutoffML(basis, Ecl, opt_nsl, EC.d), vars, ψ, hambl
end

function _optimal_mlmc(basis, FD::Union{Real,ChebInfo},
                       EC::OptimalEC{N};
                       pmax=10, ph=0.1, 
                       Q0=8, Qc=0.1,
                       ρ=guess_density(basis), 
                       kws...) where {N}
    Ecl = optimal_hierarchy(pmax, ph, Q0, EC.EcL, Qc, basis, EC; ρ, kws...)
    vars, ψ, hambl = estimate_var(basis, FD, ECutoffML(basis, Ecl, EC.nsl, EC.d); ρ, kws...)
    
    Ecl, vars, ψ, hambl
end

# Var[ϕ̂_χ^ℓ - ϕ] ≤ c1exp(-2*c2*√E_{c,ℓ})
function mlmc_cost(ecl::Function, basis::PlaneWaveBasis, 
                   c1, c2, ::OptimalEC{N}) where {N}
    dim = basis.model.n_dim
    Vl(l) = c1 * exp(-2 * c2 * sqrt(ecl(l)))
    Cl(l) = ecl(l)^(dim / 2)
      
    cost = basis.model.n_electrons * sqrt(Cl(0))
    for l = 1:N-1
        cost += (sqrt(Vl(l-1)) + sqrt(Vl(l))) * sqrt(Cl(l-1) + Cl(l))
    end

    return cost
end

function eval_conv_const(basis::PlaneWaveBasis, ::OptimalEC;
                         ρ=nothing, Ecut_ref=30, 
                         Ecuts=10:2:20, kws...)
    if isnothing(ρ)
        basis_ref = PlaneWaveBasis(basis, Ecut_ref)
        scfres_ref = self_consistent_field(basis_ref)
        smearf = FermiDirac(scfres_ref.εF, inv(basis.model.temperature))
        c1, c2 = eval_ec_const(Ecuts, scfres_ref.basis, scfres_ref.eigenvalues[1],
                               scfres_ref.ψ[1], x->sqrt(evalf(x, smearf)), scfres_ref.ρ)
    else
        basis_ref = PlaneWaveBasis(basis, Ecut_ref)
        ρref = transfer_density(ρ,basis, basis_ref)
        ham = Hamiltonian(basis_ref; ρ=ρref)
        nbands = AdaptiveBands(basis.model).n_bands_compute
        eigres = diagonalize_all_kblocks(lobpcg_hyper, ham, nbands; ψguess=nothing) 
        occupation, εF = DFTK.compute_occupation(ham.basis, eigres.λ)
        smearf = FermiDirac(εF, inv(basis.model.temperature))
        c1, c2 = eval_ec_const(Ecuts, basis_ref, eigres.λ[1], eigres.X[1], x->sqrt(evalf(x, smearf)), ρref)
    end

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
        idcsk_in, idcsk_out = transfer_mapping(basis, basis.kpoints[1], 
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

function algebraic_hierarchy(ps, Q0, QL, Qc, EC::OptimalEC{N}) where {N}
    L = N - 1
    f(l, p) = (QL - Q0) * ((l + Qc) / (L + Qc))^p + Q0
    Qlfun = [l -> f(l, p) for p in ps]
end

function optimal_hierarchy(pmax, ph, Q0, QL, Qc,
                           basis::PlaneWaveBasis{T},
                           MLMC::OptimalMLMC{N}; 
                           slope=nothing, 
                           pcustom::Union{Nothing,Real}=nothing, 
                           kws...) where {T,N}
    if !isnothing(pcustom)
        Ql = algebraic_hierarchy([pcustom], Q0, QL, Qc, MLMC)
        return Ql[1].(0:N-1)
    end

    @assert pmax > ph
    ps = ph:ph:pmax
    Ql = algebraic_hierarchy(ps, Q0, QL, Qc, MLMC)
    pind = collect(1:length(ps))

    c1, c2 = eval_conv_const(basis, MLMC; kws...)
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
    if !isnothing(slope)
        @assert slope >= 0

        ind1 = findall(x -> x < 0, cs[2:end] .- cs[1:end-1])
        ind2 = findall(x -> x > 0, cs[2:end] .- cs[1:end-1]) .+ 1
        maxima = intersect(ind1, ind2)

        range = 2:opt
        if length(maxima) > 0 && !all(x -> x > opt, maxima)
            maxind = intersect(findall(x -> x < opt, maxima), findmin(abs.(maxima .- opt))[2])[1]
            range = maxima[maxind]+1:opt
        end

        vn = 10^(SDFT.estimate_digits(cs[range[1]] - cs[range[end]]))
        ind = findlast(x -> x > cs[opt] + vn * slope, cs[range])
        isnothing(ind) && (ind = 1)
        opt = range[ind]
    end

    @show ps[opt]
    Ql[pind[opt]].(0:N-1)
end

function estimate_digits(x::Real)
    @assert x > 0
    dm = floor(log10(x))
    dr = round(x / 10^(dm + 1))
    return Int(dm + dr)
end

optimal_ns(vars, costs, tot_tol) = ceil.(Int, inv(tot_tol)^2 * sqrt.(vars ./ costs) .* sum(sqrt.(vars .* costs)))
