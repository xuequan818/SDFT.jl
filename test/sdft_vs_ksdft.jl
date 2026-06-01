using SDFT
using SDFT: estimate_fermi
include("testcase.jl")

function run_ks_time(Ecut, temperature, repeat;
                     kgrid=[1,1,1],
                     Ecut_fermi=min(10,Ecut),
                     Ecut_coarse=35*prod(repeat)^(-2/3),
                     max_band_fraction=0.33, sys="silicon")
    fun = eval(Symbol(sys, "_setup"))
    basis = fun(repeat; Ecut, temperature, kgrid)
    dof = take_dof(basis)
    ρ = guess_density(basis)
    εF = estimate_fermi(basis, ρ; Ecut_fermi)

    determine_solver(nb) = (nb / dof ≤ max_band_fraction) ? lobpcg_hyper : lapack_partial

    occupation_threshold = 1e-6
    temp = basis.model.temperature
    nbands = if temp ≥ 0.01
        basis_coarse = PlaneWaveBasis(basis, Ecut_coarse)
        ρ_coarse = DFTK.transfer_density(ρ, basis, basis_coarse)
        n_bands_coarse = determine_n_bands_ks(diag_full, basis_coarse, εF; 
                                              ρ=ρ_coarse, occupation_threshold)

        @show n_bands_coarse
        n_bands = Int(ceil((temp ≤ 0.2 ? 1.1 : 1.2) * n_bands_coarse))
        eigensolver = determine_solver(n_bands)

        determine_n_bands_ks(eigensolver, basis, εF; ρ, n_bands, occupation_threshold)
    else 
        AdaptiveBands(basis.model).n_bands_compute
    end

    eigensolver = determine_solver(nbands)

    println("  nbands: $(nbands),  eigensolver: $(eigensolver)\n")
    flush(stdout)

    t0 = time()
    ρout = compute_density_eigs(basis, εF, eigensolver, nbands; ρ)
    elapsed = round(time() - t0; digits=1)

    return elapsed, ρout, basis.dvol
end

function run_mlmcpd_time(L, Ecut, temperature, repeat; 
                         Ns=100, kgrid=[1, 1, 1], 
                         Ecut_fermi=min(10,Ecut), 
                         sys="silicon", kws...)
    fun = eval(Symbol(sys, "_setup"))
    basis = fun(repeat; Ecut, temperature, kgrid)
    ρ = guess_density(basis)
    εF = estimate_fermi(basis, ρ; Ecut_fermi)

    nsl = ceil.(Int, Ns ./ [2^i for i = 0:L]) 
    nsl = [i <= 10 ? 10 : i for i in nsl]

    t0 = time()
    ρout = compute_stoc_density(basis, εF, PDegreeML(nsl); ρ, kws...)
    elapsed = round(time() - t0; digits=1)

    return elapsed, ρout
end

function run_mlmcec_time(L, Ecut, temperature, repeat; 
                         Ns=100, kgrid=[1, 1, 1], 
                         Ecut_fermi=min(10,Ecut), 
                         sys="silicon", kws...)
    fun = eval(Symbol(sys, "_setup"))
    basis = fun(repeat; Ecut, temperature, kgrid)
    ρ = guess_density(basis)
    εF = estimate_fermi(basis, ρ; Ecut_fermi)

    nsl = ceil.(Int, Ns ./ [2^i for i = 0:L])
    nsl = [i <= 10 ? 10 : i for i in nsl]

    t0 = time()
    ρout = compute_stoc_density(basis, εF, ECutoffML(basis, nsl); ρ, kws...)
    elapsed = round(time() - t0; digits=1)

    return elapsed, ρout
end
