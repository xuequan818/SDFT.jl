using SDFT
include("testcase.jl")

function run_ks_time(Ecut, temperature, kgrid, repeat;
                     Ecut_init=8, Ecut_fermi=Ecut,
                     refine_fermi=true)
    basis = carbon_setup(repeat; Ecut, temperature, kgrid)
    ρ, εF = coarse_initial(basis; Ecut_init)

    if refine_fermi
        εF = estimate_fermi(basis, ρ, εF; Ecut_fermi)
    end

    nbands = if 0.01 ≤ basis.model.temperature < 0.5
        basis_coarse = PlaneWaveBasis(basis, Ecut_init)
        ρ_coarse = DFTK.transfer_density(ρ, basis, basis_coarse)
        n_bands_coarse = determine_n_bands_ks(basis_coarse, εF; ρ=ρ_coarse)

        determine_n_bands_ks(basis, εF; eigensolver=lobpcg_hyper, ρ, n_bands=2n_bands_coarse)
    elseif basis.model.temperature ≥ 0.5
        determine_n_bands_ks(basis, εF; eigensolver=diag_full, ρ)
    else 
        AdaptiveBands(basis.model).n_bands_compute
    end

    println("  nbands:    $(nbands)\n")
    flush(stdout)

    t0 = time()
    compute_density_eigs(basis, εF; ρ, n_bands=nbands)
    elapsed = round(time() - t0; digits=1)

    return elapsed
end

function run_mlmcpd_time(L, Ecut, temperature, repeat; 
                         Ns=500, kgrid=[1, 1, 1], 
                         Ecut_init=8, Ecut_fermi=Ecut,
                         refine_fermi=true, kws...)
    basis = carbon_setup(repeat; Ecut, temperature, kgrid)
    ρ, εF = coarse_initial(basis; Ecut_init)
    if refine_fermi
        εF = estimate_fermi(basis, ρ, εF; Ecut_fermi)
    end

    nsl = ceil.(Int, Ns ./ [2^i for i = 0:L])

    t0 = time()
    compute_stoc_density(basis, εF, PDegreeML(nsl); ρ, kws...)
    elapsed = round(time() - t0; digits=1)

    return elapsed
end

function run_mlmcec_time(L, Ecut, temperature, repeat; 
                         Ns=500, kgrid=[1, 1, 1], 
                         Ecut_init=8, Ecut_fermi=Ecut,
                         refine_fermi=true, kws...)
    basis = carbon_setup(repeat; Ecut, temperature, kgrid)
    ρ, εF = coarse_initial(basis; Ecut_init)
    if refine_fermi
        εF = estimate_fermi(basis, ρ, εF; Ecut_fermi)
    end

    nsl = ceil.(Int, Ns ./ [2^i for i = 0:L]) .+ 10

    t0 = time()
    compute_stoc_density(basis, εF, ECutoffML(basis, nsl); ρ, kws...)
    elapsed = round(time() - t0; digits=1)

    return elapsed
end

function coarse_initial(basis; Ecut_init=8)
    try
        basis_coarse = PlaneWaveBasis(basis, Ecut_init)

        scfres_coarse = self_consistent_field(
            basis_coarse;
            callback = (_) -> nothing,
        )

        ρ = DFTK.transfer_density(scfres_coarse.ρ, basis_coarse, basis)
        εF = scfres_coarse.εF

        return (; ρ, εF)
    catch e
        @warn "coarse_initial failed. Use guess_density and εF = 0.0 instead." exception=e

        ρ = guess_density(basis)
        εF = 0.0

        return (; ρ, εF)
    end
end

function estimate_fermi(basis, ρ, εF0; Ecut_fermi=10, extra_bands=30)
    basis_f = PlaneWaveBasis(basis, Ecut_fermi)
    ρf = DFTK.transfer_density(ρ, basis, basis_f)

    if basis.model.temperature ≤ 0.2
        try
            nbands_f = AdaptiveBands(basis_f.model).n_bands_compute + extra_bands
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
