using SDFT
include("testcase.jl")

function run_ks_time(Ecut, temperature, repeat)
    basis = carbon_setup(repeat; Ecut, temperature)
    ρ, εF = coarse_initial(basis)

    nbands = if basis.model.temperature ≥ 0.01
        basis_coarse = PlaneWaveBasis(basis, 5)
        n_bands_coarse = determine_n_bands_ks(basis_coarse, εF; ρ=guess_density(basis_coarse))
        determine_n_bands_ks(basis, εF; eigensolver=lobpcg_hyper, ρ, n_bands=2n_bands_coarse)
    else
        AdaptiveBands(basis.model).n_bands_compute
    end

    t0 = time()
    compute_density_eigs(basis, εF; ρ, n_bands=nbands)
    elapsed = round(time() - t0; digits=1)

    return elapsed
end

function run_mlmcpd_time(L, Ecut, temperature, repeat; Ns=500, kws...)
    basis = carbon_setup(repeat; Ecut, temperature)
    ρ, εF = coarse_initial(basis)
    nsl = ceil.(Int, Ns ./ [2^i for i = 0:L])

    t0 = time()
    compute_stoc_density(basis, εF, PDegreeML(nsl); ρ, kws...)
    elapsed = round(time() - t0; digits=1)

    return elapsed
end

function run_mlmcec_time(L, Ecut, temperature, repeat; Ns=500, kws...)
    basis = carbon_setup(repeat; Ecut, temperature)
    ρ, εF = coarse_initial(basis)
    nsl = ceil.(Int, Ns ./ [2^i for i = 0:L])

    t0 = time()
    compute_stoc_density(basis, εF, ECutoffML(basis, nsl); ρ, kws...)
    elapsed = round(time() - t0; digits=1)

    return elapsed
end

function coarse_initial(basis)
    basis_coarse = PlaneWaveBasis(basis, 8)
    scfres_coarse = self_consistent_field(basis_coarse; callback=(_) -> nothing)

    ρ = DFTK.transfer_density(scfres_coarse.ρ, basis_coarse, basis)
    εF = scfres_coarse.εF

    return (; ρint=ρ, εFint=εF)
end
