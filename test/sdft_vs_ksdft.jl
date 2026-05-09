using SDFT
include("testcase.jl")

Ecut = 20
temperature = 0.5
supercell_size = [2,2]
basis = graphene_setup(supercell_size; Ecut, temperature)

ρ = guess_density(basis);
εF = 0.0

n_bands = div(length(basis.kpoints[1].mapping),5)
nbands = determine_n_bands_ks(basis, εF; eigensolver=lobpcg_hyper, ρ, n_bands)

@time ρks = compute_density_eigs(basis, εF; ρ, n_bands=nbands);

nsl = fill(100, 2) # 2 levels
@time ρpd = compute_stoc_density(basis, εF, PDegreeML(nsl); ρ);
@show norm(ρpd - ρks)

nsl = fill(200, 2) # 2 levels
@time ρec = compute_stoc_density(basis, εF, ECutoffML(basis, nsl); ρ);
@show norm(ρec - ρks)
