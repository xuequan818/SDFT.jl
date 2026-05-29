using SDFT
using DFTK

# 1D case
a = 20
lattice = a .* [[1 0 0.0]; [0 0 0]; [0 0 0]]

positions = map(x -> [x, 0.0, 0.0], [0.0, 0.5])
gauss = ElementGaussian(1.0, 0.2)
atoms = [gauss,gauss]
n_electrons = 2
terms = [Kinetic(), AtomicLocal()]
temperature = 1.0

model = Model(lattice, atoms, positions; n_electrons, terms, temperature, spin_polarization=:spinless)
basis = PlaneWaveBasis(model; Ecut=1000, kgrid=(6, 1, 1));

εF = 0.0
eigensolver = lapack_partial
nbands = determine_n_bands_ks(eigensolver, basis, εF)
@time ρks = compute_density_eigs(basis, εF, eigensolver, nbands);

@time ρct = compute_stoc_density(basis, εF, CT(); M=10000, tol_cheb=1e-6, lb_fac=0.05, ub_fac=0.05);

@show norm(ρct-ρks)
