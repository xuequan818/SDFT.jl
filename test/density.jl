using DFTK

# 1D case
a = 20
lattice = a .* [[1 0 0.0]; [0 0 0]; [0 0 0]]

positions = map(x -> [x, 0.0, 0.0], [0.0, 0.5])
gauss = ElementGaussian(1.0, 0.2)
atoms = [gauss,gauss]
n_electrons = 2
terms = [Kinetic(), AtomicLocal()]
temperature = 1e-1

model = Model(lattice, atoms, positions; n_electrons, terms, temperature, spin_polarization=:spinless)
basis = PlaneWaveBasis(model; Ecut=1000, kgrid=(1, 1, 1));
ham = Hamiltonian(basis);

# solving eigenpaires, obtain ρ
@time eigres = diagonalize_all_kblocks(lobpcg_hyper, ham, 30; ψguess=nothing); # 100 is the number of the bands (only for test)
occupation, εF = DFTK.compute_occupation(ham.basis, eigres.λ)

"""ρout = ∑_{i}^{band}f_i|ψ(x)|^2"""
ρref = compute_density(ham.basis, eigres.X, occupation)


@time ρct = compute_stoc_density(basis, εF, CT(); M=1000, tol_cheb=1e-8,eigen_maxiter=5);
@show norm(ρct-ρref)
