# Generate next density from SDFT, Simple mixing : self consistent field.
"""
1. ρin = guess_density()
2. Update the nonlinear term by ρin.
energies, ham = energy_hamiltonian(basis, ψ, occupation;
ρ = ρin...)
3. ρout = next_density_stoctic()
4. ρin = ρout
Iteration
"""

"""
Chebyshev polynomial method.
"""
function self_consistent_field_ChebyP(basis::PlaneWaveBasis; M = 500,
    ρ=guess_density(basis),
    ψ=nothing,
    tol=1e-6,
    maxiter=100,
    solver=scf_anderson_solver(),
    eigensolver=lobpcg_hyper,
    determine_diagtol=DFTK.ScfDiagtol(),
    damping=0.8,  # Damping parameter
    mixing=SimpleMixing(), #KerkerMixing(), #LdosMixing(), #
    is_converged=DFTK.ScfConvergenceEnergy(tol),
    callback=DFTK.ScfDefaultCallback(; show_damping=false),
    compute_consistent_energies=true,
    occupation_threshold=DFTK.default_occupation_threshold(), # 1e-10
    response=ResponseOptions(),  # Dummy here, only for AD
   )
T = eltype(basis)
model = basis.model

# All these variables will get updated by fixpoint_map
occupation = nothing
eigenvalues = nothing
ρout = ρ
εF = nothing
n_iter = 0
energies = nothing
ham = nothing
info = (n_iter=0, ρin=ρ)   # Populate info with initial values
converged = false

# We do density mixing in the real representation
# TODO support other mixing types
function fixpoint_map(ρin)
        converged && return ρin  # No more iterations if convergence flagged

        n_iter += 1

        # Note that ρin is not the density of ψ, and the eigenvalues
        # are not the self-consistent ones, which makes this energy non-variational
        energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ = ρin)
        #, eigenvalues, εF)

        """Compute the Fermi level by diagonalization"""
		n_bands_compute = 100  # for convenience
        eigres = diagonalize_all_kblocks(eigensolver, ham, n_bands_compute)

        occ, εF = DFTK.compute_occupation(ham.basis, eigres.λ)

		"""Compute the Fermi level by Bisection, evaluate Tr(f_{β,μ}(H)) - Ne"""
        # εF = genmu_bisec(basis, ham, M) 

        ψ, occupation, ρout = rhoGenChebP(ham, model, εF, M)
        
        # Update info with results gathered so far
        info = (; ham, basis, converged, stage=:iterate, algorithm="SCF",
        ρin, ρout, α = damping, n_iter, occupation_threshold, ψ,
        occupation, εF)

        #Compute the energy of the new state
        if compute_consistent_energies
           energies, _ = energy_hamiltonian(basis, ψ, occupation; ρ=ρout)
        end

        info = merge(info, (; energies))

        # Apply mixing and pass it the full info as kwargs
        δρ = DFTK.mix_density(mixing, basis, ρout - ρin; info...)
        ρnext = ρin .+ T(damping) .* δρ
        info = merge(info, (; ρnext=ρnext))

        callback(info)
        is_converged(info) && (converged = true)

        ρnext
    end

    # Tolerance and maxiter are only dummy here: Convergence is flagged by is_converged
    # inside the fixpoint_map. Also we do not use the return value of fpres but rather the
    # one that got updated by fixpoint_map
    solver(fixpoint_map, ρout, maxiter; tol=eps(T))

    # We do not use the return value of fpres but rather the one that got updated by fixpoint_map
    # ψ is consistent with ρout, so we return that. We also perform
    # a last energy computation to return a correct variational energy

    energies, ham = energy_hamiltonian(basis, ψ, occupation;  ρ=ρout, εF=εF)

    # Measure for the accuracy of the SCF
    # TODO probably should be tracked all the way ...
    norm_Δρ = norm(info.ρout - info.ρin) * sqrt(basis.dvol)

    # Callback is run one last time with final state to allow callback to clean up
    info = (; ham, basis, energies, converged,      occupation_threshold,
    ρ=ρout, α=damping, occupation, εF,
    n_iter, ψ,  
    stage=:finalize, algorithm="SCF", norm_Δρ)
    # ψ, delect the - info.diagonalization
    callback(info)
    info
end

"""
Stochastic : Solve the Kohn-Sham equations with a SCF algorithm, starting at ρ.
"""

@timing function self_consistent_field_sdft(basis::PlaneWaveBasis; M = 2000, Ns = 500,
                                       ρ=guess_density(basis),
                                       ψ=nothing,
                                       tol=1e-6,
                                       maxiter=100,
                                       solver=scf_anderson_solver(),
                                       eigensolver=lobpcg_hyper,
                                       determine_diagtol=DFTK.ScfDiagtol(),
                                       damping=0.8,  # Damping parameter
                                       mixing=SimpleMixing(),#LdosMixing(),
                                       is_converged=DFTK.ScfConvergenceEnergy(tol),
                                       callback=DFTK.ScfDefaultCallback(; show_damping=false),
                                       compute_consistent_energies=true,
                                       occupation_threshold=DFTK.default_occupation_threshold(), # 1e-10
                                       response=ResponseOptions(),  # Dummy here, only for AD
                                      )
    T = eltype(basis)
    model = basis.model

    # All these variables will get updated by fixpoint_map

    occupation = nothing
    eigenvalues = nothing
    ρout = ρ
    εF = nothing
    n_iter = 0
    energies = nothing
    ham = nothing
    info = (n_iter=0, ρin=ρ)   # Populate info with initial values
    converged = false
    
    # We do density mixing in the real representation
    # TODO support other mixing types
    function fixpoint_map(ρin)
        converged && return ρin  # No more iterations if convergence flagged

        n_iter += 1

        # Note that ρin is not the density of ψ, and the eigenvalues
        # are not the self-consistent ones, which makes this energy non-variational
        
        if n_iter == 1  # since input ψ is nothing
            energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ = ρin)
        else 
            energies, ham = energy_hamiltonian(basis, ψ/sqrt(Ns), occupation;
            ρ = ρin)
        end    
       
		"""Compute the Fermi level by diagonalization"""
		n_bands_compute = 100  # for convenience, given directly
        eigres = diagonalize_all_kblocks(eigensolver, ham, n_bands_compute)
        occ, εF = DFTK.compute_occupation(ham.basis, eigres.λ)

		"""Compute the Fermi level by Bisection, evaluate Tr(f_{β,μ}(H)) - Ne"""
        # εF = genmu_bisec(basis, ham, M)  

        ψ, occupation, ρout = rhoGenStoc(ham, model, εF, M, Ns) 
        
        # Update info with results gathered so far
        info = (; ham, basis, converged, stage=:iterate, algorithm="SCF",
        ρin, ρout, α = damping, n_iter,  occupation_threshold, ψ,
        occupation, εF)

        #Compute the energy of the new state
        if compute_consistent_energies
           energies, _ = energy_hamiltonian(basis, ψ/sqrt(Ns), occupation; ρ=ρout, εF=εF)
        end

        info = merge(info, (; energies))

        # Apply mixing and pass it the full info as kwargs
        δρ = DFTK.mix_density(mixing, basis, ρout - ρin; info...)
        ρnext = ρin .+ T(damping) .* δρ
        info = merge(info, (; ρnext=ρnext))

        callback(info)
        is_converged(info) && (converged = true)

        ρnext
    end

    # Tolerance and maxiter are only dummy here: Convergence is flagged by is_converged
    # inside the fixpoint_map. Also we do not use the return value of fpres but rather the
    # one that got updated by fixpoint_map
    solver(fixpoint_map, ρout, maxiter; tol=eps(T))

    # We do not use the return value of fpres but rather the one that got updated by fixpoint_map
    # ψ is consistent with ρout, so we return that. We also perform
    # a last energy computation to return a correct variational energy

    energies, ham = energy_hamiltonian(basis, ψ/sqrt(Ns), occupation; ρ=ρout, εF=εF)

    # Measure for the accuracy of the SCF
    # TODO probably should be tracked all the way ...
    norm_Δρ = norm(info.ρout - info.ρin) * sqrt(basis.dvol)

    # Callback is run one last time with final state to allow callback to clean up
    info = (; ham, basis, energies, converged,      occupation_threshold, ρ=ρout, α=damping, occupation, εF, n_iter, ψ, stage=:finalize, algorithm="SCF", norm_Δρ)
    callback(info)
    info
end

