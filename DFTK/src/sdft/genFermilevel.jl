# Compute the Fermi level εF by evaluating Trace(f_{β,εF}(H))

import Roots

function genmu_bisec(basis::PlaneWaveBasis, ham; M = 500)
    Ne = basis.model.n_electrons
	filled_occ = DFTK.filled_occupation(basis.model)
	εF = zeros(Float64, length(basis.kpoints))
	temp = basis.model.temperature
	β  = iszero(temp) ? Float64(Inf) : 1/temp
	for k = 1 : length(basis.kpoints)
        H = Matrix(ham.blocks[k])
        Emin, Umin = eigsolve(H, 2*Ne, :SR); # solving the lowest 2Ne eigenvalues 
        Emax, Umax = eigsolve(H, 1, :LR);
		minε = real.(Emin[1])
		maxε = real.(Emin[end]) # narrow the range of bisection
		Elb = real.(Emin[1]) - 0.1
		Eub = real.(Emax[1]) + 0.1
		E1 = (Elb + Eub ) / 2
		E2 = (Eub - Elb) / 2
		H = (H - E1 * I) ./ E2   # scaled H 
 
		εF[k] = Roots.find_zero(μ -> filled_occ * compute_nelec_trace(H, E1, E2, β, μ, M) - Ne, (minε, maxε),
									Roots.Bisection(), atol=eps(Float64))
	end
	DFTK.weighted_ksum(basis, εF)
end

function compute_nelec_trace(H::Matrix{ComplexF64}, E1::Float64, E2::Float64, β, μ, M)
	
	FD = FermiDirac((μ - E1) / E2, β * E2)
	ChebP = ChebyshevP(M, FD)
	cf = ChebP.coef

	npw = size(H, 1) 
	ψ = DFTK.ortho_qr(randn(ComplexF64, npw, npw)) 
	nelec = 0.
	for i = 1 : npw
		u0 = ψ[:, i]
        u1 = H * u0
	    z = cf[1] .* u0 + cf[2] .* u1
		for m = 3 : M + 1
			u2 = H * u1
			u2 = 2.0 * u2 - u0
			z += cf[m] * u2
		
			u0 = u1
			u1 = u2
		end
    	nelec += real(z' * z)
    end
    return nelec
end
