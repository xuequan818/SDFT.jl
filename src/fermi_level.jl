import Roots

# compute fermi level by THD obtained from hamiltonian
function compute_fermi_level(basis::PlaneWaveBasis;
    cal_way=:cal_mat, kws...)
    # Fixed fermi level
    !isnothing(basis.model.εF) && return basis.model.εF

    ham = Hamiltonian(basis; kws...).blocks[1]
    λmin, λmax = eigs_minmax(ham, Val(cal_way), 0.5)
    εF0 = (λmin + λmax) / 2
    THD, Cheb = compute_ham_cheb_trace(basis, εF0; cal_way, kws...)
    compute_fermi_level(basis, THD, Cheb; kws...)
end

# compute fermi level by THD obtained from TH
function compute_fermi_level(basis::PlaneWaveBasis{T},
							 THD, Cheb::ChebInfo;
							 tol_n_elec=max(T(1e-6),100eps(T)),
							 kws...) where {T}
    # Fixed fermi level
    !isnothing(basis.model.εF) && return basis.model.εF

    inv_temp = inv(basis.model.temperature)
    excess(εF) = excess_n_electrons(basis, THD, Cheb, εF, inv_temp)

    εmin, εmax = bracket_fermi_level(excess, Cheb, T(basis.model.temperature))

    tol_εF = sqrt(eps(T)) * max(one(T), abs(εmin), abs(εmax))
    εF = Roots.find_zero(excess, (εmin, εmax), Roots.Bisection(), atol=tol_εF)
    abs(excess(εF)) > tol_n_elec && error("This should not happen ...")
    εF
end

function bracket_fermi_level(excess, Cheb, temperature::T; εF_maxiter=80) where {T}
    # Chebyshev scaling:
    # Hs = (H - E1*I) / E2
    # so the approximate spectral interval is [E1 - E2, E1 + E2].
    center = T(Cheb.E1)
    radius = T(Cheb.E2)

    # Add an energy margin outside the spectral interval.
    # At high temperature, the Fermi level may be far away from the spectrum,
    # so the margin should scale with temperature.
    margin = max(one(T), 20 * temperature)

    # Initial bracket.
    εmin = center - radius - margin
    εmax = center + radius + margin

    fmin = excess(εmin)
    fmax = excess(εmax)

    for _ in 1:εF_maxiter
        # We need excess(εmin) < 0 < excess(εmax).
        if fmin < 0 && fmax > 0
            return εmin, εmax
        end

        # Enlarge the searching interval.
        margin *= 2

        # If the lower endpoint still gives too many electrons,
        # move it further to the left.
        if fmin >= 0
            εmin = center - radius - margin
            fmin = excess(εmin)
        end

        # If the upper endpoint still gives too few electrons,
        # move it further to the right.
        if fmax <= 0
            εmax = center + radius + margin
            fmax = excess(εmax)
        end
    end

    error("Failed to bracket Fermi level: excess(εmin) = $fmin, excess(εmax) = $fmax")
end

function excess_n_electrons(basis, THD, Cheb, εF, inv_temp)
    compute_dos_given_fermi_level(basis, THD, Cheb, εF, inv_temp) - basis.model.n_electrons
end

function compute_dos_given_fermi_level(basis, THD, Cheb, εF, inv_temp)
    smearf = FermiDirac(εF, inv_temp)
    _, coef = genCheb(smearf, Cheb; is_sqrt=false, Npt_ratio = 4, tol_cheb=nothing)
    filled_occ = filled_occupation(basis.model)
    return filled_occ * real(dot(coef, THD))
end

function compute_ham_cheb_trace(basis::PlaneWaveBasis, εF::Real; 
							    M=Int(1e5), tol_cheb=1e-6,
								cal_way=:cal_mat, kws...)
    smearf = FermiDirac(εF, inv(basis.model.temperature))
    is_sqrt = false
    if length(basis.kpoints) == 1
        ham = Hamiltonian(basis; kws...)
        Cheb = chebyshev_info(ham.blocks[1], smearf, M, cal_way; is_sqrt, tol_cheb, kws...)
    else
        Cheb = chebyshev_info(basis, smearf, M, cal_way; is_sqrt, tol_cheb, kws...)
    end

    THD = compute_ham_cheb_trace(basis, Cheb; cal_way, kws...)

	return THD, Cheb
end

function compute_ham_cheb_trace(basis::PlaneWaveBasis{T}, Cheb::ChebInfo; cal_way=:cal_mat, kws...) where {T}
    nk = length(basis.kpoints)
	hambls = Hamiltonian(basis; kws...).blocks
    TT = complex(T)

	THD = zeros(TT, Cheb.order + 1)
	for ik = 1:nk
        Hs = S2_ham(hambls[ik], Val(cal_way), Cheb.E1, Cheb.E2)
		THk = compute_TH_trace(Hs, Cheb.order, Cheb.E1, Cheb.E2)
		THD .+= basis.kweights[ik] .* THk
	end

	return real.(THD)
end

function compute_TH_trace(H, M::Int64, E1, E2)
    inv_E2 = inv(E2)
    SE2 = 2*inv_E2
	n = size(H,1)
	T = eltype(H)
    THD = zeros(T, M + 1)
	THD[1] = n

    u0 = zeros(T, n)
    u1 = similar(u0)
    u2 = similar(u0)
	for i = 1:n
        if M > 0
			@. u0 = zero(T)
			u0[i] = one(T)			
            S2_mul!(u1, H, u0, E1, inv_E2)
            THD[2] += u1[i]
            for k = 3:M+1
                # compute u2 = 2 * H * u1 - u0
                S2_mul!(u2, H, u1, E1, SE2)
                broadcast!(-, u2, u2, u0)
                THD[k] += u2[i]

				u0, u1, u2 = u1, u2, u0
            end
        end
    end
    
    THD
end
