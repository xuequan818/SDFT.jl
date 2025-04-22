function estimate_var(basis::PlaneWaveBasis,
                      εF::Real, ST::SDFTMethod;
                      cal_way=:cal_mat, M=Int(5e4),
					  tol_cheb=1e-6, kws...)
    ham = Hamiltonian(basis; kws...).blocks[1]
    smearf = FermiDirac(εF, inv(basis.model.temperature))
    Cheb = chebyshev_info(ham, smearf, M, cal_way; tol_cheb, kws...)

    estimate_var(basis, Cheb, ST; cal_way, kws...)
end

function estimate_var(basis::PlaneWaveBasis,
                      Cheb::ChebInfo, ST::SDFTMethod;
                      cal_way=:cal_mat, kws...) 
    occ = filled_occupation(basis.model)
    hambl = [iham.blocks[1] for iham in sdft_hamiltonian(basis, ST; kws...)]
    ψ = compute_wavefun(hambl, cal_way, Cheb, ST)
    
    return occ^2 .* estimate_var(basis, ψ, ST), ψ, hambl
end

estimate_var(basis::PlaneWaveBasis, ψ, ST::MC) = variance(ψ[1])

function estimate_var(basis::PlaneWaveBasis{T}, ψ, 
					  ST::PDegreeML{N}) where {T,N}
	var = zeros(T,N)
	var[1] = variance(ψ[1])
	for l = 2:N
		var[l] = variance(ψ[2l-2],ψ[2l-1])
	end
	var
end

function estimate_var(basis::PlaneWaveBasis{T}, ψ, 
					  ST::ECutoffML{N}) where {T,N}
    basisl = ST.basisl
	var = zeros(T,N)
	var[1] = variance(ψ[1])
	for l = 2:N
        ψ1 = transfer_blochwave_kpt(ψ[2l-2], basisl[l-1], 
									basisl[l-1].kpoints[1],
							   		basisl[l], basisl[l].kpoints[1])
        var[l] = variance(ψ[2l-1], ψ1)
	end
	var
end

# Compute the variance of X[:,i] = (Aχ_i)*(Aχ_i)' and
# use the Welford algorithm (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
function variance(X::Matrix{T}) where {T}
    n, N = size(X)

    mean = zeros(T, n, n)
    old_mean = copy(mean)
    Xi = copy(mean)
    var = copy(mean)
    for i = 1:N
        @views xi = X[:, i]
        outervec!(Xi, xi)

        copy!(old_mean, mean)
        mean_update(mean, Xi, inv(i))
        var_update(var, Xi, mean, old_mean)
    end

    real(sum(var)) / N
end

# Compute the variance of X[:,i]-Y[:,i] = (Aχ_i)*(Aχ_i)' - (Bχ_i)*(Bχ_i)' and
# use the Welford algorithm (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
function variance(X::Matrix{T}, Y::Matrix{T}) where {T}
	@assert size(X) == size(Y)
    n, N = size(X)

    mean = zeros(T, n, n)
    old_mean = copy(mean)
    XYi = copy(mean)
    var = copy(mean)
    for i = 1:N
        @views xi, yi = X[:, i], Y[:, i]
        outervecmxy!(XYi, xi, yi)

        copy!(old_mean, mean)
        mean_update(mean, XYi, inv(i))
        var_update(var, XYi, mean, old_mean)
    end

    real(sum(var)) / N
end

# result = x * x'
function outervec!(result::AbstractMatrix, x::AbstractVector)
    is, js = axes(result)
    if (is != js) || (is != axes(x, 1))
        error("mismatched array sizes")
    end
    for j in js
        for i in is
            @inbounds ci, cj = x[i], x[j]
            @inbounds result[i, j] = ci * conj(cj)
        end
    end
    result
end

# result = x * x' - y * y'
function outervecmxy!(result::AbstractMatrix, x::AbstractVector, y::AbstractVector)
    is, js = axes(result)
    if (is != js) || (is != axes(x, 1)) || (axes(x,1) != axes(y,1))
        error("mismatched array sizes")
    end
    for j in js
        for i in is
            @inbounds ci, ri, cj, rj = x[i], y[i], x[j], y[j]
            @inbounds result[i, j] = ci * conj(cj) - ri * conj(rj)
        end
    end
    result
end

# mean += (mean - X) .* nc
function mean_update(mean::Matrix, X::Matrix, nc::Real)
    is, js = axes(mean)
    for j in js
        for i in is
            @inbounds x, m = X[i, j], mean[i, j]
            @inbounds mean[i, j] = muladd(x - m, nc, m)
        end
    end
end

# var += (X-mean) .* conj.(X-old_mean)
function var_update(var::Matrix, X::Matrix, mean::Matrix, old_mean::Matrix)
    is, js = axes(var)
    for j in js
        for i in is
            @inbounds x, m1, m2 = X[i, j], mean[i, j], old_mean[i, j]
            @inbounds var[i, j] = muladd(x - m1, conj(x - m2), var[i, j])
        end
    end
end
