#------------------------------------------------
# compute the Chebyshev information
#------------------------------------------------
struct ChebInfo{T<:Real}
    E1::T
    E2::T
    order::Integer
    coef::Matrix{T}
end

function chebyshev_info(ham::HamiltonianBlock,
                        smearfs, M, cal_way::Symbol;
                        is_sqrt=true, cheb_method=KPM(),
                        Npt=round(Int,1.1M), kws...) 
    TT = real(eltype(ham))
    E1, E2 = S2_bound(ham, cal_way; kws...)
    pt = cos.(range(0, 2pi - pi / Npt, length=2Npt))
    newM, coef = genCheb(smearfs, is_sqrt, M, pt, E1, E2, cheb_method; kws...)
    if iszero(newM)
        error("No eigs in this range. S2 bound : ($(round(E1, digits=1)), $(round(E2, digits=1)))")
    end
    
    ChebInfo(TT(E1),TT(E2),newM,TT.(coef))
end

abstract type ChebyshevMethod end

struct KPM <: ChebyshevMethod end
struct JacksonKPM <: ChebyshevMethod end

function genCheb(funs::Vector{<:Function}, 
				 M::Integer, pt, ::KPM; 
				 tol_cheb=nothing, kws...)
    # f(x) = \sum_{n = 0}^{M}a_n*T_n(x) 
    coef = zeros(length(funs), M + 1)
    cM = (M + 1) * ones(Int, length(funs))
    @views for (i, ifun) in enumerate(funs)
        cfi = cheb_coef_by_fft(ifun, M, pt)
        coef[i, :] = cfi
        coef[i, 1] /= 2
        !isnothing(tol_cheb) && (
            if norm(cfi, Inf) <= tol_cheb
                cM[i] = 1
            else
                ciM = findlast(x -> abs(x) > tol_cheb, cfi)
                isnothing(ciM) || (cM[i] = ciM)
            end
        )
    end
    Mmax = maximum(cM)

    return Mmax - 1, coef[:, 1:Mmax]
end

function genCheb(funs::Vector{<:Function}, 
                 M::Integer, pt, ::JacksonKPM;
				 tol_cheb=nothing, kws...)    
	# f(x) = \sum_{n = 0}^{M}a_n*T_n(x) 
    function JacksonDamping(M)
        aM = pi / (M + 2)
        g(m) = ((1 - m / (M + 2))sin(aM)cos(m * aM) + (1 / (M + 2))cos(aM)sin(m * aM)) / sin(aM)
        g.(0:M)
    end
    
    JDM = JacksonDamping(M)
    coef = zeros(length(funs), M + 1)
    cM = (M + 1) * ones(Int, length(funs))
    @views for (i, ifun) in enumerate(funs)
        cfi = cheb_coef_by_fft(ifun, M, pt)
        cfi[1] /= 2
        @. cfi = cfi * JDM
        coef[i, :] = cfi
        !isnothing(tol_cheb) && (
            if norm(cfi, Inf) <= tol_cheb
                cM[i] = 1
            else
                ciM = findlast(x -> abs(x) > tol_cheb, cfi)
                isnothing(ciM) || (cM[i] = ciM)
            end
        )       
    end

    Mmax = maximum(cM) - 1
    if Mmax != M
        coef = coef[:, 1:Mmax+1] .* rvec(JacksonDamping(Mmax))
    end
    
    return Mmax, coef
end

function genCheb(smearfs::Vector{<:SmearFunction}, is_sqrt,
                 M::Integer, pt::Vector{T}, E1::T, 
				 E2::T, cheb_method::ChebyshevMethod; 
				 kws...) where {T<:Real}
    S2_fs = [x -> is_sqrt ? sqrt(S2_evalf(x, f, E1, E2)) : 
                            S2_evalf(x, f, E1, E2) for f in smearfs]
    genCheb(S2_fs, M, pt, cheb_method; kws...)
end

function genCheb(smearf::SmearFunction, is_sqrt,
                 M::Integer, pt::Vector{T},E1::T, 
                 E2::T, cheb_method::ChebyshevMethod; 
				 kws...) where {T<:Real}
    genCheb([smearf], is_sqrt, M, pt, E1, E2, cheb_method; kws...)
end

function genCheb(smearfs, M::Integer,
                 E1::T, E2::T;
                 is_sqrt=true,
                 cheb_method=KPM(),
                 Npt=round(Int,1.1M), 
				 kws...) where {T<:Real}
    pt = cos.(range(0, 2pi - pi / Npt, length=2Npt))
    genCheb(smearfs, is_sqrt, M, pt, E1, E2, cheb_method; kws...)
end
genCheb(smearfs, chebinfo::ChebInfo; kws...) = genCheb(smearfs, chebinfo.order, chebinfo.E1, chebinfo.E2; kws...)

function cheb_coef_by_fft(f::Function, M::Integer, pt::Vector{<:Real})
    fpt = @. complex(f(pt))
    FFTW.fft!(fpt)
    @views coef = real.(fpt[1:M+1])
    c = 2 / length(pt)
    @. coef *= c
end

# scaling and shifting (S2) operations
function S2_bound(ham::HamiltonianBlock, cal_way::Symbol;
                  lb_fac=0.2, ub_fac=0.2, eigen_maxiter=1, kws...)
    vmin, vmax = eigs_minmax(ham, Val(cal_way), eigen_maxiter)
	vmin = vmin - lb_fac
    vmax = vmax + ub_fac

    E1 = (vmax + vmin) / 2
    E2 = (vmax - vmin) / 2
    @assert !iszero(E2)

	return E1, E2
end

function eigs_minmax(ham::HamiltonianBlock, ::Val{:cal_op}, maxiter)
    T = eltype(ham)
    ψ0 = rand(T, size(ham,1))
    ψ0 = ψ0 ./ norm(ψ0)
    H_ψ(ψ) = ham * ψ
    vmin = real(KrylovKit.eigsolve(H_ψ, ψ0, 1, :SR; maxiter)[1][1])
    vmax = real(KrylovKit.eigsolve(H_ψ, ψ0, 1, :LR; maxiter)[1][1])
    return vmin, vmax
end

function eigs_minmax(ham::HamiltonianBlock, ::Val{:cal_mat}, maxiter)
    H = Matrix(ham)
    vmin = real(KrylovKit.eigsolve(H, 1, :SR; maxiter)[1][1])
    vmax = real(KrylovKit.eigsolve(H, 1, :LR; maxiter)[1][1])
    return vmin, vmax
end

S2_evalf(x, smearf::SmearFunction, E1, E2) = evalf(E2*x+E1, smearf)

S2_ham(ham::HamiltonianBlock, ::Val{:cal_op}, E1, E2) = ham
function S2_ham(ham::HamiltonianBlock, ::Val{:cal_mat}, E1, E2)
    # H - E1*I
    H = Matrix(ham)
    mul!(H, -E1, I, true, true)
end

@inline function S2_mul!(Hψ, H::HamiltonianBlock, ψ, E1, inv_E2)
    mul!(Hψ, H, ψ)
    @. Hψ = (Hψ - E1 * ψ) * inv_E2
end

@inline function S2_mul!(Hψ, H::AbstractArray, ψ, E1, inv_E2)
    mul!(Hψ, H, ψ, inv_E2, false)
end

function dense_to_sparse(A::Matrix, tolerance::Real)
    is, js = axes(A)
    for j in js
        for i in is
            @inbounds val = A[i, j]
            @inbounds if abs2(val) < tolerance
                A[i, j] = 0
            end
        end
    end
    sparse(A)
end
