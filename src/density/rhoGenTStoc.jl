# generate ρ by tempering stochastic method

function genTCP(mu::Float64, T::Vector{Float64}, tM::Vector{Int64})

    Nt = length(T)
    tCP = zeros(maximum(tM) + 1, Nt)
    f1(x) = evaluateDos(FermiDirac(mu, T[1]), x)
    @. tCP[1:tM[1]+1, 1] = ChebyshevP(tM[1], f1).coef

    for i = 2:Nt
        f2(x) = evaluateDos(FermiDirac(mu, T[i]), x) #- evaluateDos(FermiDirac(mu, t[i-1]), x)
        @. tCP[1:tM[i]+1, i] = ChebyshevP(tM[i], f2).coef
    end

    return tCP
end

function rhoGenTStoc(Nsw::Int64, Nsc::Int64, tM::Vector{Int64}, tCP::Matrix{Float64}, pw, ind::Vector{Int64}, H)
    n_fftw = pw.n_fftw
    n_eigs = pw.n_eigs
    dim = length(n_fftw)
    nf = ntuple(x -> vcat(n_fftw, n_fftw)[x], 2dim)
    N = prod(nf)
    ufft = zeros(ComplexF64, nf)
    u = zeros(ComplexF64, nf)
    ρw = zeros(ComplexF64, nf)
    ρc = zeros(ComplexF64, nf)
    Nc = pw.Nc
    Nt = size(tCP, 2)
    npw = length(ind)
    d = Uniform(0, 2pi)

    z = zeros(ComplexF64, npw)
    zt = zeros(ComplexF64, npw, Nt)
    u1 = zeros(ComplexF64, npw)
    u2 = zeros(ComplexF64, npw)
    u0 = zeros(ComplexF64, npw)

    wcf = tCP[:, 1]
    wM = tM[1]
    for l = 1:npw#Nsw
        #u0 = exp.(im .* rand(d, npw))
        @. u0 = 0.0 + 0.0im
        u0[l] = 1 + 0.0im
        mul!(u1, H, u0)
        @. z = wcf[1] * u0 + wcf[2] * u1
        for k = 3:wM+1
            mul!(u2, H, u1)
            @. u2 = 2.0 * u2 - u0
            @. z += wcf[k] * u2
    
            u0 = copy(u1)
            u1 = copy(u2)
        end
        @. ufft = 0.0 + 0.0im
        ufft[ind] = z
        ifft!(ufft)
        @. u = conj(ufft) * ufft * N^2
        broadcast!(+, ρw, ρw, u)
    end
    @. ρw = (ρw * Nc) #/ Nsw

    for l = 1:Nsc
        u0 = exp.(im .* rand(d, npw))
        #@. u0 = 0.0 + 0.0im
        #u0[l] = 1 + 0.0im
        mul!(u1, H, u0)
        @. zt = u0 * tCP[1, :]' + u1 * tCP[2, :]'
        for k = 3:maximum(tM)+1
            mul!(u2, H, u1)
            @. u2 = 2.0 * u2 - u0
            @. zt += u2 * tCP[k, :]'

            u0 = copy(u1)
            u1 = copy(u2)
        end
        for t = 1:Nt
            @. ufft = 0.0 + 0.0im
            ufft[ind] = zt[:, t]
            ifft!(ufft)
            @. u = conj(ufft) * ufft * N^2 * (-1)^t
            broadcast!(+, ρc, ρc, u)
        end
    end
    @. ρc = (ρc * Nc) / Nsc

    ρ = @. ρw + ρc

    return ρ
end

