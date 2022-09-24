# generate ρ by stochastic methods
function rhoGenStoc(Nv::Int64, ChebP::ChebyshevP, pw, ind::Vector{Int64}, H)#H_v::Function)
    n_fftw = pw.n_fftw
    n_eigs = pw.n_eigs
    dim = length(n_fftw)
    nf = ntuple(x -> vcat(n_fftw, n_fftw)[x], 2dim)
    N = prod(nf)
    ufft = zeros(ComplexF64, nf)
    u = zeros(ComplexF64, nf)
    ρ = zeros(ComplexF64, nf)
    Nc = pw.Nc

    M = ChebP.M
    cf = ChebP.coef
    npw = length(ind)
    z = zeros(ComplexF64, npw)
    u1 = zeros(ComplexF64, npw)
    u2 = zeros(ComplexF64, npw)

    #d = Bernoulli()
    d = Uniform(0, 2pi)
    for l = 1:Nv
        #u0 = 2 .* rand(d, npw) .- 1
        u0 = exp.(im .* rand(d, npw))
        #u0 = randn(npw)
        #u1 = H_v(u0)
        mul!(u1, H, u0)
        @. z = cf[1] * u0 + cf[2] * u1
        for k = 3:M+1
            #u2 = H_v(u1)
            mul!(u2, H, u1)
            @. u2 = 2.0 * u2 - u0
            @. z += cf[k] * u2
    
            u0 = copy(u1)
            u1 = copy(u2)
        end
        @. ufft = 0.0 + 0.0im
        ufft[ind] = z
        ifft!(ufft)
        @. u = conj(ufft) * ufft * N^2
        broadcast!(+, ρ, ρ, u)
    end
    @. ρ = (ρ * Nc) / Nv

    return ρ
end

function rhoGenStocFree(Nv::Int64, ChebP::ChebyshevP,
    pw, ind::Vector{Int64}, H_v::Function,
    E1::Float64, E2::Float64)
    n_fftw = pw.n_fftw
    n_eigs = pw.n_eigs
    dim = length(n_fftw)
    nf = ntuple(x -> vcat(n_fftw, n_fftw)[x], 2dim)
    N = prod(nf)
    ufft = zeros(ComplexF64, nf)
    u = zeros(ComplexF64, nf)
    ρ = zeros(ComplexF64, nf)
    Nc = pw.Nc

    M = ChebP.M
    cf = ChebP.coef
    npw = length(ind)
    z = zeros(ComplexF64, npw)
    u1 = zeros(ComplexF64, npw)
    u2 = zeros(ComplexF64, npw)

    for l = 1:Nv
        u0 = randn(ComplexF64, npw)
        u1 = H_v(u0)
        @. u1 = (u1 - E1 * u0) / E2
        @. z = cf[1] * u0 + cf[2] * u1
        for k = 3:M+1
            u2 = H_v(u1)
            @. u2 = (u2 - E1 * u1) / E2
            @. u2 = 2.0 * u2 - u0
            @. z += cf[k] * u2

            u0 = copy(u1)
            u1 = copy(u2)
        end
        @. ufft = 0.0 + 0.0im
        ufft[ind] = z
        ifft!(ufft)
        @. u = conj(ufft) * ufft * N^2
        broadcast!(+, ρ, ρ, u)
    end
    @. ρ = (ρ * Nc) / Nv

    return ρ
end
