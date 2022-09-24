# generate ρ by energy windows stochastic methods

function genEwCP(ew::Vector{Float64}, beta::Float64, M::Int64)

    Nw = length(ew)
    f1(x) = evaluateDos(FermiDirac(ew[1], beta), x)
    ewCP = ChebyshevP(M, f1).coef

    for i = 2:Nw
        f2(x) = evaluateDos(FermiDirac(ew[i], beta), x) - evaluateDos(FermiDirac(ew[i-1], beta), x)
        ewCP = hcat(ewCP, ChebyshevP(M, f2).coef)
    end
    
    return ewCP
end

#use different stochastic orbitals
function rhoGenEwStocDiff(Nv::Int64, ewCP::Matrix{Float64}, pw, ind::Vector{Int64}, H)
    n_fftw = pw.n_fftw
    n_eigs = pw.n_eigs
    dim = length(n_fftw)
    nf = ntuple(x -> vcat(n_fftw, n_fftw)[x], 2dim)
    N = prod(nf)
    ufft = zeros(ComplexF64, nf)
    u = zeros(ComplexF64, nf)
    ρ = zeros(ComplexF64, nf)
    Nc = pw.Nc
    Nw = size(ewCP, 2)
    M = size(ewCP, 1) - 1
    npw = length(ind)

    z = zeros(ComplexF64, npw)
    u1 = zeros(ComplexF64, npw)
    u2 = zeros(ComplexF64, npw)

    d = Uniform(0, 2pi)
    for w = 1:Nw
        cf = ewCP[:, w]
        for l = 1:Nv
            u0 = exp.(im .* rand(d, npw))
            mul!(u1, H, u0)
            @. z = cf[1] * u0 + cf[2] * u1
            for k = 3:M+1
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
    end
    @. ρ = (ρ * Nc) / Nv

    return ρ
end

#use same stochastic orbitals
function rhoGenEwStocSame(Nv::Int64, ewCP::Matrix{Float64}, pw, ind::Vector{Int64}, H)
    n_fftw = pw.n_fftw
    n_eigs = pw.n_eigs
    dim = length(n_fftw)
    nf = ntuple(x -> vcat(n_fftw, n_fftw)[x], 2dim)
    N = prod(nf)
    ufft = zeros(ComplexF64, nf)
    u = zeros(ComplexF64, nf)
    ρ = zeros(ComplexF64, nf)
    Nc = pw.Nc
    Nw = size(ewCP, 2)
    M = size(ewCP, 1) - 1
    npw = length(ind)

    z = zeros(ComplexF64, npw, Nw)
    u1 = zeros(ComplexF64, npw)
    u2 = zeros(ComplexF64, npw)

    d = Uniform(0, 2pi)
    for l = 1:Nv
        u0 = exp.(im .* rand(d, npw))
        mul!(u1, H, u0)
        @. z = u0 * ewCP[1, :]' + u1 * ewCP[2, :]'
        for k = 3:M+1
            mul!(u2, H, u1)
            @. u2 = 2.0 * u2 - u0
            @. z += u2 * ewCP[k, :]'

            u0 = copy(u1)
            u1 = copy(u2)
        end
        for w = 1:Nw
            @. ufft = 0.0 + 0.0im
            ufft[ind] = z[:, w]
            ifft!(ufft)
            @. u = conj(ufft) * ufft * N^2
            broadcast!(+, ρ, ρ, u)
        end
    end
    @. ρ = (ρ * Nc) / Nv

    return ρ
end

function rhoGenEwSD(Nv::Int64, ewCP::Matrix{Float64}, pw, ind::Vector{Int64}, H)
    n_fftw = pw.n_fftw
    n_eigs = pw.n_eigs
    d = length(n_fftw)
    nf = ntuple(x -> vcat(n_fftw, n_fftw)[x], 2d)
    N = prod(nf)
    ufft = zeros(ComplexF64, nf)
    u = zeros(ComplexF64, nf)
    ρD = zeros(ComplexF64, nf)
    ρS = zeros(ComplexF64, nf)
    Nc = pw.Nc
    Nw = size(ewCP, 2)
    M = size(ewCP, 1) - 1
    npw = length(ind)

    u0 = zeros(ComplexF64, npw)
    u1 = zeros(ComplexF64, npw)
    u2 = zeros(ComplexF64, npw)
    ewD = ewCP[:, 1:end-1]
    ewS = ewCP[:, end]
    zD = zeros(ComplexF64, npw, Nw - 1)
    zS = zeros(ComplexF64, npw)

    for l = 1:npw
        @. u0 = 0.0 + 0.0im
        u0[l] = 1 + 0.0im
        mul!(u1, H, u0)
        @. zD = u0 * ewD[1, :]' + u1 * ewD[2, :]'
        for k = 3:M+1
            mul!(u2, H, u1)
            @. u2 = 2.0 * u2 - u0
            @. zD += u2 * ewD[k, :]'

            u0 = copy(u1)
            u1 = copy(u2)
        end
        for w = 1:Nw-1
            @. ufft = 0.0 + 0.0im
            ufft[ind] = zD[:, w]
            ifft!(ufft)
            @. u = conj(ufft) * ufft * N^2
            broadcast!(+, ρD, ρD, u)
        end
    end
    @. ρD = ρD * Nc

    d = Uniform(0, 2pi)
    for l = 1:Nv
        u0 = exp.(im .* rand(d, npw))
        #@. u0 = 0.0 + 0.0im
        #u0[l] = 1 + 0.0im
        mul!(u1, H, u0)
        @. zS = ewS[1] * u0 + ewS[2] * u1
        for k = 3:M+1
            mul!(u2, H, u1)
            @. u2 = 2.0 * u2 - u0
            @. zS += ewS[k] * u2
    
            u0 = copy(u1)
            u1 = copy(u2)
        end
        @. ufft = 0.0 + 0.0im
        ufft[ind] = zS
        ifft!(ufft)
        @. u = conj(ufft) * ufft * N^2
        broadcast!(+, ρS, ρS, u)
    end
    @. ρS = (ρS * Nc) / Nv

    ρ = ρD + ρS

    return ρ
end