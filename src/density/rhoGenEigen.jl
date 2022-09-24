# generate ρ by direct summation of eigenfunctions

function rhoGen(fE::Vector{Float64}, U::Vector{Vector{ComplexF64}}, pw, ind::Vector{Int64})
    n_fftw = pw.n_fftw
    n_eigs = pw.n_eigs
    d = length(n_fftw)
    nf = ntuple(x -> vcat(n_fftw, n_fftw)[x], 2d)
    N = prod(nf)
    ufft = zeros(ComplexF64, nf)
    u = zeros(ComplexF64, nf)
    ρ = zeros(ComplexF64, nf)
    Nc = pw.Nc
    for j = 1:n_eigs
        @. ufft = 0.0 + 0.0im
        ufft[ind] = U[j]
        ifft!(ufft)
        @. u = fE[j] * N^2 * conj(ufft) * ufft
        broadcast!(+, ρ, ρ, u)
    end
    @. ρ = ρ * Nc
    return ρ
end