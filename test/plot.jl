using Plots, Plots.Measures
using LaTeXStrings

const markers = [:circle, :utriangle, :diamond, :star4]
const colors = palette(:tab10)


# variance versus the number of electrons
include("sdft_var.jl")
Ne, Var, VarT = run_var(2; Ecut=10, Ns=200)

P = plot(; guidefontsize=22, tickfontsize=20, legendfontsize=16, legend=:topleft, grid=:off, box=:on, size=(700, 500), titlefontsize=25, left_margin=4mm, right_margin=4mm, top_margin=3mm, bottom_margin=3mm, xlabel=L"N^2", ylabel=L"\mathbb{V}[\phi(P,\chi,f^{\frac{1}{2}})]", dpi=500)
plot!(P, Ne .^ 2, Var, xticks=Ne .^ 2, color=colors[1], markershape=:c, label="sDFT", markersize=8, st=:scatter, markerstrokecolor=colors[1], alpha=0.9)
plot!(P, Ne .^ 2, VarT, color=colors[2], m=:utriangle, st=:scatter, label="sDFT, Theory", markersize=8, markerstrokecolor=colors[2], alpha=0.9)
plot!(P, Ne[1]^2:Ne[end]^2, Ne[1]^2:Ne[end]^2, color=:black, lw=3, ls=:dash, alpha=0.5, label="")
P

# error versus the number of stochastic orbital
include("sdft_err.jl")
Ne, Ns, Error = run_err_rho(100:100:400; Nmax=2, Ecut=10, tol_cheb=1e-4)

P = plot(; guidefontsize=22, tickfontsize=20, legendfontsize=16, legend=:topleft, grid=:off, box=:on, size=(700, 500), titlefontsize=25, left_margin=4mm, right_margin=4mm, top_margin=3mm, bottom_margin=3mm, xlabel=L"(N/|\mathbb{S}|)^{1/2}", ylabel=L"\Delta\rho", dpi=500)
for (i,ne) in enumerate(Ne)
    err = Error[i]
    xs = @. sqrt(Ne[i]) / sqrt(Ns)
    plot!(P, xs, err, color=colors[i], markershape=markers[i], label=L"N=%$ne", markersize=6, st=:scatter, markerstrokecolor=colors[i], alpha=0.9)
end
xxs = vcat([@. sqrt(ne) / sqrt(Ns) for ne in Ne]...)
sort!(unique!(xxs))
err = Error[end]
xs = @. sqrt(Ne[end]) / Ns .^ (1 / 2)
ks = [(err[j+1] - err[j]) / (xs[j+1] - xs[j]) for j = 1:length(xs)-1]
k = sum(ks) / length(ks) + 0.05
fx(x) = k * (x - xs[end]) + err[end] 
plot!(P, xxs, fx.(xxs), color=:black, lw=3, ls=:dash, alpha=0.5, label="")


# mlmc variance
include("mlmc_var.jl")
# polynomial degree
L = 3
varpd, Qlpd, _ = run_mlmcpd_var(L; Nmax=1, Ecut=10, Q0=100, Qc=0);

P = plot(yscale=:log10, xlabel="ℓ", ylabel="", guidefontsize=22, title="", label="", ylims=(10^(round(log10(minimum(varpd[1]))-0.5)), 10^(0.2+ceil(log10(maximum(varpd[2]))))), tickfontsize=20, legendfontsize=19, legend=:bottomleft, grid=:off, box=:on, size=(770, 660), titlefontsize=20, left_margin=2mm, right_margin=2mm, top_margin=4mm, dpi=500)
plot!(P, 0:L, xticks=collect(0:L), varpd[2], yscale=:log10, lw=4, markershape=:c, label=L"\mathbb{V}[\widehat{\phi}^{(\ell)}_\chi]", markersize=10, ls=:dash)
plot!(P, 0:L, varpd[1], lw=4, m=:utriangle, label=L"\mathbb{V}[\widehat{\phi}^{(\ell)}_\chi- \widehat{\phi}^{(\ell-1)}_\chi]", markersize=10)

# energy cutoff
L = 2
varec, Qlec, ψ, basis, Cheb, ρ = run_mlmcec_var(L; Nmax=1, Q0=7.1, Ecut=12, tol_cheb=1e-4);

P = plot(yscale=:log10, xlabel="ℓ", ylabel="", guidefontsize=22, title="", label="", tickfontsize=20, legendfontsize=19, legend=:bottomleft, grid=:off, box=:on, size=(770, 660), titlefontsize=20, left_margin=2mm, right_margin=2mm, top_margin=4mm, dpi=500)
plot!(P, 0:L, xticks=collect(0:L), varec[2], yscale=:log10, lw=4, markershape=:c, label=L"\mathbb{V}[\widehat{\phi}^{(\ell)}_\chi]", markersize=10, ls=:dash)
plot!(P, 0:L, varec[1], lw=4, m=:utriangle, label=L"\mathbb{V}[\widehat{\phi}^{(\ell)}_\chi- \widehat{\phi}^{(\ell-1)}_\chi]", markersize=10)

# mlmc cost
include("mlmc_cost.jl")
# polynomial degree
N1s = [1,2,3]
N2s = [1]
Ecuts=[10.0]
temperatures = [1e-2, 1e-3]
lne = length(N1s) * length(N2s)
Ls = reshape(fill(1, lne) .* [2,3]', lne, 1, 2)
pd_t, mc_t, Ne, ns, Ms = run_mlmc_costs(:mlmcpd; Ls, N1s, N2s, Ecuts, temperatures, tol_cheb=2e-4)

P = plot(xlabel=L"$nN$", ylabel="Wall time (s)", guidefontsize=22, title="",  tickfontsize=20, legendfontsize=19, legend=:topleft, grid=:off, box=:on, size=(770, 660), titlefontsize=20, left_margin=2mm, right_margin=2mm, top_margin=4mm, dpi=500, scale=:log10)
for ti in 1:2
    xs = Ne[:,:,ti] .* ns[:,:,ti]
    plot!(P, xs, pd_t[:,:,ti], lw=1, markershape=:c, label= L"β=%$(inv(temperatures[ti]))", color=colors[ti], markersize=8, st=:scatter)
    plot!(P, xs, mc_t[:,:,ti], lw=1, markershape=:utriangle, label= L"β=%$(inv(temperatures[ti]))", color=colors[ti], markersize=8, st=:scatter)
end
P

# energy cutoff
N1s = [1,2,3]
N2s = [1]
Ecuts=[8.0, 12.0]
temperatures = [1e-2]
lne = length(N1s) * length(N2s)
Ls = reshape(fill(1, lne) .* [1,2]', lne, 2, 1)
ec_t, mc_t2, Ne2, ns2, Ms2 = run_mlmc_costs(:mlmcec; Ls, N1s, N2s, Ecuts, temperatures, tol_cheb=1e-3, cal_way=:cal_op)

P = plot(xlabel=L"$NM$", ylabel="Wall time (s)", guidefontsize=22, title="",  tickfontsize=20, legendfontsize=19, legend=:topleft, grid=:off, box=:on, size=(770, 660), titlefontsize=20, left_margin=2mm, right_margin=2mm, top_margin=4mm, dpi=500, scale=:log10)
for ei in 1:2
    xs = Ne2[:,ei,:] .* Ms2[:,ei,:]
    plot!(P, xs, ec_t[:,ei,:], lw=1, markershape=:c, label= L"E_{\rm c}=%$(Ecuts[ei])", color=colors[ei], markersize=8, st=:scatter)
    plot!(P, xs, mc_t2[:,ei,:], lw=1, markershape=:utriangle, label= L"E_{\rm c}=%$(Ecuts[ei])", color=colors[ei], markersize=8, st=:scatter)
end
P
