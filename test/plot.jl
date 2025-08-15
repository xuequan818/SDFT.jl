using Plots, Plots.Measures
using LaTeXStrings

# comparison of variance
include("variance.jl")

Ne, Var, VarT = run_var(2; Ecut=10)
colors = palette(:tab10)
P = plot(; guidefontsize=22, tickfontsize=20, legendfontsize=16, legend=:topleft, grid=:off, box=:on, size=(700, 500), titlefontsize=25, left_margin=4mm, right_margin=4mm, top_margin=3mm, bottom_margin=3mm, xlabel=L"N^2", ylabel=L"\mathbb{V}[\phi(P,\chi,f^{\frac{1}{2}})]", dpi=500)
plot!(P, Ne .^ 2, Var, xticks=Ne .^ 2, color=colors[1], markershape=:c, label="sDFT", markersize=8, st=:scatter, markerstrokecolor=colors[1], alpha=0.9)
plot!(P, Ne .^ 2, VarT, color=colors[2], m=:utriangle, st=:scatter, label="sDFT, Theory", markersize=8, markerstrokecolor=colors[2], alpha=0.9)
plot!(P, Ne[1]^2:Ne[end]^2, Ne[1]^2:Ne[end]^2, color=:black, lw=3, ls=:dash, alpha=0.5, label="")
P
