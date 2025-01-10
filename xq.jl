L = 5  # height of the simulation box
kgrid = [1, 1, 1]
#kgrid = [6, 6, 1]
Ecut = 25
temperature = 1e-3

## Define the geometry and pseudopotential
a = 4.66  # lattice constant
a1 = a * [1 / 2, -sqrt(3) / 2, 0]
a2 = a * [1 / 2, sqrt(3) / 2, 0]
a3 = L * [0, 0, 1]
lattice = [a1 a2 a3]
C1 = [1 / 3, -1 / 3, 0.0]  # in reduced coordinates
C2 = -C1
positions = [C1, C2]
# We can shoose lda or pbe as the psp data 
C = ElementPsp(:C, psp=load_psp("hgh/pbe/c-q4"))
atoms = [C, C]
model = model_PBE(lattice, atoms, positions; temperature)
# remove the Entropy, since we don't solving eigenvalues, Entropy needs eigenvalues as input.
# change the "struct Model" in DFTK/src/Model.jl to "mutable struct Model".
filter!(x -> x != Entropy(), model.term_types)
basis = PlaneWaveBasis(model; Ecut, kgrid)
ρin = guess_density(basis)
hamL = Hamiltonian(basis; ρ=ρin)
H = Matrix(hamL.blocks[1])
eigensolver = lobpcg_hyper
eigres = diagonalize_all_kblocks(eigensolver, hamL, 30; ψguess=nothing) # 100 is the number of the bands (only for test)
occupation, εF = DFTK.compute_occupation(hamL.basis, eigres.λ)
f(x) = evaluateDos(FermiDirac(εF, 1 / temperature), x)
e, u = eigsolve(H, 10, :SR);
P = zeros(ComplexF64, length(u[1]), length(u[1]))
for i = 1:10
    P += f(e[i]) * u[i] * u[i]'
end
tr = sum(diag(P))^2
sum(diag(P) .^ 2)


cut = [15, 8]
Ns = [200, 100]
rcut = rhoCutStoc(100, cut, Ns, model; kgrid);
ham1 = rcut.hamL[1];
ham2 = rcut.hamL[2];
basis1 = ham1.basis
basis2 = ham2.basis
dofL = rcut.dofL
H1 = Matrix(ham1.blocks[1])
H2 = Matrix(ham2.blocks[1])
occupation, εF = DFTK.compute_occupation(ham1.basis, eigres.λ)
f(x) = evaluateDos(FermiDirac(εF, 1 / temperature), x)
e1, u1 = eigsolve(H1, 10, :SR);
P1 = zeros(ComplexF64, dofL[1, 1], dofL[1, 1])
for i = 1:10
    P1 += f(e1[i]) * u1[i] * u1[i]'
end
e2, u2 = eigsolve(H2, 10, :SR);
P2 = zeros(ComplexF64, dofL[1, 2], dofL[1, 2])
for i = 1:10
    P2 += f(e2[i]) * u2[i] * u2[i]'
end
P3 = P1[rcut.same_ind[1, 1], rcut.same_ind[1, 1]]

z3 = zero(z_l1)
z3[ind, :] = z_l2
norm(diag(z_l1 * z_l1' - z3 * z3'), 1)

s1 = basis1.kpoints[1].mapping
s2 = basis2.kpoints[1].mapping
u1 = rand(dofL[1, 1])
u1 = u1 / norm(u1)

"""ρout = ∑_{i}^{band}f_i|ψ(x)|^2"""
r1 = compute_density(basis1, [u1], [[2.0]])
u2 = zeros(dofL[1, 1])
u2[rcut.same_ind[1, 1]] = u1[rcut.same_ind[1, 1]]
u3 = u1 - u2
r2 = compute_density(basis1, [u2], [[2.0]])
r3 = compute_density(basis1, [u3], [[2.0]])

norm(r1 - (r2 + r3))

n = 100
a = rand(n, n)
a = (a + a') / 2
b = rand(n, n)
b = (b + b') / 2
b = a + 0.1 * b
A = a * a
B = b * b
AB = a * b

l1 = 0.0
for i = 1:n
    for j = setdiff(1:n, i)
        l1 += abs((A[i, i] - B[i, i]) * (A[j, j] - B[j, j]))
    end
end

l2 = 0.0
for i = 1:n
    for j = setdiff(1:n, i)
        l2 += A[i, i] * A[j, j] + B[i, i] * B[j, j] - 2 * AB[i, i] * AB[j, j]
    end
end
println("l1:$(l1), l2:$(l2), l3:$(norm(b-a,1)+l1)")


f(x) = @. 1 / (2 + sin(x))
L = 6pi
x = range(0, L, length=300)
ft = fft(f.(x)) ./ length(x)
plot!(st=:scatter, norm.(ft)[1:10])

L = 5  # height of the simulation box
kgrid = [1, 1, 1]
# kgrid = [6, 6, 1]
Ecut = 30
temperature = 1e-3 #1e-2

## Define the geometry and pseudopotential
a = 4.66  # lattice constant
a1 = a * [1 / 2, -sqrt(3) / 2, 0]
a2 = a * [1 / 2, sqrt(3) / 2, 0]
a3 = L * [0, 0, 1]
lattice = [a1 a2 a3]
C1 = [1 / 3, -1 / 3, 0.0]  # in reduced coordinates
C2 = -C1
positions = [C1, C2]
# We can shoose lda or pbe as the psp data 
C = ElementPsp(:C, psp=load_psp("hgh/pbe/c-q4"))
atoms = [C,C]

model = model_PBE(lattice, atoms, positions; temperature)
# remove the Entropy, since we don't solving eigenvalues, Entropy needs eigenvalues as input.
filter!(x -> x != Entropy(), model.term_types)

basis = PlaneWaveBasis(model; Ecut, kgrid)


"""scf: SimpleMixing + Anderson"""

"""solving eigenpaire"""
scfres = self_consistent_field(basis);
H = Matrix(scfres.ham.blocks[1])
f(x) = evaluateDos(FermiDirac(scfres.εF, 1 / temperature), x)
e, u = eigsolve(H, 14, :SR);
P = zeros(ComplexF64, length(u[1]), length(u[1]))
for i = 1:10
    P += f(e[i]) * u[i] * u[i]'
end
@show tr = sum(diag(P))^2 - sum(diag(P) .^ 2)

f1(x) = @. evaluateDos(FermiDirac(scfres.εF, 20), x)
f2(x) = @. evaluateDos(FermiDirac(scfres.εF, 10), x)
xx = collect(-0.5:0.001:1.5)
plot(xx,f1.(xx))
plot!(xx,f2.(xx))
g(x) = @. (sqrt(f1(x))-sqrt(f2(x)))^2
plot!(xx,g.(xx))
plot!(real.(e[1:10]),real.(g.(e[1:10])),st=:scatter)

n = 1000
d = Normal(0,1)
xn = zeros(n)
xn[1] = 100.
a = 1.05
for i = 1:n-1
    xn[i+1] = 1/xn[i] + rand(d)
end
plot(1:n,xn)

xx = collect(0:0.1:10)
f(x) = 1-1/1+exp(x)
plot(xx,f.(xx))

FD = FermiDirac(0.1, 10.)
f1(x) = sqrt(evaluateDos(FD, x))
xx = collect(-1.0:0.001:1.)
plot(xx,f1.(xx))

function f2(x; cp=ChebyshevP(1000, FD))
    cf = cp.coef
    M = cp.M
    t0 = 1.
    t1 = x
    val = cf[1]*t0 + cf[2]*t1
    for i = 3:M
        t2 = 2*x*t1 - t0
        val += cf[i]*t2

        t0 = t1
        t1 = t2
    end
    return val
end
plot!(xx, f2.(xx))


M = collect(10:10:1000)
e = zeros(length(M))
for i = 1:length(M)
    cp = ChebyshevP(M[i],FD)
    e[i] = norm(abs.(f2.(xx;cp=cp)-f1.(xx)),Inf)
end
plot(M,log.(e))
rho = pi / FD.β + sqrt((pi / FD.β)^2 + 1)
plot!(M,-log(rho)*M)

 
beta = 1000
M = 5000
s = round(pi / beta + sqrt((pi / beta)^2 + 1), digits=3)
#e = 1e-6
#f(p, l) = sqrt((s^(-2 * (p^(l - 1))) - s^(-2 * p^l))p^l)
pp = collect(2:0.1:10)
pm(p, l) = @. round(p^l)#round(2*l*log(p)/log(s))
#dd(p, l) = @. s^(-2 * pm(p, l - 1)) - s^(-2 * pm(p, l))
dd(p, l) = @. (s^(-pm(p, l - 1)) + s^(-pm(p, l)))^2
f(p,l) = @. (dd(p, l) * pm(p, l))
fL(p, l) = ((s^(-1 * pm(p, l))+s^(-1 * M)))^2 * M
f0(p,l) = (s^(-2 * pm(p, l-1)) * pm(p, l-1))
#LL = collect(2:20)
y = zeros(length(pp))
#for j = 1:length(LL)
L = 20
xx = collect(1:L)
for i = 1:length(pp)
    p = pp[i]
    pmi = pm(p,xx)
    #c = pmi[1] < 200 ? round(200/pmi[1],digits=3) : 1.
    #pmi = c .* pmi
    ls = findfirst(x -> x >= 50, pmi)
    lu = findfirst(x -> x >= M, pmi)
    l1 = ls == nothing ? 1 : ls
    l2 = lu == nothing ? length(xx) : lu-1
    L2 = xx[l1:l2]
    @show pmi[l1:l2]
    y[i] = (f0(p,xx[1])+sum(f.(p, L2    )) + fL(p,L2[end]))#*(length(L2)+1)
end
plot(pp,y)
markers = filter((m -> begin
        m in Plots.supported_markers()
    end), Plots._shape_keys)
cols = collect(palette(:Dark2_5))
p = plot(pp, y, lw=2, m=markers[1], markersize=4, label="", ylabel=L"\mathrm{sum}", legend=:topright, grid=:off, box=:on, xlabel=L"p", size=(600, 450), legendfontsize=18, tickfontsize=16, guidefontsize=20)
savefig("1000beta_p.pdf")


l = collect(1:10)
plot(l,f.(4,l))

x = collect(0:0.1:100)
plot(x,s.^x)

FD = FermiDirac(0., 1000.0)
f1(x) = evaluateDos(FD, x)
xx = collect(-1.0:0.001:1.0)
beta = collect(100:1000)
ee = zeros(length(beta))
for i = 1:length(beta)
    FD2 = FermiDirac(0., beta[i])
    f2(x) = evaluateDos(FD2, x)
    ee[i] = norm(f1.(xx).-f2.(xx),Inf)
end
plot(beta,ee)

a = [1 -1;sqrt(3) sqrt(3)]
b = a^(-1)

using Plots, Measurements
plot(0:2, [6, 10, 2], yerr = [1, 2, 3], msc = 1, label = "", ytick = 0:2:12)
plot(1:10, rand(10), showaxis=true, draw_arrow=false)

function aluminium_setup(repeat=1; Ecut=7.0, kgrid=[2, 2, 2])
    a = 7.65339
    lattice = a * Matrix(I, 3, 3)
    Al = ElementPsp(:Al, psp=load_psp("hgh/lda/al-q3"))
    atoms     = [Al, Al, Al, Al]
    positions = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]

    ## Make supercell in ASE:
    ## We convert our lattice to the conventions used in ASE
    ## and then back ...
    supercell = ase_atoms(lattice, atoms, positions) * (repeat, 1, 1)
    lattice   = load_lattice(supercell)
    positions = load_positions(supercell)
    atoms = fill(Al, length(positions))

    ## Construct an LDA model and discretise
    ## Note: We disable symmetries explicitly here. Otherwise the problem sizes
    ##       we are able to run on the CI are too simple to observe the numerical
    ##       instabilities we want to trigger here.
    model = model_LDA(lattice, atoms, positions; temperature=1e-3, symmetries=false)
    PlaneWaveBasis(model; Ecut, kgrid)
end;

# As part of the code we are using a routine inside the ASE,
# the [atomistic simulation environment](https://wiki.fysik.dtu.dk/ase/index.html)
# for creating the supercell and make use of the two-way interoperability of
# DFTK and ASE. For more details on this aspect see the documentation
# on [Input and output formats](@ref).

# Write an example supercell structure to a file to plot it:
setup = aluminium_setup(5)
ase_atoms(setup.model).write("al_supercell.png")


L = 10  # height of the simulation box
kgrid = [1, 1, 1]
Ecut = 15
temperature = 5e-3

model, basis = graphene_setup(2; Ecut=12.0, kgrid=[1, 1, 1])

scfres = self_consistent_field(basis)
ham = Hamiltonian(basis; ρ=scfres.ρ)
npw = length(G_vectors(ham.basis, ham.basis.kpoints[1]))
println("dof : $(npw)")

# solving eigenpaires, obtain ρ
eigensolver = lobpcg_hyper
eigres = diagonalize_all_kblocks(eigensolver, ham, 30; ψguess=nothing) # 100 is the number of the bands (only for test)

function select_eigenpairs_all_kblocks(eigres, rg)
    merge(eigres, (; λ=[λk[rg] for λk in eigres.λ],
        X=[Xk[:, rg] for Xk in eigres.X],
        residual_norms=[resk[rg] for resk in eigres.residual_norms]))
end

merge((; basis=bs_basis), select_eigenpairs_all_kblocks(eigres, 1:6))
occupation, εF = DFTK.compute_occupation(ham.basis, eigres.λ)

"""ρout = ∑_{i}^{band}f_i|ψ(x)|^2"""
ρout = compute_density(ham.basis, eigres.X, occupation);

M = 2000
""" ρout = rhoCheb """
rc = rhoCheb(M)
@time ψ, occupation, ρout_ChebP = rhoGen(ham, model, εF, rc);
pm = ψ[1] * ψ[1]'
rhop = [pm[i,i] for i = 1:npw]
sp = sum(rhop)^2
sr = 0.0
for i = 1:npw
    for j = 1:npw
        if abs(j - i) < npw / 2
            sr += rhop[i] * rhop[j]
        end
    end
end

function PHGen(ham::Hamiltonian, model::Model, εF::Float64, rhoG::rhoCutStoc)
    M = rhoG.M
    β = 1 / model.temperature
    nL = rhoG.nL
    hamL = rhoG.hamL
    full_ind = rhoG.full_ind
    dofL = rhoG.dofL

    filled_occ = DFTK.filled_occupation(model)
    kpoints = ham.basis.kpoints
    occ_sdft = Matrix{Any}(undef, length(kpoints), nL)
    ψ = Matrix{Any}(undef, length(kpoints), nL)

    k = 1

    # find the bound of eigenvalues
    H = Matrix(ham.blocks[k])
    Emin, Umin = eigsolve(H, 1, :SR)
    Emax, Umax = eigsolve(H, 1, :LR)
    Elb = real.(Emin[1]) - 0.1
    Eub = real.(Emax[1]) + 0.1
    E1 = (Elb + Eub) / 2
    E2 = (Eub - Elb) / 2

    FD_s = FermiDirac((εF - E1) / E2, β * E2)
    ChebP = ChebyshevP(M, FD_s)
    cf = ChebP.coef

    for l = 1:nL
        z_full1 = zeros(ComplexF64, dofL[k, l], dofL[k, l])
        H_l1 = Matrix(hamL[l].blocks[k])
        ind = full_ind[k, l]
        u0_l1 = Matrix{ComplexF64}(I, dofL[k, l], dofL[k, l])
        u1_l1 = copy(u0_l1)
        u2_l1 = copy(u0_l1)

        z = zeros(ComplexF64, dofL[1], dofL[1])
        z[full_ind[k, l], full_ind[k, l]] = ChebRecur(M, cf, u0_l1, u1_l1, u2_l1, H_l1, E1, E2)
        ψ[k, l] = z
    end

    return ψ
end

cut = collect(Ecut:-2.0:4.0)
Ns = ones(Int,length(cut))
rcut = rhoCutStoc(M, cut, Ns, model; kgrid);
Pcut = PHGen(ham, model, εF, rcut);

e = [norm(Pcut[1]-Pcut[i])^2 for i = length(cut):-1:2]
plot(cut[end:-1:2],log10.(e))


s = 0.
for i = 1:npw
    for j = 1:npw
        if abs(j-i) < npw/2
            s += tr[i]*tr[j]
        end
    end
end


""" ρout = rhoStoc """
sterrTwo = []
sterrInf = []
rep = collect(1:6)
K = 3
M = 3000
rc = rhoCheb(M)
rs = rhoStoc(M, 400)
for ne in rep
    veInf = 0.0
    veTwo = 0.0

    model, basis = graphene_setup(ne; Ecut=12.0, kgrid=[1, 1, 1])
    scfres = self_consistent_field(basis)
    ham = Hamiltonian(basis; ρ=scfres.ρ)
    npw = length(G_vectors(ham.basis, ham.basis.kpoints[1]))
    println("dof : $(npw)")
    # solving eigenpaires, obtain ρ
    @time eigres = diagonalize_all_kblocks(lobpcg_hyper, ham, 30; ψguess=nothing) # 100 is the number of the bands (only for test)
    occupation, εF = DFTK.compute_occupation(ham.basis, eigres.λ)
    #=
    H = Matrix(ham.blocks[1])
    β = 1 / model.temperature
    FD = FermiDirac(εF, β)
    f(x) = evaluateDos(FD,x)
    E, U = eigsolve(H, 10, :SR)
    P1 = zeros(ComplexF64,npw,npw)
    for i = 1:length(E)
        P1 += f(E[i]) * U[i] * U[i]'
    end
    =#

    @time ψ, occupation, ρout_ChebP = rhoGen(ham, model, εF, rc)
    P1 = ψ[1] * ψ[1]'

    for k = 1:K
        println("k-$(k),  Ne-$(ne)")
        @time ψ, occupation, ρout_sdft = rhoGen(ham, model, εF, rs)
        P2 = ψ[1] * ψ[1]'
        veTwo = norm(P1 - P2)^2
        veInf = norm(P1 - P2,1)
    end
    push!(sterrTwo, sqrt(veTwo / K))
    push!(sterrInf, sqrt(veInf / K))
end
P = plot(rep, sterrInf, scale=:log10, ylabel="Error", xlabel=L"N_e", guidefontsize=22, st=:scatter, label="", tickfontsize=20, legendfontsize=20, legend=(0.13, 0.88), grid=:off, box=:on, size=(800, 600), titlefontsize=30, margin=3mm, marker=:circle, markersize=6, markercolor=:white, markerstrokecolor=:black)
c = 6.5
plot!(P, rep, c .* (rep) .^ (0.5), color=:red, lw=1.5, label=L"N_e^{1/2}")


model, basis = graphene_setup(1; Ecut=6.0, kgrid=[1, 1, 1])
scfres = self_consistent_field(basis)
ham = Hamiltonian(basis; ρ=scfres.ρ)
npw = length(G_vectors(ham.basis, ham.basis.kpoints[1]))
println("dof : $(npw)")
# solving eigenpaires, obtain ρ
@time eigres = diagonalize_all_kblocks(lobpcg_hyper, ham, 30; ψguess=nothing) # 100 is the number of the bands (only for test)
occupation, εF = DFTK.compute_occupation(ham.basis, eigres.λ)
H = Matrix(ham.blocks[1])
β = 1 / model.temperature
FD = FermiDirac(εF, β)
f(x) = evaluateDos(FD, x)
E, U = eigsolve(H, 10, :SR)
P1 = zeros(ComplexF64, npw, npw)
for i = 1:length(E)
    P1 += sqrt(f(E[i])) * U[i] * U[i]'
end

