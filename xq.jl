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

FD = FermiDirac(0.1, 100.)
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


s = 1.00314
e = 1e-6
#f(p,l) = sqrt((s^(-2(l-1)^p)-s^(-2l^p))l^p)
f(p, l) = sqrt((s^(-2*(p^(l-1))) - s^(-2*p^l))p^l)
f0(l) = sqrt(s^(-2))
pp = collect(2:1:10)
y = zeros(length(pp))
for i = 1:length(pp)
    p = pp[i]
    #L = ceil((log(e^(-1)) / log(s))^(1 / p))-1
    L = ceil(log(log(e^(-1)) / log(s))/log(p))   
    println("$(L),")
    xx = collect(1:L)
    y[i] = sum(f.(p,xx))
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
