### 2D tests
## variance versus the number of electronss
include("sdft_var.jl")
sys = "stone_wales" # choose from "graphene", "stone_wales", "doping"
Nmax = 2 # max supercell size
Ecut = 10 # energy cutoff
Ns = 200 # number of stochastic orbitals
Ne, Var, VarT = run_var(Nmax; sys, Ecut, Ns)


## error versus the number of stochastic orbital
include("sdft_err.jl")
sys = "stone_wales" # choose from "graphene", "stone_wales", "doping"
Nmax = 2 # max supercell size
Ecut = 10 # energy cutoff
Ns = 100:100:400 # numbers of stochastic orbitals
tol_cheb = 1e-4 # truncation tolerance of Chebyshev expansion
Ne, Ns, Error = run_err_rho(Ns; sys, Nmax, Ecut, tol_cheb)


## mlmc cost
include("mlmc_cost.jl")

sys = "stone_wales" # choose from "graphene", "stone_wales", "doping"
L = 2 # number of levels
Ecut = 10 # energy cutoff
temperature = 0.1 
repeat = [1, 1] # supercell size
tot_tol = 1.0 # sampling tolerance
tol_cheb = 1e-4

# energy cutoff
rltec = run_mlmcec_cost(L, Ecut, temperature, repeat; tot_tol, tol_cheb, sys) 

# polynomial degree
rltpd = run_mlmcpd_cost(L, Ecut, temperature, repeat; tot_tol, tol_cheb, sys)

### 3D tests
##  Wall time comparison between sDFT with KS-DFT
include("sdft_vs_ksdft.jl")
repeat = [1, 1, 1]
Ecut = 10
temperature = 0.1
dp_ratio = 0.01 # doping ratio
el_dp = :C # doping element
L = 1 # number of levels
tot_tol = 0.5
tol_cheb = 1e-6
basis = silicon_setup(repeat; Ecut, temperature, dp_ratio, el_dp)
ρin = guess_density(basis)
εF = estimate_fermi(basis, ρin)

# KS-DFT
tks, ρks, dvol = run_ks_time(basis.model, Ecut, repeat, ρin, εF)

# energy cutoff
tec, ρec = run_mlmcec_time(L, basis.model, Ecut, repeat, ρin, εF; tot_tol, tol_cheb)
err_ec = norm(ρec - ρks) * sqrt(dvol)

# polynomial degree
tpd, ρpd = run_mlmcpd_time(L, basis.model, Ecut, repeat, ρin, εF; tot_tol, tol_cheb)
err_pd = norm(ρpd - ρks) * sqrt(dvol)

# single-level sDFT 
tmc, ρmc = run_mlmcpd_time(0, basis.model, Ecut, repeat, ρin, εF; tot_tol, tol_cheb)
