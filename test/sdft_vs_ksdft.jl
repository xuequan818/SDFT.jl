using SDFT
using SDFT: estimate_fermi
using Dates
include("testcase.jl")

function run_ks_time(model, Ecut, repeat, ρ, εF;
                     kgrid=[1,1,1], max_band_fraction=0.33,
                     Ecut_coarse=35*prod(repeat)^(-2/3))
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    dof = take_dof(basis)

    determine_solver(nb) = (nb / dof ≤ max_band_fraction) ? lobpcg_hyper : lapack_partial

    occupation_threshold = 1e-6
    temp = basis.model.temperature
    nbands = if temp ≥ 0.01
        basis_coarse = PlaneWaveBasis(basis, Ecut_coarse)
        ρ_coarse = DFTK.transfer_density(ρ, basis, basis_coarse)
        n_bands_coarse = determine_n_bands_ks(diag_full, basis_coarse, εF; 
                                              ρ=ρ_coarse, occupation_threshold)

        @show n_bands_coarse
        n_bands = Int(ceil((temp ≤ 0.2 ? 1.1 : 1.2) * n_bands_coarse))
        eigensolver = determine_solver(n_bands)

        determine_n_bands_ks(eigensolver, basis, εF; ρ, n_bands, occupation_threshold)
    else 
        AdaptiveBands(basis.model).n_bands_compute
    end

    eigensolver = determine_solver(nbands)

    println("  nbands: $(nbands),  eigensolver: $(eigensolver)\n")
    flush(stdout)

    t0 = time()
    ρout = compute_density_eigs(basis, εF, eigensolver, nbands; ρ)
    elapsed = round(time() - t0; digits=1)

    return elapsed, ρout, basis.dvol
end

function run_mlmcpd_time(L, model, Ecut, repeat, ρ, εF;
                         Ns=100, kgrid=[1, 1, 1], kws...)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    nsl = ceil.(Int, Ns ./ [2^i for i = 0:L]) 
    nsl = [i <= 10 ? 10 : i for i in nsl]

    t0 = time()
    ρout = compute_stoc_density(basis, εF, PDegreeML(nsl); ρ, kws...)
    elapsed = round(time() - t0; digits=1)

    return elapsed, ρout
end

function run_mlmcec_time(L, model, Ecut, repeat, ρ, εF;
                         Ns=100, kgrid=[1, 1, 1], kws...)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    nsl = ceil.(Int, Ns ./ [2^i for i = 0:L])
    nsl = [i <= 10 ? 10 : i for i in nsl]

    t0 = time()
    ρout = compute_stoc_density(basis, εF, ECutoffML(basis, nsl); ρ, kws...)
    elapsed = round(time() - t0; digits=1)

    return elapsed, ρout
end

function extract_silicon_information(basis; Ecut_fermi=min(10,basis.Ecut))
    ρ = guess_density(basis)
    εF = estimate_fermi(basis, ρ; Ecut_fermi)

    lattice = basis.model.lattice
    positions = basis.model.positions
    atoms = basis.model.atoms

    ind_pf = findall(x -> Symbol(x.species) == :Si, atoms)
    ind_dp = findall(x -> Symbol(x.species) != :Si, atoms)
    if !isempty(ind_dp)
        el_dp = Symbol(atoms[ind_dp[1]].species)
    else
        el_dp = nothing
    end

    return (; ρ, εF, lattice, positions, ind_pf, ind_dp, el_dp)
end

function generate_silicon_data(repeats, Ecut, temperature; dp_ratio=0.0, 
                               el_dp=nothing, Ecut_fermi=min(10, Ecut))
    ρ_list = []
    εF_list = []
    lat_list = []
    pos_list = []
    ind_pf_list = []
    ind_dp_list = []
    for (i, repeat) in enumerate(repeats)
        basis = silicon_setup(repeat; Ecut, temperature, dp_ratio, el_dp)
        ρ, εF, lat, pos, ind_pf, ind_dp, el_dp = extract_silicon_information(basis; Ecut_fermi)
        push!(ρ_list, ρ)
        push!(εF_list, εF)
        push!(lat_list, lat)
        push!(pos_list, pos)
        push!(ind_pf_list, ind_pf)
        push!(ind_dp_list, ind_dp)
    end

    function save_output(outdir)
        date_str = Dates.format(now(), "yyyymmdd_HH_MM")
        dpstr = isnothing(el_dp) ? "" : string(el_dp)
        output_file = joinpath(outdir, "Si$(dpstr)_data_$(date_str).jld2")
        jldsave(output_file; Ecut, temperature, repeats, el_dp, ρ_list, εF_list, lat_list, pos_list, ind_pf_list, ind_dp_list)
    end

    try
        outdir = joinpath(@__DIR__, "..", "data")
        save_output(outdir)
    catch
        save_output(@__DIR__)
    end
end

function build_model(lat, pos, ind_pf, ind_dp, el_dp; temperature=1e-3)         
    psp = PseudoFamily("cp2k.nc.sr.pbe.v0_1.semicore.gth")
    atoms = Vector{DFTK.Element}(undef, length(ind_pf) + length(ind_dp))
    for ipf in ind_pf
        atoms[ipf] = ElementPsp(:Si, psp)
    end
    if length(ind_dp) > 0
        for idp in ind_dp
            atoms[idp] = ElementPsp(el_dp, psp)
        end
    end

    model = model_DFT(lat, atoms, pos; functionals=PBE(),
                      temperature, symmetries=false)
end
