using PseudoPotentialData

export graphene_setup, stone_wales_setup, doping_setup

function graphene_setup(repeats=[1,1]; Ecut=7.0, temperature=1e-3)
    lattice, atoms, positions = graphene_supercell(repeats)
    
    model = model_DFT(lattice, atoms, positions; functionals=LDA(), temperature)
    PlaneWaveBasis(model; Ecut, kgrid=[1,1,1])
end

function stone_wales_setup(repeats=[1, 1]; Ecut=7.0, temperature=1e-3)
    lattice, atoms, positions = graphene_supercell(repeats)
    rot = [0 -1 0; 1 0 0; 0 0 0]
    at1 = lattice * positions[1]
    at2 = lattice * positions[2]
    atc = (at1 + at2) / 2
    new_at1 = rot * (at1 - atc) + atc
    new_at2 = rot * (at2 - atc) + atc
    positions[1] = inv(lattice) * new_at1
    positions[2] = inv(lattice) * new_at2

    model = model_DFT(lattice, atoms, positions; functionals=LDA(), temperature)
    PlaneWaveBasis(model; Ecut, kgrid=[1, 1, 1])
end

function doping_setup(repeats=[1, 1]; Ecut=7.0, temperature=1e-3)
    lattice, atoms, positions = graphene_supercell(repeats)
    n_atoms = length(atoms)
    nd = cld(n_atoms, 10)
    ind = sample(1:n_atoms, nd, replace=false)
    ind_N = ind[1:cld(nd, 2)]
    ind_B = setdiff(ind, ind_N)
    psp = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
    for iN in ind_N
        atoms[iN] = ElementPsp(:N, psp)
    end
    for iB in ind_B
        atoms[iB] = ElementPsp(:B, psp)
    end

    model = model_DFT(lattice, atoms, positions; functionals=LDA(), temperature)
    PlaneWaveBasis(model; Ecut, kgrid=[1,1,1])
end

function graphene_supercell(repeats)
    L = 20  # height of the simulation box
    a = 4.66  # lattice constant
    a1 = a * [1 / 2, -sqrt(3) / 2, 0] * repeats[1]
    a2 = a * [1 / 2, sqrt(3) / 2, 0] * repeats[2]
    a3 = L * [0, 0, 1]
    lattice = [a1 a2 a3]
    C1 = [1 / 3, -1 / 3]  # in reduced coordinates
    C2 = [2 / 3, -2 / 3]
    C = ElementPsp(:C, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
    atoms = fill(C, 2 * prod(repeats))

    positions = typeof(C1)[]
    for rpt1 in 0:repeats[1]-1
        for rpt2 in 0:repeats[2]-1
            rpt = [rpt1, rpt2]
            push!(positions, C1 + rpt)
            push!(positions, C2 + rpt)
        end
    end
    map!(x -> x ./ repeats, positions, positions)
    positions = vcat.(positions, 0)

    (; lattice, atoms, positions)
end
