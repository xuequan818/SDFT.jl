using DFTK
using LinearAlgebra
using StaticArrays
using PseudoPotentialData
using UnitfulAtomic, Unitful
using AtomsBase
using AtomsBuilder
using Distributions

## 3D systems

function silicon_setup(repeats=[1, 1, 1]; Ecut=7.0, kgrid=[1, 1, 1], temperature=1e-3)
    ## Use AtomsBuilder to setup silicon cubic unit cell (8 Si atoms)
    ## with provided lattice constant, see [AtomsBase integration](@ref) for details.
    unit_cell = bulk(:Si; cubic=true)
    supercell = unit_cell * tuple(repeats...)  # Make a supercell

    pseudopotentials = PseudoFamily("cp2k.nc.sr.pbe.v0_1.semicore.gth")
    model = model_DFT(supercell; pseudopotentials, functionals=PBE(),
                      temperature, symmetries=false)
    PlaneWaveBasis(model; Ecut, kgrid)
end

## graphene-type systems

function graphene_setup(repeats=[1,1]; Ecut=7.0, temperature=1e-3, 
                        psp=PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
    lattice, atoms, positions = build_graphene_supercell(repeats; psp)
    
    model = model_DFT(lattice, atoms, positions; functionals=LDA(), temperature)
    PlaneWaveBasis(model; Ecut, kgrid=[1,1,1])
end

function stone_wales_setup(repeats=[1, 1]; Ecut=7.0, temperature=1e-3, 
                           psp=PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
    lattice, atoms, positions = build_graphene_supercell(repeats; psp)
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

function doping_setup(repeats=[1, 1]; Ecut=7.0, temperature=1e-3, 
                      psp=PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
    lattice, atoms, positions = build_graphene_supercell(repeats; psp)
    n_atoms = length(atoms)
    nd = cld(n_atoms, 10)
    ind = sample(1:n_atoms, nd, replace=false)
    ind_N = ind[1:cld(nd, 2)]
    ind_B = setdiff(ind, ind_N)
    for iN in ind_N
        atoms[iN] = ElementPsp(:N, psp)
    end
    for iB in ind_B
        atoms[iB] = ElementPsp(:B, psp)
    end

    model = model_DFT(lattice, atoms, positions; functionals=LDA(), temperature)
    PlaneWaveBasis(model; Ecut, kgrid=[1,1,1])
end

function build_graphene_supercell(repeats; psp=PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
    L = 20  # height of the simulation box

    # Define the geometry and pseudopotential
    a = 4.66  # lattice constant
    a1 = a * [1 / 2, -sqrt(3) / 2, 0]
    a2 = a * [1 / 2, sqrt(3) / 2, 0]
    a3 = L * [0, 0, 1]
    lattice = [a1 a2 a3]
    C1 = [1 / 3, -1 / 3, 0.0]  # in reduced coordinates
    C2 = [2 / 3, -2 / 3, 0.0]
    positions = [C1, C2]
    C = ElementPsp(:C, psp)
    atoms = [C, C]

    model = model_DFT(lattice, atoms, positions; functionals=PBE())
    sys_bohr = DFTK.periodic_system(model)

    cell_ang = [ustrip.(u"Å", cv) * u"Å" for cv in sys_bohr.cell.cell_vectors]
    atoms_ang = map(enumerate(sys_bohr.particles)) do (i, atom)
        pos_ang = ustrip.(u"Å", atom.position) * u"Å"
        AtomsBase.Atom(species(atom), pos_ang; mass=AtomsBase.mass(atom))
    end
    sys_ang = AtomsBase.FlexibleSystem(
        atoms_ang;
        cell_vectors=cell_ang,
        periodicity=sys_bohr.cell.periodicity,
        data=sys_bohr.data,
    )

    sys = sys_ang * tuple(repeats..., 1)

    lattice_sc = get_lattice(sys)
    positions_sc = fractional_coordinates(sys)
    atoms_sc = fill(C, length(positions_sc))

    return (; lattice=lattice_sc, atoms=atoms_sc, positions=positions_sc)
end

function get_lattice(sys)
    return UnitfulAtomic.austrip.(stack(cell_vectors(sys)))
end

function fractional_coordinates(sys)
    lattice = UnitfulAtomic.austrip.(stack(cell_vectors(sys)))
    positions = map(sys) do atom
        coordinate = zeros(Float64, 3)
        coordinate[1:3] = lattice[1:3, 1:3] \ Float64.(austrip.(position(atom)))
        SVector{3,Float64}(coordinate)
    end
    return positions
end
