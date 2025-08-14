export graphene_setup

function graphene_setup(repeats=[1,1]; Ecut=7.0, temperature=1e-3)
    L = 20  # height of the simulation box
    a = 4.66  # lattice constant
    a1 = a*[1/2,-sqrt(3)/2, 0] * repeats[1]
    a2 = a*[1/2, sqrt(3)/2, 0] * repeats[2]
    a3 = L*[0  , 0        , 1]
    lattice = [a1 a2 a3]
    C1 = [1/3,-1/3]  # in reduced coordinates
    C2 = [2/3,-2/3] 
    C = ElementPsp(:C, load_psp("hgh/lda/c-q4"))
    atoms = fill(C, 2*prod(repeats))

    positions = typeof(C1)[]
    for rpt1 in 0:repeats[1]-1
        for rpt2 in 0:repeats[2]-1
            rpt = [rpt1, rpt2]
            push!(positions, C1 + rpt)
            push!(positions, C2 + rpt)
        end
    end
    map!(x -> x./repeats, positions, positions)
    positions = vcat.(positions, 0)
    
    model = model_DFT(lattice, atoms, positions; functionals=LDA(), temperature)
    PlaneWaveBasis(model; Ecut, kgrid=[1,1,1])
end
