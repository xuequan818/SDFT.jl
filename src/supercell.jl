export graphene_setup

function aluminium_setup(repeat=1; Ecut=7.0, kgrid=[2, 2, 2])
    a = 7.65339
    lattice = a * Matrix(I, 3, 3)
    Al = ElementPsp(:Al; psp=load_psp("hgh/lda/al-q3"))
    atoms = [Al, Al, Al, Al]
    positions = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    unit_cell = periodic_system(lattice, atoms, positions)

    ## Make supercell in ASE:
    ## We convert our lattice to the conventions used in ASE, make the supercell
    ## and then convert back ...
    supercell_ase = convert_ase(unit_cell) * pytuple((repeat, 1, 1))
    supercell = pyconvert(AbstractSystem, supercell_ase)

    ## Unfortunately right now the conversion to ASE drops the pseudopotential information,
    ## so we need to reattach it:
    supercell = attach_psp(supercell; Al="hgh/lda/al-q3")

    ## Construct an LDA model and discretise
    ## Note: We disable symmetries explicitly here. Otherwise the problem sizes
    ##       we are able to run on the CI are too simple to observe the numerical
    ##       instabilities we want to trigger here.
    model = model_LDA(supercell; temperature=1e-3, symmetries=false)
    return model, PlaneWaveBasis(model; Ecut, kgrid)
end;

function graphene_setup(repeat=1; Ecut=7.0, kgrid=[2, 2, 2], L = 10.)
    a = 4.66
    a1 = a * [1 / 2, -sqrt(3) / 2, 0]
    a2 = a * [1 / 2, sqrt(3) / 2, 0]
    a3 = L * [0, 0, 1]
    lattice = [a1 a2 a3]
    C = ElementPsp(:C, psp=load_psp("hgh/lda/c-q4"))
    C1 = [1 / 3, -1 / 3, 0.0]  # in reduced coordinates
    C2 = -C1
    positions = [C1, C2]
    atoms = [C, C]

    unit_cell = periodic_system(lattice, atoms, positions)

    ## Make supercell in ASE:
    ## We convert our lattice to the conventions used in ASE, make the supercell
    ## and then convert back ...
    supercell_ase = convert_ase(unit_cell) * pytuple((repeat, 1, 1))
    supercell = pyconvert(AbstractSystem, supercell_ase)

    ## Unfortunately right now the conversion to ASE drops the pseudopotential information,
    ## so we need to reattach it:
    supercell = attach_psp(supercell; C="hgh/lda/c-q4")

    ## Construct an LDA model and discretise
    ## Note: We disable symmetries explicitly here. Otherwise the problem sizes
    ##       we are able to run on the CI are too simple to observe the numerical
    ##       instabilities we want to trigger here.
    model = model_LDA(supercell; temperature=1e-3, symmetries=false)
    return model, PlaneWaveBasis(model; Ecut, kgrid)
end;