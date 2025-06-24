# Creates a new basis identical to `basis`, but with a new Ecut.
function DFTK.PlaneWaveBasis(basis::PlaneWaveBasis, Ecut::Number)
    model = basis.model
    kgrid = basis.kgrid
    variational = basis.variational
    symmetries_respect_rgrid = basis.symmetries_respect_rgrid
    if symmetries_respect_rgrid
        fft_size = nothing
    else 
        fft_size = basis.fft_size
    end
    use_symmetries_for_kpoint_reduction = basis.use_symmetries_for_kpoint_reduction
    comm_kpts = basis.comm_kpts
    architecture = basis.architecture
    PlaneWaveBasis(model; Ecut, kgrid, variational, 
                   fft_size, symmetries_respect_rgrid,
                   use_symmetries_for_kpoint_reduction, 
                   comm_kpts, architecture)
end
