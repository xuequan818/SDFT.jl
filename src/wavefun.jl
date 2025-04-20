function compute_wavefun(ham, cal_way, Cheb, ST::MC)
    H = S2_ham(ham[1], Val(cal_way), Cheb.E1, Cheb.E2)
    X = random_orbital(eltype(H), size(H, 1), ST)
    ψ = compute_cheb_recur(H, X, Cheb.coef, Cheb.E1, Cheb.E2)
end

function compute_wavefun(ham, cal_way, Cheb, PD::PDegreeML{N}) where {N}
    coef = Cheb.coef
    E1 = Cheb.E1
    E2 = Cheb.E2
    H = S2_ham(ham[1], Val(cal_way), E1, E2)
    T = eltype(H)
    dof = size(H, 1)

    Ml = PD.Ml
    ψml = Vector{Matrix{T}}(undef, 2N - 1)

    X0 = random_orbital(T, dof, PD, 1)
    ψml[1] = compute_cheb_recur(H, X0, coef[1:Ml[1]+1], E1, E2)
    for l = 2:N
        Xl = random_orbital(T, dof, PD, l)
        ψl1, U0, U1, U2 = compute_cheb_recur(H, Xl, coef[1:Ml[l-1]+1],
            E1, E2, true)
        ψml[2l-2] = copy(ψl1)
        ψml[2l-1] = compute_cheb_recur!(ψl1, H, U0, U1, U2,
            coef[Ml[l-1]+2:Ml[l]+1],
            E1, E2, false)
    end

    return ψml
end

function compute_wavefun(ham, cal_way, Cheb, EC::ECutoffML{N}) where {N}
    coef = Cheb.coef
    E1 = Cheb.E1
    E2 = Cheb.E2
    H = [S2_ham(iham, Val(cal_way), E1, E2) for iham in ham]
    T = eltype(H[1])

    ψml = Vector{Matrix{T}}(undef, 2N - 1)

    X0 = random_orbital(T, size(H[1], 1), EC, 1)
    ψml[1] = compute_cheb_recur(H[1], X0, coef, E1, E2)
    for l = 2:N
        Xl2 = random_orbital(T, size(H[l], 1), EC, l)
        Xl1 = transfer_blochwave_kpt(Xl2, ham[l].basis, ham[l].kpoint, ham[l-1].basis, ham[l-1].kpoint)

        ψml[2l-2] = compute_cheb_recur(H[l-1], Xl1, coef, E1, E2)
        ψml[2l-1] = compute_cheb_recur(H[l], Xl2, coef, E1, E2)
    end

    return ψml
end

function compute_cheb_recur(H, U0, coef, E1, E2, Ureturn=false)
    m1, m2 = size(H, 1), size(U0, 2)
    TH = coef[1] * U0
    inv_E2 = inv(E2)
    SE2 = 2 * inv_E2

    U1 = similar(U0)
    U2 = similar(U0)
    S2_mul!(U1, H, U0, E1, inv_E2)
    mul!(TH, coef[2], U1, true, true)

    compute_cheb_recur!(TH, H, U0, U1, U2, coef[3:end], E1, E2, Ureturn)
end

function compute_cheb_recur!(TH, H, U0, U1, U2, coef,
    E1, E2, Ureturn::Bool)
    @assert size(coef, 2) == 1
    inv_E2 = inv(E2)
    SE2 = 2 * inv_E2

    for ic in coef
        # compute U2 = 2 * H * U1 - U0
        S2_mul!(U2, H, U1, E1, SE2)
        broadcast!(-, U2, U2, U0)
        mul!(TH, ic, U2, true, true)

        copy!(U0, U1)
        copy!(U1, U2)
    end

    if Ureturn
        return TH, U0, U1, U2
    else
        return TH
    end
end
