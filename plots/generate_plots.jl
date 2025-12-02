using JLD2
using LaTeXStrings
using PythonPlot
const plt = PythonPlot
using SDFT

begin
    DATA_DIR = "plots/data"
    IMAGE_DIR = "plots/images"
    CASES = ["graphene", "stone_wales", "doping"]
end

begin
    width = 426 / 72.27
    golden = (1 + 5^0.50) / 3
    figsize = (width, width / golden)
    matplotlib.rcParams["text.usetex"] = true
    matplotlib.rcParams["text.latex.preamble"] = raw"\usepackage{amsmath}\usepackage{amssymb}\usepackage{upgreek}"
    linefit = (; ls="-.", color="#4F4F4F", alpha=0.7, linewidth=2)
end

begin
    plt.rc("font", family="serif")
    plt.rc("axes", titlesize=24, labelsize=22, grid=false)
    plt.rc("axes.spines", top=false, right=false)
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.rc("legend", fontsize=22, frameon=false)
    plt.rc("lines", linewidth=1.5)
    plt.rc("savefig", dpi=500)
    plt.rc("text", usetex="True")
end

sdft_var = let
    for (i, ic) in enumerate(CASES)
        # variance
        files = let
            files = readdir(DATA_DIR)[1:end]
            #files = filter(e -> (contains(e, "dos")), files)
            files = filter(e -> contains(e, "variance_$(ic)"), files)
            joinpath.(DATA_DIR, files)
        end
        data = load(files[1])
        Ne = data["Ne"]
        ind = Int[] 
        for element in unique(Ne)
            index = findfirst(==(element), Ne)
            push!(ind, index)
        end
        Ne = Ne[ind]
        Var = data["Var"][ind]
        VarT = data["VarT"][ind]
        xs = Ne.^2

        fig, ax = plt.subplots(figsize=(figsize[1], figsize[2]))
        ax.scatter(xs, Var, marker=".", s=390, alpha=0.9, label="sDFT")
        ax.scatter(xs, VarT, marker="^", s=130, alpha=0.9, label="sDFT prediction")
        ax.legend()
        ax.set_yscale("log")
        ax.set_xscale("log")
        xs2 = minimum(xs)-10:100:maximum(xs)+1000
        ax.plot(xs2, xs2; linefit...)
        ax.set_xlabel(L"N^2")
        ax.set_ylabel(L"{\mathbb{V}}[\upphi(P,\chi,f^{\frac{1}{2}})]")

        if i == 1
            title_c = "Supercell"
        elseif i == 2
            title_c = "Stone-Wales"
        elseif i == 3
            title_c = "Doping"
        end
        #ax.set_title("$(title_c)")


        fig.tight_layout()
        filename = joinpath(IMAGE_DIR, "sdft_var_$i.pdf")
        fig.savefig(filename, bbox_inches="tight")
        println("Saved plot: $filename")
    end
end

sdft_var_combined = let
    sdft = (; marker=".", s=360, alpha=0.9, label="sDFT", color="C0")
    sdft_theory = (; marker="^", s=120, alpha=0.9, label="sDFT theory", color="C1")

    num_case = 3

    fig, axs = plt.subplots(1, num_case;
        figsize=(figsize[1] * num_case, figsize[2]),
        sharey=false,
    )
    flat_axs = axs.flatten()
    local handles, labels

    for (i, ic) in enumerate(CASES)
        py_index = i - 1
        ax = axs.__getitem__(py_index)

        files = let
            files = readdir(DATA_DIR)[1:end]
            #files = filter(e -> (contains(e, "dos")), files)
            files = filter(e -> contains(e, "variance_$(ic)"), files)
            joinpath.(DATA_DIR, files)
        end
        data = load(files[1])
        Ne = data["Ne"]
        ind = Int[]
        for element in unique(Ne)
            index = findfirst(==(element), Ne)
            push!(ind, index)
        end
        Ne = Ne[ind]
        Var = data["Var"][ind]
        VarT = data["VarT"][ind]
        xs = Ne .^ 2

        ax.scatter(xs, Var; sdft...)
        ax.scatter(xs, VarT; sdft_theory...)

        ax.set_yscale("log")
        ax.set_xscale("log")
        xs2 = minimum(xs)-10:100:maximum(xs)+1000
        ax.plot(xs2, xs2; linefit...)
        ax.set_xlabel(L"N^2")

        if i == 1
            title_c =  "Supercell"
        elseif i == 2
            title_c =  "Stone-Wales"
        elseif i == 3
            title_c =  "Doping"
        end
        ax.set_title("$(title_c)")

        ax.set_xlabel(L"N^2")

        if i == 1
            handles, labels = ax.get_legend_handles_labels()
        end
    end

    axs.__getitem__(0).set_ylabel(L"{\mathbb{V}}[\phi(P,\chi,f^{\frac{1}{2}})]")
    fig.legend(handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.1),
    )

    fig.tight_layout()
    filename = joinpath(IMAGE_DIR, "sdft_variance_combined.pdf")
    fig.savefig(filename, bbox_inches="tight")
    println("Saved plot: $filename")
end

sdft_density_error = let
    neplot = [(; marker="v", s=100, alpha=0.9, color="C0"),
              (; marker="D", s=100, alpha=0.9, color="C1"),
              (; marker="*", s=120, alpha=0.9, color="C2")]

    for (i, ic) in enumerate(CASES)
        # variance
        files = let
            files = readdir(DATA_DIR)[1:end]
            #files = filter(e -> (contains(e, "dos")), files)
            files = filter(e -> contains(e, "density_$(ic)"), files)
            joinpath.(DATA_DIR, files)
        end
        data = load(files[1])

        ind = 2:4
        Ne = data["Ne"][ind]
        Ns = data["Ns"]
        Err = data["Error"][ind]

        fig, ax = plt.subplots(figsize=(figsize[1], figsize[2]))
        for (j, ne) in enumerate(Ne)
            err = Err[j]
            xs = @. sqrt(Ne[j]) / sqrt(Ns)
            ns = j+1
            ax.scatter(xs, err, label=L"%$(ns) \times %$(ns)"; neplot[j]...)
        end

        xxs = vcat([@. sqrt(Ne[i]) / sqrt(Ns) for i = 1:3]...)
        sort!(unique!(xxs))
        err = Err[end]
        xs = @. sqrt(Ne[end]) / Ns .^ (1 / 2)
        ks = [(err[j+1] - err[j]) / (xs[j+1] - xs[j]) for j = 1:length(xs)-1]
        k = sum(ks) / length(ks) + 0.05
        fx(x) = k * (x - xs[end]) + err[end] - 0.01
        ax.plot(xxs, fx.(xxs); linefit...)


        ax.legend()
        ax.set_xlabel(L"(N/|\mathbb{S}|)^{1/2}")
        ax.set_ylabel(L"\Delta\rho")

        if i == 1
            title_c = "Supercell"
        elseif i == 2
            title_c = "Stone-Wales"
        elseif i == 3
            title_c = "Doping"
        end
        #ax.set_title("$(title_c)")


        fig.tight_layout()
        filename = joinpath(IMAGE_DIR, "sdft_error_$i.pdf")
        fig.savefig(filename, bbox_inches="tight")
        println("Saved plot: $filename")
    end
end

mlmc_pd_cost = let
    pdplot = [(; marker="v", s=100, alpha=0.9, color="C0"),
              (; marker="D", s=100, alpha=0.9, color="C1"),
              (; marker="*", s=120, alpha=0.9, color="C2")]

    for (i, ic) in enumerate(CASES)
        # variance
        files = let
            files = readdir(DATA_DIR)[1:end]
            #files = filter(e -> (contains(e, "dos")), files)
            files = filter(e -> contains(e, "mlmcpd_cost_$(ic)"), files)
            joinpath.(DATA_DIR, files)
        end
        data_cost = load(files[1])

        Nefull = data_cost["Ne_mat"][:, 1, 1]
        ind = indexin(unique(Nefull), Nefull)
        Ecuts = data_cost["Ecuts"]
        Ne = data_cost["Ne_mat"][ind, :, :]
        ns = data_cost["ns_mat"][ind, :, :]
        Ms = data_cost["Ms_mat"][ind, :, :]
        pd_t = data_cost["mlmc_time"][ind, :, :]
        mc_t = data_cost["mc_time"][ind, :, :]
        temperatures = data_cost["temperatures"]
        #M0 = data_cost["Ql_mat"][1][1]

        fig, ax = plt.subplots(figsize=(figsize[1], figsize[2]))
        for (j, tp) in enumerate(temperatures)
            xs = Ne[:, :, j] .* ns[:, :, j]
            ib = Int(log10(inv(tp)))
            ax.scatter(xs, pd_t[:, :, j], label=L"\beta=10^%$ib"; pdplot[j]...)
        end

        xxs = sort(unique(Ne .*  ns))
        if i == 1
            k = 1.3
        elseif i == 2
            k = 1.1
        elseif i == 3
            k = 0.7
        end
        fx(x) = x / xxs[1] * (k * pd_t[1,1,1])
        ax.plot(xxs, fx.(xxs); linefit...)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(10^(log10(xxs[1])-0.2), 10^(log10(xxs[end])+0.2))
        ax.legend()
        ax.set_xlabel(L"nN")
        ax.set_ylabel("Cost")

        if i == 1
            title_c = "Supercell"
        elseif i == 2
            title_c = "Stone-Wales"
        elseif i == 3
            title_c = "Doping"
        end
        #ax.set_title("$(title_c)")

        fig.tight_layout()
        filename = joinpath(IMAGE_DIR, "pdml_cost_$i.pdf")
        fig.savefig(filename, bbox_inches="tight")
        println("Saved plot: $filename")
    end
end

mlmc_ec_cost = let
    ecplot = [(; marker="v", s=100, alpha=0.9, color="C0"),
        (; marker="D", s=100, alpha=0.9, color="C1"),
        (; marker="*", s=120, alpha=0.9, color="C2")]

    for (i, ic) in enumerate(CASES)
        # variance
        files = let
            files = readdir(DATA_DIR)[1:end]
            #files = filter(e -> (contains(e, "dos")), files)
            files = filter(e -> contains(e, "mlmcec_cost_$(ic)"), files)
            joinpath.(DATA_DIR, files)
        end
        data_cost = load(files[1])

        Nefull = data_cost["Ne_mat"][:, 1, 1]
        ind = indexin(unique(Nefull), Nefull)
        Ecuts = data_cost["Ecuts"]
        Ne = data_cost["Ne_mat"][ind, :, :]
        ns = data_cost["ns_mat"][ind, :, :]
        Ms = data_cost["Ms_mat"][ind, :, :]
        ec_t = data_cost["mlmc_time"][ind, :, :]
        mc_t = data_cost["mc_time"][ind, :, :]

        #=
        N12s = [[n1, n2] for n1 in 1:5 for n2 in 1:n1]
        N12s = N12s[findall(x -> prod(x) <= 10, N12s)][ind]
        n0 = similar(ns)
        Qls = data_cost["Ql_mat"]
        take_dof(basis) = length(basis.kpoints[1].mapping)
        for k = 1:length(N12s), l = 1:length(Ecuts)
            basis = graphene_setup(N12s[k]; Ecut=Qls[k, l][1])
            n0[k, l, 1] = take_dof(basis)
        end
        =#

        fig, ax = plt.subplots(figsize=(figsize[1], figsize[2]))
        for (ei, ecut) in enumerate(Ecuts)
            xs = Ne[:, ei, :] .^ 2 .* Ms[:, ei, :]
            ax.scatter(xs, ec_t[:, ei, :], label=L"E_{\rm c}=%$ecut"; ecplot[ei]...)
        end

        xxs = sort(unique(Ne .^2 .* Ms))
        if i == 1
            ks = 1.2
        elseif i == 2
            ks = 1.3
        elseif i == 3
            ks = 0.9
        end
        fx(x) = (x / xxs[1]) * (ks * ec_t[1])
        ax.plot(xxs, fx.(xxs); linefit...)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(10^(log10(xxs[1]) - 0.2), 10^(log10(xxs[end]) + 0.2))
        ax.set_xticks(10 .^[4.5, 5.5, 6.5])
        ax.set_xticklabels([L"10^{4.5}", L"10^{5.5}", L"10^{6.5}"])
        ax.legend()
        ax.set_xlabel(L"N^2M")
        ax.set_ylabel("Cost")

        if i == 1
            title_c = "Supercell"
        elseif i == 2
            title_c = "Stone-Wales"
        elseif i == 3
            title_c = "Doping"
        end
        #ax.set_title("$(title_c)")

        fig.tight_layout()
        filename = joinpath(IMAGE_DIR, "ecml_cost_$i.pdf")
        fig.savefig(filename, bbox_inches="tight")
        println("Saved plot: $filename")
    end
end


mlmc_pd_var = let
    for (i, ic) in enumerate(CASES)
        # variance
        files = let
            files = readdir(DATA_DIR)[1:end]
            #files = filter(e -> (contains(e, "dos")), files)
            files = filter(e -> contains(e, "mlmcpd_cost_$(ic)"), files)
            joinpath.(DATA_DIR, files)
        end
        data = load(files[1])
        var = data["var_mat"][end]
        L = length(var[1]) - 1
        xs = 0:L

        fig, ax = plt.subplots(figsize=(figsize[1], figsize[2]))
        ax.plot(xs, var[2], linestyle="--", 
                linewidth=2, 
                marker="o", 
                markersize=11,
                label=L"\mathbb{V}[\widehat{\upphi}^{(\ell)}_\chi]")
        ax.plot(xs, var[1], linestyle="-",
                linewidth=2,
                marker="^",
                markersize=11,
                label=L"\mathbb{V}[\widehat{\upphi}^{(\ell)}_\chi- \widehat{\upphi}^{(\ell-1)}_\chi]")
        ax.legend()
        ax.set_yscale("log")
        ax.set_xlabel(L"\ell")
        ax.set_ylabel("Variance")

        if i == 1
            title_c = "Supercell"
        elseif i == 2
            title_c = "Stone-Wales"
        elseif i == 3
            title_c = "Doping"
        end
        #ax.set_title("$(title_c)")


        fig.tight_layout()
        filename = joinpath(IMAGE_DIR, "mlmc_pd_var_$i.pdf")
        fig.savefig(filename, bbox_inches="tight")
        println("Saved plot: $filename")
    end
end


mlmc_ec_var = let
    for (i, ic) in enumerate(CASES)
        # variance
        files = let
            files = readdir(DATA_DIR)[1:end]
            #files = filter(e -> (contains(e, "dos")), files)
            files = filter(e -> contains(e, "mlmcec_cost_$(ic)"), files)
            joinpath.(DATA_DIR, files)
        end
        data = load(files[1])
        var = data["var_mat"][end]
        L = length(var[1]) - 1
        xs = 0:L

        fig, ax = plt.subplots(figsize=(figsize[1], figsize[2]))
        ax.plot(xs, var[2], linestyle="--",
            linewidth=2,
            marker="o",
            markersize=11,
            label=L"\mathbb{V}[\widehat{\upphi}^{(\ell)}_\chi]")
        ax.plot(xs, var[1], linestyle="-",
            linewidth=2,
            marker="^",
            markersize=11,
            label=L"\mathbb{V}[\widehat{\upphi}^{(\ell)}_\chi- \widehat{\upphi}^{(\ell-1)}_\chi]")
        ax.legend()
        ax.set_yscale("log")
        ax.set_xlabel(L"\ell")
        ax.set_ylabel("Variance")

        if i == 1
            title_c = "Supercell"
        elseif i == 2
            title_c = "Stone-Wales"
        elseif i == 3
            title_c = "Doping"
        end
        #ax.set_title("$(title_c)")


        fig.tight_layout()
        filename = joinpath(IMAGE_DIR, "mlmc_ec_var_$i.pdf")
        fig.savefig(filename, bbox_inches="tight")
        println("Saved plot: $filename")
    end
end