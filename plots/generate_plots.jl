using JLD2
using LaTeXStrings
using PythonPlot
const plt = PythonPlot
using SDFT
using SDFT: graphene_supercell

begin
    DATA_DIR = "plots/data"
    IMAGE_DIR = "plots/images"
    CASES = ["graphene", "graphene", "graphene"]
end

begin
    width = 426 / 72.27
    golden = (1 + 5^0.50) / 3
    figsize = (width, width / golden)
    matplotlib.rcParams["text.usetex"] = true
    matplotlib.rcParams["text.latex.preamble"] = raw"\usepackage{amsmath}\usepackage{amssymb}"
    linefit = (; ls="-.", color="#4F4F4F", alpha=0.7, linewidth=2)
end

begin
    plt.rc("font", family="serif")
    plt.rc("axes", titlesize=24, labelsize=20, grid=false)
    plt.rc("axes.spines", top=false, right=false)
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.rc("legend", fontsize=18, frameon=false)
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
        ax.scatter(xs, Var, marker=".", s=360, alpha=0.9, label="sDFT")
        ax.scatter(xs, VarT, marker="^", s=120, alpha=0.9, label="sDFT theory")
        ax.legend()
        ax.set_yscale("log")
        ax.set_xscale("log")
        xs2 = minimum(xs)-10:100:maximum(xs)+1000
        ax.plot(xs2, xs2; linefit...)
        ax.set_xlabel(L"N^2")
        ax.set_ylabel(L"{\mathbb{V}}[\phi(P,\chi,f^{\frac{1}{2}})]")

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
            ax.scatter(xs, err, label=L"N=%$ne"; neplot[j]...)
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