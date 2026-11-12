using Statistics
using Measures

using ImageFiltering
using LaTeXStrings
using CairoMakie


include("./plotting.jl")
include("./file_management.jl")
include("./variables_to_compute.jl")


# Approximate physical dimensions: 250 × 175 mm at 96 dpi.
const FIG_SIZE = (945, 661)

function initialize_histogram_insets!(
    fig::CairoMakie.Figure;
    selected_t = [1, 8, 15],
    inset_width = 0.27,
    inset_height = 0.19,
    yfrac = 0.72,
    xmargin = 0.08,
    gap = 0.025,
)
    W, H = FIG_SIZE
    inset_w = round(Int, inset_width * W)
    inset_h = round(Int, inset_height * H)

    inset_axes = Dict{Int, CairoMakie.Axis}()
    n = length(selected_t)

    total_width = n * inset_width + (n - 1) * gap
    start_x = (1 - total_width) / 2

    for (k, t) in enumerate(selected_t)
        xfrac = start_x + (k - 1) * (inset_width + gap)

        # Figure-relative position, expressed in pixels.
        x0 = round(Int, xfrac * W)
        y0 = round(Int, yfrac * H)

        ax = CairoMakie.Axis(
            fig.scene;
            bbox = CairoMakie.BBox(
                x0, y0,
                x0 + inset_w, y0 + inset_h,
            ),
            title = "t = $t",
            ylabel = k == 1 ? "N" : "",
            titlesize = 12,
            ylabelsize = 11,
            xticklabelsize = 10,
            yticklabelsize = 10,
            xscale = identity,
            yscale = identity,
            xgridvisible = false,
            ygridvisible = false,
            backgroundcolor = :white,
            spinewidth = 1,
        )

        if k > 1
            ax.yticksvisible = false
            ax.yticklabelsvisible = false
        end

        inset_axes[t] = ax
    end

    return inset_axes
end

function add_histogram_curves!(
    inset_axes::Dict{Int, CairoMakie.Axis},
    selected_t,
    centers,
    edge_counts,
    other_counts,
    Dr_label::String,
    color;
    sigma = 1.0,
)
    isempty(centers) && return inset_axes

    # Ensure that x coordinates increase from left to right.
    order = sortperm(centers)
    x = centers[order]

    # Restrict the insets to the central part of the system.
    i1 = clamp(floor(Int, 0.30 * length(x)), 1, length(x))
    i2 = clamp(ceil(Int, 0.70 * length(x)), 1, length(x))
    xlims_hist = (x[i1], x[i2])

    for t in selected_t
        hist_idx = t + 1

        isempty(edge_counts[hist_idx]) && continue
        isempty(other_counts[hist_idx]) && continue

        counts = edge_counts[hist_idx] .+ other_counts[hist_idx]
        smoothed = smooth_counts(counts; sigma)

        ax = inset_axes[t]

        CairoMakie.lines!(
            ax,
            x,
            smoothed[order];
            color = color,
            linewidth = 2,
        )

        CairoMakie.xlims!(ax, xlims_hist)
        CairoMakie.ylims!(ax, 0, nothing)
    end

    return inset_axes
end


function initialize_msd_figure(; selected_t = [1, 8, 15])
    fig = CairoMakie.Figure(size = FIG_SIZE, fontsize = 14, backgroundcolor = :white)

    gl = fig[1, 1] = CairoMakie.GridLayout()
    CairoMakie.rowsize!(gl, 1, CairoMakie.Fixed(0.25 * FIG_SIZE[2]))  # inset row
    # main axis in row 2
    ax = CairoMakie.Axis(gl[2, 1];
        xlabel = "Time, t",
        ylabel = L"\mathrm{MSD}/d^2",
        xscale = log10, yscale = log10,
        xlabelsize = 16, ylabelsize = 16,
        xticklabelsize = 13, yticklabelsize = 13,
        xgridvisible = true, ygridvisible = true,
        spinewidth = 1.2,
    )
    # insets in row 1, laid out horizontally
    inset_axes = initialize_histogram_insets!(fig; selected_t)
    return fig, ax, inset_axes
end

function load_msd_jld2(
    filename::AbstractString;
    vars = nothing,
)
    return jldopen(filename, "r") do file
        selected = isnothing(vars) ? keys(file) : string.(vars)

        Dict{Symbol, Any}(
            Symbol(name) => file[name]
            for name in selected
        )
    end
end

function plot_mean_parameters(
    main_folder::String,
    code_directory::String,
    Ls::Vector{Float64},
    inputf_orig::Vector{Int64},
)
    figs_folder = joinpath(main_folder, "mean_params_figs_various_L")
    mkpath(figs_folder)

    selected_t = [1, 8, 15]

    fig, ax, inset_axes = initialize_msd_figure(; selected_t)

    params = Params()

    for (which_L, L) in enumerate(Ls)
        inp = inputf_orig[which_L]

        inputs_file = joinpath(
            main_folder,
            "input_parameters$(inp)_test_wound7.txt",
        )

        params = Params()
        load_input_parameters!(inputs_file, params)
        load_model!(code_directory, params)

        Base.invokelatest(
            plot_with_model,
            main_folder,
            params,
            ax,
            inset_axes,
            which_L,
            selected_t,
        )
    end

    axislegend(
        ax;
        position = :rb,
        framevisible = false,
        labelsize = 12,
    )

    CairoMakie.save(
        joinpath(figs_folder, "msd2.pdf"),
        fig;
        pt_per_unit = 1,
    )

    return fig
end

function smooth_counts(counts::AbstractVector; sigma=1.0)
    radius = ceil(Int, 3sigma)
    offsets = -radius:radius
    kernel = exp.(-0.5 .* (offsets ./ sigma).^2)
    kernel ./= sum(kernel)

    n = length(counts)
    smoothed = similar(Float64.(counts))

    for i in 1:n
        total = 0.0
        weight_sum = 0.0

        for (k, offset) in enumerate(offsets)
            j = i + offset
            if 1 <= j <= n
                total += kernel[k] * counts[j]
                weight_sum += kernel[k]
            end
        end

        smoothed[i] = total / weight_sum
    end

    return smoothed
end


function plot_with_model(main_folder::String, params::Params, ax::Axis, inset_axes::Dict{Int, Axis}, L::Int64, selected_t::Vector{Int64})

    model = get_model()
    aux_force_variables = model_aux_structs(params, model)
    new_main_folder::String = load_input_folder(main_folder,params, model)
    println(new_main_folder)

    nseeds::Int64 = count(i->(occursin("output_part1.txt",i)),readdir(new_main_folder))+count(i->(occursin("output.txt",i)),readdir(new_main_folder))+count(i->(occursin("output_short_part1.txt",i)),readdir(new_main_folder))

    ctx = AnalysisContext()
    ctx.n_times_to_compute::Int64 = 15
    ctx.save_folder = new_main_folder

    colors = palette(:brg, 3)


    edge_counts = [[] for t in 0:15]
    other_counts = [[] for t in 0:15]
    centers = []

    times = [0.0 for t in 0:15]

    for s in 1:nseeds 
        println(params.seed)


        for t in 0:15 #n_times_to_compute
            csv_file = ctx.save_folder*"particle_binning$(params.seed)_t$t.csv"
            if !isfile(csv_file) || filesize(csv_file) == 0
                println("No particle binning file found inside folder $new_main_folder.")
                continue
            end

            df = CSV.read(csv_file, DataFrame)

            # Access individual columns
            centers = df.center
            if isempty(edge_counts[t+1])
                edge_counts[t+1] = df.edge_counts
                other_counts[t+1] = df.other_counts
            else
                edge_counts[t+1] .+= df.edge_counts
                other_counts[t+1] .+= df.other_counts
            end
            times[t+1] = df.t[1]

        end


        params.seed += 1 #(ojo con esto? conviene sacarlo del nombre del archivo mejor?)
        #GC.gc() #**ver con el profiler si conviene o si ayuda
    end

    data = load_msd_jld2(ctx.save_folder*"msd_data.jld2")#; vars = (:msd_all_seeds,),)
    data_wound = load_msd_jld2(ctx.save_folder*"msd_wound_data.jld2")#; vars = (:msd_edge_initial_all_seeds, :msd_bulk_initial_all_seeds,),)

    times = data[:times]
    msd_all = data[:msd_all_seeds]
    edge = data_wound[:msd_edge_initial_all_seeds]
    bulk = data_wound[:msd_bulk_initial_all_seeds]

    olors = Makie.wong_colors()

    if L == 1
        lines!(
            ax,
            times[3:end],
            0.2 .* times[3:end];
            linestyle = :dash,
            color = :black,
            linewidth = 2,
            label = "slope = 1",
        )

        lines!(
            ax,
            times[3:end],
            0.2 .* times[3:end] .^ 2;
            linestyle = :dot,
            color = :black,
            linewidth = 2,
            label = "slope = 2",
        )
    end

    τr_value = 2 / params.ηr^2
    τr_label = "τᵣ = $(round(τr_value, sigdigits=2))"

    CairoMakie.scatter!(
        ax,
        times[3:end],
        msd_all[3:end];
        label = τr_label,
        color = colors[L],
        alpha = 0.65,
        markersize = 8,
    )

    # Use the central 30–70% of the histogram bins.
    if !isempty(centers)
        i1 = clamp(floor(Int, 0.30 * length(centers)), 1, length(centers))
        i2 = clamp(ceil(Int, 0.70 * length(centers)), 1, length(centers))
        xlims_hist = (centers[i1], centers[i2])

        add_histogram_curves!(
            inset_axes,
            selected_t,
            centers,
            edge_counts,
            other_counts,
            τr_label,
            colors[L];
            sigma = 1.0,
            #xlims_hist,
        )
    end

    """
    for t in 0:15
        nbins = length(centers)

        dx = params.Lx / nbins
        # Overlay lightly smoothed curves
        sigma = 1.0  # smoothing width, measured in bins

        for (label, counts) in zip(["Dr= ],[edge_counts[t+1].+other_counts[t+1]]) #
            smoothed = smooth_counts(counts; sigma=1.0)

            plot!(
                figs_hist[t+1],
                centers,
                smoothed,
                linewidth = 2.5,
                label = label,
                color=colors[L], title = isnothing(times[t+1]) ? "Particle distribution" : "Particle distribution at t = "
            )
        end
    end


    """
end



Ls = [0.25, 1.0] #0.0001,

plot_mean_parameters(joinpath(@__DIR__, "prueba/"), joinpath(@__DIR__), Ls, [2,3]) #0,



