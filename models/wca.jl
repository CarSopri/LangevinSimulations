#=
    models/wca.jl
    ==============
    New file — nothing existing is modified. Follows the exact same
    shape as dummy_model.jl / your other models/*.jl files, so it drops
    into `load_model!` unchanged: set `params.model = "wca"` (or
    whatever you name this file) and everything else (get_forces_external!,
    the invokelatest wall in main_simulation, etc.) already knows what to
    do with it.

    WHAT THIS MODEL IS
    --------------------
    A Lennard-Jones potential, truncated and shifted at its own minimum
    (the standard "WCA" construction: Weeks-Chandler-Andersen), so the
    interaction is purely repulsive but the force is still continuous —
    it reaches exactly zero at the cutoff instead of being clipped.

        U(r) = 4ε[(σ/r)^12 - (σ/r)^6] + ε   for r < r_cut = 2^(1/6)σ
        U(r) = 0                             for r ≥ r_cut

    Differentiating gives the force actually used below — the "+ε" shift
    only affects U, not the force, so it doesn't appear in
    `get_force_prueba` at all (only relevant if you later want to track
    potential energy; see the note at the bottom).

    WHY a == b IS REQUIRED (AND HOW IT'S ENFORCED)
    -------------------------------------------------
    A single distance-dependent radial force has no natural notion of
    "which way an ellipse is pointing" — it only makes physical sense
    for round particles, where σ (the interaction diameter) is a single
    number `2a` instead of a function of the angle between the two
    particles' orientations and their separation. Rather than adding an
    `if params.a != params.b` branch inside the force calculation (which
    would silently compute *something* for ellipses, just not
    excluded-volume-correct), the check lives once, in
    `model_aux_structs` — the one place already guaranteed to run before
    a single force gets computed (see run_simulation / 1simulation.jl).
    A mismatched a/b throws immediately, with a message that says why,
    instead of a working-until-you-look-closely elliptical WCA imitation.

    NO TORQUE, BY CONSTRUCTION
    ----------------------------
    Since the force doesn't depend on either particle's orientation,
    `τ_i` and `τ_j` are always 0.0 — round particles don't rotate from
    this interaction (they can still rotate from noise/other forces
    elsewhere in the pipeline, this model just contributes nothing to
    that).

    WORKING IN r² (PERFORMANCE)
    ------------------------------
    Every quantity below is computed from r² (`sr2 = σ²/r²`, etc.) —
    no `sqrt` anywhere in the hot path. The force magnitude divided by r
    (`f_over_r`) is what's actually needed to get (fx,fy) = f_over_r*(dx,dy),
    so r itself is never needed at all.
=#

using StaticArrays
include(joinpath(@__DIR__, "..", "particle_objects.jl"))

struct WCAModel <: AbstractModel end
get_model() = WCAModel()

Base.@kwdef mutable struct GeometryVarsWCA <: GeometryVarsType
end

Base.@kwdef mutable struct NematicVarsWCA <: NematicVarsType
end

Base.@kwdef mutable struct AuxVarsWCA <: AuxVarsType
    disp::Float64 = 0.0
end

Base.@kwdef mutable struct ForceVarsWCA <: ForceVarsType
    τ_i::Float64 = 0.0
    fi_axis::MVector{2,Float64} = MVector(0.0, 0.0)
    fi::MVector{2,Float64} = MVector(0.0, 0.0)
end

Base.@kwdef mutable struct ConstantVarsWCA <: ConstantVarsType
    a::Float64 = Float64
    b::Float64 = Float64
    ε::Float64 = Float64

    viscosity_sum::Float64 = a+b
    viscosity_substract::Float64 = a-b

    diam::Float64 = 2*b


    # Interaction strength. Params has no generic "extra model parameter"
    # slot, so this is the one knob to edit by hand if you want a
    # stiffer/softer repulsion — change the default here, or pass
    # `ε=...` through to `ConstantVars(...)` in `model_aux_structs` below.
    #ε::Float64 = 12*ηa*ηa

    σ2::Float64 = diam^2          # σ = particle diameter = 2a (a==b enforced below)
    r_cut2::Float64 = 2.0^(1/3) * σ2 # (2^(1/6)σ)²

    potential_max_threshold::Float64 = Float64
    force_max_value::Float64 = 24*ε*(potential_max_threshold^(1/6)/diam)*(2*potential_max_threshold*potential_max_threshold-potential_max_threshold) # 1.99e4

end


function model_aux_structs(params::Params, model::WCAModel)
    @assert params.a ≈ params.b "El modelo wca (volumen excluido) sólo está definido para partículas redondas: se necesita params.a == params.b."
    aux_force_variables = ModelWorkspace(GeometryVarsWCA(), NematicVarsWCA(), AuxVarsWCA(), ForceVarsWCA(), ConstantVarsWCA(a=params.a, b=params.b, ε=2*params.b*params.c/(24*params.μa), potential_max_threshold=(0.5*params.b/params.c/params.Δt)^(6/13)))
    return aux_force_variables
end

function make_model_specific_subfolder_name(params::Params, force_constants::ConstantVarsWCA, subfolder_name::String, model::WCAModel)
    subfolder_name = "$(subfolder_name)_eps_$(round(force_constants.ε,sigdigits=2))_etar_$(round(params.ηr,sigdigits=2))"
    return subfolder_name
end

function make_gif_title(params::Params, force_constants::ConstantVarsWCA, model::WCAModel)
    fig_ellipse_title::String = "np=$(params.np), v0=$(round(params.c,sigdigits=2)), ε=$(round(force_constants.ε,sigdigits=2)), η=$(round(params.ηr,sigdigits=2))"
    return fig_ellipse_title
end


function get_force_prueba(params::Params, ::WCAModel,
                        geom_ws::GeometryVarsWCA, nematic_ws::NematicVarsWCA, aux_ws::AuxVarsWCA, force_ws::ForceVarsWCA, constant_ws::ConstantVarsWCA,
                        pos_i::MVector{2,Float64}, angle_i::Float64, pos_j::MVector{2,Float64}, angle_j::Float64,
                        cos_i::Float64, sin_i::Float64)

    # dx,dy = pos_i - pos_j, periodic (minimum image), via the same
    # d_ij_periodic already defined in model_forces.jl.
    dx = d_ij_periodic(pos_j[1], pos_i[1], params.Lx)
    dy = d_ij_periodic(pos_j[2], pos_i[2], params.Ly)
    r2 = dx*dx + dy*dy

    if r2 == 0.0 || r2 >= constant_ws.r_cut2
        force_ws.fi_axis[1] = 0.0 #**se supone que debería ser mejor usar fx pero por algún motivo el profiler me tira consistentemente que esto es más rápido idk
        force_ws.fi_axis[2] = 0.0
        force_ws.τ_i = 0.0
        τ_j = 0.0
        return force_ws.fi_axis[1], force_ws.fi_axis[2], force_ws.τ_i, τ_j
    end

    sr2  = constant_ws.σ2 / r2
    sr6  = sr2*sr2*sr2

    if sr6>constant_ws.potential_max_threshold
        #adim_r=10
        f_over_r = constant_ws.force_max_value

        inv_r = 1/sqrt(r2)
        force_ws.fi_axis[1] = f_over_r*dx*inv_r
        force_ws.fi_axis[2] = f_over_r*dy*inv_r
    else
        sr12 = sr6*sr6
        # F(r)/r, so that (fx,fy) = f_over_r .* (dx,dy) without ever taking a sqrt.
        f_over_r = 24*constant_ws.ε*(2*sr12 - sr6) / r2

        force_ws.fi_axis[1] = f_over_r*dx
        force_ws.fi_axis[2] = f_over_r*dy
    end

    force_ws.τ_i = 0.0 # no orientation dependence -> no torque
    τ_j = 0.0

    return force_ws.fi_axis[1], force_ws.fi_axis[2], force_ws.τ_i, τ_j
end


function MSD_theory(times::Vector{Float64}, force_constants::ConstantVarsType, params::Params, model::WCAModel)
    ηr2 = params.ηr*params.ηr
    ηa2 = params.ηa*params.ηa
    ηb2 = params.ηb*params.ηb
    if params.ηr>0.0
        return (ηa2+ηb2).*times .+ 8*params.c*params.c/(ηr2*ηr2).*(0.5*ηr2.*times.-1.0.+exp.((-0.5*ηr2).*times))
    else
        return (ηa2+ηb2).*times .+ params.c*params.c .*(times.*times)
    end
end


function MSD_passive_theory(times::Vector{Float64}, msd::Vector{Float64}, force_constants::ConstantVarsType, params::Params, model::WCAModel)
    mask = isfinite.(times) .& isfinite.(msd) .& (msd.> 0)
    logt = log10.(times[mask])
    logmsd = log10.(msd[mask])

    @. loglog_linear(x, p) = p[1] + p[2] * x

    p0 = [1.0, 1.0]

    lower = [-Inf, 0.0]
    upper = [Inf,  Inf]

    fit = curve_fit(loglog_linear, logt, logmsd, p0, lower=lower, upper=upper)
    a, n = fit.param

    # Fitted width evaluated at all original times
    println("Fitted exponent for msd(t): ", n)

    theory_log = loglog_linear.(log10.(times), Ref(fit.param))
    theory = 10 .^ theory_log

    return theory, n

end

function width_theory(times::Vector{Float64}, width::Vector{Float64}, close_time::Float64, force_constants::ConstantVarsType, params::Params, model::WCAModel)
    # Only positive, finite widths can be used in the exponential fit
    mask = isfinite.(times) .& isfinite.(width) .& (width .> 0) .& (times.<close_time)

    t = times[mask]
    w = width[mask]
    w0 = w[1]

    # Exponential model: w(t) = A * exp(-k*t)
    #exponential(t, p) = w0 .* exp.(-2*p[1]/w0 .* t)
    exponential(t, p) = p[1] .* exp.(-p[2] .* t)

    # Initial guesses
    p0 = [ w0, 1 / (t[end] - t[1])]

    fit = curve_fit(exponential, t, w, p0)

    #v_edge = fit.param
    A, k = fit.param

    # From w(t) = w0 exp(-2 v_edge t / w0)
    v_edge = A * k / 2

    # Fitted width evaluated at all original times
    width_fit = exponential(times, fit.param)

    return width_fit, v_edge
end

#=
    NOT NEEDED BY THE CURRENT PIPELINE, HERE FOR REFERENCE ONLY:
    if you ever revive the commented-out `upot` bookkeeping in
    1simulation.jl, the shifted potential for this model is

        U(r) = 4ε*(sr12 - sr6) + ε    for r < r_cut, else 0

    i.e. exactly `4*constant_ws.ε*(sr12-sr6) + constant_ws.ε` using the
    same sr6/sr12 computed above.

    OPTIONAL FOLLOW-UP (not needed for this to work, just an efficiency
    note): 1simulation.jl currently sets `params.rc` with
    `params.model == "vandamme" ? ... : 5.0*params.a`, so this model
    falls into the `5.0*params.a` branch. That's comfortably larger than
    r_cut = 2^(1/6)*2a ≈ 2.24a, so neighbor lists will be correct — just
    slightly bigger than strictly necessary (a few more candidate pairs
    checked per particle than needed). Only worth tightening if neighbor
    list overhead shows up in a profile.
=#
