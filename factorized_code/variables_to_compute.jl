using Random
using Distributions #MvNormal en noise
using StaticArrays 
using LinearAlgebra
using BioStatPhys
using Statistics
using JLD2


include("./particle_objects.jl")


#***además de gc, tengo que cerrar todos los plots como hago en pyton mepa


#**capaz me conviene separar por tipo de observables que usen distintas funciones ? tipo las que se calculan para todo t y se promedian por seeds, las que se usan para histogramas, las que se plotean para algunos t y se promedian por seeds.
#**crosscorrel es como msdev, y todas las velocidades son auxiliares que van a un histograma para tv, mientras que los nongaussian se promedian para algunos specific times

#**hay muchos de estos que guardo en un vector de vectores afuera del loop, para poder plotear promedios en seed, así que tener en cuenta !!! no sé si no debería ser una función extra que vaya guardando cada elemento?

"""
esto está adentro del loop en k y en for line in open(file):
if analyize_velocities #**ver bien esto xq no sé si es que solo guardo el último t ?
                    empty!(v_tot)
                    empty!(vx_tot)
                    empty!(vy_tot)
                    empty!(vpar_tot)
                    empty!(vperp_tot)
                end
"""

#***me está faltando times_non_gaussian

#(v_tot::Vector{Float64},vx_tot::Vector{Float64},vy_tot::Vector{Float64},vpar_tot::Vector{Float64},vperp_tot::Vector{Float64},
# kurtosis_all::Vector{Float64},skewness_all::Vector{Float64},kurtosis_allx::Vector{Float64},skewness_allx::Vector{Float64},kurtosis_ally::Vector{Float64},skewness_ally::Vector{Float64},
# kurtosis_all_par::Vector{Float64},skewness_all_par::Vector{Float64},kurtosis_all_perp::Vector{Float64},skewness_all_perp::Vector{Float64},
# cross_correl::Vector{Float64},times_nongaussian::Vector{Float64}, times::Vector{Float64},tv::Real, checkpoints_file::AbstractString)

const NULL_CHAR = "\0"


@inline fastparse(x::Float64) = x
@inline fastparse(x::AbstractString)  = parse(Float64, x)


abstract type AbstractObservable end
"""
# Each observable implements:

initialize!(obs, np, nseeds)

for s in nseeds
    files = {file1_seed_s, file2_seed_s, ..., filek_output_seed_s}

    initialize_per_file!(obs, ctx, n_total_rows) 
    initialize_specific_times!(obs, ctx, n_times_to_compute) (n_times_to_compute depende del tamaño del archivo)

    for file_k in files
        open file
            for line in file 

                if line is corrupted
                    error_handling!(obs, ctx) 
                    continue
                end

                begin_timestep!(obs, ctx)
                begin_timestep_specific_times!(obs, ctx)

                for particle in line

                    #***estas dos funciones podrían ser una sola donde se toma de input un ::Val{True} que hace una cosa (specific_times) y otra con ::Val{False} usando multiple dispatch que haga otra, o que directamente ignore ese input
                    
                    accumulate!(obs, particle, particle0, particle_unwrapped, particle) #**chequear si convendría hacer así además de pasar ctx
                    accumulate_specific_times!(obs, particle, particle0, particle_unwrapped, particle) #**chequear si convendría hacer así además de pasar ctx

                end

                finish_timestep!(obs, ctx, t)
                finish_timestep_specific_times!(obs, ctx, t)
            end
        end
    end

    finish_seed_stats!(obs, ctx, s)
end

finish_loop_in_seed!(obs, ctx, nseeds)

plot_everything(obs, ctx, save_folder)

"""



#***capaz ctx debería tener params?
# así me evito guardar seed y np, inv_Lx y inv_Ly los saco de ahí directamente idk
# y podría tener el modelo guardado acá también, para poder agregar cálculos teóricos
# ** probablemente acá tenga que agregar las cosas que se calculan en 3 y 4 !! (correlaciones)
#**también creo que tendría que eliminar todas las funciones de specific_times y pasarlo como variable cuando lo uso solamente, 
# donde if false corre para todo t, if true corre sólo para algunos 
Base.@kwdef mutable struct AnalysisContext #**este se inicializa en initialize_context
    particles::Vector{ParticleData} = Vector{ParticleData}() #***creeeo que conviene que sea un struct de Particles, que sería más eficiente que tener un vector de structs !!
    particles_unwrapped::Vector{ParticleData} = Vector{ParticleData}()
    particles_initial_config::Vector{ParticleData} = Vector{ParticleData}()

    row::Vector{SubString{String}} = Vector{SubString{String}}()

    t::Int64 = 0
    time_val::Float64 = 0.0

    tv::Float64 = 0 
    t_split::Int64 = 0
    n_times_to_compute::Int64 = 0

    times::Vector{Float64} = Vector{Float64}()
    specific_times::Vector{Float64} = Vector{Float64}()

    n_complete_rows::Int64 = 0 #****cambiar nombre xq nunca puedo distinguirla de n_complete_rows !!
    np::Int64 = 0
    
    inv_Lx::Float64 = 0.0 #***esto puede salir de params ahora para cuando lo necesito, guardar en ese obs solamente
    inv_Ly::Float64 = 0.0

    seed::Int64 = 0

    params::Params = Params()
    force_constants = ConstantVarsType
    model = AbstractModel

    save_folder::String = ""
end



function save_msd_jld2(obs::AbstractObservable, ctx::AnalysisContext, filename::AbstractString; obs_vars = (), ctx_vars = (),)
    data = Dict{Symbol, Any}()

    for var in obs_vars
        hasfield(typeof(obs), var) ||
            throw(ArgumentError("Observable $(typeof(obs)) has no field :$var")) #***printear todos los que sí están? en vez de tirar error ?
        data[var] = getfield(obs, var)
    end

    for var in ctx_vars
        hasfield(typeof(ctx), var) ||
            throw(ArgumentError("AnalysisContext has no field :$var"))
        data[var] = getfield(ctx, var)
    end

    mkpath(dirname(abspath(filename)))

    jldopen(filename, "w") do file
        for (name, value) in data
            file[string(name)] = value
        end
    end

    return filename
end




Base.@kwdef mutable struct MSD <: AbstractObservable
    values::Vector{Float64} = Vector{Float64}()
    msd::Float64 = 0.0

    msd_all_seeds::Vector{Float64} = Vector{Float64}()
    fig_msd::Plots.Plot = Plots.Plot()
end

function initialize!(obs::MSD, np::Int64, nseeds::Int64)
    obs.fig_msd = plot(layout = (1, 2), size = (1200, 500), legend=:outertopright)
end
function initialize_specific_times!(obs::MSD, ctx::AnalysisContext, n_times_to_compute::Int64)
end
function initialize_per_file!(obs::MSD, ctx::AnalysisContext, n_total_rows::Int64)
    obs.values = Vector{Float64}(undef,n_total_rows) 

    if length(obs.msd_all_seeds)<n_total_rows
        obs.msd_all_seeds = vcat(obs.msd_all_seeds, zeros(n_total_rows-length(obs.msd_all_seeds)))
    end
end

function begin_timestep!(obs::MSD, ctx::AnalysisContext)
    obs.msd = 0.0
end
function begin_timestep_specific_times!(obs::MSD, ctx::AnalysisContext)
end

function error_handling!(obs::MSD, ctx::AnalysisContext) #***pensar si es mejor usar ctx.t directamente o si pasarle i=t desde afuera
    #obs.values[ctx.t] = NaN
end

function accumulate!(obs::MSD, ctx::AnalysisContext, i::Int64) #**será muy ineficiente llamar adentro de dos structs?
    obs.msd += (ctx.particles_unwrapped[i].x-ctx.particles_initial_config[i].x)^2 + (ctx.particles_unwrapped[i].y-ctx.particles_initial_config[i].y)^2
end
function accumulate_specific_times!(obs::MSD, ctx::AnalysisContext, i::Int64) 
end

function finish_timestep!(obs::MSD, ctx::AnalysisContext, t::Int64)
    obs.values[t] = obs.msd / ctx.np
end
function finish_timestep_specific_times!(obs::MSD, ctx::AnalysisContext, t::Int64)
end

function finish_seed_stats!(obs::MSD, ctx::AnalysisContext, seed::Int64)
    plot!(obs.fig_msd[1],ctx.times,obs.values, label="seed=$(seed)", alpha=0.5) #[t*real_t+1 for t in 1:length(msdev)]

    obs.msd_all_seeds .+= obs.values
end

function finish_loop_in_seed!(obs::MSD, ctx::AnalysisContext, nseeds::Int64)
    obs.msd_all_seeds ./= nseeds   
end

#***acá tendría que poder sacar el modelo de ctx para poder plotear la cuenta teórica?
# tendría que agregar una función MSD_theory(model_constants, times, msd) y label capaz
function plot_everything(obs::MSD, ctx::AnalysisContext, save_folder::String)
    times = ctx.times
    msd   = obs.msd_all_seeds

    valid = isfinite.(times) .& isfinite.(msd) .& (times .> 0.0) .& (msd .> 0)

    t_valid = times[valid]
    msd_valid = msd[valid]

    scatter!(obs.fig_msd[1],times,msd, label="all", color=:black, alpha=0.5, linewidth=3) #[t*real_t+1 for t in 1:length(msdev_all)]

    theory = nothing

    try
        if ctx.params.activity == false
            theory, n = MSD_passive_theory(times, msd, ctx.force_constants, ctx.params, ctx.model)

            valid_theory = isfinite.(theory) .& (times .> 0.0) .& (theory.> 0)
            println(t_valid,theory[valid_theory ])
            plot!(obs.fig_msd[2],times[valid_theory],(theory[valid_theory]), label="t^"*string(round(n, sigdigits=2)), color=:red, alpha=0.5, linewidth=3)
        else
            theory = MSD_theory(times, ctx.force_constants, ctx.params, ctx.model)
            plot!(obs.fig_msd[2],t_valid,theory[valid], label="theory", color=:red, alpha=0.5, linewidth=3)
        end
    catch
    end


    xlabel!(obs.fig_msd[1], "t")
    ylabel!(obs.fig_msd[1], "MSD")

    scatter!(obs.fig_msd[2],t_valid,msd_valid, label="all", color=:black, alpha=0.5, linewidth=3, xscale = :log10, yscale = :log10) #[t*real_t+1 for t in 1:length(msdev_all)]
    

    if !isempty(t_valid)
        tmin = minimum(t_valid)
        tmax = maximum(t_valid)

        # Anchor both reference lines around the middle
        t0 = sqrt(tmin * tmax)
        y0 = exp((log(minimum(msd_valid)) + log(maximum(msd_valid))) / 2)
        tref = [tmin, tmax]

        y1 = y0 .* (tref ./ t0)
        y2 = y0 .* (tref ./ t0).^2

        plot!(obs.fig_msd[2], tref, y1, linestyle = :dash, color = :blue, linewidth = 2, label = "slope = 1")

        plot!(obs.fig_msd[2], tref, y2, linestyle = :dot, color = :green, linewidth = 2, label = "slope = 2")
    end

    xlabel!(obs.fig_msd[2], "log t")
    ylabel!(obs.fig_msd[2], "log MSD")

    
    display(obs.fig_msd)
    savefig(obs.fig_msd,save_folder*"msd.png")

    save_msd_jld2(obs, ctx, ctx.save_folder*"msd_data.jld2"; obs_vars = (:msd_all_seeds,), ctx_vars = (:times,))
end



Base.@kwdef mutable struct TimeVal <: AbstractObservable #**creo que este obs sería obligatorio porque es parte del loop principal
end

function initialize!(obs::TimeVal, np::Int64, nseeds::Int64)
end
function initialize_per_file!(obs::TimeVal, ctx::AnalysisContext, n_total_rows::Int64)
    ctx.t = 1
    ctx.time_val = 1.0 

    println(n_total_rows, ctx.n_times_to_compute)
    ctx.n_times_to_compute = n_total_rows÷(n_total_rows÷ctx.n_times_to_compute)
    ctx.t_split = maximum([1,floor(Int,n_total_rows/ctx.n_times_to_compute)])

    println(ctx.t_split," a ", ctx.tv,"  ",ctx.n_times_to_compute)
    ctx.times = Vector{Float64}(undef,n_total_rows) #****creo que tengo que hacer como con msd que si algún seed tiene nrows más grande, se acomode
end
function initialize_specific_times!(obs::TimeVal, ctx::AnalysisContext, n_times_to_compute::Int64)
    ctx.specific_times = zeros(n_times_to_compute)
end

function begin_timestep!(obs::TimeVal, ctx::AnalysisContext)
    ctx.time_val = fastparse(ctx.row[1])
end
function begin_timestep_specific_times!(obs::TimeVal, ctx::AnalysisContext)
end

function error_handling!(obs::TimeVal, ctx::AnalysisContext)
    #ctx.times[ctx.t] = NaN
end

function accumulate!(obs::TimeVal, ctx::AnalysisContext, i::Int64) 
end
function accumulate_specific_times!(obs::TimeVal, ctx::AnalysisContext, i::Int64) 
end
                
function finish_timestep!(obs::TimeVal, ctx::AnalysisContext, t::Int64)
    ctx.times[t] = ctx.time_val
end
function finish_timestep_specific_times!(obs::TimeVal, ctx::AnalysisContext, t::Int64)
    ctx.specific_times[t] = ctx.time_val
end

function finish_seed_stats!(obs::TimeVal, ctx::AnalysisContext, seed::Int64)
end

function finish_loop_in_seed!(obs::TimeVal, ctx::AnalysisContext, nseeds::Int64)
end

function plot_everything(obs::TimeVal, ctx::AnalysisContext, save_folder::String)
end


Base.@kwdef mutable struct VelocityStats <: AbstractObservable 

    # Auxiliares, se acumulan por partículas y se van a acumular en realidad para un mega vector para hacer histogramas no?
    speeds_in_t::Vector{Float64} = Vector{Float64}()
    final_speed_all_seeds::Vector{Float64} = Vector{Float64}()
    final_theta_all_seeds::Vector{Float64} = Vector{Float64}()

    vx_in_zero::Vector{Float64} = Vector{Float64}()
    vy_in_zero::Vector{Float64} = Vector{Float64}()

    vx_in_t::Vector{Float64} = Vector{Float64}()
    vy_in_t::Vector{Float64} = Vector{Float64}()

    vparallel_in_t::Vector{Float64} = Vector{Float64}()
    vperpendicular_in_t::Vector{Float64} = Vector{Float64}()

    # Son un escalar que se calcula para specific_times
    kurtosis::Vector{Float64} = Vector{Float64}()
    skewness::Vector{Float64} = Vector{Float64}()

    kurtosis_x::Vector{Float64} = Vector{Float64}()
    skewness_x::Vector{Float64} = Vector{Float64}()
    kurtosis_y::Vector{Float64} = Vector{Float64}()
    skewness_y::Vector{Float64} = Vector{Float64}()

    kurtosis_parallel::Vector{Float64} = Vector{Float64}()
    skewness_parallel::Vector{Float64} = Vector{Float64}()
    kurtosis_perpendicular::Vector{Float64} = Vector{Float64}()
    skewness_perpendicular::Vector{Float64} = Vector{Float64}()

    fig_v::Plots.Plot = Plots.Plot()
    fig_theta::Plots.Plot = Plots.Plot()
end

function initialize!(obs::VelocityStats, np::Int64, nseeds::Int64) 
    obs.fig_v = plot(legend=:outertopright)
    obs.fig_theta = plot(legend=:outertopright)

    obs.speeds_in_t = Vector{Float64}(undef,np)
    obs.final_speed_all_seeds = Vector{Float64}(undef,nseeds*np)
    obs.final_theta_all_seeds = Vector{Float64}(undef,nseeds*np)


    obs.vx_in_zero = Vector{Float64}(undef,np)
    obs.vy_in_zero = Vector{Float64}(undef,np)

    obs.vx_in_t = Vector{Float64}(undef,np)
    obs.vy_in_t = Vector{Float64}(undef,np)

    obs.vparallel_in_t = Vector{Float64}(undef,np)
    obs.vperpendicular_in_t = Vector{Float64}(undef,np)
end
function initialize_specific_times!(obs::VelocityStats, ctx::AnalysisContext, n_times_to_compute::Int64)
    obs.kurtosis = zeros(n_times_to_compute) 
    obs.skewness = zeros(n_times_to_compute) 

    obs.kurtosis_x = zeros(n_times_to_compute)
    obs.skewness_x = zeros(n_times_to_compute) 
    obs.kurtosis_y = zeros(n_times_to_compute)
    obs.skewness_y = zeros(n_times_to_compute) 

    obs.kurtosis_parallel = zeros(n_times_to_compute)
    obs.skewness_parallel = zeros(n_times_to_compute) 
    obs.kurtosis_perpendicular = zeros(n_times_to_compute)
    obs.skewness_perpendicular = zeros(n_times_to_compute) 


end
function initialize_per_file!(obs::VelocityStats, ctx::AnalysisContext, n_total_rows::Int64) 
end

function begin_timestep!(obs::VelocityStats, ctx::AnalysisContext)
    ctx.time_val = fastparse(ctx.row[1])
end
function begin_timestep_specific_times!(obs::VelocityStats, ctx::AnalysisContext)
end

function error_handling!(obs::VelocityStats, ctx::AnalysisContext)
end

function accumulate!(obs::VelocityStats, ctx::AnalysisContext, i::Int64) # Hace falta para calcular las correlaciones además de esto, así q me ahorro recalcularlo    
end
function accumulate_specific_times!(obs::VelocityStats, ctx::AnalysisContext, i::Int64) 
    vx = ctx.particles[i].vx
    vy = ctx.particles[i].vy 
    
    obs.speeds_in_t[i] = sqrt(vx*vx+vy*vy) 

    obs.vx_in_t[i] = ctx.particles[i].vx
    obs.vy_in_t[i] = ctx.particles[i].vy

    cos_theta = cos(ctx.particles[i].θ) #**capaz esto lo podría cargar en ctx? creo q particle tiene cosenos y senos no? sólo tendría que cargarlos igual antes
    sin_theta = sin(ctx.particles[i].θ)
    obs.vparallel_in_t[i] = vx*cos_theta-vy*sin_theta
    obs.vperpendicular_in_t[i] = vx*sin_theta+vy*cos_theta
#    println(p.x," ",x_t_previous," ",p0.x," ",dx," ",dy)
end

function finish_timestep!(obs::VelocityStats, ctx::AnalysisContext, t::Int64)
end
function finish_timestep_specific_times!(obs::VelocityStats, ctx::AnalysisContext, t::Int64) #***y después al final del loop en seeds, a todos estos los divido por el n_seeds para tener el promedio para plotear !
    
    obs.kurtosis[t] += kurtosis(obs.speeds_in_t)
    obs.skewness[t] += skewness(obs.speeds_in_t)
    obs.kurtosis_x[t] += kurtosis(obs.vx_in_t)
    obs.skewness_x[t] += skewness(obs.vx_in_t)
    obs.kurtosis_y[t] += kurtosis(obs.vy_in_t)
    obs.skewness_y[t] += skewness(obs.vy_in_t)
    obs.kurtosis_parallel[t] += kurtosis(obs.vparallel_in_t)
    obs.skewness_parallel[t] += skewness(obs.vparallel_in_t)
    obs.kurtosis_perpendicular[t] += kurtosis(obs.vperpendicular_in_t)
    obs.skewness_perpendicular[t] += skewness(obs.vperpendicular_in_t)
    
    #end
end


function finish_seed_stats!(obs::VelocityStats, ctx::AnalysisContext, seed::Int64)
    obs.final_speed_all_seeds[1+ctx.np*(seed-1):seed*ctx.np] = obs.speeds_in_t
    obs.final_theta_all_seeds[1+ctx.np*(seed-1):seed*ctx.np] = atan.(obs.vy_in_t, obs.vx_in_t)

    histogram!(obs.fig_v, obs.speeds_in_t, bins=maximum([minimum([ctx.np÷3,100]),1]), label="$(ctx.seed) files, t=$(ctx.tv)", alpha = 0.5)
    histogram!(obs.fig_theta, atan.(obs.vy_in_t, obs.vx_in_t), bins=maximum([minimum([ctx.np÷3,100]),1]), label="$(ctx.seed) files, t=$(ctx.tv)", alpha = 0.5)
end

function finish_loop_in_seed!(obs::VelocityStats, ctx::AnalysisContext, nseeds::Int64)
    obs.kurtosis ./= nseeds
    obs.skewness ./= nseeds
    obs.kurtosis_x ./= nseeds
    obs.skewness_x ./= nseeds
    obs.kurtosis_y ./= nseeds
    obs.skewness_y ./= nseeds
    obs.kurtosis_parallel ./= nseeds
    obs.skewness_parallel ./= nseeds
    obs.kurtosis_perpendicular ./= nseeds
    obs.skewness_perpendicular ./= nseeds
end

function plot_everything(obs::VelocityStats, ctx::AnalysisContext, save_folder::String)
    fig_nongaussian::Plots.Plot = plot(legend=:outertopright)
    scatter!(fig_nongaussian,ctx.specific_times,obs.kurtosis, label="k", color=:black, alpha=0.7, linewidth=3, markershape=:circle) #[t*real_t+1 for t in 1:length(msdev_all)]
    scatter!(fig_nongaussian,ctx.specific_times,obs.skewness, label="s", color=:blue, alpha=0.7, linewidth=3, markershape=:diamond)
    scatter!(fig_nongaussian,ctx.specific_times,obs.kurtosis_x, label="kx", color=:red, alpha=0.7, linewidth=3, markershape=:cross) #[t*real_t+1 for t in 1:length(msdev_all)]
    scatter!(fig_nongaussian,ctx.specific_times,obs.skewness_x, label="sx", color=:green, alpha=0.7, linewidth=3, markershape=:utriangle)
    scatter!(fig_nongaussian,ctx.specific_times,obs.kurtosis_y, label="ky", color=:maroon, alpha=0.7, linewidth=3, markershape=:pentagon) #[t*real_t+1 for t in 1:length(msdev_all)]
    scatter!(fig_nongaussian,ctx.specific_times,obs.skewness_y, label="sy", color=:yellow, alpha=0.7, linewidth=3, markershape=:dtriangle)
    scatter!(fig_nongaussian,ctx.specific_times,obs.kurtosis_parallel, label="k∥", color=:pink, alpha=0.7, linewidth=3, markershape=:cross) #[t*real_t+1 for t in 1:length(msdev_all)]
    scatter!(fig_nongaussian,ctx.specific_times,obs.skewness_parallel, label="s∥", color=:lightgreen, alpha=0.7, linewidth=3, markershape=:utriangle)
    scatter!(fig_nongaussian,ctx.specific_times,obs.kurtosis_perpendicular, label="k⟂", color=:orange, alpha=0.7, linewidth=3, markershape=:pentagon) #[t*real_t+1 for t in 1:length(msdev_all)]
    scatter!(fig_nongaussian,ctx.specific_times,obs.skewness_perpendicular, label="s⟂", color=:gray, alpha=0.7, linewidth=3, markershape=:dtriangle)
    
    plot!(fig_nongaussian,ctx.specific_times,obs.kurtosis, label="",color=:black, alpha=0.5, linewidth=3) #[t*real_t+1 for t in 1:length(msdev_all)]
    plot!(fig_nongaussian,ctx.specific_times,obs.skewness, label="",color=:blue, alpha=0.5, linewidth=3)
    plot!(fig_nongaussian,ctx.specific_times,obs.kurtosis_x, label="",color=:red, alpha=0.5, linewidth=3) #[t*real_t+1 for t in 1:length(msdev_all)]
    plot!(fig_nongaussian,ctx.specific_times,obs.skewness_x, label="",color=:green, alpha=0.5, linewidth=3)
    plot!(fig_nongaussian,ctx.specific_times,obs.kurtosis_y, label="",color=:orange, alpha=0.5, linewidth=3) #[t*real_t+1 for t in 1:length(msdev_all)]
    plot!(fig_nongaussian,ctx.specific_times,obs.skewness_y, label="",color=:yellow, alpha=0.5, linewidth=3)
    plot!(fig_nongaussian,ctx.specific_times,obs.kurtosis_parallel, label="", color=:pink, alpha=0.5, linewidth=3) #[t*real_t+1 for t in 1:length(msdev_all)]
    plot!(fig_nongaussian,ctx.specific_times,obs.skewness_parallel, label="", color=:lightgreen, alpha=0.5, linewidth=3)
    plot!(fig_nongaussian,ctx.specific_times,obs.kurtosis_perpendicular, label="", color=:orange, alpha=0.5, linewidth=3) #[t*real_t+1 for t in 1:length(msdev_all)]
    plot!(fig_nongaussian,ctx.specific_times,obs.skewness_perpendicular, label="", color=:gray, alpha=0.5, linewidth=3)
    xlabel!(fig_nongaussian,"t")
    title!(fig_nongaussian,"Skewness and kurtosis of velocity distribution")
    display(fig_nongaussian)
    savefig(fig_nongaussian,save_folder*"nongaussian.png")


    histogram!(obs.fig_v, obs.final_speed_all_seeds, bins=maximum([minimum([ctx.np÷3,100]),1]), label="all files, t=$(ctx.tv)", color=:red, alpha = 0.5)
    #display(obs.fig_v)
    savefig(obs.fig_v,save_folder*"speed_dist_t$(ctx.tv).png")

    histogram!(obs.fig_theta, obs.final_theta_all_seeds, bins=maximum([minimum([ctx.np÷3,100]),1]), label="all files, t=$(ctx.tv)", color=:red, alpha = 0.5)
    #display(obs.fig_v)
    savefig(obs.fig_theta,save_folder*"theta_dist_t$(ctx.tv).png")

    output_hist_v = open(save_folder * "data_hist_v.txt", "w+")# "a+")
    println(output_hist_v, "t\tMean v\tStd v\tMedian v\tMode v")
    println(output_hist_v, "$(ctx.tv)\t$(StatsBase.mean(obs.final_speed_all_seeds))\t$(StatsBase.std(obs.final_speed_all_seeds))\t$(StatsBase.median(obs.final_speed_all_seeds))\t$(StatsBase.mode(obs.final_speed_all_seeds))")
        
    close(output_hist_v) 


end






Base.@kwdef mutable struct Speed <: AbstractObservable 
    speed_self_correlation::Vector{Float64} = Vector{Float64}()
    speed2::Vector{Float64} = Vector{Float64}()

    speed2_from_displacement::Vector{Float64} = Vector{Float64}()

    vx_in_zero::Vector{Float64} = Vector{Float64}()
    vy_in_zero::Vector{Float64} = Vector{Float64}()

    vx_in_t::Vector{Float64} = Vector{Float64}()
    vy_in_t::Vector{Float64} = Vector{Float64}()

    particles_previous_t::Vector{ParticleData} = Vector{ParticleData}()

    fig_self_correl::Plots.Plot = Plots.Plot()
    fig_speed2::Plots.Plot = Plots.Plot()
end

function initialize!(obs::Speed, np::Int64, nseeds::Int64) 
    obs.fig_self_correl = plot(legend=:outertopright)
    obs.fig_speed2 = plot(legend=:outertopright)

    obs.vx_in_zero = Vector{Float64}(undef,np)
    obs.vy_in_zero = Vector{Float64}(undef,np)

    obs.vx_in_t = Vector{Float64}(undef,np)
    obs.vy_in_t = Vector{Float64}(undef,np)


end
function initialize_specific_times!(obs::Speed, ctx::AnalysisContext, n_times_to_compute::Int64)
end
function initialize_per_file!(obs::Speed, ctx::AnalysisContext, n_total_rows::Int64) 
    obs.speed_self_correlation = zeros(n_total_rows)
    obs.speed2 = zeros(n_total_rows)
    
    obs.speed2_from_displacement = zeros(n_total_rows)
end

function begin_timestep!(obs::Speed, ctx::AnalysisContext)
end
function begin_timestep_specific_times!(obs::Speed, ctx::AnalysisContext)
end

function error_handling!(obs::Speed, ctx::AnalysisContext)
end

function accumulate!(obs::Speed, ctx::AnalysisContext, i::Int64) # Hace falta para calcular las correlaciones además de esto, así q me ahorro recalcularlo    
    obs.vx_in_t[i] = ctx.particles[i].vx
    obs.vy_in_t[i] = ctx.particles[i].vy
end
function accumulate_specific_times!(obs::Speed, ctx::AnalysisContext, i::Int64) 
end

function finish_timestep!(obs::Speed, ctx::AnalysisContext, t::Int64)
    if t ==1
        obs.vx_in_zero .= obs.vx_in_t
        obs.vy_in_zero .= obs.vy_in_t
    else
        obs.speed2_from_displacement[t] = Statistics.mean([(ctx.particles_unwrapped[i].x-obs.particles_previous_t[i].x)^2 + (ctx.particles_unwrapped[i].y-obs.particles_previous_t[i].y)^2  for i in 1:length(ctx.particles_unwrapped)] ./ (ctx.times[t]-ctx.times[t-1])^2)
    end

    #***debería promediar o plotear por seed
    obs.speed_self_correlation[t] = Statistics.mean(obs.vx_in_t.*obs.vx_in_zero.+obs.vy_in_t.*obs.vy_in_zero)
    obs.speed2[t] =  Statistics.mean(obs.vx_in_t.*obs.vx_in_t.+obs.vy_in_t.*obs.vy_in_t)

    obs.particles_previous_t = deepcopy(ctx.particles_unwrapped)
end
function finish_timestep_specific_times!(obs::Speed, ctx::AnalysisContext, t::Int64) #***y después al final del loop en seeds, a todos estos los divido por el n_seeds para tener el promedio para plotear !
end


function finish_seed_stats!(obs::Speed, ctx::AnalysisContext, seed::Int64)
end

function finish_loop_in_seed!(obs::Speed, ctx::AnalysisContext, nseeds::Int64)
end

function plot_everything(obs::Speed, ctx::AnalysisContext, save_folder::String)
    
    plot!(obs.fig_self_correl,ctx.times,obs.speed_self_correlation, label="", color=:gray, alpha=0.5, linewidth=3)
    try 
        plot!(obs.fig_self_correl,ctx.times,v_self_correl_theory(ctx.times, ctx.force_constants, ctx.params, ctx.model), label="theory", color=:red, alpha=0.5, linewidth=3)
    catch
    end
    xlabel!(obs.fig_self_correl,"t")
    title!(obs.fig_self_correl,"Velocity time self-correlation")
    display(obs.fig_self_correl)
    savefig(obs.fig_self_correl,save_folder*"v_self_correl.png")


    
    plot!(obs.fig_speed2,ctx.times,obs.speed2, label="", color=:gray, alpha=0.5, linewidth=3)
    plot!(obs.fig_speed2,ctx.times,obs.speed2_from_displacement, label="from disp", color=:green, alpha=0.5, linewidth=3)
    #plot!(obs.fig_speed2,ctx.times,[ctx.params.c*ctx.params.c+0.5*(ctx.params.ηa^2+ctx.params.ηb^2)/(ctx.times[2]-ctx.times[1]) for _ in ctx.times], label="from disp estimate", color=:blue, alpha=0.5, linewidth=3)
    ηr2 = ctx.params.ηr*ctx.params.ηr
    ηa2 = ctx.params.ηa*ctx.params.ηa
    ηb2 = ctx.params.ηb*ctx.params.ηb
    Δt = (ctx.times[2]-ctx.times[1])
    #***este tendría que ser teórico también
    plot!(obs.fig_speed2,ctx.times,[(ηa2+ηb2)/Δt + 8*ctx.params.c*ctx.params.c/(ηr2*ηr2*Δt^2)*(0.5*ηr2*Δt-1.0+exp((-0.5*ηr2)*Δt)) for _ in ctx.times], label="from disp estimate", color=:blue, alpha=0.5, linewidth=3)

    try 
        plot!(obs.fig_speed2,ctx.times,speed2_theory(ctx.times, ctx.force_constants, ctx.params, ctx.model), label="theory", color=:red, alpha=0.5, linewidth=3)
    catch
    end
    xlabel!(obs.fig_speed2,"t")
    title!(obs.fig_speed2,"Velocity squared")
    ylims!(obs.fig_speed2,(-Inf,maximum([Statistics.median(obs.speed2), Statistics.median(obs.speed2_from_displacement)])*1.5))
    display(obs.fig_speed2)
    savefig(obs.fig_speed2,save_folder*"speed2.png")
end




Base.@kwdef mutable struct CrossCorrelation <: AbstractObservable #**creo q no lo puedo hacer adaptable xq no puedo pasarle a un obs otro obs, debería poner esto en ctx en todo caso
    x_variable1::Vector{Float64} = Vector{Float64}()
    y_variable1::Vector{Float64} = Vector{Float64}()

    x_variable2::Vector{Float64} = Vector{Float64}()
    y_variable2::Vector{Float64} = Vector{Float64}()

    values::Vector{Float64} = Vector{Float64}()

end


#***entonces, esto se suponía que me iba a facilitar calcular correlaciones....... pensar bien cómo incorporaría otra igual acá, sobre todo pensando: podré reciclar variables? capaz me bastará con agregarle variables a este struct?

Base.@kwdef mutable struct OrientationCorrelation <: AbstractObservable 
    values::Vector{Float64} = Vector{Float64}()

    phi::Float64 = 0.0
    cosφ::Float64 = 0.0
    sinφ::Float64 = 0.0
    cos2θ::Float64 = 0.0
    sin2θ::Float64 = 0.0

    mean_cos_variable1::Vector{Float64} = Vector{Float64}() #***lo podría separar en orientation ellipse y orientation v según los requiera?
    mean_sin_variable1::Vector{Float64} = Vector{Float64}()
    mean_cos2_variable2::Vector{Float64} = Vector{Float64}()
    mean_sin2_variable2::Vector{Float64} = Vector{Float64}()

    correl_all_seeds::Vector{Float64} = Vector{Float64}()
    fig_correl::Plots.Plot = Plots.Plot()

end 

function initialize!(obs::OrientationCorrelation, np::Int64, nseeds::Int64)
    obs.fig_correl = plot(legend=:outertopright)

    obs.mean_cos_variable1  = Vector{Float64}(undef,np) 
    obs.mean_sin_variable1  = Vector{Float64}(undef,np)
    obs.mean_cos2_variable2 = Vector{Float64}(undef,np)
    obs.mean_sin2_variable2 = Vector{Float64}(undef,np)
end
function initialize_per_file!(obs::OrientationCorrelation, ctx::AnalysisContext, n_total_rows::Int64)
    obs.values =  zeros(n_total_rows)

    if length(obs.correl_all_seeds)<n_total_rows #***capaz renombrando las variables puedo hacer que estas funciones sean iguales a las del struct de MSD!
        obs.correl_all_seeds = vcat(obs.correl_all_seeds, zeros(n_total_rows-length(obs.correl_all_seeds)))
    end
end
function initialize_specific_times!(obs::OrientationCorrelation, ctx::AnalysisContext, n_times_to_compute::Int64)
end

function begin_timestep!(obs::OrientationCorrelation, ctx::AnalysisContext)
    obs.mean_cos_variable1  = zeros(ctx.np) 
    obs.mean_sin_variable1  = zeros(ctx.np)
    obs.mean_cos2_variable2 = zeros(ctx.np)
    obs.mean_sin2_variable2 = zeros(ctx.np)
end
function begin_timestep_specific_times!(obs::OrientationCorrelation, ctx::AnalysisContext)
end

function error_handling!(obs::OrientationCorrelation, ctx::AnalysisContext)
    #obs.values[ctx.t] = NaN
end

function accumulate!(obs::OrientationCorrelation,ctx::AnalysisContext, i::Int64) #**Asegurate de que VelocityStats corra ANTES que este porque sino ctx.particles.vx y vy van a estar con el valor anterior!
    phi = atan(ctx.particles[i].vy,ctx.particles[i].vx)
    cosφ, sinφ = cos(phi), sin(phi)
    obs.mean_cos_variable1[i] += cosφ
    obs.mean_sin_variable1[i] += sinφ
    cos2θ, sin2θ = cos(2*ctx.particles[i].θ), sin(2*ctx.particles[i].θ)
    obs.mean_cos2_variable2[i] += cos2θ
    obs.mean_sin2_variable2[i] += sin2θ
end
function accumulate_specific_times!(obs::OrientationCorrelation, ctx::AnalysisContext, i::Int64)
end

function finish_timestep!(obs::OrientationCorrelation, ctx::AnalysisContext, t::Int64)
    obs.values[t] = sum((obs.mean_cos2_variable2.-StatsBase.mean(obs.mean_cos2_variable2)).*(obs.mean_cos_variable1.-StatsBase.mean(obs.mean_cos_variable1)).+(obs.mean_sin2_variable2.-StatsBase.mean(obs.mean_sin2_variable2)).*(obs.mean_sin_variable1.-StatsBase.mean(obs.mean_sin_variable1)) ) / ctx.np
end 
function finish_timestep_specific_times!(obs::OrientationCorrelation, ctx::AnalysisContext, t::Int64)
end 

function finish_seed_stats!(obs::OrientationCorrelation, ctx::AnalysisContext, seed::Int64)
    obs.correl_all_seeds .+= obs.values
end

function finish_loop_in_seed!(obs::OrientationCorrelation, ctx::AnalysisContext, nseeds::Int64)
    obs.correl_all_seeds ./= nseeds   
end

function plot_everything(obs::OrientationCorrelation, ctx::AnalysisContext, save_folder::String)
    scatter!(obs.fig_correl,ctx.times,obs.correl_all_seeds, label="all", color=:black, alpha=0.5, linewidth=3) #[t*real_t+1 for t in 1:length(msdev_all)]
    display(obs.fig_correl)
    savefig(obs.fig_correl,save_folder*"cross_correl.png")
end




# Some extra functions

Base.@kwdef mutable struct OrientationChannel
    mean_sin::Vector{Float64}  = Vector{Float64}()
    mean_cos::Vector{Float64}  = Vector{Float64}()
    mean_sin2::Vector{Float64} = Vector{Float64}()
    mean_cos2::Vector{Float64} = Vector{Float64}()

    orientations::Vector{SVector{2,Float64}}      = Vector{SVector{2,Float64}}()
    vector_for_correl::Vector{SVector{2,Float64}} = Vector{SVector{2,Float64}}()
end

function OrientationChannel(np::Int64)
    OrientationChannel(
        mean_sin  = zeros(np),
        mean_cos  = zeros(np),
        mean_sin2 = zeros(np),
        mean_cos2 = zeros(np),
        orientations      = Vector{SVector{2,Float64}}(undef, np),
        vector_for_correl = Vector{SVector{2,Float64}}(undef, np),
    )
end

@inline function reset_channel!(c::OrientationChannel)
    fill!(c.mean_sin, 0.0);  fill!(c.mean_cos, 0.0)
    fill!(c.mean_sin2, 0.0); fill!(c.mean_cos2, 0.0)
    return c
end

@inline function accumulate_channel!(c::OrientationChannel, i::Int64, angle::Float64,
                                      vec_for_correl::SVector{2,Float64})
    cosα, sinα   = cos(angle),  sin(angle)
    cos2α, sin2α = cos(2angle), sin(2angle)

    c.mean_cos[i]  += cosα
    c.mean_sin[i]  += sinα
    c.mean_cos2[i] += cos2α
    c.mean_sin2[i] += sin2α

    c.orientations[i]      = SVector(cos2α, sin2α)
    c.vector_for_correl[i] = vec_for_correl
    return nothing
end

@inline function order_parameters(c::OrientationChannel)
    P = hypot(Statistics.mean(c.mean_sin),  Statistics.mean(c.mean_cos))
    Q = hypot(Statistics.mean(c.mean_sin2), Statistics.mean(c.mean_cos2))
    return P, Q
end

@inline function accumulate_orientation_channels!(ctx::AnalysisContext, phi::OrientationChannel,
                                                   theta::OrientationChannel,
                                                   positions::Vector{SVector{2,Float64}}, i::Int64)
    p = ctx.particles[i]

    x = ctx.row[i*5-3]; y = ctx.row[i*5-2]
    p.x = fastparse(x); p.y = fastparse(y)
    θ = ctx.row[i*5-1]
    p.θ = fastparse(θ)
    vx = ctx.row[i*5]; vy = ctx.row[i*5+1]
    p.vx = fastparse(vx); p.vy = fastparse(vy)

    positions[i] = SVector(p.x, p.y)

    φ = atan(p.vy, p.vx)
    accumulate_channel!(phi,   i, φ,   SVector(p.vx, p.vy))
    accumulate_channel!(theta, i, p.θ, SVector(cos(p.θ), sin(p.θ)))
    return nothing
end



Base.@kwdef mutable struct PhiThetaCorrelation <: AbstractObservable
    Δr::Float64 = 0.0
    rectangle::PeriodicHyperCube{2} = PeriodicRectangle(1.0, 1.0) # overwritten in initialize_per_file!

    phi::OrientationChannel   = OrientationChannel()
    theta::OrientationChannel = OrientationChannel()
    positions::Vector{SVector{2,Float64}} = Vector{SVector{2,Float64}}()

    # Freshly created every timestep (see begin_timestep!) since file 3
    # writes one *independent* correlation per row, not an average.
    corr_vel_phi   = nothing
    corr_nem_phi   = nothing
    corr_vel_theta = nothing
    corr_nem_theta = nothing

    # Set from outside (by whoever opens/rotates the output file), same
    # spirit as `fig_msd` being built in `initialize!` but drawn on from
    # `plot_everything` — here the "figure" is a text file handle.
    output::IO = devnull
end

function initialize!(obs::PhiThetaCorrelation, np::Int64, nseeds::Int64)
end
function initialize_specific_times!(obs::PhiThetaCorrelation, ctx::AnalysisContext, n_times_to_compute::Int64)
end

function initialize_per_file!(obs::PhiThetaCorrelation, ctx::AnalysisContext, n_total_rows::Int64)
    np = ctx.np
    obs.Δr        = sqrt(ctx.params.Lx * ctx.params.Ly / np) # separación media entre partículas
    obs.rectangle = ctx.params.rectangle

    obs.phi   = OrientationChannel(np)
    obs.theta = OrientationChannel(np)
    obs.positions = Vector{SVector{2,Float64}}(undef, np)
end

function begin_timestep!(obs::PhiThetaCorrelation, ctx::AnalysisContext)
    reset_channel!(obs.phi)
    reset_channel!(obs.theta)

    obs.corr_vel_phi   = space_correlation(obs.rectangle, obs.Δr, connected=true)
    obs.corr_nem_phi   = space_correlation(obs.rectangle, obs.Δr, connected=true)
    obs.corr_vel_theta = space_correlation(obs.rectangle, obs.Δr, connected=true)
    obs.corr_nem_theta = space_correlation(obs.rectangle, obs.Δr, connected=true)
end
function begin_timestep_specific_times!(obs::PhiThetaCorrelation, ctx::AnalysisContext)
end

function error_handling!(obs::PhiThetaCorrelation, ctx::AnalysisContext)
    # The original had a partial-recovery attempt for the truncated last
    # particle on a corrupted line, self-flagged in a comment as
    # "**no está bien esto" (not correct). Simplified here to match file
    # 2's approach: skip the whole row instead of guessing at bad data.
end

function accumulate!(obs::PhiThetaCorrelation, ctx::AnalysisContext, i::Int64)
    accumulate_orientation_channels!(ctx, obs.phi, obs.theta, obs.positions, i)
end
function accumulate_specific_times!(obs::PhiThetaCorrelation, ctx::AnalysisContext, i::Int64)
end

function finish_timestep!(obs::PhiThetaCorrelation, ctx::AnalysisContext, t::Int64)
    phi_p, phi_q     = order_parameters(obs.phi)
    theta_p, theta_q = order_parameters(obs.theta)

    space_correlation!(obs.corr_vel_phi,   obs.positions, obs.phi.vector_for_correl)
    space_correlation!(obs.corr_nem_phi,   obs.positions, obs.phi.orientations)
    space_correlation!(obs.corr_vel_theta, obs.positions, obs.theta.vector_for_correl)
    space_correlation!(obs.corr_nem_theta, obs.positions, obs.theta.orientations)

    cv_phi   = correlations(obs.corr_vel_phi)
    cn_phi   = correlations(obs.corr_nem_phi)
    cv_theta = correlations(obs.corr_vel_theta)
    cn_theta = correlations(obs.corr_nem_theta)

    println(obs.output,
        "$(ctx.time_val)\t$(phi_p)\t$(phi_q)\t$(theta_p)\t$(theta_q)\t",
        "$(cv_phi[1])\t$(cv_phi[2])\t$(cv_phi[3])\t$(cv_phi[4])\t",
        "$(cn_phi[2])\t$(cn_phi[3])\t$(cn_phi[4])\t",
        "$(cv_theta[2])\t$(cv_theta[3])\t$(cv_theta[4])\t",
        "$(cn_theta[2])\t$(cn_theta[3])\t$(cn_theta[4])")
end
function finish_timestep_specific_times!(obs::PhiThetaCorrelation, ctx::AnalysisContext, t::Int64)
end

function finish_seed_stats!(obs::PhiThetaCorrelation, ctx::AnalysisContext, seed::Int64)
end
function finish_loop_in_seed!(obs::PhiThetaCorrelation, ctx::AnalysisContext, nseeds::Int64)
end
function plot_everything(obs::PhiThetaCorrelation, ctx::AnalysisContext, save_folder::String)
end






# IF WOUND

Base.@kwdef mutable struct WoundWidthCalculator
    L::Float64
    dx::Float64
    nbins::Int

    density::Vector{Float64}
    rho_bulk::Float64 = 0.0
    threshold::Float64 = 0.0

    left_bin::Int = 0
    right_bin::Int = 0
end

function WoundWidthCalculator(L::Float64, nbins::Int)
    dx = L / nbins
    density = zeros(Float64, nbins)
    return WoundWidthCalculator(L=L, dx=dx, nbins=nbins, density=density)
end

Base.@kwdef mutable struct Wound <: AbstractObservable
    width::Vector{Float64} = Float64[]

    msd_edge_current::Vector{Float64} = Float64[]
    msd_bulk_current::Vector{Float64} = Float64[]

    msd_edge_initial::Vector{Float64} = Float64[]
    msd_bulk_initial::Vector{Float64} = Float64[]

    # Particle classification at t = 0
    initial_edge::Vector{Bool} = Bool[]
    initial_bulk::Vector{Bool} = Bool[]

    width_all_seeds::Vector{Float64} = Float64[]

    msd_edge_current_all_seeds::Vector{Float64} = Float64[]
    msd_bulk_current_all_seeds::Vector{Float64} = Float64[]
    msd_edge_initial_all_seeds::Vector{Float64} = Float64[]
    msd_bulk_initial_all_seeds::Vector{Float64} = Float64[]

    fig_width::Plots.Plot = Plots.Plot()
    p_w1::Plots.Plot = Plots.Plot()
    p_w2::Plots.Plot = Plots.Plot()
    p_w3::Plots.Plot = Plots.Plot()
    p_w4::Plots.Plot = Plots.Plot()
    fig_msd::Plots.Plot = Plots.Plot()
    p_msd1::Plots.Plot = Plots.Plot()
    p_msd2::Plots.Plot = Plots.Plot()

    p::Vector{Plots.Plot} = Plots.Plot[]
    dfs::Vector{AbstractDataFrame} = AbstractDataFrame[]


    calc::WoundWidthCalculator = WoundWidthCalculator(1.0, 1)
end


@inline function find_wound_edge_bins(density::Vector{Float64}, threshold::Float64, nbins::Int)
    center = nbins ÷ 2 + 1
    left_bin = 0
    @inbounds for i in center:-1:2
        if density[i] <= threshold && density[i-1] > threshold
            left_bin = i
            break
        end
    end
    right_bin = 0
    @inbounds for i in center:nbins-1
        if density[i] < threshold && density[i+1] >= threshold
            right_bin = i
            break
        end
    end
    return left_bin, right_bin
end

@inline function is_edge_bin(bin::Int, left_bin::Int, right_bin::Int)
    return (
        bin == left_bin - 1 ||
        bin == left_bin ||
        bin == right_bin ||
        bin == right_bin + 1
    )
end

@inline function is_bulk_bin(bin::Int, nbins::Int)
    quarter = nbins ÷ 4
    return bin <= quarter || bin > nbins - quarter
end

function wound_width!(calc::WoundWidthCalculator, particles::Vector{ParticleData}, np::Int) # x::AbstractVector{Float64})
    L       = calc.L
    dx      = calc.dx
    nbins   = calc.nbins
    density = calc.density

    #fill!(density, zero(Float64))

    # Histogram x positions
    @inbounds for i in 1:np
        x = particles[i].x
        
        bin = floor(Int, x / dx) + 1 # assumes 0 ≤ xi < L
        bin = min(bin, nbins) # Safety for xi == L due to roundoff

        density[bin] += 1.0
    end

    # Convert counts to density
    inv_dx = inv(dx)

    @inbounds for i in 1:nbins
        density[i] *= inv_dx
    end

    # Bulk density from the outer quarters
    quarter = nbins ÷ 4

    rho_bulk = 0.0

    @inbounds for i in 1:quarter
        rho_bulk += density[i]
        rho_bulk += density[nbins - i + 1]
    end

    rho_bulk /= 2 * quarter

    threshold = 0.5 * rho_bulk

    calc.rho_bulk = rho_bulk
    calc.threshold = threshold

    left_bin, right_bin =
        find_wound_edge_bins(density, threshold, nbins)

    calc.left_bin = left_bin
    calc.right_bin = right_bin

    if left_bin == 0 || right_bin == 0
        return 0.0
    end

    # Left interface
    ρ1 = density[left_bin - 1]
    ρ2 = density[left_bin]

    frac_left = (threshold - ρ1) / (ρ2 - ρ1)

    x_left = (left_bin - 2 + frac_left) * dx

    # Right interface
    ρ1 = density[right_bin]
    ρ2 = density[right_bin + 1]

    frac_right = (threshold - ρ1) / (ρ2 - ρ1)

    x_right = (right_bin - 1 + frac_right) * dx

    return x_right - x_left
end

function wound_msd!(obs::Wound, ctx::AnalysisContext, t::Int)
    calc = obs.calc

    left_bin = calc.left_bin
    right_bin = calc.right_bin

    # If wound is already closed
    if left_bin == 0 || right_bin == 0
        obs.msd_edge_current[t] = NaN
        obs.msd_bulk_current[t] = NaN
        obs.msd_edge_initial[t] = NaN
        obs.msd_bulk_initial[t] = NaN
        return
    end

    dx = calc.dx
    nbins = calc.nbins
    np = ctx.np

    edge_current_sum = 0.0
    bulk_current_sum = 0.0
    edge_initial_sum = 0.0
    bulk_initial_sum = 0.0

    n_edge_current = 0
    n_bulk_current = 0
    n_edge_initial = 0
    n_bulk_initial = 0

    @inbounds for i in 1:np

        # --------------------------------------------------------
        # Current position → current region
        # --------------------------------------------------------

        x = ctx.particles[i].x

        bin = floor(Int, x / dx) + 1
        bin = min(bin, nbins)

        current_edge =
            bin == left_bin - 1 ||
            bin == left_bin ||
            bin == right_bin ||
            bin == right_bin + 1

        quarter = nbins ÷ 4

        current_bulk =
            bin <= quarter ||
            bin > nbins - quarter

        # --------------------------------------------------------
        # Displacement from initial position
        # --------------------------------------------------------

        dxp = ctx.particles_unwrapped[i].x -
              ctx.particles_initial_config[i].x

        dyp = ctx.particles_unwrapped[i].y -
              ctx.particles_initial_config[i].y

        dr2 = dxp * dxp + dyp * dyp

        # --------------------------------------------------------
        # Current populations
        # --------------------------------------------------------

        if current_edge
            edge_current_sum += dr2
            n_edge_current += 1
        elseif current_bulk
            bulk_current_sum += dr2
            n_bulk_current += 1
        end

        # --------------------------------------------------------
        # Initial populations
        # --------------------------------------------------------

        if obs.initial_edge[i]
            edge_initial_sum += dr2
            n_edge_initial += 1
        elseif obs.initial_bulk[i]
            bulk_initial_sum += dr2
            n_bulk_initial += 1
        end
    end

    obs.msd_edge_current[t] =
        n_edge_current > 0 ?
        edge_current_sum / n_edge_current : NaN

    obs.msd_bulk_current[t] =
        n_bulk_current > 0 ?
        bulk_current_sum / n_bulk_current : NaN

    obs.msd_edge_initial[t] =
        n_edge_initial > 0 ?
        edge_initial_sum / n_edge_initial : NaN

    obs.msd_bulk_initial[t] =
        n_bulk_initial > 0 ?
        bulk_initial_sum / n_bulk_initial : NaN
end


function initialize_wound_populations!(obs::Wound, ctx::AnalysisContext)
    calc = obs.calc

    dx = calc.dx
    nbins = calc.nbins

    left_bin = calc.left_bin
    right_bin = calc.right_bin

    quarter = nbins ÷ 4

    @inbounds for i in 1:ctx.np

        x = ctx.particles[i].x

        bin = floor(Int, x / dx) + 1
        bin = min(bin, nbins)

        obs.initial_edge[i] =
            bin == left_bin - 1 ||
            bin == left_bin ||
            bin == right_bin ||
            bin == right_bin + 1

        obs.initial_bulk[i] =
            bin <= quarter ||
            bin > nbins - quarter
    end
end

function plot_wound_binning(obs::Wound, ctx::AnalysisContext; t=nothing)
    calc = obs.calc
    dx = calc.dx
    nbins = calc.nbins
    np = ctx.np

    left_bin = calc.left_bin
    right_bin = calc.right_bin

    # Count particles in each spatial bin
    counts = zeros(Int, nbins)

    @inbounds for i in 1:np
        x = ctx.particles[i].x
        bin = floor(Int, x / dx) + 1
        bin = min(bin, nbins)

        if 1 <= bin <= nbins
            counts[bin] += 1
        end
    end

    centers = ((1:nbins) .- 0.5) .* dx
    quarter = nbins ÷ 4

    # Assign each bin to the same mutually exclusive categories
    # used in wound_msd!: edge takes precedence over bulk.
    edge_counts = zeros(Int, nbins)
    bulk_counts = zeros(Int, nbins)
    other_counts = zeros(Int, nbins)

    if left_bin != 0 && right_bin != 0
        for bin in 1:nbins
            is_edge = bin == left_bin - 1 ||
                      bin == left_bin ||
                      bin == right_bin ||
                      bin == right_bin + 1

            is_bulk = bin <= quarter || bin > nbins - quarter

            if is_edge
                edge_counts[bin] = counts[bin]
            elseif is_bulk
                bulk_counts[bin] = counts[bin]
            else
                other_counts[bin] = counts[bin]
            end
        end
    else
        other_counts .= counts
    end

    # Stacked histogram: each bar shows the number of particles
    # in a spatial bin, with its region indicated by color.
    p = bar(
        centers,
        hcat(edge_counts, other_counts), #bulk_counts
        #bar_position = :stack,
        bar_width = dx,
        label = ["Wound edge" "Other"], #"Bulk"
        xlabel = "x",
        ylabel = "Number of particles",
        title = isnothing(t) ? "Particle distribution" :
                "Particle distribution at t = $t",
        legend = :topright,
        alpha = 0.8,
    )

    # Mark the wound boundaries and the bins used for the edge.
    if left_bin != 0 && right_bin != 0
        vline!(
            p,
            [(left_bin - 1) * dx, left_bin * dx,
             right_bin * dx, (right_bin + 1) * dx],
            color = :black,
            linestyle = :dash,
            linewidth = 1,
            label = "",
        )
    end

    # Export histogram data
    df = DataFrame(
        center = centers,
        edge_counts = edge_counts,
        other_counts = other_counts,
        bulk_counts = bulk_counts,
        t = t,
    )

    return p, df
end




function initialize!(obs::Wound, np::Int64, nseeds::Int64)
    obs.fig_width = plot(legend=:topright, layout=(2,2), size = (1200, 1200))
    obs.p_w1 = plot(legend=:topright)
    obs.p_w2 = plot(legend=:topright)#, xscale=:log10, yscale=:log10)
    obs.p_w3 = plot(legend=:topright)#, yscale=:log10)
    obs.p_w4 = plot(legend=:topright)
    obs.fig_msd = plot(legend=:topright, layout=(1,2), size = (1200, 500))
    obs.p_msd1 = plot(legend=:topright)
    obs.p_msd2 = plot(legend=:topright, xscale=:log10, yscale=:log10)

    obs.initial_edge = Vector{Bool}(undef, np)
    obs.initial_bulk = Vector{Bool}(undef, np)
end
function initialize_specific_times!(obs::Wound, ctx::AnalysisContext, n_times_to_compute::Int64)
    obs.p = Vector{Plots.Plot}(undef,n_times_to_compute)
    obs.dfs = Vector{AbstractDataFrame}(undef,n_times_to_compute)
end
function initialize_per_file!(obs::Wound, ctx::AnalysisContext, n_total_rows::Int64)
    obs.width = Vector{Float64}(undef, n_total_rows)
    obs.msd_edge_initial = fill(NaN, n_total_rows)
    obs.msd_bulk_initial = fill(NaN, n_total_rows)
    obs.msd_edge_current = fill(NaN, n_total_rows)
    obs.msd_bulk_current = fill(NaN, n_total_rows)

    if length(obs.width_all_seeds) < n_total_rows
        nnew = n_total_rows - length(obs.width_all_seeds)

        append!(obs.width_all_seeds, zeros(nnew)) #obs.width_all_seeds = vcat(obs.width_all_seeds, zeros(n_total_rows-length(obs.width_all_seeds)))
        append!(obs.msd_edge_initial_all_seeds, zeros(nnew))
        append!(obs.msd_bulk_initial_all_seeds, zeros(nnew))
        append!(obs.msd_edge_current_all_seeds, zeros(nnew))
        append!(obs.msd_bulk_current_all_seeds, zeros(nnew))
    end

    nbins = Int(ctx.params.Lx ÷ (2 * ctx.params.b))

    obs.calc = WoundWidthCalculator(ctx.params.Lx, nbins)
    
end

function begin_timestep!(obs::Wound, ctx::AnalysisContext)
    #obs.width = 0.0
end
function begin_timestep_specific_times!(obs::Wound, ctx::AnalysisContext)
end

function error_handling!(obs::Wound, ctx::AnalysisContext) #***pensar si es mejor usar ctx.t directamente o si pasarle i=t desde afuera
    #obs.width[ctx.t] = NaN
end

function accumulate!(obs::Wound, ctx::AnalysisContext, i::Int64) #**será muy ineficiente llamar adentro de dos structs?
    #obs.width += (ctx.particles_unwrapped[i].x-ctx.particles_initial_config[i].x)^2 + (ctx.particles_unwrapped[i].y-ctx.particles_initial_config[i].y)^2
end
function accumulate_specific_times!(obs::Wound, ctx::AnalysisContext, i::Int64) 
end

function finish_timestep!(obs::Wound, ctx::AnalysisContext, t::Int64)
    obs.width[t] = wound_width!(obs.calc, ctx.particles, ctx.np)

    if t == 2
        initialize_wound_populations!(obs, ctx)
    elseif t>2
        wound_msd!(obs, ctx, t)
    end

    
    #obs.values[t] = obs.width / ctx.np
end
function finish_timestep_specific_times!(obs::Wound, ctx::AnalysisContext, t::Int64)
    obs.p[t], obs.dfs[t] = plot_wound_binning(obs, ctx; t=ctx.specific_times[t])
    #println(obs.dfs[t])
end

function finish_seed_stats!(obs::Wound, ctx::AnalysisContext, seed::Int64)
    plot!(obs.fig_width[1],ctx.times,obs.width, label="seed=$(seed)", alpha=0.5) #[t*real_t+1 for t in 1:length(widthev)]
    obs.width_all_seeds .+= obs.width

    #****initial podría seguir la curva, no cortarse como current !!
    #plot!(obs.fig_msd,ctx.times,obs.msd_edge_current, label="seed=$(seed)", alpha=0.5) #[t*real_t+1 for t in 1:length(msd_edge_currentev)]
    obs.msd_edge_current_all_seeds .+= obs.msd_edge_current
    #plot!(obs.fig_msd,ctx.times,obs.msd_bulk_current, label="", alpha=0.5, linestyle=:dash) #[t*real_t+1 for t in 1:length(msd_bulk_currentev)]
    obs.msd_bulk_current_all_seeds .+= obs.msd_bulk_current
    plot!(obs.p_msd1,ctx.times,obs.msd_edge_initial, label="", alpha=0.5, linestyle=:dot) #[t*real_t+1 for t in 1:length(msd_edge_initialev)]
    obs.msd_edge_initial_all_seeds .+= obs.msd_edge_initial
    plot!(obs.p_msd1,ctx.times,obs.msd_bulk_initial, label="", alpha=0.5, linestyle=:dashdotdot) #[t*real_t+1 for t in 1:length(msd_bulk_initialev)]
    obs.msd_bulk_initial_all_seeds .+= obs.msd_bulk_initial

    for (t,hist_plot) in enumerate(obs.p)
        savefig(hist_plot, ctx.save_folder*"density_histogram_$(ctx.params.seed)_t$(t-1).png")
        CSV.write(ctx.save_folder*"particle_binning$(ctx.params.seed)_t$(t-1).csv", obs.dfs[t])
    end
end

function finish_loop_in_seed!(obs::Wound, ctx::AnalysisContext, nseeds::Int64)
    obs.width_all_seeds ./= nseeds   
    obs.msd_edge_current_all_seeds ./= nseeds   
    obs.msd_bulk_current_all_seeds ./= nseeds   
    obs.msd_edge_initial_all_seeds ./= nseeds   
    obs.msd_bulk_initial_all_seeds ./= nseeds

end


function plot_everything(obs::Wound, ctx::AnalysisContext, save_folder::String)


    tmax_idx = findfirst(x->x==0, obs.width_all_seeds)
    tmax_idx_true = isnothing(tmax_idx) ? length(ctx.times) : tmax_idx<3 ? length(ctx.times) : tmax_idx
    xmaxlim = ctx.times[tmax_idx_true] + ctx.times[2]

    if tmax_idx_true>3

       lo = -ctx.times[2]
       hi = xmaxlim
       if !(hi > lo) || !isfinite(hi)
           hi = maximum(ctx.times)
       end
       xlims!(obs.p_msd1, (lo, hi))

       print(tmax_idx, "  ",tmax_idx_true, "  ",xmaxlim)

       scatter!(obs.p_w1,ctx.times,obs.width_all_seeds, label="all", color=:black, alpha=0.5, linewidth=3) #[t*real_t+1 for t in 1:length(widthev_all)]
       scatter!(obs.p_w2,ctx.times[3:tmax_idx_true-1],obs.width_all_seeds[3:tmax_idx_true-1], label="log-log", color=:black, alpha=0.5, linewidth=3)
       scatter!(obs.p_w3,ctx.times[3:tmax_idx_true-1],obs.width_all_seeds[3:tmax_idx_true-1], label="ylog", color=:black, alpha=0.5, linewidth=3)
    #scatter!(obs.p_w4,ctx.times[3:tmax_idx_true-1],exp.(obs.width_all_seeds[3:tmax_idx_true-1]), label="all", color=:black, alpha=0.5, linewidth=3)     
    

       try 
           width_fit = width_theory(ctx.times, obs.width_all_seeds, ctx.times[tmax_idx_true]/5, ctx.force_constants, ctx.params, ctx.model)
           plot!(obs.p_w1,ctx.times, width_fit[1], label="theory v0=$(width_fit[2])", color=:red, alpha=0.5, linewidth=3)
       catch
       end
    
    #xlims!(obs.p_w1,(-ctx.times[2], xmaxlim ))
    #xlims!(obs.fig_width[2],(ctx.times[2], xmaxlim ))

       fig_width = plot(obs.p_w1, obs.p_w2, obs.p_w3; layout=(1,3), size=(1600,500))
    
       display(fig_width)
       savefig(fig_width,save_folder*"width.png")

       #println(obs.msd_edge_initial_all_seeds, obs.msd_bulk_initial_all_seeds)

    #scatter!(obs.fig_msd,ctx.times,obs.msd_edge_current_all_seeds, label="wound edge", color=:black, alpha=0.5, linewidth=3) #[t*real_t+1 for t in 1:length(widthev_all)]
    #scatter!(obs.fig_msd,ctx.times,obs.msd_bulk_current_all_seeds, label="bulk", color=:red, alpha=0.5, linewidth=3) #[t*real_t+1 for t in 1:length(widthev_all)]
       scatter!(obs.p_msd1,ctx.times,obs.msd_edge_initial_all_seeds, label="wound edge", color=:blue, alpha=0.5, linewidth=3) #[t*real_t+1 for t in 1:length(widthev_all)]
       scatter!(obs.p_msd1,ctx.times,obs.msd_bulk_initial_all_seeds, label="bulk", color=:green, alpha=0.5, linewidth=3) #[t*real_t+1 for t in 1:length(widthev_all)]
    #xlims!(obs.p_msd1,(-ctx.times[2], xmaxlim ))

       scatter!(obs.p_msd2,ctx.times[3:end-1],obs.msd_edge_initial_all_seeds[3:end-1], label="wound edge", color=:blue, alpha=0.5, linewidth=3) #[t*real_t+1 for t in 1:length(widthev_all)]
       scatter!(obs.p_msd2,ctx.times[3:end-1],obs.msd_bulk_initial_all_seeds[3:end-1], label="bulk", color=:green, alpha=0.5, linewidth=3) #[t*real_t+1 for t in 1:length(widthev_all)]
      
       plot!(obs.p_msd2, ctx.times[3:end], ctx.times[3:end], linestyle = :dash, color = :blue, linewidth = 2, label = "slope = 1")
       plot!(obs.p_msd2, ctx.times[3:end], ctx.times[3:end].*ctx.times[3:end], linestyle = :dot, color = :green, linewidth = 2, label = "slope = 2")


       fig_msd = plot(obs.p_msd1, obs.p_msd2; layout=(1,2), size=(1200,500))

       display(fig_msd)
       savefig(fig_msd,save_folder*"msd_per_d_to_wound.png")

       save_msd_jld2(obs, ctx, ctx.save_folder*"msd_wound_data.jld2"; obs_vars = (:msd_edge_initial_all_seeds, :msd_bulk_initial_all_seeds,), ctx_vars = (:times,))
   end
end



observables = (
        TimeVal(),
        MSD(),
        #VelocityStats(),
        #Speed(),
        #Correlation(),
        #PositionVelocityCorrelation(),
        #OrientationCorrelation(),
        #Wound()
    )

