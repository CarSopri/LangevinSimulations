
using StatsBase

include("./plotting.jl")
include("./file_management.jl")
include("./variables_to_compute.jl")


function load_initial_config!(inputs_file::String, n::Int64, ctx::AnalysisContext)
    open(inputs_file) do file
        # Skip header
        readline(file)
        
        # Read first data line
        line = readline(file)
        row = split(line,"\t")
        #@views row = df[1,:] #Para que no cree una copia
        for i in 1:n #simd auto-vectorize operations when iterations are independent
    
            idx = colx(i) 
            p = ctx.particles_initial_config[i] #**ver si funciona esto, si efectivamente se modifican
            x = row[idx]; y = row[idx+1]
            p.x = fastparse(x) #fastparse( x)
            p.y = fastparse(y)
            θ = row[idx+2]
            p.θ = fastparse(θ)

            vx = row[idx+3]; vy = row[idx+4] 
            p.vx = fastparse(vx)
            p.vy = fastparse(vy)
        end 

    end
    ctx.particles_unwrapped = deepcopy(ctx.particles_initial_config)
    ctx.particles = deepcopy(ctx.particles_initial_config)
end


function make_gif_and_calculate_obs_file_k(inputs_file::String, params::Params, ctx::AnalysisContext, observables::Tuple{Vararg{AbstractObservable}}, tot_row_length::Int64, tmax_saved::Int64, 
    do_plot_orig::Bool, nf::Int64, anim::Animation, fig_ellipse::Plots.Plot, fig_ellipse_title::String)
    
    do_plot = do_plot_orig
    #plot_density::Bool = false

    dx::Float64 = 0.0; dy::Float64 = 0.0; msd::Float64 = 0.0 ## uso dx y dy ?***
    vx::String = "0.0"; vy::String = "0.0"; θ::String = "0.0"; x::String = "0.0"; y::String = "0.0" ##**acá qué onda, me conviene inicializar o directamente pasar a un struct también? estos están bastante fijos
    
    open(inputs_file) do file
        for line in Iterators.drop(eachline(file), 1) # skip first line
            ctx.row = split(line,"\t")
            time_val = fastparse(ctx.row[1]) #**este lo cargo en ctx recién en begin_timestep mmm

            analyize_specific_times = ctx.t%ctx.t_split==1 #|| t==1
            do_plot = do_plot && time_val>=tmax_saved && ctx.t%(nf)==0 ##

            n_non_corrupted = params.np 

            if length(ctx.row)<tot_row_length || (ctx.row[tot_row_length] isa String && occursin(NULL_CHAR, ctx.row[tot_row_length])) 
                println(ctx.t," ",time_val, " Something went wrong (not sure what) but we can ignore this line.")
                foreach(obs->error_handling!(obs,ctx), observables)

                n_non_corrupted = params.np-1
                continue                
            end

            foreach(obs->begin_timestep!(obs,ctx), observables)
            if analyize_specific_times #***capaz debería haber directamente dos versiones de este loop para no chequear tantas veces este if ni adentro del loop en partículas !
                foreach(obs->begin_timestep_specific_times!(obs,ctx), observables)
            end
            
            @simd for i in 1:n_non_corrupted
                p  = ctx.particles[i] ####**creo q mejor no usarlos estos? preguntar al chat si es o no más eficiente, prob se pueda dejar así xq es local idk. idem con declarar x,y,theta,etc afuera (xq x_tprevious xej no los declaré previamente)
                p_u = ctx.particles_unwrapped[i]

                # Store previous position BEFORE updating
                x_t_previous = p.x
                y_t_previous = p.y


                # Update current position from row data
                x = ctx.row[i*5-3]; y = ctx.row[i*5-2]
                p.x = fastparse(x) #fastparse( x)
                p.y = fastparse(y)
                θ = ctx.row[i*5-1]
                p.θ = fastparse(θ)
                p.vx = fastparse(ctx.row[i*5]) #ctx.particles
                p.vy = fastparse(ctx.row[i*5+1] ) 

                if do_plot 
                    plot_ellipse_better(fig_ellipse,params.a,params.b,p.x,p.y,p.θ, params.Lx,params.Ly)
                end

                # Calculate displacement with periodic boundary correction
                dx = p.x - x_t_previous
                dy = p.y - y_t_previous

                dx -= round(dx*ctx.inv_Lx) * params.Lx #params.Lx * floor(dx/params.Lx + 0.5)
                dy -= round(dy*ctx.inv_Ly) * params.Ly #params.Ly * floor(dy/params.Ly + 0.5)

                # Update unwrapped position
                p_u.x += dx
                p_u.y += dy

                # Calculate MSD from initial position
                foreach(obs->accumulate!(obs,ctx,i), observables)

                if analyize_specific_times
                    foreach(obs->accumulate_specific_times!(obs,ctx, i), observables)
                end
            end

            foreach(obs->finish_timestep!(obs,ctx, ctx.t), observables)
            if analyize_specific_times
                foreach(obs->finish_timestep_specific_times!(obs,ctx,ctx.t÷ctx.t_split+1), observables)
            end

            if do_plot
                gif_settings(fig_ellipse, params, ctx.time_val, fig_ellipse_title, anim)
            end

            if (ctx.t - 1) % (ctx.n_complete_rows÷5) == 0
                println("Progress: $(ctx.time_val)") # / $(params.Δt*params.steps_output*(n_complete_rows-1))") #n-1 xq el n=1 es t=0
            end
            ctx.t+=1
        end
    end
end



function initialize_context(params::Params, checkpoints_file::String, ctx::AnalysisContext)

    n::Int64 = params.np
    tot_row_length::Int64 = 5*n+1
    ctx.np = n

    ctx.inv_Lx = 1/params.Lx ## 
    ctx.inv_Ly = 1/params.Ly ##

    ctx.row = Vector{SubString{String}}(undef,5*n+1)

    ctx.particles = Vector{ParticleData}(undef,n)
    ctx.particles_unwrapped = Vector{ParticleData}(undef,n) 
    ctx.particles_initial_config = Vector{ParticleData}(undef,n) 

    for i in 1:n
        ctx.particles[i] = ParticleData()
        ctx.particles_initial_config[i] =  ParticleData() 
    end

    ctx.seed = params.seed

    ctx.params = params

    tmax_saved, times_saved, k_checkpoint = load_initial_configuration(checkpoints_file) 
    return tot_row_length, tmax_saved, times_saved, k_checkpoint
end

function find_output_files(main_folder::String, special_filename::String) #**cambiarle el nombre a esta función xd
    n_total_rows::Int64 = 1
    n_complete_rows::Int64 = 1 #**ver qué onda
    n_non_corrupted::Int64 = 1

    k_output::Int64 = count(i->(occursin(special_filename*"_output",i)),readdir(main_folder))


    for k in 1:k_output 
        if isfile(main_folder*special_filename*"_output_short_part$(k).txt")
            n_total_rows+=countlines(main_folder*special_filename*"_output_short_part$(k).txt")-1
        elseif isfile(main_folder*special_filename*"_output_part$(k).txt")
            n_total_rows+=countlines(main_folder*special_filename*"_output_part$(k).txt")-1
        end
    end

    println("Total rows: ",n_total_rows)
    return k_output, n_total_rows
end

function load_initial_configuration(checkpoints_file::String)
    if isfile(checkpoints_file) ####
        tmax_saved, times_saved, k_checkpoint = load_latest_time(checkpoints_file)
    else
        k_checkpoint = 0
        tmax_saved = 0
        times_saved = []
    end

    println("tmax saved: ",tmax_saved," k_checkpoint: ",k_checkpoint)
    return tmax_saved, times_saved, k_checkpoint 
end





function run_parameters_calculations(params::Params, model::AbstractModel, force_constants::ConstantVarsType, main_folder::String, 
                checkpoints_file::String,
                ctx::AnalysisContext, #;
				observables::Tuple{Vararg{AbstractObservable}} #AbstractObservable...
				)

	# Esto es lo que va a ir variando con el seed
    special_filename::String = "$(params.model)_$(params.seed)"
    k_output::Int64, n_total_rows::Int64 = find_output_files(main_folder, special_filename)

    tot_row_length, tmax_saved, times_saved, k_checkpoint = initialize_context(params, checkpoints_file, ctx) 

    do_plot::Bool = params.make_gif || params.plot_particles
    nf::Int64 = maximum([1,floor(params.Lx/params.a/10)])

    ctx.tv = (n_total_rows-2)*params.Δt*params.step
    ctx.model = model
    ctx.force_constants = force_constants
    
    foreach(obs->initialize_per_file!(obs,ctx,n_total_rows-1), observables)
    foreach(obs->initialize_specific_times!(obs,ctx,ctx.n_times_to_compute+1), observables)

    for k in 1:k_output #**k_output es básicamente el nro de files en realidad, cambiar nombre
        print(k,"/",k_output," files. ")

        inputs_file, ctx.n_complete_rows = load_output_log(main_folder*special_filename, k)
        
        if k == 1 
            load_initial_config!(inputs_file, params.np, ctx)            
        end 

        if do_plot
            anim = Animation()
            fig_ellipse::Plots.Plot = plot(aspect_ratio=:equal, legend=true)
            fig_ellipse_title::String = make_gif_title(params, force_constants, model) 
        else
            anim = Animation()
            fig_ellipse = plot()
            fig_ellipse_title = ""
        end

        make_gif_and_calculate_obs_file_k(inputs_file, params, ctx, observables, tot_row_length, tmax_saved, do_plot, nf, anim, fig_ellipse, fig_ellipse_title)
        
        
        # Lo que hacemos acá es cortar el gif cuando termina de analizar un output_partk (k sería k_output) y actualizar el checkpoints_file de make_gif, loadeando el último punto temporal guardado x las dudas
        if do_plot
            which_plot = "elipses"
            #if plot_density 
            #    which_plot = "densities"
            #end
            try
                gif(anim, main_folder*which_plot*"_"*special_filename*"_part"*string(k)*".mp4", fps=10)	
                empty!(fig_ellipse)
                empty!(anim.frames) # Desalocar memoria
                anim = nothing
                save_checkpoint_gif(checkpoints_file, ctx.time_val, k)
                tmax_saved, times_saved, k_checkpoint = load_latest_time(checkpoints_file)
            catch y
                println(y,"Gif could not be made.")
            end
        end
        GC.gc()
    end		
end



function run_just_make_gif(params::Params, model::AbstractModel, force_constants::ConstantVarsType, main_folder::String, checkpoints_file::String)
    # Esto es lo que va a ir variando con el seed
    special_filename::String = "$(params.model)_$(params.seed)"
    inputs_file::String = main_folder*special_filename*"_output.txt"
    #output_file::String = main_folder * special_filename * "_P_Q_correl.txt"

    k_output::Int64 = count(i->(occursin(special_filename*"_output",i)),readdir(main_folder))

    # **pensar cómo agregar estos después al variables_input
    plot_density::Bool = false
    display_plots::Bool = false

    do_plot_orig::Bool = params.make_gif || display_plots
    do_plot::Bool = do_plot_orig

    buf::EllipseBuffers = EllipseBuffers()

    n::Int64 = params.np
    tot_row_length::Int64 = 5*n+1
    
    if do_plot
        anim = Animation()
    end
    fig_ellipse::Plots.Plot = plot(aspect_ratio=:equal, legend=true)
    fig_ellipse_title::String = ""

    model_forces = model
    fig_ellipse_title = make_gif_title(params, force_constants, model)

    n_total_rows::Int64 = 1
    n_complete_rows::Int64 = 1
    n_non_corrupted::Int64 = 1 
    
    for k in 1:k_output
        if isfile(main_folder*special_filename*"_output_short_part$(k).txt")
            n_total_rows+=countlines(main_folder*special_filename*"_output_short_part$(k).txt")-1
        elseif isfile(main_folder*special_filename*"_output_part$(k).txt")
            n_total_rows+=countlines(main_folder*special_filename*"_output_part$(k).txt")-1
        end
    end

    t::Int64 = 1
    time_val::Float64 = 1.0

    println("Total rows: ",n_total_rows)

    row::Vector{SubString{String}} = Vector{SubString{String}}(undef,5*n+1)
    
    θ_df::String = "0.0"; x_df::String = "0.0"; y_df::String = "0.0" ##
    θ::Float64 = 0.0; x::Float64 = 0.0; y::Float64 = 0.0

    m = 0

    p::ParticleData = ParticleData()
    particles::Vector{ParticleData} = Vector{ParticleData}(undef,n)

    for i in 1:n
        particles[i] = p  #ParticleData()
    end


    if isfile(checkpoints_file) ####
        tmax_saved, times_saved, k_checkpoint = load_latest_time(checkpoints_file)
    else
        k_checkpoint = 0
        tmax_saved = 0.0
        times_saved = []
    end

    println("tmax saved: ",tmax_saved," k_checkpoint: ",k_checkpoint)

    for k in 1:k_output
        do_plot = do_plot_orig && k>=k_checkpoint ##

        if do_plot
            anim = Animation()
        else
            anim = nothing
        end

        if !isfile(main_folder*special_filename*"_output.txt")
            if isfile(main_folder*special_filename*"_output_part$(k).txt")
                inputs_file = main_folder*special_filename*"_output_part$(k).txt"#"_output_short_part$(k).txt"
            elseif isfile(main_folder*special_filename*"_output_short_part$(k).txt")
                inputs_file = main_folder*special_filename*"_output_short_part$(k).txt"
            end
        end

        println(k,"/",k_output," files. Now calculating:  ",inputs_file)

        n_complete_rows = countlines(inputs_file)#nrow(df)

        println("Complete rows: ",n_complete_rows)
  

        open(inputs_file) do file
            for line in Iterators.drop(eachline(file), 1) # skip first line
                
                row = split(line,"\t")
                #@views row = df[t,:] #Para que no cree una copia
                time_val = fastparse(row[1])

                do_plot = do_plot_orig && time_val>=tmax_saved ##

                n_non_corrupted = n

                if length(row)<tot_row_length || (row[tot_row_length] isa String && occursin(NULL_CHAR, row[tot_row_length]))  #typeof(row[n*5+1])<:String && occursin("\0", row[n*5+1])
                    println(t," ",time_val, " Something went wrong (not sure what) but we can ignore this line.")
                    continue
                end

                @simd for i in 1:n_non_corrupted
                    x_df = row[i*5-3]
                    y_df = row[i*5-2]
                    θ_df = row[i*5-1]

                    x = fastparse(x_df)
                    y = fastparse(y_df)
                    θ = fastparse(θ_df)

                    if do_plot
                        plot_ellipse_better(fig_ellipse,buf,model_forces,params.a,params.b,x,y,θ, params.Lx,params.Ly)
                    end
                end

                if plot_density
                    plot_spatial_density(params.Lx,params.Ly,particles)
                end

                if do_plot
                    scatter!(fig_ellipse,[0,0], [0,1], zcolor=[0,2], markersize=0.0, label=false,colorbar_ticks=[0,0.5,1.0,1.5], c=:hsv, colorbar_title="Angles (1/π)")
                    
                    xlims!(-params.b*2,params.Lx+2*params.b)
                    ylims!(-params.a*2,params.Ly+2*params.a)
                    title!(fig_ellipse,"t=$(time_val)"*fig_ellipse_title)

                    if params.make_gif
                        frame(anim,fig_ellipse)
                        empty!(fig_ellipse)
                    end
                    if display_plots
                        display(fig_ellipse)
                        gui(); sleep(0.1)
                        empty!(fig_ellipse)
                    end 
                end
                if (t - 1) % (n_complete_rows÷5) == 0
                    println("Progress: $time_val") # / $(params.Δt*params.steps_output*(n_complete_rows-1))") #n-1 xq el n=1 es t=0
                end
                t+=1 
            end
        end

        which_plot::String=" "
        if do_plot
            which_plot = "elipses"
        end
        if plot_density 
            which_plot = "densities"
        end
        println(params.make_gif," ",tmax_saved," ",time_val)
        if do_plot && tmax_saved<time_val ####
            try
                gif_files = count(i->(occursin(".mp4",i)),readdir(main_folder))+1
                gif(anim, main_folder*which_plot*"_"*special_filename*"_part"*string(gif_files)*".mp4", fps=10)	
                empty!(fig_ellipse)
                empty!(anim.frames) # Desalocar memoria
                anim = nothing
                save_checkpoint_gif(checkpoints_file, time_val, k)
                tmax_saved, times_saved, k_checkpoint = load_latest_time(checkpoints_file)
            catch y
                println(y,"Gif could not be made.")
            end
        end
        GC.gc()
    end
    
end

function main_config_analysis(main_folder::String, code_directory::String; just_plot::Bool=true)
    for inputf in 0:0#count(i->(occursin("input_parameters",i)),readdir(main_folder))
        for Li in ["test_wound2"] #["L1","L2","L3","L4","L5"]
            if isempty(ARGS)
                L = Li
            else
                L = ARGS[1]
                inputf = parse(Int64, ARGS[2])
                just_plot = parse(Bool, ARGS[3])
            end
            inputs_file::String = "$(main_folder)input_parameters$(inputf)_$L.txt"
            print(inputs_file," in  ")

            params = read_input_parameters(inputs_file, main_folder)   # only to learn the model name
            println(inputs_file, "  ",code_directory)
            load_model!(code_directory, params)
            Base.invokelatest(config_analysis_with_model, main_folder, params; just_plot)   # ← the single wall
        end
    end
end

function config_analysis_with_model(main_folder::String, params::Params; just_plot::Bool=true)

    model = get_model()    
    aux_force_variables = model_aux_structs(params, model)           
    new_main_folder::String = load_input_folder(main_folder,params, model)
    println(new_main_folder)
    #println(main_folder, " ",count(i->(occursin("output_part1.txt",i)),readdir(main_folder)))

    nseeds::Int64 = 1#count(i->(occursin("output_part1.txt",i)),readdir(new_main_folder))+count(i->(occursin("output.txt",i)),readdir(new_main_folder))+count(i->(occursin("output_short_part1.txt",i)),readdir(new_main_folder))

    checkpoints_file::String = "$(new_main_folder)gif_checkpoint.txt"

    
    if just_plot
        for i in 1:nseeds
            if i==1
                params.make_gif = true
                run_just_make_gif(params, model, aux_force_variables.constant, new_main_folder, checkpoints_file)
            else
                params.make_gif = false
            end
            println(params.seed)
            params.seed = params.seed + 1 #**capaz el seed debería leerlo del nombre del archivo o alguasi ? sólo importa para los nombres acá no hay nada random
        end
    else

        ctx = AnalysisContext()
        ctx.n_times_to_compute::Int64 = 15
        ctx.save_folder = new_main_folder

        foreach(obs->initialize!(obs,params.np, nseeds), observables)

        for s in 1:nseeds
            if s==1
                params.make_gif = true
            else
                params.make_gif = false
            end
            println(params.seed)
            run_parameters_calculations(params, model, aux_force_variables.constant, new_main_folder, checkpoints_file, ctx, observables)

            foreach(obs->finish_seed_stats!(obs, ctx, s), observables)
            params.seed += 1 
        end

        foreach(obs->finish_loop_in_seed!(obs,ctx, nseeds), observables)
        foreach(obs->plot_everything(obs,ctx, new_main_folder), observables)

        GC.gc()

    end
end

main_config_analysis(joinpath(@__DIR__, "prueba/"), @__DIR__, just_plot=false)

