include(joinpath(@__DIR__, "wca.jl"))
include(joinpath(@__DIR__, "move_particles.jl"))
include(joinpath(@__DIR__, "plotting.jl"))
include(joinpath(@__DIR__, "file_management.jl"))


function run_simulation(params::Params, main_folder::String)
    # Esto es lo que va a ir variando con el seed
    special_filename::String = params.model*"_"*string(params.seed)
    output_file::String = main_folder*special_filename*"_output.txt"
    checkpoints_file::String = main_folder*special_filename*"_particles_checkpoint.jld2"

    Random.seed!(params.seed) # si lo saco, cambia la condición inicial idk

    k_output::Int64 = count(i->(occursin(special_filename*"_output",i)),readdir(main_folder))
    k_output = k_output==0 ? 1 : k_output
    output_file = main_folder*special_filename*"_output_part$(k_output).txt"

    particles = Particles(n=params.np)
    particle_i = Particle()

    t0::Int64 = 1
    tf::Int64 = params.n_of_points

    fig_ellipse1 = plot(aspect_ratio=:equal, legend=true)
    
    xlims!(fig_ellipse1,-params.b*2,params.Lx+2*params.b)
    ylims!(fig_ellipse1,-params.a*2,params.Ly+2*params.a)
    scatter!(fig_ellipse1,[0,0], [0,1], zcolor=[0,2], markersize=0.0, label=false,colorbar_ticks=[0,0.5,1.0,1.5], c=:hsv, colorbar_title="Angles (1/π)")
        
    
    model_forces = get_model()
    aux_force_variables = model_aux_structs(params, model_forces)

    #if params.ηa > 0.0
    noise_variables = NoiseVariables(params=params)
    #end
    

    geom_ws::GeometryVarsType = aux_force_variables.geom
    nematic_ws::NematicVarsType = aux_force_variables.nematic
    aux::AuxVarsType = aux_force_variables.aux
    force_ws::ForceVarsType = aux_force_variables.force
    constant_ws::ConstantVarsType = aux_force_variables.constant


    orientation_config  = get_orientation_config()
    position_config     = get_position_config()


    if params.start_run != 0.0 && isfile(checkpoints_file) # Si no se llegó a crear el archivo de checkpoints, arranca de 0

        time::Int64, data::Particles, state_rng::Xoshiro = load_latest(checkpoints_file) # load_latest si start_run = -1.0, si no, debería ser load_checkpoint(checkpoints_file,params.start_run)

        copy!(Random.default_rng(),state_rng)

        particles = data

        println("Time from checkpoints file: ", time)
        
        if isfile(output_file)
            if params.make_output_file
                # La función truncate_file_after_match borra las líneas después del t de checkpoint
                # Y devuelve el índice para sacar ese último t

                if params.start_run == -1.0
                    #t0 = truncate_file_after_match(output_file,format_time_string(time-1,params.Δt)) * params.step +1 #+1 porque este paso ya lo hiciste !
                    t0 = round(Int64,truncate_file_after_time_value(output_file,time*params.Δt) / params.Δt) # quiero el índice desde donde seguir
                
                    while t0 == (-999999999999999.0 / params.Δt) #(t0-1)/ params.step == -999999999999999.0 # quiere decir que no encontró en el output part$k la línea con t = time del checkpoint
                        k_output -= 1
                        output_file = main_folder*special_filename*"_output_part$(k_output).txt"
                        if isfile(output_file)
                            #t0 = truncate_file_after_match(output_file,format_time_string(time-1,params.Δt)) * params.step +1 
                            t0 = round(truncate_file_after_time_value(output_file,time*params.Δt) / params.Δt)
                            
                            rm(main_folder*special_filename*"_output_part$(k_output+1).txt")
                        else
                            # No se puede tener un time que no tenga una línea correspondiente en el archivo de output (conviene empezar de nuevo si se corrompió el output)
                            throw("ERROR while reading outputs: couldn't find output file part$(k_output).")
                        end
                    end
                else
                    # **bueno, como acá podría tener un t intermedio, va a estar bien o mal cortar el archivo de outputs dependiendo de si quiero o no sacar datos intermedios !!!
                    t0 = time #time##**tiene que depender del time, que sea el que más cerca está (por abajo) de start_run. 
                end

                output = open(output_file, "a+")
            else
                if params.start_run == -1.0
                    t0 = time#last_time_from_file(output_file) / params.Δt #countlines(output_file)* params.step -1
                else
                    t0 = params.start_run
                end
                output = open(main_folder*"input_parameters.txt", "r")
            end
        else
            if params.make_output_file
                output = open(output_file, "w")
                print(output, "t")

                for i in 1:params.np
                    print(output,"\tx$(i)\ty$(i)\tθ$(i)\tvx$(i)\tvy$(i)")
                end
                print(output,"\n0.0")

                time = 1
            else
                output = open(main_folder*"input_parameters.txt", "r")
            end

            init_particles!(particles, params, output, fig_ellipse1, model_forces, orientation_config, position_config, main_folder)
        end

        println("from checkpoint: ",time," from file: ",t0)

        t0 = t0+1 # I want t0 to be the step following the last one saved in the checkpoint

        if time+1 != t0
            println("REQUETE RECONTRA ERROR")
            throw("ERROR no sé qué pasó***")
        end

        
        total_n_of_points = parse(Int,split(find_line!(main_folder*"input_parameters.txt", "n_of_points"),",")[2])

        # If the code stopped before the n_of_points from the inputs file was completed, then run the time steps missing
        if t0 < total_n_of_points
            params.start_run = -1.0
            tf = total_n_of_points - time 
            println("Loop was not finished. Completing time steps for this file. Steps missing: $(tf) out of $(total_n_of_points).")
        end

    #elseif params.start_run == -1.0 && isfile(main_folder*special_filename*"_particles_initial.jld2")
        
    else

        if params.make_output_file
            output = open(output_file, "w")
            print(output, "t")

            for i in 1:params.np
                print(output,"\tx$(i)\ty$(i)\tθ$(i)\tvx$(i)\tvy$(i)")
            end
            print(output,"\n0.0")
        else
            output = open(main_folder*"input_parameters.txt")
        end

        start_from_ref = Bool

        # Maybe you want to load a checkpoints file from another run as the particles initial condition
        if isfile(main_folder*special_filename*"_particles_initial.jld2")
            # En esta situación, el t0 no es el del checkpoint sino 1, porque queremos reiniciar la simulación con el tf de otro lado.
            # Para agregar seeds, hay que hacer un copy+paste con el nombre del seed que querés para el mismo archivo

            time, data, state_rng = load_latest(main_folder*special_filename*"_particles_initial.jld2")

            copy!(Random.default_rng(),state_rng)

            particles = data

            start_from_ref = true

        else 
            start_from_ref = false
        end

        #t0 = 1
        init_particles!(particles, params, output, fig_ellipse1, model_forces, orientation_config, position_config, main_folder, start_from_ref)

    end


    if params.make_gif
        anim = Animation()
        if t0 == 1
            frame(anim,fig_ellipse1)
        end
        
        fig_ellipse_title = make_gif_title(params, constant_ws, model_forces)
    end

    k::Int64 = count(i->(occursin(".mp4",i) && occursin("elipses_"*special_filename,i)),readdir(main_folder))
    params.rc = params.model == "vandamme" ? 2.0*params.a + constant_ws.potential_threshold-2*params.b : 5.0*params.a
    NL = NeighbourList(params.rectangle, particles.pos, params.rc) 

    neighbors::Vector{Vector{Int64}} = []
    for i in 1:params.np
        push!(neighbors,[])
        foreach_neighbor(NL, i) do j,d
            push!(neighbors[i],j)
        end
    end
    
    maxdisp::Float64 = 0.0

    nf::Int64 = maximum([1,floor(Int, 0.03 * minimum([params.Lx, params.Ly]) ÷ (params.c*params.Δt))]) # avanza el 3% de la caja ?
    if params.activity == false
        nf = params.step
    end
    if occursin("wound", params.initial_position)
        nf = maximum([1,floor(Int, 0.0025 * maximum([params.Lx, params.Ly]) ÷ (params.c*params.Δt))])
    end

    output_condition = Bool
    do_plot::Bool = params.plot_particles || params.make_gif
   
    println("Running from step ",t0," to step ",t0+tf-1)
    for t in t0:t0+tf-1 # -1 porque querés que saque n_of_points datos (just julia things)
        output_condition = params.make_output_file && t%params.step==0 && (!params.is_there_block || 0<=(t%params.block_step)<=params.block_length) #(t-t0+1)

        if output_condition
            print(output,"\n"*format_time_string(t,params.Δt))
            #println("particles2 ",particles)
        end
        if do_plot
            if t%nf==0 # si hay un gif de output, no tiene sentido acortar los frames xq no se entiende cómo se mueven !
                println(t)
                fig_ellipse = plot(aspect_ratio=:equal, legend=true)
            end        end

        get_forces_external!(particles, params, model_forces, geom_ws, nematic_ws, aux, force_ws, constant_ws, neighbors)
        post_pair_forces!(model_forces, particles, params,
                           geom_ws, nematic_ws, aux, force_ws, constant_ws,
                           neighbors)

        for i in 1:params.np
            particle_i.pos .= particles.pos[i]
            particle_i.vel .= particles.vel[i]
            particle_i.θ = particles.angles[i]
            particle_i.omega = particles.omega[i]
            particle_i.cos_i = particles.cosines[i]
            particle_i.sin_i = particles.sines[i]

            maxdisp = move_particle_prueba!(params, particle_i, geom_ws, nematic_ws, aux, force_ws, constant_ws, noise_variables, maxdisp)#, fig_ellipse) 

            particles.pos[i] .= particle_i.pos
            particles.vel[i] .= particle_i.vel
            particles.angles[i] = particle_i.θ
            particles.omega[i] = particle_i.omega
            particles.cosines[i] = particle_i.cos_i
            particles.sines[i] = particle_i.sin_i

            if do_plot
                if t%nf==0
                    plot_ellipse_better(fig_ellipse,params.a,params.b, particle_i.pos[1],particle_i.pos[2],particle_i.θ, params.Lx,params.Ly)
                    #plot_ellipse(fig_ellipse,params.a,params.b, particle_i.pos[1],particle_i.pos[2],particle_i.θ, params.Lx,params.Ly)
                end
            end
            if output_condition
                print(output,"\t",particle_i.pos[1],"\t",particle_i.pos[2],"\t",particle_i.θ,"\t",particle_i.vel[1],"\t",particle_i.vel[2])
            end
        end
        update_NL_and_neighbors!(NL, sqrt(maxdisp), neighbors, t, particles)

        if do_plot
            if t%nf==0
                scatter!(fig_ellipse,[0,0], [0,1], zcolor=[0,2], markersize=0.0, label=false,  colorbar_ticks=[0,0.5,1.0,1.5],c=:hsv, colorbar_title="Angles (1/π)")

                xlims!(fig_ellipse,-params.b*2,params.Lx+2*params.b)
                ylims!(fig_ellipse,-params.b*2,params.Ly+2*params.b)
                title!(fig_ellipse,"t=$(format_time_string(t, params.Δt)), $(fig_ellipse_title)")
                
                if params.plot_particles
                    display(fig_ellipse)
                    gui(); sleep(0.1)
                end

                if params.make_gif
                    frame(anim,fig_ellipse)
                end
            end
        end
            # Checkpoint every 5e6 outputs (5e6/np lines or times saved; it's approximately 500MiB worth of data in the output.txt)
        #println(t,"   ",t%floor(params.step*1e4/params.np))
        if t%(floor(params.step*5e6/params.np))==0 && t!=(t0+tf-1) # Si step es menor que 1000, no hace falta sacar checkpoint we

            local state = copy(Random.default_rng())
            save_checkpoint(checkpoints_file, t, particles, state)

            # We create another file if filesize is over 1GiB
            if params.make_output_file && filesize(output_file)>1e9#**1e9

                k_output += 1

                output_file = main_folder*special_filename*"_output_part"*string(k_output)*".txt"

                close(output)

                output = open(output_file, "w")
                print(output, "t")

                for i in 1:params.np
                    print(output,"\tx$(i)\ty$(i)\tθ$(i)\tvx$(i)\tvy$(i)")
                end



            end

        end



    end

    

    close(output)
    

    if params.make_gif 
        gif(anim, main_folder*"elipses_"*special_filename*"_part"*string(k+1)*".mp4")
        empty!(anim.frames) # Desalocar memoria
        anim = nothing
       
    end

    t = t0+tf-1 #t0+params.n_of_points
    state = copy(Random.default_rng())
    save_checkpoint(checkpoints_file, t, particles, state)
    GC.gc()  # force garbage collection
end


function main_simulation(main_folder, code_directory)
    for inputf in 0:0#count(i->(occursin("input_parameters",i)),readdir(main_folder))
        for Li in ["test_wound7"] #["L1","L2","L3","L4","L5"]
            if isempty(ARGS)
                L = Li
            else
                L = ARGS[1]
                inputf = parse(Int64, ARGS[2])
            end
            inputs_file::String = main_folder*"input_parameters"*string(inputf)*"_$L.txt"
            print(inputs_file," in  ")

            params = read_input_parameters(inputs_file, main_folder)   # only to learn the model name
            load_model!(code_directory, params)
            load_initial_orientation!(code_directory, params)
            load_initial_position!(code_directory, params)
            Base.invokelatest(run_with_model, main_folder, params)   # ← the single wall
	    end
    end
end

function run_with_model(main_folder, params)
    model = get_model()                                         # native call, new world
    new_main_folder = initialize_input(params, main_folder, model)

    println(new_main_folder)
    #initial_conditions = ["random", "polarized_ordered", "nematic_ordered"]

    n_of_checkpoint_files = count(i->(occursin("particles_checkpoint.jld2",i)),readdir(new_main_folder))
    new_input_parameters = new_main_folder*"input_parameters.txt"

    first_file = 1
    last_file = params.n_files_to_av
    
    # Priorizo que se mantenga el step frente a n_of_points, al cual modifico para que sea divisible por step
    if params.n_of_points%params.step!=0
        params.n_of_points = floor(Int,params.n_of_points/params.step)*params.step
    end

    if n_of_checkpoint_files < params.n_files_to_av && n_of_checkpoint_files>0
        println("Loop was not finished. Completing original sequence.")
        
        first_file = n_of_checkpoint_files + 1

        params.seed = params.seed + n_of_checkpoint_files

        if parse(Int,split(find_line!(new_input_parameters, "n_files_to_av"),",")[2]) < params.n_files_to_av
            replace_line!(new_input_parameters, "n_files_to_av", "n_files_to_av,"*string(params.n_files_to_av))
        end

    else
        if params.start_run == 0.0
            #quiero que siga corriendo más archivos para sacar promedios
            params.seed = params.seed + n_of_checkpoint_files
            replace_line!(new_input_parameters, "n_files_to_av", "n_files_to_av,"*string(params.n_files_to_av + n_of_checkpoint_files))
        
        else
            if n_of_checkpoint_files>0
                #quiero que vuelva a los .jld2 que ya corrieron y los siga de ahí
                old_n_of_points = split(find_line!(new_input_parameters, "n_of_points"),",")[2]
                replace_line!(new_input_parameters, "n_of_points", "n_of_points,"*string(params.n_of_points+parse(Int,old_n_of_points)))
            end
        end
    end

    for repetition in first_file:last_file
        if repetition > 1
            params.make_gif = false
            params.plot_particles = false
        end
        
        println("Seed: ",params.seed)
        run_simulation(params, new_main_folder) 
        params.seed = params.seed + 1
    end
end

main_simulation(joinpath(@__DIR__, "prueba/"), @__DIR__)
