using JLD2
#using FileIO
using DataFrames
using CSV


include("./particle_objects.jl")
include("./model_forces.jl")


#**creo que algo que me falta agregarle es chequear que los archivos de input de afuera y el que potencialmente
# esté adentro de la carpeta correspondiente sean iguales 


# Esto sí correr una sola vez
function assign_input_parameters_from_file!(input_parameters_file::String, params::Params, main_folder::String)
    if !isfile(input_parameters_file) # ***capaz debería comparar que el archivo de input tenga los inputs que corresponden ?

        #**de hecho los inputs deberían ser algo legible, y este jld2 debería ser
        # solamente para RNG, x,y,θ,t. el checkpoint no debería sobreescribir el archivo 
        # anterior x las dudas que se corte mientras escribe, pero sí debería ir borrando los archivos viejos

        # **seed lo pongo por las dudas en variables file, tipo la simulación arrancó acá (el RNG no puedo xd)
        variables = open(input_parameters_file,"w")
        print(variables,"a,"*string(params.a)*"\n")
        print(variables,"b,"*string(params.b)*"\n")

        print(variables,"Lx,"*string(params.Lx)*"\n")
        print(variables,"Ly,"*string(params.Ly)*"\n")
        print(variables,"np,"*string(params.np)*"\n")

        print(variables,"v,"*string(params.c)*"\n")

        print(variables,"eta_a,"*string(params.ηa)*"\n")
        print(variables,"eta_b,"*string(params.ηb)*"\n")
        print(variables,"eta_r,"*string(params.ηr)*"\n")

        print(variables,"Delta_t,"*string(params.Δt)*"\n")
        print(variables,"n_of_points,"*string(params.n_of_points)*"\n")
        print(variables,"steps_output,"* join(params.steps_output,"/")*"\n") # string(params.steps_output)*"\n")
        print(variables,"n_files_to_av,"*string(params.n_files_to_av)*"\n")

        print(variables,"seed,"*string(params.seed)*"\n")
        print(variables,"model,"*string(params.model)*"\n")

        print(variables,"initial_position,"*params.initial_position*"\n")
        print(variables,"initial_orientation,"*params.initial_orientation*"\n")

        print(variables,"activity,"*string(params.activity)*"\n")

        print(variables,"make_gif,"*string(params.make_gif)*"\n")
        print(variables,"plot_particles,"*string(params.plot_particles)*"\n")
        print(variables,"make_output_file,"*string(params.make_output_file)*"\n")

        #print(variables,"start_run,"*string(params.start_run)*"\n")
        print(variables,"start_run,"*string(params.start_run)*"\n") # Si no existe el archivo de inputs en la carpeta, arrancamos de 0
        

        

        close(variables)
    else
        variables = CSV.File(input_parameters_file, delim=",", header=false) |> DataFrame

        #**what if me confundo y start_run está en main_folder en vez de new_main_folder ? x ej si el inputs no existe y lo crea de acá
        if variables[22,1] == "start_run" # file is from new_main_folder
            # a, b, Lx, Ly, np, c, k, γ, α, β, η, model, activity and initial_condition are inside new_main_folder's name
            # so we don't need to check if they are the same as the input_file inside of it.
            # make_gif, plot_particles and make_output_file are not very relevant if changed
            # **i think that seed and n_files_to_av might be different because you want to run more files
            
            if params.Δt != parse(Float64,variables[10,2]) #|| params.steps_output != parse(Int64,variables[16,2]) #params.n_of_points != parse(Int64,variables[15,2])
                # If any of the time step characteristic is different, we can't use this input file

                #assign_input_parameters_from_file!("second_run_"*input_parameters_file, params::Params)
                if  count(i->(occursin("particles_checkpoint.jld2",i)),readdir(main_folder)) > 0 # If checkpoint was made with another Δt, raise error
                    throw("ERROR: input_file from main_folder is different from input_file inside $(main_folder). \nCheck input_file from example to see the structure required to run.")
                else
                    # Delete input file and replace it with new one
                    rm(input_parameters_file)
                    assign_input_parameters_from_file!(input_parameters_file, params, main_folder)
                end

            # Else (si los params relevantes son todos iguales), no hace falta hacer nada porque ya cargaste 
            # los params cuando corriste en main_folder
                
            end

            #*params.n_files_to_av != parse(Int64,variables[17,2]) y params.seed son raros
        elseif variables[22,1] == "finished_loop" # file is from main_folder

            #******** acá creo que debería invocar la función load_input_parameters


            # no need to do this if input file exists inside new_main_folder; we already loaded these params from main_folder
            params.a = parse(Float64,variables[1,2])
            params.b = parse(Float64,variables[2,2])

            params.inv_a2 = 1/(params.a*params.a)
            params.inv_b2 = 1/(params.b*params.b)
            params.diag2 = 1/(params.a*params.a+params.b*params.b)
            params.ϵ = (params.a*params.a-params.b*params.b)*params.diag2

            params.Lx = parse(Float64,variables[3,2])
            params.Ly = parse(Float64,variables[4,2])
            params.rectangle = PeriodicRectangle(params.Lx,params.Ly)
            params.np = parse(Int64,variables[5,2])

            params.c = parse(Float64,variables[6,2])

            params.ηa = parse(Float64,variables[7,2])
            params.ηb = parse(Float64,variables[8,2])
            params.ηr = parse(Float64,variables[9,2])
            if params.a == params.b 
                params.μa = 1/params.a
                params.μb = 1/params.a
                params.μr = 0.75/params.a^3
            elseif params.a < params.b
                throw("ERROR: a=$(params.a) (major axis) is smaller than b=$(params.b) (minor axis)")
            else
                params.μa = 0.75*(-params.a*params.diag2/params.ϵ+(2*params.a*params.a-params.b*params.b)*(params.diag2/params.ϵ)^1.5*log(params.a/params.b+sqrt(params.a*params.a*params.inv_b2-1)))              # Mobility in the major axis direction
                params.μb = 0.75*(params.a*0.5*params.diag2/params.ϵ+(params.a*params.a-1.5*params.b*params.b)*(params.diag2/params.ϵ)^1.5*log(params.a/params.b+sqrt(params.a*params.a*params.inv_b2-1)))              # Mobility in the minor axis direction
                params.μr = 1.5*params.μa*params.diag2              # Mobility in the rotation
                #println(params.μa, " ",params.μb, " ", params.μr)
            end

            params.Δt = parse(Float64,variables[10,2])
            params.n_of_points = parse(Int64,variables[11,2])
            params.steps_output = parse.(Int,split(variables[12,2],"/"))  # parse(Int64,variables[16,2])
            params.is_there_block = length(params.steps_output)==3 #****esto creo que no lo usé al final o sí ?
            params.step = params.steps_output[1]         
            params.block_length = params.is_there_block ? params.steps_output[2] : 1           
            params.block_step = params.is_there_block ? params.steps_output[3] : 1 
            params.n_files_to_av = parse(Int64,variables[13,2])

            params.seed = parse(Int64,variables[14,2])
            params.model = variables[15,2]

            params.initial_position = variables[16,2]
            params.initial_orientation = variables[17,2]

            #*************
            #if params.model == "vandamme"
            #    params.μa = 1/params.b
            #    params.μb = params.μa
            #    params.μr = 0.75*params.μa*params.inv_b2
            #end
            #*************

            params.activity = variables[18,2] == "false" ? false : variables[18,2] == "true" ? true : throw("ERROR: not a valid activity input. Change it to either true or false in inputs_file.")#parse(Bool,variables[19,2])

            #params.start_run = parse(Bool,variables[20,2])
            params.make_gif = variables[19,2] == "false" ? false : variables[19,2] == "true" ? true : throw("ERROR: not a valid make_gif input. Change it to either true or false in inputs_file.")#parse(Bool,variables[21,2])
            params.plot_particles = variables[20,2] == "false" ? false : variables[20,2] == "true" ? true : throw("ERROR: not a valid plot_particles input. Change it to either true or false in inputs_file.")#parse(Bool,variables[22,2])
            params.make_output_file = variables[21,2] == "false" ? false : variables[21,2] == "true" ? true : throw("ERROR: not a valid make_output_file input. Change it to either true or false in inputs_file.")#parse(Bool,variables[23,2])

            params.start_run = variables[22,2] == "false" ? -1.0 : variables[22,2] == "true" ? 0.0 : parse(Float64,variables[22,2]) # throw("ERROR: not a valid finished_loop input. Change it to either true or false in inputs_file.")#false    # Si el archivo de inputs ya existe en la carpeta, asumo que quiero seguir en más t
            

            params.main_folder = main_folder
        else 
            throw("ERROR: order or number of parameters inside input_file is wrong. Check input_file from example to see the structure required to run.")
            
        end
        # ahora comparar si las variables son todas iguales. algunas pueden no serlo como finished_loop con start_run, o n_files_to_av. 
        
        

    end
end

function format_time_string(t, Δt)
    string_title::String = string(round(t*Δt,digits=length(string(Δt))))
    digits_title::Int64 = length(split(string_title,".")[2])
    digits_step::Int64 = length(split(string(Δt),".")[2])
    if digits_title != digits_step
        string_title = string_title*"0"^(digits_step-digits_title)
    end
    return string_title
end

function read_input_parameters(inputs_file::String, main_folder::String)
    params = Params()
    assign_input_parameters_from_file!(inputs_file, params, main_folder)
    return params                                   # no model needed
end

function initialize_input(params::Params, main_folder::String, model::AbstractModel)
    #params::Params = Params()

    # De inputs_file toma los inputs, pero después lo corro de nuevo para generar el input_file en la carpeta correspondiente
    #assign_input_parameters_from_file!(inputs_file, params, main_folder)
    
    aux_force_variables = model_aux_structs(params, model)
    
    subfolder_name::String = "$(params.model)/act_$(params.activity)_pos_$(params.initial_position)_ang_$(params.initial_orientation)_a_$(params.a)_b_$(params.b)_Lx_$(params.Lx)_Ly_$(params.Ly)_np_$(params.np)_v_$(params.c)_eta_$(params.ηa)"
    subfolder_name = Base.invokelatest(make_model_specific_subfolder_name, params, aux_force_variables.constant, subfolder_name, model)
    params.subfolder_name = subfolder_name

    mkpath(main_folder*subfolder_name) # Create a directory (if it doesn't exist already) to save the files
    new_main_folder::String = main_folder*subfolder_name*"/"
    input_parameters_file::String = new_main_folder*"input_parameters.txt" 

    assign_input_parameters_from_file!(input_parameters_file, params, new_main_folder)

    return new_main_folder
end

# **ponerle nombre bien a la función we
function truncate_file_after_match(filename::String, match_string::String)
    # Read all lines from the file
    lines = readlines(filename)
    println("Time from checkpoints file: ", match_string)

    first_time = split(lines[2],"\t")[1]
    last_time = split(lines[end],"\t")[1]


    println("First line from output file: ",first_time,"; Last time from output file: ",last_time)

    # Find the first occurrence of a line that starts with `match_string`
    cutoff_index = findfirst(l -> startswith(l, match_string), lines)
    #println(cutoff_index)

    if cutoff_index !== nothing 
        # Keep only the lines before the matched line
        if cutoff_index < length(lines)
            lines = lines[1:cutoff_index]
        else # no hace falta correr la parte que recorta
            return length(lines)-2 #-2 si k=1, -1 otherwise *******
        end
    else # no sé qué pudo haber salido mal pero es un error si el times que guardó el checkpoint es mayor que el t de la última línea de output
        # o si se guardó un checkpoint con un t intermedio a los t guardados en el output, también salió mal algo
        # si se guardó un times de checkpoint menor que el t de la última línea de output, se trunca desde ahí, pero podría haber un cutoff_index == nothing
        # si pasó eso, es porque el t correcto está en el output con k anterior.

        if parse(Float64,first_time) > parse(Float64, match_string)
            return -999999999999999999
        else
        # if el primer t del file es mayor que el time de jld2, entonces mirar lo mismo en el file anterior y borrar el file actual una vez que encuentre el t correspondiente !
            throw("ERROR while reading outputs: checkpoint saved time larger than last time written in output file.")
        end
    end

    # Write the modified content back to the file
    open(filename, "w") do file
        for line in lines[1:end-1]
            println(file, line)  # Write each line back
        end
        #println(chomp(lines[end]))
        print(file,chomp(lines[end])) # chomp te saca de encima el \n
    end
    return cutoff_index-2
end



function truncate_file_after_time_value(filename::String, time::Float64)
    # Read all lines from the file

    first_time = String
    last_time = String

    t_this_line = 0.0
    t_previous_line = 0.0
    t_closest_to_checkpoint = -999999999999999.0

    l = 1
    cutoff_index = 0

    lines = readlines(filename)

    first_time = split(lines[2],"\t")[1]
    last_time = split(lines[end],"\t")[1]

    # if el primer t del file es mayor que el time de jld2, entonces mirar lo mismo en el file anterior y borrar el file actual una vez que encuentre el t correspondiente !
    if parse(Float64,first_time) > time
        return -999999999999999.0
    elseif parse(Float64,last_time) <= time
        return time
    end

    for line in lines[2:end]
        l+=1

        t_this_line = parse(Float64,split(line,"\t")[1])

        # si start_run != -1.0, esto no se corre, así que seguro quiero empezar desde el último checkpoint que se guardó
        if t_this_line >= time >= t_previous_line
            t_closest_to_checkpoint = time
            cutoff_index = l
            break
        end

        t_previous_line = t_this_line
    end

    println("First line from output file: ",first_time,"; Last time from output file: ",last_time)

    if cutoff_index != 0 
        # Keep only the lines before the matched line
        if cutoff_index < length(lines)
            lines = lines[1:cutoff_index]
        else # no hace falta correr la parte que recorta
            return t_closest_to_checkpoint #length(lines)-2 #-2 si k=1, -1 otherwise 
        end
    else # creo que este error no debería saltar nunca
        #if t > last_time, en realidad no sería un error si el time es mayor que el t de la última línea de output !!! podría seguir sacando outputs y listo total tengo el checkpoint con la data
        
        # no sé qué pudo haber salido mal pero es un error si el times que guardó el checkpoint es mayor que el t de la última línea de output
        # o si se guardó un checkpoint con un t intermedio a los t guardados en el output, también salió mal algo
        # si se guardó un times de checkpoint menor que el t de la última línea de output, se trunca desde ahí, pero podría haber un cutoff_index == nothing
        # si pasó eso, es porque el t correcto está en el output con k anterior.

        throw("ERROR while reading outputs: couldn't find time closest to checkpoint.")

    end

    # Write the modified content back to the file
    open(filename, "w") do file
        for line in lines[1:end-1]
            println(file, line)  # Write each line back
        end
        #println(chomp(lines[end]))
        print(file,chomp(lines[end])) # chomp te saca de encima el \n
    end
    return t_closest_to_checkpoint #cutoff_index-2
end





function load_input_folder(main_folder::String, params::Params, model::AbstractModel) ###***** chequear si no es igual a initialize_input -- no porque no crea nada nuevo, pero capaz se podría reusar este código
    
    aux_force_variables = model_aux_structs(params, model)
    
    subfolder_name::String = "$(params.model)/act_$(params.activity)_pos_$(params.initial_position)_ang_$(params.initial_orientation)_a_$(params.a)_b_$(params.b)_Lx_$(params.Lx)_Ly_$(params.Ly)_np_$(params.np)_v_$(params.c)_eta_$(params.ηa)"
    subfolder_name = Base.invokelatest(make_model_specific_subfolder_name, params, aux_force_variables.constant, subfolder_name, model)
    new_main_folder::String = main_folder*subfolder_name*"/"
    #input_parameters_file = new_main_folder*"input_parameters.txt" 
    params.subfolder_name = subfolder_name

    return new_main_folder
end

function load_input_parameters!(input_parameters_file::String, params::Params)
    # Get all of the relevant input parameters
    variables = CSV.File(input_parameters_file, delim=",", header=false) |> DataFrame


    # no need to do this if input file exists inside new_main_folder; we already loaded these params from main_folder
    params.a = parse(Float64,variables[1,2])
    params.b = parse(Float64,variables[2,2])

    params.inv_a2 = 1/(params.a*params.a)
    params.inv_b2 = 1/(params.b*params.b)
    params.diag2 = 1/(params.a*params.a+params.b*params.b)
    params.ϵ = (params.a*params.a-params.b*params.b)*params.diag2

    params.Lx = parse(Float64,variables[3,2])
    params.Ly = parse(Float64,variables[4,2])
    params.rectangle = PeriodicRectangle(params.Lx,params.Ly)
    params.np = parse(Int64,variables[5,2])

    params.c = parse(Float64,variables[6,2])

    params.ηa = parse(Float64,variables[7,2])
    params.ηb = parse(Float64,variables[8,2])
    params.ηr = parse(Float64,variables[9,2])

    if params.a == params.b 
        params.μa = 1/params.a
        params.μb = 1/params.a
        params.μr = 0.75/params.a^3
    elseif params.a < params.b
        throw("ERROR: a=$(params.a) (major axis) is smaller than b=$(params.b) (minor axis)")
    else
        params.μa = 0.75*(-params.a*params.diag2/params.ϵ+(2*params.a*params.a-params.b*params.b)*(params.diag2/params.ϵ)^1.5*log(params.a/params.b+sqrt(params.a*params.a*params.inv_b2-1)))              # Mobility in the major axis direction
        params.μb = 0.75*(params.a*0.5*params.diag2/params.ϵ+(params.a*params.a-1.5*params.b*params.b)*(params.diag2/params.ϵ)^1.5*log(params.a/params.b+sqrt(params.a*params.a*params.inv_b2-1)))              # Mobility in the minor axis direction
        params.μr = 1.5*params.μa*params.diag2              # Mobility in the rotation
        #println(params.μa, " ",params.μb, " ", params.μr)
    end

    params.Δt = parse(Float64,variables[10,2])
    params.n_of_points = parse(Int64,variables[11,2])
    params.steps_output = parse.(Int,split(variables[12,2],"/"))  # parse(Int64,variables[16,2])
    params.is_there_block = length(params.steps_output)==3
    params.step = params.steps_output[1]         
    params.block_length = params.is_there_block ? params.steps_output[2] : 1           
    params.block_step = params.is_there_block ? params.steps_output[3] : 1 
    params.n_files_to_av = parse(Int64,variables[13,2])

    params.seed = parse(Int64,variables[14,2])
    params.model = variables[15,2]

    params.initial_position = variables[16,2]
    params.initial_orientation = variables[17,2]

    #*************
    #if params.model == "vandamme"
    #    params.μa = 1/params.b
    #    params.μb = params.μa
    #    params.μr = 0.75*params.μa*params.inv_b2
    #end
    #*************

    params.activity = variables[18,2] == "false" ? false : variables[18,2] == "true" ? true : throw("ERROR: not a valid activity input. Change it to either true or false in inputs_file.")#parse(Bool,variables[19,2])

    #params.start_run = parse(Bool,variables[20,2])
    #params.make_gif = variables[19,2] == "false" ? false : variables[22,2] == "true" ? true : throw("ERROR: not a valid make_gif input. Change it to either true or false in inputs_file.")#parse(Bool,variables[21,2])
    #params.plot_particles = variables[20,2] == "false" ? false : variables[23,2] == "true" ? true : throw("ERROR: not a valid plot_particles input. Change it to either true or false in inputs_file.")#parse(Bool,variables[22,2])
    #params.make_output_file = variables[21,2] == "false" ? false : variables[24,2] == "true" ? true : throw("ERROR: not a valid make_output_file input. Change it to either true or false in inputs_file.")#parse(Bool,variables[23,2])

    #params.start_run = variables[22,2] == "false" ? -1.0 : variables[21,2] == "true" ? 0.0 : parse(Float64,variables[21,2]) # throw("ERROR: not a valid finished_loop input. Change it to either true or false in inputs_file.")#false    # Si el archivo de inputs ya existe en la carpeta, asumo que quiero seguir en más t

        
end

function load_output_log(folder::String, k::Int64)
    if !isfile(folder*"_output.txt")
        if isfile(folder*"_output_part$(k).txt")
            inputs_file = folder*"_output_part$(k).txt"#"_output_short_part$(k).txt"
        elseif isfile(folder*"_output_short_part$(k).txt")
            inputs_file = folder*"_output_short_part$(k).txt"
        end
    end
    println("Now calculating:  ",inputs_file)

    n_complete_rows = countlines(inputs_file)#nrow(df)
    println("All rows: ",n_complete_rows) 

    return inputs_file, n_complete_rows
end

function last_time_from_file(filename::String) #########********* creo que esta función no la uso al final
    open(filename, "r") do io
        # go to the end
        seekend(io)
        pos = position(io)

        # step backwards until we find a newline or reach start
        while pos > 0
            pos -= 1
            seek(io, pos)
            if read(io, Char) == '\n'
                break
            end
        end

        # now read the last line
        lastline = readline(io)

        # split by tab and take the first element
        return parse(Float64,split(lastline, '\t')[1])
    end
end


# Save checkpoint
function save_checkpoint(file, time::Int, particles::Particles, rng::AbstractRNG)
    jldopen(file, "a+") do f
        # keep a list of times
        #times = haskey(f, "times") ? f["times"] : Int[]
        #push!(times, time)
        #f["times"] = times
        println("New checkpoint at step ",time)

        # overwrite 'time'
        if haskey(f, "grp/$time")
            delete!(f, "grp/$time")
        end

        # store checkpoint in its own group
        f["grp/$time/particles"] = particles
        f["grp/$time/rng"] = rng
        #grp = g_create(f, "data/t$(time)")
        #grp["particles"] = particles
        #grp["rng"] = rng
    end
end


# Save checkpoint
function save_checkpoint_correl(file, time::Float64, f_seed::Int64, correl)
    jldopen(file, "a+") do f
        
        if haskey(f,"grp")
            #println("Checkpoints made for ",parse.(Int, keys(f["grp"]))," times")
            l = maximum(parse.(Int, keys(f["grp"])))
            # overwrite 'time'
            if haskey(f, "grp/$l")

                times = f["grp/$l/times"]
                l += 1
                println("New time added: ",time," from file $(f_seed), in total: ",l,".")# "Times meaned were: ",times)
                delete!(f, "grp/$l")
            end
        else 
            times = []
            l=1
        end

        push!(times,[time,f_seed])

        f["grp/$l/times"] = sort(times)

        # store checkpoint in its own group
        f["grp/$l/correl"] = correl

        #println(sort(times),correl)
    end
end

function save_checkpoint_gif(file, time::Float64, k::Int64)
    jldopen(file, "a+") do f
        if haskey(f,"grp")
            #println("Checkpoints made for ",parse.(Int, keys(f["grp"]))," times")
            #l = maximum(parse.(Int, keys(f["grp"])))
            # overwrite 'time'
            
            if haskey(f, "grp/$time")
                
                delete!(f, "grp/$time")
            end
            if haskey(f, "grp/times")
                times = f["grp/times"]
                delete!(f, "grp/times")
            end
        else 
            times = []
        end

        push!(times,time)#[time,k])
        println("New time added: ",time," from file $(k), in total: $(length(times)), $times.")# "Times meaned were: ",times)

        f["grp/times"] = sort(times)
        f["grp/$time/k"] = k
    end
end

function all_times(file)
    jldopen(file, "r") do f
        return f["grp/times"] #***bueno claro es un 
    end
end

function how_many_times(file)
    jldopen(file, "r") do f
        return maximum(parse.(Int, keys(f["grp"])))
    end
end

function latest_time(file)
    #println("Checkpoints made at steps: ",all_times(file))
    maximum(all_times(file))
end

# Load latest checkpoint
function load_latest(file) # returns t, particles, rng
    jldopen(file, "r") do f

        if haskey(f, "t") && !haskey(f, "grp") # el formato viejo
            return f["t"], f["particles"], f["state"]
        end

        #times = f["times"]# keys(f["times"])
        #println(times)
        tmax = how_many_times(file)#maximum(times)
        #grp = f["data/t$(tmax)"]
        #return (f["times/$t/particles"], f["times/$t/rng"])
        return tmax, f["grp/$tmax/particles"], f["grp/$tmax/rng"]
    end
end


# Load latest checkpoint
function load_latest_correl(file) # returns t, particles, rng
    jldopen(file, "r") do f

        #times = f["times"]# keys(f["times"])
        #println(times)
        tmax = how_many_times(file)#maximum(times)
        #grp = f["data/t$(tmax)"]
        #return (f["times/$t/particles"], f["times/$t/rng"])
        return tmax, f["grp/$tmax/times"], f["grp/$tmax/correl"]
    end
end

function load_latest_time(file) # returns t, particles, rng
    jldopen(file, "r") do f
        tmax = latest_time(file)
        return tmax, f["grp/times"], f["grp/$tmax/k"]
    end
end

# Load checkpoint at specific time
function load_checkpoint(file, time::Int)
    jldopen(file, "r") do f

        if haskey(f, "particle") && !haskey(f, "grp") # el formato viejo
            return f["particles"], f["state"]
        end

        grp = f["grp/$(time)"]
        return grp["particles"], grp["rng"]
    end
end



function replace_line!(filepath::String, start_line::String, replace_line::String)
    # Read the file into an array of lines
    lines = readlines(filepath)

    # Replace the first line that starts with start_line
    for (i, line) in enumerate(lines)
        if startswith(line, start_line)
            lines[i] = replace_line
            break  # Only replace the first match
        end
    end

    # Write the modified lines back to the file
    open(filepath, "w") do io
        for line in lines
            println(io, line)
        end
    end
end

function find_line!(filepath::String, start_line::String)
    # Read the file into an array of lines
    lines = readlines(filepath)

    # Replace the first line that starts with start_line
    for (i, line) in enumerate(lines)
        if startswith(line, start_line)
            return lines[i]
        end
    end
end
