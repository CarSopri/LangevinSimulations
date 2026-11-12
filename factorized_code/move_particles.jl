#using BioStatPhys
using Active

include("./particle_objects.jl")
include("./model_forces.jl") #**creo que al final no lo uso porque moví la def de AbstractModel

#************* creo que debería haber un archivo más con las posibles condiciones iniciales como funciones ? 
#*** hay algunas cosas como el plot de la elipse que debería depender del modelo
#***also tener cuidado con a==b


#****************revisar los MVector, dice chatgpt que es para declarar no para alocar !
#=
    move_particles_REFACTORED.jl — init_particles! only
    ======================================================
    For review only, nothing wired into your real code. Companions:
    particle_objects_ADDITIONS.jl, model_forces_ADDITIONS.jl, and the
    four configurations_*_EXAMPLE.jl files (what an actual config file
    in "configurations/orientations/" or "configurations/positions/"
    looks like — one with a workspace, one without, per hierarchy).

    WHAT CHANGED
    -------------
    Both five-way if/elseif chains (`initial_orientation`,
    `initial_position`) are gone from `init_particles!`. In their
    place: `initial_angle(...)` and `initial_position(...)`, dispatched
    on whichever `AbstractInitialOrientation`/`AbstractInitialPosition`
    concrete type the user's file in "configurations/" defined — same
    relationship `get_force_prueba` has to `AbstractModel` already.

    `start_from_ref` is untouched: it's still the one real `if` in the
    function, still skips initial-condition setup entirely when true,
    exactly as before.

    Two small, genuine cleanups that fell out of the restructuring
    rather than being separately decided (flagging rather than
    bundling silently):
      - The original computed `mean_particle_distance`, `x_shift`,
        `y_shift`, and a first pass at `x_particle_distance`/
        `y_particle_distance` unconditionally at the top of the
        function, for *every* orientation/position combination — but
        `square_grid` and `triangular_grid` immediately overwrote
        `x_particle_distance`/`y_particle_distance` with their own
        values, and no position option other than those two ever read
        `x_shift`/`y_shift`'s outer values either. That whole
        precomputation only ever mattered for the two grid configs, and
        now lives in exactly those two configs' `build_position_workspace`
        — nothing else pays for it.
      - `rand_if_ordered` was likewise computed unconditionally even
        for orientations that never read it (`random`, `zero`,
        `diagonal`). It's now built only inside `nematic.jl`/`polar.jl`'s
        own `build_orientation_workspace`.
      Net effect: exactly the same values are computed for whichever
      orientation/position you actually pick, and nothing extra is
      computed for the others — a real (if minor) win, not just a
      reorganization.

    HOW init_particles! GETS CALLED NOW
    --------------------------------------
    Same shape as `model_forces::AbstractModel` — the config instances
    are built once, above the simulation loop, and threaded in:

        load_model!(code_directory, params)
        load_initial_orientation!(code_directory, params)
        load_initial_position!(code_directory, params)

        Base.invokelatest(run_one_seed, params, code_directory, main_folder, ...)

        # inside run_one_seed:
        model_forces        = get_model()
        orientation_config   = get_orientation_config()
        position_config      = get_position_config()
        init_particles!(particles, params, output, fig_ellipse, model_forces,
                         orientation_config, position_config, main_folder, start_from_ref)

    The `Base.invokelatest` has to wrap the *whole* call chain that
    ends up calling `initial_angle`/`initial_position`/`get_model`/
    `get_orientation_config`/`get_position_config` — not just one of
    them — because all of those are newly `include`d in the current
    world age, same reasoning as `Base.invokelatest(get_correlations_with_model, ...)`
    in get_correlations.jl. Since I don't have your actual top-level
    driver file, I can't point at the exact line to wrap — but wherever
    you currently call `load_model!` followed eventually by something
    that calls `init_particles!`, that whole span needs to move inside
    one `Base.invokelatest`, same as it already does for the model
    alone today.
=#


"""
    load_initial_orientation!(code_directory, params)

Loads the user's orientation config from
`configs/initial_orientations/<params.initial_orientation>.jl`, same
convention as `load_model!` in model_forces.jl.
"""
function load_initial_orientation!(code_directory::String, params::Params)
    include(joinpath(code_directory, "configs", "initial_orientations", params.initial_orientation * ".jl"))
end

"""
    load_initial_position!(code_directory, params)

Loads the user's position config from
`configs/initial_positions/<params.initial_position>.jl`.
"""
function load_initial_position!(code_directory::String, params::Params)
    include(joinpath(code_directory, "configs", "initial_positions", params.initial_position * ".jl"))
end


function init_particles!(particles::Particles, params::Params, output::IOStream, fig_ellipse, model_forces::AbstractModel,
                          orientation_config::AbstractInitialOrientation, position_config::AbstractInitialPosition,
                          main_folder::String, start_from_ref=false)
    """
    Initial particle positions and orientations, delegated to whatever
    `orientation_config`/`position_config` the caller loaded from
    configurations/. If `start_from_ref` is true, this whole step is
    skipped and particles keep whatever positions/angles they already
    have (unchanged from the original).
    """

    if !start_from_ref
        orientation_state = orientation_setup(params, orientation_config)
        position_state = position_setup(params, position_config)
    end

    for i in 1:params.np

        if !start_from_ref
            angle_i = get_initial_angle(params, orientation_config, orientation_state, i)
            pos_ix, pos_iy = get_initial_position(params, position_config, position_state, i, angle_i)

            particles.angles[i] = angle_i
            particles.cosines[i] = cos(angle_i)
            particles.sines[i] = sin(angle_i)

            if params.activity
                vel_ix = params.c * particles.cosines[i]
                vel_iy = params.c * particles.sines[i]
            else
                vel_ix = 0.0
                vel_iy = 0.0
            end

            particles.pos[i] = @MVector [pos_ix, pos_iy]
            particles.vel[i] = @MVector [vel_ix, vel_iy]
        end

        pos_ix = particles.pos[i][1]
        pos_iy = particles.pos[i][2]
        angle_i = particles.angles[i]
        vel_ix = particles.vel[i][1]
        vel_iy = particles.vel[i][2]

        plot_ellipse_better(fig_ellipse, params.a, params.b, pos_ix, pos_iy, angle_i, params.Lx, params.Ly)

        if params.make_output_file
            print(output, "\t", pos_ix, "\t", pos_iy, "\t", angle_i, "\t", vel_ix, "\t", vel_iy)
        end
    end
    title!("t=0")
    if params.plot_particles || params.make_gif
        display(fig_ellipse)
        gui(); sleep(0.1)
    end
    savefig(fig_ellipse, main_folder * "init_particles.png")
end


#**creo que esta función ya no la uso
function move_particle!(particles::Particles, params::Params, p_i::Particle, model_forces::AbstractModel, 
    geom_ws::GeometryVarsType, nematic_ws::NematicVarsType, aux::AuxVarsType, force_ws::ForceVarsType, constant_ws::ConstantVarsType,
    noise_v::NoiseVariables, neighbors_i::Vector{Int64}, maxdisp::Float64)#, fig_ellipse)

    """
    Move a particle according to interaction potential and self-activity.
    """

    #pos_i::Vector{Float64} = particles.pos[i]
    #vel_i::Vector{Float64} = particles.vel[i]
    #angle_i::Float64 = particles.angles[i]

    p_i.cos_i = cos(p_i.θ)
    p_i.sin_i = sin(p_i.θ)
    #cos_i::Float64 = particles.cosines[i]
    #sin_i::Float64 = particles.sines[i]

    #println(@allocated get_forces(particles, params, model_forces, aux_force_variables, i, pos_i, vel_i, angle_i, cos_i, sin_i, neighbors_i))
    fx, fy, τ = get_forces(particles, params, model_forces, geom_ws, nematic_ws, aux, force_ws, constant_ws, p_i.pos, p_i.vel, p_i.θ, p_i.cos_i, p_i.sin_i, neighbors_i)#, fig_ellipse) #, upot_i

    ####print(" velocities: ",fx," ",fy)
    ####print(" omega: ",τ)

    #p_i.vel[1] = fx
    #p_i.vel[2] = fy

    p_i.θ = p_i.θ + τ*params.Δt
    p_i.cos_i = cos(p_i.θ)
    p_i.sin_i = sin(p_i.θ)

    if params.activity
        fx += params.c*p_i.cos_i
        fy += params.c*p_i.sin_i
    end

    #**podrían ser más aux qsy
    disp_ix = fx*params.Δt
    disp_iy = fy*params.Δt


    if params.ηa!=0.0 || params.ηb !=0.0 || params.ηr!=0.0
        # Get noise
        
        noise_v.noise_r = randn()*noise_v.mv_noise_r #rand(noise_v.mv_noise_r)[1] #**da distinto que en old_ooooold.jl

        #noise_v.noise_va, noise_v.noise_ra = rand(noise_v.mv_noise_a) 
        #noise_v.noise_vb, noise_v.noise_rb = rand(noise_v.mv_noise_b)

        rand!(noise_v.mv_noise_a, noise_v.buf_a)
        rand!(noise_v.mv_noise_b, noise_v.buf_b)
        noise_v.noise_va, noise_v.noise_ra = noise_v.buf_a
        noise_v.noise_vb, noise_v.noise_rb = noise_v.buf_b

        #if i==1
        #    println(noise_r,"  ",noise_va," ",noise_vb)
        #end

        # Add noise
        fx += (noise_v.noise_va*p_i.cos_i + noise_v.noise_vb*p_i.sin_i) 
        fy += (noise_v.noise_va*p_i.sin_i - noise_v.noise_vb*p_i.cos_i) 

        disp_ix = disp_ix + (noise_v.noise_ra*p_i.cos_i + noise_v.noise_rb*p_i.sin_i)
        disp_iy = disp_iy + (noise_v.noise_ra*p_i.sin_i - noise_v.noise_rb*p_i.cos_i)
        p_i.θ = p_i.θ + noise_v.noise_r

        #println(noise_v.noise_r," ",noise_v.noise_va," ", noise_v.noise_ra," ", noise_v.noise_vb," ", noise_v.noise_rb  )
        
    end

    #p_i.pos[1] = p_i.pos[1] + disp_ix ###
    #p_i.pos[2] = p_i.pos[2] + disp_iy

    disp2 = disp_ix^2 + disp_iy^2 # es más rápido para acceder
    aux.disp = disp2

    #aux.disp = disp_ix^2 + disp_iy^2
    maxdisp = max(maxdisp,aux.disp)
    #if aux.disp > maxdisp
    #    maxdisp = aux.disp
    #end

    # Periodic boundary conditions
    #p_i.pos[1] = p_i.pos[1] - floor(p_i.pos[1]/params.Lx)*params.Lx #mod(p_i.pos[1] + disp_ix, params.Lx) 
    #p_i.pos[2] = p_i.pos[2] - floor(p_i.pos[2]/params.Ly)*params.Ly #mod(p_i.pos[2] + disp_iy, params.Ly) 

    @inline function wrap!(x::Float64, L::Float64)
        if x ≥ L
            x -= L
        elseif x < 0.0
            x += L
        end
        return x
    end
    #p_i.pos = @MVector [wrap!(p_i.pos[1] + disp_ix, params.Lx),wrap!(p_i.pos[2] + disp_iy, params.Ly)]
    p_i.pos[1] = wrap!(p_i.pos[1] + disp_ix, params.Lx)
    p_i.pos[2] = wrap!(p_i.pos[2] + disp_iy, params.Ly)

    #p_i.vel = @MVector [fx,fy]
    p_i.vel[1] = fx
    p_i.vel[2] = fy

    #println(p_i.pos,p_i.vel)


    #particles.pos[i] = p_i.pos
    #particles.vel[i] = p_i.vel
    #particles.angles[i] = p_i.θ
    #particles.cosines[i] = cos_i
    #particles.sines[i] = sin_i

    #**********************************************************************************
    #if params.b>0.25
    #    params.b-=0.00001
    #end
    return maxdisp#, upot_i
   
end

###############*********************************************************************************

function move_particle_prueba!(#particles::Particles, 
    params::Params, p_i::Particle, #model_forces::AbstractModel, 
    geom_ws::GeometryVarsType, nematic_ws::NematicVarsType, aux::AuxVarsType, force_ws::ForceVarsType, constant_ws::ConstantVarsType,
    noise_v::NoiseVariables, #neighbors_i::Vector{Int64}, 
    maxdisp::Float64)#, fig_ellipse)

    """
    Move a particle according to interaction potential and self-activity.
    """


    #pos_i::Vector{Float64} = particles.pos[i]
    #vel_i::Vector{Float64} = particles.vel[i]
    #angle_i::Float64 = particles.angles[i]

    #p_i.cos_i = cos(p_i.θ)
    #p_i.sin_i = sin(p_i.θ)
    #cos_i::Float64 = particles.cosines[i]
    #sin_i::Float64 = particles.sines[i]


    #********************************************* no guardar fx,fy en un nuevo vector, sino guardar en vx,vy y después sumarle ruido y autopropulsión a cada partícula
    #********************************************* creo que tendría que haber 2 loops de i 1:np, uno para calcular las fuerzas sin repetir, y otro para mover
    #********************************************* claro capaz lo mejor es sacar get_forces de esta función, y ponerlo afuera incluso del loop en i 1:np que llama move_particles!
    #*********************** (ponerlo afuera y antes de llamar esta función, tal que get_forces incluya el loop de i 1:np y el de j i+1:np; if P_j in NL_i)

    #geom_ws, nematic_ws, aux_ws, force_ws, constant_ws, 
    #println(@allocated get_forces(particles, params, model_forces, aux_force_variables, i, pos_i, vel_i, angle_i, cos_i, sin_i, neighbors_i))
    """
    fx, fy, τ = get_forces(particles, params, model_forces,  aux_force_variables,   p_i.pos, p_i.vel, p_i.θ, p_i.cos_i, p_i.sin_i, neighbors_i)#, fig_ellipse) #, upot_i
    """

    cos_theta = p_i.cos_i
    sin_theta = p_i.sin_i

    cos2 = cos_theta*cos_theta-sin_theta*sin_theta
    sin2 = 2*sin_theta*cos_theta

    #nematic_ws.Qi = nematic(cos_theta, sin_theta) # no puedo usar el de p_i(theta) xq todavía no lo calculé (guardé sólo el theta)
    Q1 = (constant_ws.viscosity_substract*cos2)
    Q2 = (constant_ws.viscosity_substract*sin2)

    ####println("post: ",p_i.vel,p_i.pos,p_i.omega)
    #force_ws.fi = (constant_ws.viscosity_sum*I2+constant_ws.viscosity_substract*nematic_ws.Qi)*0.5 * MVector(p_i.vel[1], p_i.vel[2])
    force_ws.fi[1] = ((constant_ws.viscosity_sum+Q1)*p_i.vel[1] + Q2*p_i.vel[2])*0.5
    force_ws.fi[2] = (Q2*p_i.vel[1] + (constant_ws.viscosity_sum-Q1)*p_i.vel[2])*0.5

    ####print("   ",force_ws.fi[1],"   ", force_ws.fi[2],"   ", params.μr * force_ws.τ_i, "   " )

    fx = force_ws.fi[1] # ***podría llamar a lo de arriba directamente fx y fy
    fy = force_ws.fi[2]
    τ = p_i.omega*params.μr

    if params.activity
        fx += params.c*cos_theta
        fy += params.c*sin_theta
    end

    #println(" forces:  ",fx,"   ",fy,"   ",τ, "     particle data:  ",p_i)

    #p_i.vel[1] = fx
    #p_i.vel[2] = fy


    #****creo que toda esta parte tendría que estar en otro loop externo, sobre i de 1:np, pero donde ya tenga en cuenta todas las contribuciones de fij,
    # que ya se habrían sumado en el get_forces anterior. o capaz debería sacar get_forces de acá adentro y ponerlo afuera en otro loop anterior sobre i de 1:np
    #***bueno, además, adentro de get_forces tendría que hacer un loop que no vaya sólo sobre los vecinos, sino que vaya de:
    # for j in (i+1):params.np, if p_j in neighbors, (fx,fy)ij+=interaction y (fx,fy)ji+=interaction  (la misma); else +=0.0
    #*******bueno y chequear que quede bien acomodado para todas las fuerzas que calculo. si fij!= fji, debería tener otro formato !!!! (como jamous lpm)
    #***bueno tendría que sacar de get_forces todo lo que está afuera del loop en j, porque eso lo aplicaría al final, una vez que las f están todas calculadas (sin repetir je no como ahora)
    #****y despuéeeees chequear si en el grosmman de L=10 que se formaban las olas se siguen formando (creería que sí but we'll see).

    p_i.θ = p_i.θ + τ*params.Δt

    #**podrían ser más aux qsy
    disp_ix = fx*params.Δt #0.0 #
    disp_iy = fy*params.Δt #0.0 #

    if params.ηa!=0.0 || params.ηb !=0.0 ||params.ηr!=0.0
        # Get noise
        
        noise_v.noise_r = randn()*noise_v.mv_noise_r #rand(noise_v.mv_noise_r)[1]

        noise_v.noise_va, noise_v.noise_ra = rand(noise_v.mv_noise_a) 
        noise_v.noise_vb, noise_v.noise_rb = rand(noise_v.mv_noise_b)

        #rand!(noise_v.mv_noise_a, noise_v.buf_a)
        #rand!(noise_v.mv_noise_b, noise_v.buf_b)
        #noise_v.noise_va, noise_v.noise_ra = noise_v.buf_a
        #noise_v.noise_vb, noise_v.noise_rb = noise_v.buf_b

        #if i==1
        #    println(noise_r,"  ",noise_va," ",noise_vb)
        #end

        # Add noise
        fx += (noise_v.noise_va*cos_theta + noise_v.noise_vb*sin_theta) 
        fy += (noise_v.noise_va*sin_theta - noise_v.noise_vb*cos_theta) 

        disp_ix = disp_ix + (noise_v.noise_ra*cos_theta + noise_v.noise_rb*sin_theta)
        disp_iy = disp_iy + (noise_v.noise_ra*sin_theta - noise_v.noise_rb*cos_theta)
        p_i.θ = p_i.θ + noise_v.noise_r
    end

    p_i.cos_i = cos(p_i.θ) # lo calculo de nuevo xq ahora lo moví !
    p_i.sin_i = sin(p_i.θ)

    
    #disp_ix = fx*params.Δt
    #disp_iy = fy*params.Δt


    #p_i.pos[1] = p_i.pos[1] + disp_ix
    #p_i.pos[2] = p_i.pos[2] + disp_iy

    disp2 = disp_ix^2 + disp_iy^2 # es más rápido para acceder
    #aux_ws.disp = disp2

    #aux.disp = disp_ix^2 + disp_iy^2
    #maxdisp = max(maxdisp,aux_ws.disp)

    aux.disp = disp2
    maxdisp = max(maxdisp,aux.disp)
    #if aux.disp > maxdisp
    #    maxdisp = aux.disp
    #end

    #***********probar si wrap es más rápido con profiler 

    # Periodic boundary conditions
    p_i.pos[1] = mod(p_i.pos[1] + disp_ix, params.Lx) #p_i.pos[1] - floor(p_i.pos[1]/params.Lx)*params.Lx
    p_i.pos[2] = mod(p_i.pos[2] + disp_iy, params.Ly) #p_i.pos[2] - floor(p_i.pos[2]/params.Ly)*params.Ly

    p_i.vel[1] = fx
    p_i.vel[2] = fy


    #particles.pos[i] = p_i.pos
    #particles.vel[i] = p_i.vel
    #particles.angles[i] = p_i.θ
    #particles.cosines[i] = cos_i
    #particles.sines[i] = sin_i

    #**********************************************************************************
    #if params.b>0.25
    #    params.b-=0.00001
    #end
    return maxdisp#, upot_i
   
end



function foreach_neighbor(f,NN::NeighbourList, i::Int)
    dsq = 0.
    for j ∈ eachindex(NN.pos)
        i!=j || continue
        dsq = distancesq(NN.region,NN.pos[i],NN.pos[j]) #****ver si está tomando raíz, prob no hace falta. yo sí tomo raíz mmmm
        if dsq<= NN.rcsq
            f(j,dsq)
        end
    end
end


function update_neighbors!(neighbors,NL::NeighbourList)
    for i in eachindex(neighbors)
        neighbors[i] = []
        foreach_neighbor(NL, i) do j,d
            push!(neighbors[i],j)
        end
    end
    #println(neighbors)
end


function update_NL_and_neighbors!(NL::NeighbourList,maxdisp::Number, neighbors::Vector{Vector{Int64}}, t, particles)
    NL.accdisp += maxdisp
    if NL.accdisp > NL.δr/2 
        Active.update!(NL) 
        update_neighbors!(neighbors, NL)
        #println(t, NL.pos, particles.vel)
    end
end
