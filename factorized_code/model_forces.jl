using StaticArrays 
using LinearAlgebra

#***buscar y sacar todos los if params.model y poner una versión de esa función en los archivos de modelo/fuerza


#######******************************* probablemente rc debería ser un input externo
include("./particle_objects.jl")


@inline function d_ij_periodic(x1::Float64, x2::Float64, L::Float64)
    temp::Float64 = x2-x1
    return temp-L*round(temp/L)
    #if abs(d)<L/2
    #    return d
    #else
    #    return d-L*sign(d)
    #end
    #if d >  L/2
    #    return d - L
    #elseif d < -L/2
    #    return d + L
    #else
    #    return d
    #end
end


const I2 = @SMatrix [1.0 0.0; 0.0 1.0]




Base.@kwdef mutable struct ModelWorkspace{G, N, A, F, C}
    geom::G # = GrossmannGeometryVars()
    nematic::N # = GrossmannNematicVars()
    aux::A # = GrossmannAuxVars()
    force::F # = ForceVars()
    constant::C
end

struct Model <: AbstractModel end 
function load_model!(code_directory::String, params::Params)
    include(joinpath(code_directory, "factorized_code","models", params.model * ".jl"))
end
function select_model(params::Params) #**no me parece lo mejor esto
    include("./factorized_code/models/$(params.model).jl")
    return get_model()
end


function get_forces_external!(ps::Particles, params::Params, model_forces::AbstractModel, 
    geom_ws::GeometryVarsType, nematic_ws::NematicVarsType, aux::AuxVarsType, force_ws::ForceVarsType, constant_ws::ConstantVarsType,
    neighbors::Vector{Vector{Int64}})
    

    #fx_aux = [0.0 for i in 1:params.np]
    #fy_aux = [0.0 for i in 1:params.np]
    #omega_aux = [0.0 for i in 1:params.np]

    for i in 1:params.np
        
        #** este llevar a 0 las velocidades lo tengo que hacer o antes afuera (mmmm) o al final después de calcular todos (pero se tiene que inicializar siempre en 0) (porque al j le sumás en otro orden!!)
        # a menos que... el j vaya desde los j<i?
        ps.vel[i][1] = 0.0 # no puedo ponerle un alias acá porque necesito 
        ps.vel[i][2] = 0.0
        ps.omega[i] = 0.0    
        #************ prob poner alias a las variables porque no parece muy sostenible estar llamando una y otra vez a que busque el elemento i en cada cosa

        #for j in 1:(i-1) #(1+i):params.np #***entonces cambiar el loop de adentro de TODOS los get_forces para que sea un if j in neighbors_i ****eventualmente tener más cuidado con jamous, porque ahí fij!=fji !!!!!
            #if j in neighbors_i
            
        p_i = ps.pos[i]
        thetai = ps.angles[i]
        cos_i = ps.cosines[i]#cos(thetai) # creo que no estoy guardando senos y cosenos en particles
        sin_i = ps.sines[i]#sin(thetai)

        #************los senos y cosenos calcular de afuer ! o sea no repetir para cada j dentro de i
        ####println(i,"   ",neighbors[i])
        for j in neighbors[i] # if j in neighbors_i, entonces sí o sí i in neighbors_j
            if j<i
                fx_ij, fy_ij, τ_ij, τ_ji = get_force_prueba(params, model_forces, geom_ws, nematic_ws, aux, force_ws, constant_ws, p_i, thetai, ps.pos[j], ps.angles[j], cos_i, sin_i)
            
                #fx_aux[i] += fx_ij
                #fy_aux[i] += fy_ij
                #omega_aux[i] += τ_ij

                #fx_aux[j] -= fx_ij
                #fy_aux[j] -= fy_ij
                #omega_aux[j] -= τ_ij


                ps.vel[i][1] += fx_ij
                ps.vel[i][2] += fy_ij
                ps.omega[i] += τ_ij   


                #**sé que es más fácil acceder al vector en orden pero acá aprovecho que calculé get_force(i,j)
                ps.vel[j][1] -= fx_ij
                ps.vel[j][2] -= fy_ij
                ps.omega[j] += τ_ji

            end
            
        end
        

        
        # Multiplicar por Δt para que sea vel en vez de f lo hacemos en move_particles!

        #aux_force_variables.force.fi = (aux_force_variables.constant.viscosity_sum*I2+aux_force_variables.constant.viscosity_substract*aux_force_variables.nematic.Qi)*0.5 * MVector(fx_aux[i], fy_aux[i])

        #print("   ",afv.force.fi[1],"   ", afv.force.fi[2],"   ", params.μr * afv.force.τ_i, "   " )

    end

    #println("velocities: ",ps.vel)
    #println("omega: ",ps.omega)

    #********** cuestión que una vez que corre esto, ya ponemos correr move_particles! no? donde en vez de llamar fx_i, fy_i, τ_i = get_force(...), hacemos fx_i, fy_i, τ_i = ps.pos[i], ps.vel[i], ps.omega[i]

end



