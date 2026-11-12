using Random
using Distributions #MvNormal en noise
using StaticArrays 
using LinearAlgebra
using BioStatPhys


#*********encargate del MvNormal en noise 

Base.@kwdef mutable struct Params
    """
    Save relevant constants for easy access.
    """
    a::Float64 = 1.0            # Ellipse's major axis
    b::Float64 = 0.5            # Ellipse's minor axis

    inv_a2::Float64 = 1/(a*a)
    inv_b2::Float64 = 1/(b*b)
    diag2::Float64 = 1/(a*a+b*b)
    ϵ::Float64 = (a*a-b*b)*diag2
    area_particle::Float64 = π*a*b

    Lx::Float64 = 10             # Box horizontal length (1 = a)**
    Ly::Float64 = 10             # Box vertical length
    rectangle::PeriodicHyperCube{2} = PeriodicRectangle(Lx,Ly)
    np::Int64 = 70              # Number of particles

    c::Float64 = 10             # Preferred velocity (same for every particle)
    # If Grossmann2020

    # Neighbor threshold
    rc::Float64 = 5.0*a

    # Noise parameters
    ηa::Float64 = 0.5            # Noise in the major axis direction
    ηb::Float64 = 0.5            # Noise in the minor axis direction
    ηr::Float64 = 0.5            # Noise in the rotation
    μa::Float64 = 0.75*(-a*diag2/ϵ+(2*a*a-b*b)*(diag2/ϵ)^1.5*log(a/b+sqrt(a*a*inv_b2-1)))              # Mobility in the major axis direction
    μb::Float64 = 0.75*(a*0.5*diag2/ϵ+(a*a-1.5*b*b)*(diag2/ϵ)^1.5*log(a/b+sqrt(a*a*inv_b2-1)))              # Mobility in the minor axis direction
    μr::Float64 = 1.5*μa*diag2              # Mobility in the rotation

    Δt::Float64 = 0.001         # Time steps for integration **calculate according to c, alpha and beta values
    n_of_points::Int64 = 20     # Total time = n_of_points*Δt
    steps_output::Vector{Int64} = [10,1,1]    # Might have up to three elements: (expected to be of increasing value; if not, only first element is relevant)
    step::Int64 = steps_output[1]         #             Step: time steps to wait until output, 
    block_length::Int64 = steps_output[2] #             Block length: time steps to finish a block, then skip output until next block,
    block_step::Int64 = steps_output[3]   #             Block steps: time steps from the start of a block until the next one.
    is_there_block::Bool = false
                                          # Checkpoints will be made when output.txt has approx 500MiB of size.
    n_files_to_av::Int64 = 5   # Number of files to output, to get average parameters later by running "read_outputs.jl"
    #**maybe check that steps_output<n_of_points!!

    seed::Int64 = 1234
    RNG::AbstractRNG = Random.Xoshiro(seed)

    model::String = "jamous"
    activity::Bool = true
    start_run::Float64 = -1.0 
    make_gif::Bool = false #**el video debería salir de a partes ? para q no sature la memoria
    plot_particles::Bool = false
    make_output_file::Bool = true

    initial_position::String = "random" # random, square_grid, triangular_grid, circle, center
    initial_orientation::String = "nematic" # random, polar, nematic, zero

    main_folder::String = ""
    subfolder_name::String = ""

end


Base.@kwdef mutable struct Particles
    """
    Every particle will have a position and orientation defined here 
    and modified for every time step. Starts randomly.
    """
    #angles = Float64[]
    #pos = MVector{2,Float64}[]
    #vel = MVector{2,Float64}[]

    n = Int64

    angles::Vector{Float64} = Vector{Float64}(undef,n)
    pos::Vector{MVector{2,Float64}} = Vector{MVector{2,Float64}}(undef,n)
    vel::Vector{MVector{2,Float64}} = Vector{MVector{2,Float64}}(undef,n)
    omega::Vector{Float64} = Vector{Float64}(undef,n)

    cosines::Vector{Float64} = Vector{Float64}(undef,n)
    sines::Vector{Float64} = Vector{Float64}(undef,n)

end

Base.@kwdef mutable struct Particle
    """
    Every particle will have a position and orientation defined here 
    and modified for every time step. Starts randomly.
    """
    θ::Float64 = 0.0
    omega::Float64 = 0.0
    #x::Float64 = 0.0
    #y::Float64 = 0.0
    pos::MVector{2,Float64} = MVector{2,Float64}(0.0,0.0)
    vel::MVector{2,Float64} = MVector{2,Float64}(0.0,0.0)

    cos_i::Float64 = 0.0
    sin_i::Float64 = 0.0

    #vx::Float64 = 1.0
    #vy::Float64 = 1.0

end


abstract type AbstractModel end

abstract type GeometryVarsType end

abstract type NematicVarsType end

abstract type AuxVarsType end

abstract type ForceVarsType end

abstract type ConstantVarsType end



abstract type AbstractInitialOrientation end
abstract type AbstractInitialPosition end

"""
Called once per `init_particles!` call to build
whatever a config needs to share across every particle — e.g. "polar"
and "nematic" both draw one reference angle that every particle's
orientation is defined relative to. Most configs don't need this at
all, so the default is `nothing`; only override it if your config's
`get_initial_angle` needs something computed once, before the particle
loop starts.
"""
orientation_setup(params::Params, ::AbstractInitialOrientation) = nothing
position_setup(params::Params, ::AbstractInitialPosition) = nothing


Base.@kwdef mutable struct InteractionForces
    
    n::Int64 = Int64

    fx::Matrix{MVector{2,Float64}} = Matrix{MVector{2,Float64}}(undef,n,n)
    fy::Matrix{MVector{2,Float64}} = Matrix{MVector{2,Float64}}(undef,n,n)

    #cosines = Float64[]
    #sines = Float64[]

end




Base.@kwdef mutable struct NoiseVariables
    
    params::Params = Params #**esto crea una copia de params? me conviene pasar sólo las ctes que uso acá?

    noise_r::Float64 = 0.0

    noise_va::Float64 = 0.0
    noise_ra::Float64 = 0.0
    noise_vb::Float64 = 0.0
    noise_rb::Float64 = 0.0

    sigma2a::Float64 = params.ηa^2 #*params.μa
    sigma2b::Float64 = params.ηb^2 #*params.μb

    gaussian_center::SVector{2,Float64} = SVector(0.0,0.0)

    mv_noise_r::Float64 = params.ηr*sqrt(params.Δt) #MvNormal([0.0], [params.ηr*sqrt(params.μr*params.Δt)])
    mv_noise_a  = params.ηa>0.0 ? MvNormal(gaussian_center, @SMatrix [sigma2a 0.0; 0.0 sigma2a*params.Δt]) : MvNormal(gaussian_center,[1.0 0.0;0.0 1.0]) # el PDMats sirve envolviendo la matriz si no es diagonal
    mv_noise_b  = params.ηb>0.0 ? MvNormal(gaussian_center, @SMatrix [sigma2b 0.0; 0.0 sigma2b*params.Δt]) : MvNormal(gaussian_center,[1.0 0.0;0.0 1.0])

    # To preallocate memory
    buf_a::MVector{2,Float64} = MVector{2,Float64}(0.0,0.0)
    buf_b::MVector{2,Float64} = MVector{2,Float64}(0.0,0.0)
    #**como el ruido es decorrelacionado, conviene hacer esto en realidad: (chequear si da exactamente igual)

    # 1D Gaussian
    #noise_v.noise_r = sqrt(params.ηr*params.μr*params.Δt) * randn()

    # 2D Gaussian
    #√σa = sqrt(sigma2a)
    #noise_v.noise_va = √σa * randn()
    #noise_v.noise_ra = √σa * sqrt(params.Δt) * randn()

    #√σb = sqrt(sigma2b)
    #noise_v.noise_vb = √σb * randn()
    #noise_v.noise_rb = √σb * sqrt(params.Δt) * randn()

end



Base.@kwdef mutable struct ParticleData
    """
    Every particle will have a position and orientation defined here 
    and modified for every time step. Starts randomly.
    """
    θ::Float64 = rand()*π*2
    x::Float64 = rand()*5+2.5
    y::Float64 = rand()*5+2.5

    vx::Float64 = 1
    vy::Float64 = 1
end
