using Plots
using LsqFit

include("./particle_objects.jl")
include("./model_forces.jl")



function check_boundary_ellipse(r::Float64,L::Float64)
    """
    Make sure ellipses stay within box boundaries enforcing periodicity.
    **Might be faster than mod(r,L) for large boxes where most particles are within box boundaries
    **Gotta be careful with how much particles are allowed to move due to force interactions**
    """
    if 0<r<L
        return r, true#,true
    elseif r≤0 
        return L+r, false#,true
    else
        return r-L, false#,false
    end
end


# Global precomputed angle vectors (better done once, outside the loop)
const NPTS = 20
const angle_range::SVector{NPTS,Float64} = range(0, stop=2π, length=NPTS)
const cos_angle_range::SVector{NPTS,Float64} = cos.(angle_range)
const sin_angle_range::SVector{NPTS,Float64} = sin.(angle_range)


# Vectorized boundary wrap
@inline periodic(x, L) = mod(x, L)

mutable struct EllipseBuffers
    rx_p::Vector{Float64}
    ry_p::Vector{Float64}
    rx_1::Vector{Float64}
    ry_1::Vector{Float64}
    rx_2::Vector{Float64}
    ry_2::Vector{Float64}
    rx_3::Vector{Float64}
    ry_3::Vector{Float64}
end

function EllipseBuffers(N)
    EllipseBuffers(
        Vector{Float64}(undef, 0), Vector{Float64}(undef, 0),
        Vector{Float64}(undef, 0), Vector{Float64}(undef, 0),
        Vector{Float64}(undef, 0), Vector{Float64}(undef, 0),
        Vector{Float64}(undef, 0), Vector{Float64}(undef, 0),
    )
end


# Global precomputed angle vectors (better done once, outside the loop)
const NPTS = 20
const angle_range::SVector{NPTS,Float64} = range(0, stop=2π, length=NPTS)
const cos_angle_range::SVector{NPTS,Float64} = cos.(angle_range)
const sin_angle_range::SVector{NPTS,Float64} = sin.(angle_range)


# Vectorized boundary wrap
@inline periodic(x, L) = mod(x, L)

mutable struct EllipseBuffers
    rx_p::Vector{Float64}
    ry_p::Vector{Float64}
    rx_1::Vector{Float64}
    ry_1::Vector{Float64}
    rx_2::Vector{Float64}
    ry_2::Vector{Float64}
    rx_3::Vector{Float64}
    ry_3::Vector{Float64}
end

EllipseBuffers() = EllipseBuffers(
    Float64[], Float64[],
    Float64[], Float64[],
    Float64[], Float64[],
    Float64[], Float64[],
)

function plot_particles_correctly_in_boundaries(rx::Float64,ry::Float64,Lx::Float64,Ly::Float64,buf::EllipseBuffers)

    xw, xin = check_boundary_ellipse(rx, Lx)
    yw, yin = check_boundary_ellipse(ry, Ly)

    if xin & yin
        push!(buf.rx_p, xw); push!(buf.ry_p, yw)
    elseif xin
        push!(buf.rx_1, xw); push!(buf.ry_1, yw)
    elseif yin
        push!(buf.rx_2, xw); push!(buf.ry_2, yw)
    else
        push!(buf.rx_3, xw); push!(buf.ry_3, yw)
    end
end


#**est aes la que uso al final pero no sé si está muy chequeado lo del diámetro, y creo que prefiero que quede el make_particle_shape como dependiente del modelo no?
#***es una porquería cómo calcula el linewidth a mano esto lo odio !!!
function plot_ellipse_better(fig_ellipse, a::Float64, b::Float64, x0::Float64, y0::Float64, θ::Float64, Lx::Float64, Ly::Float64)

    sθ, cθ = sincos(θ)

    half_rect = a - b
    
    # Center line endpoints
    x1 = x0 + half_rect * cθ
    y1 = y0 + half_rect * sθ
    x2 = x0 - half_rect * cθ
    y2 = y0 - half_rect * sθ
    
    # Plot as a thick line with rounded caps    
    angle_color = HSV_GRAD[mod(θ,2π)/(2π)]

    plot!(fig_ellipse, [x1, x2], [y1, y2], linewidth=25*28/(Lx+8)*b, # Thickness = diameter
            color=angle_color, alpha=0.7, label=false, linecap=:round)  # Rounded ends

end

#****no la uso esta al final y varias de acá tampoco
function make_particle_shape!(::AbstractModel,a::Float64,b::Float64,x0::Float64,y0::Float64,cθ::Float64,sθ::Float64,Lx::Float64,Ly::Float64,buf::EllipseBuffers)

    @inbounds for i in 1:NPTS

        x = a * cos_angle_range[i]
        y = b * sin_angle_range[i]

        rx = cθ*x - sθ*y + x0
        ry = sθ*x + cθ*y + y0

        plot_particles_correctly_in_boundaries(rx,ry,Lx,Ly,buf)
    end

end

const HSV_GRAD = cgrad(:hsv)
const twoπ = 2π

#pyplot()
#plotlyjs()


function plot_ellipse_better2(fig_ellipse, buf::EllipseBuffers, model_forces::AbstractModel, a::Float64, b::Float64, x0::Float64, y0::Float64, θ::Float64, Lx::Float64, Ly::Float64)

    sθ, cθ = sincos(θ)

    empty!(buf.rx_p); empty!(buf.ry_p)
    empty!(buf.rx_1); empty!(buf.ry_1)
    empty!(buf.rx_2); empty!(buf.ry_2)
    empty!(buf.rx_3); empty!(buf.ry_3)

    sizehint!(buf.rx_p, NPTS)  # optional but helps
    sizehint!(buf.rx_1, NPTS)
    sizehint!(buf.rx_2, NPTS)
    sizehint!(buf.rx_3, NPTS)

    make_particle_shape!(model_forces,a,b,x0,y0,cθ,sθ,Lx,Ly,buf)
    
    angle_color = HSV_GRAD[mod(θ,2π)/(2π)]

    plot!(fig_ellipse, Shape(buf.rx_p, buf.ry_p), color=angle_color, alpha=0.7, label=false)

    if !isempty(buf.rx_1)
        plot!(fig_ellipse, Shape(buf.rx_1, buf.ry_1), color=angle_color, alpha=0.7, label=false)
    end
    if !isempty(buf.rx_2)
        plot!(fig_ellipse, Shape(buf.rx_2, buf.ry_2), color=angle_color, alpha=0.7, label=false)
    end
    if !isempty(buf.rx_3)
        plot!(fig_ellipse, Shape(buf.rx_3, buf.ry_3), color=angle_color, alpha=0.7, label=false)
    end
end


function plot_ellipse(fig_ellipse, a::Float64, b::Float64, x0::Float64, y0::Float64, θ::Float64, Lx::Float64, Ly::Float64)
    """
    Create ellipse contour coordinates to plot within box boundaries.
    """
    
    x::SVector{NPTS,Float64} = a * cos_angle_range
    y::SVector{NPTS,Float64} = b * sin_angle_range

    #**acá puedo pasar p_i en vez de x,y, theta y calcular de nuevo sin y cos
    
    rx::SVector{NPTS,Float64} = cos(θ).*x-sin(θ).*y.+x0
    ry::SVector{NPTS,Float64} = sin(θ).*x+cos(θ).*y.+y0

    
    rx_periodic::Vector{Float64} = []
    ry_periodic::Vector{Float64} = []
    rx_aux1::Vector{Float64} = []
    ry_aux1::Vector{Float64} = []
    rx_aux2::Vector{Float64} = []
    ry_aux2::Vector{Float64} = []
    rx_aux3::Vector{Float64} = []
    ry_aux3::Vector{Float64} = []
    for (i,x) in enumerate(rx)
        aux_x::Tuple{Float64, Bool, Bool} = check_boundary_ellipse(x,Lx)
        aux_y::Tuple{Float64, Bool, Bool} = check_boundary_ellipse(ry[i],Ly)
        if aux_x[2] && aux_y[2]
            push!(rx_periodic, aux_x[1])
            push!(ry_periodic, aux_y[1])
        elseif aux_x[2] && !aux_y[2]
            push!(rx_aux1, aux_x[1])
            push!(ry_aux1, aux_y[1])
        elseif !aux_x[2] && aux_y[2]
            push!(rx_aux2, aux_x[1])
            push!(ry_aux2, aux_y[1])
        else 
            push!(rx_aux3, aux_x[1])
            push!(ry_aux3, aux_y[1])
        end
    end

    #rx_periodic = [check_boundary_ellipse(i,Lx) for i in rx]
    #ry_periodic = [check_boundary_ellipse(i,Ly) for i in ry]
    angle_color = cgrad(:hsv)[mod(θ,2*π)/(2*π)]

    plot!(fig_ellipse,Shape(rx_periodic,ry_periodic), label=false, color=angle_color, alpha=0.7 ) #romaO; vik; hsv; twilight; cyclic_mygbm_30_95_c78_n256_s25; cyclic_tritanopic_cwrk_40_100_c20_n256  #color=RGB(choose_color_t(i, params.np)...))
    if !isempty(rx_aux1)
        plot!(fig_ellipse,Shape(rx_aux1,ry_aux1), label=false, color=angle_color, alpha=0.7 ) #romaO; vik; hsv; twilight; cyclic_mygbm_30_95_c78_n256_s25; cyclic_tritanopic_cwrk_40_100_c20_n256  #color=RGB(choose_color_t(i, params.np)...))
    elseif !isempty(rx_aux2)
        plot!(fig_ellipse,Shape(rx_aux2,ry_aux2), label=false, color=angle_color, alpha=0.7 )
    elseif !isempty(rx_aux3)
        plot!(fig_ellipse,Shape(rx_aux3,ry_aux3), label=false, color=angle_color, alpha=0.7 )
    end

    if a==b 
        plot!(fig_ellipse,[x0%Lx,x0%Lx+a*0.9*cos(θ)],[y0%Ly,y0%Ly+b*0.9*sin(θ)],arrow=true,color=:black,linewidth=1,label="")
    end 

        #return rx_periodic,ry_periodic
end







function choose_color_t(s::Float64, total_steps::Int64) 
    """
    Particle color changes as time s passes, 
    or, if s ==> n and total_steps ==> np, then you select a color for every particle.
    """
    if s<total_steps/4.0
        red = 4.0*s/total_steps
        green = 0.0
        blue = 0.0
    elseif s<2.0*total_steps/4.0
        red = 1.0
        green = 4.0*(s-total_steps/4.0)/total_steps
        blue = 0.0
    elseif s<3.0*total_steps/4.0
        red = 4.0*(3.0*total_steps/4.0-s)/total_steps
        green = 1.0
        blue = 4.0*(s-total_steps*2.0/4.0)/total_steps
    else
        red = 0.0
        green = 4.0*(total_steps-s)/total_steps
        blue = 1.0
    end
    return red,green,blue
    
end



"""
            if params.make_gif && filesize(main_folder*"elipses_"*special_filename*"_part"*string(k)*".mp4")>10000000 # Si no existe, da 0
                gif(anim, main_folder*"elipses_"*special_filename*"_part"*string(k)*".mp4")
                # **concatenar acá también
                # **me faltaría poner un if the si el archivo anterior es de un tamaño razonable, concatenar ahí
                k+=1 # **Es malísimo hacer esto, capaz se arregla metiendo en una función
                anim = Animation() # Reseteás la animación si te pasás de tamaño
            elseif params.make_gif && filesize(main_folder*"elipses_"*special_filename*"_part"*string(k)*".mp4")<10000000
                if !isfile(main_folder*"elipses_"*special_filename*"_part"*string(k)*".mp4")
                    gif(anim, main_folder*"elipses_"*special_filename*"_part"*string(k)*".mp4")
                    #anim = Animation()
                else
                    # **Acá intenté ir concatenando los gifs en vez de crearlos de 0 pero no me salió we
                    gif(anim, main_folder*"elipses_"*special_filename*"_part"*string(k)*".mp4")
                    #gif(anim, main_folder*"new_frames.mp4")
                    #output_gif = main_folder*"elipses_"*special_filename*"_part"*string(k)*".mp4"
                    #new_frames_gif = main_folder*"new_frames.mp4"
                    #final_output_gif = main_folder*"elipses_"*special_filename*"_part"*string(k)*".mp4"
                    #open("file_list.txt", "w") do io
            """
                    #    write(io, "file '$output_gif'\n")
                    #    write(io, "file '$new_frames_gif'\n")
                    #end
                    #run(`ffmpeg -y -f concat -safe 0 -i file_list.txt -c:v libx264 -crf 23 -preset fast -c:a aac -b:a 192k final_output.mp4`)
                    #run(`ffmpeg -y -i "concat:$output_gif|$new_frames_gif" -c copy $final_output_gif`)
                    #anim = Animation()
            """
                end
            end
            """

    #plot!(fig_upot, [t*params.Δt for t in t0:t0+tf-1], upot, label="upot")
    #for i in 1:params.np
    #    plot!(fig_upot, [t*params.Δt for t in t0:t0+tf-1], upot_todos[i], label="i=$i")
    #end
    #xlabel!("t")
    #display(fig_upot)
    #savefig(fig_upot,main_folder*"upot.png")
    #plot!(fig_v2, [t*params.Δt for t in t0:t0+tf-1], v2, label="v^2")
    #xlabel!("t")
    #display(fig_v2)
    #savefig(fig_v2,main_folder*"v2.png")

    # **se pueden mergear dos gifs (mejor en vez de crear uno de 0 cada vez) ???
    """
    Save your first batch of frames as output.gif.
    Generate new frames and save them as new_frames.gif.
    Use convert from ImageMagick to merge them.

    # Merge two gifs into a single file
    run(`convert -loop 0 output.gif new_frames.gif final_output.gif`)
    """


# **esto va a haber que optimizarlo bien para sistemas mucho más grandas (tanto de caja como de np)
function plot_spatial_density(Lx,Ly,particles)
    x::Vector{Float64} = [p.x for p in particles]
    y::Vector{Float64} = [p.y for p in particles]

    #x = [i for i in 1:Lx for j in 1:Lx]
    #y = [j for i in 1:Ly for j in 1:Ly]
    
    # Define grid resolution
    Nx, Ny = 100, 100  # Number of grid points in x and y
    X = range(0, Lx, length=Nx)
    Y = range(0, Ly, length=Ny)

    # Define Gaussian width (bandwidth)
    σx, σy = 1.0, 1.0  # Controls smoothing in x and y

    # Compute density field
    density = zeros(Nx, Ny)

    for i in 1:Nx
        for j in 1:Ny
            x_grid, y_grid = X[i], Y[j]
            density[i, j] = sum(exp.(-(d_ij_periodic.(x_grid .- x,Lx).^2) / (2 * σx^2) - (d_ij_periodic.(y_grid .- y,Ly).^2) / (2 * σy^2)))
        end
    end

    # Normalize density
    density ./= (2π * σx * σy * params.np)

    density = density ./ maximum(density)*2
    # Plot density map
    contourf!(X, Y, density', color=:plasma, levels=50)

    #histogram2d!(x,y,bins=(0:Lx/5:Lx,0:Ly/5:Ly),show_empty_bins=true)
    
    #extended_x::Vector{Float64} = vcat(x.-Lx, x, x.+Lx,x.-Lx, x, x.+Lx,x.-Lx, x, x.+Lx)
    #extended_y::Vector{Float64} = vcat(y.-Ly,y.-Ly,y.-Ly,y,y,y,y.+Ly,y.+Ly,y.+Ly)
    # Compute the 2D kernel density estimate
    #kde_result = kde((extended_x, extended_y); npoints=(40,40), kernel=Normal, bandwidth = (0.5,0.5))

    #println(maximum(kde_result.density),"  ",minimum(kde_result.density))

    #density = kde_result.density ./ maximum(kde_result.density)*2  # Scale to [0,1]
    # Plot the density as a heatmap
    #contourf!(kde_result.x, kde_result.y, density, color=:plasma,levels=100)#, color=cgrad(:plasma,scale=log))

    # Add scatter points on top
    scatter!(x, y, markersize=4, alpha=1.0, color=:white, label=false)
    # And the mirrored particles
    scatter!(x.-Lx, y.-Ly, markersize=4, alpha=0.3, color=:white, label=false)
    scatter!(x.-Lx, y, markersize=4, alpha=0.3, color=:white, label=false)
    scatter!(x.-Lx, y.+Ly, markersize=4, alpha=0.3, color=:white, label=false)
    scatter!(x, y.-Ly, markersize=4, alpha=0.3, color=:white, label=false)
    scatter!(x, y.+Ly, markersize=4, alpha=0.3, color=:white, label=false)
    scatter!(x.+Lx, y.-Ly, markersize=4, alpha=0.3, color=:white, label=false)
    scatter!(x.+Lx, y, markersize=4, alpha=0.3, color=:white, label=false)
    scatter!(x.+Lx, y.+Ly, markersize=4, alpha=0.3, color=:white, label=false)
    
    scatter!([0,0], [0,1], zcolor=[0,1.0], alpha=0.0, label=false,c=:hsv, colorbar_title="Density (norm over max)")

end


function gif_settings(fig_ellipse, params::Params, time_val::Float64, fig_ellipse_title::String, anim::Animation) #**poner bien los tipos
    scatter!(fig_ellipse,[0,0], [0,1], zcolor=[0,2], markersize=0.0, label=false,colorbar_ticks=[0,0.5,1.0,1.5], c=:hsv, colorbar_title="Angles (1/π)")
    #scatter!([0,0], [0,1], zcolor=[0,2], alpha=0.0, label=false,  colorbar_ticks=[0,0.5,1.0,1.5],c=:hsv, colorbar_title="Angles (1/π)")

    xlims!(-params.b*2,params.Lx+2*params.b)
    ylims!(-params.a*2,params.Ly+2*params.a)
    title!(fig_ellipse,"t=$(time_val)"*fig_ellipse_title)

    if params.make_gif
        frame(anim,fig_ellipse)
        empty!(fig_ellipse)
    end                    
end


function plot_spatial_correlation(rs, mean_data; xlabel="", title_str="", save_name="")
    plt = plot(legend=true, grid=true)
    
    color = :red
    plot!(plt, rs, mean_data, markerstrokecolor=color,
        color=color, alpha=0.7, marker=:circle, line=:solid)

    plot!(plt, xlabel=xlabel, legend=:topright)
    title!(plt, title_str)
    #display(plt)

    savefig(plt,save_name)
end


#***hacer transparente el legend !!!
function plot_fourier_correlation(rs, mean_data; xlabel="", ylabel="", title_str="", save_name="", n_points=4)
    # Define linear model: y = a + b*x
    @. linearmodel(x, p) = p[1] + p[2] * x    
    # Initial guess for parameters [intercept, slope]
    p0 = [0.0, 1.0]
    lrs = length(rs)

    x_fit = rs[1:minimum([n_points,lrs])]
    y_fit = mean_data[1:minimum([n_points,lrs])]

    # Fit the model
    fit = curve_fit(linearmodel, x_fit, y_fit, p0)

    # Get fitted parameters
    a, b = fit.param
    a_err, b_err = stderror(fit)  # Standard errors
    println("Fitted parameters:")
    println("Intercept (a): $a +- $(a_err)")
    println("Slope (b): $b +- $(b_err)")

    # Calculate R²
    y_pred = a .+ b .* x_fit
    ss_res = sum((y_fit .- y_pred).^2)
    ss_tot = sum((y_fit .- Statistics.mean(y_fit)).^2)
    r2 = 1 - ss_res / ss_tot


    # Generate points for the fitted line
    x_line = range(rs[1],rs[minimum([5,lrs])],length=2)#range(minimum(rs[1:minimum([10,lrs])]), maximum(mean_data[1:minimum([10,lrs])]), length=2)
    y_line = a .+ b .* x_line

    plt = plot(legend=true, grid=true)
    
    color = :red
    plot!(plt, rs, mean_data, markerstrokecolor=color,
        color=color, alpha=0.7, marker=:circle, line=:solid)

    # Plot the fitted line
    plot!(x_line, y_line, color=:blue, linewidth=2, label="Linear fit (first $(n_points) points)\ny=$b x+$a")

    plot!(plt, xlabel=xlabel, legend=:topright)
    title!(plt, title_str)
    #display(plt)

    savefig(plt,save_name)

    return a,b,a_err,b_err,r2
    
end

function propagate_ratio_error(a, b, Δa, Δb; correlation=0.0)
    """
    Propagate errors for x = a/b
    
    Arguments:
    - a, b: measured values
    - Δa, Δb: uncertainties in a and b (standard deviations)
    - correlation: correlation coefficient between a and b (-1 to 1)
    
    Returns:
    - x: computed ratio
    - Δx: propagated uncertainty
    - status: flag indicating if the result is reliable (0=ok, 1=problematic, 2=infinite)
    """
    

    # Handle zero or near-zero b (prevent division by zero)
    if abs(b) < eps(Float64)
        if abs(a) < eps(Float64)
            return (x=NaN, Δx=NaN, status=2, message="Both a and b are zero")
        else
            # If b is zero but a ≠ 0, the ratio is infinite
            x = (a > 0 ? Inf : -Inf)
            return (x=x, Δx=Inf, status=2, message="Division by zero")
        end
    end

    # Handle infinite uncertainties
    status = 0
    if isinf(Δa) && isinf(Δb)
        return (x=a/b, Δx=Inf, status=2, message="Both uncertainties are infinite")
    elseif isinf(Δa)
        # Only Δa is infinite, Δb is finite
        return (x=a/b, Δx=Inf, status=2, message="Δa is infinite")
    elseif isinf(Δb)
        # Only Δb is infinite
        # Special case: if b has infinite uncertainty, the result is unreliable
        # But we can still compute x with an infinite uncertainty
        x = a / b
        return (x=x, Δx=Inf, status=2, message="Δb is infinite")
    end
    
   
    
    # Calculate ratio
    x = a / b
    
    # Propagate error using formula: (Δx/x)^2 = (Δa/a)^2 + (Δb/b)^2 - 2*correlation*(Δa/a)*(Δb/b)
    # For uncorrelated variables (correlation=0): Δx = |x| * sqrt((Δa/a)^2 + (Δb/b)^2)
    
    # Check for zero or very small a
    if abs(a) < eps(Float64) && abs(Δa) > 0
        # Special case: a ≈ 0
        # Use alternative formula: Δx = |Δa| / |b| (if a=0)
        Δx = abs(Δa) / abs(b)
        # Add contribution from b error if b is also small
        if abs(b) < eps(Float64)
            Δx = Inf
            status = 1
        end
    else
        # Standard error propagation
        relative_error_squared = (Δa / a)^2 + (Δb / b)^2 - 2 * correlation * (Δa / a) * (Δb / b)
        
        # Ensure we don't get negative due to numerical issues
        if relative_error_squared < 0 && relative_error_squared > -1e-12
            relative_error_squared = 0.0
        end
        
        if relative_error_squared < 0
            # Negative due to high correlation, take absolute value
            relative_error_squared = abs(relative_error_squared)
            status = 1
        end
        
        Δx = abs(x) * sqrt(relative_error_squared)
    end
    
    # Check for NaN or infinite results
    if isnan(Δx)
        Δx = Inf
        status = 2
    end
    
    return (x=x, Δx=Δx, status=status, message="")
end


function plot_mean_spatial_correlation(times, rs, mean_data, std_data; xlabel="", ylabel="", title_str="", save_name="")
    plt = plot(legend=true, grid=true)
    t_min = length(times)
    for t in 1:t_min
        if ((t-1) % max(1, t_min ÷ 16) == 0 && t!=1) || t==2
            color = cgrad(:rainbow)[t/t_min]
            plot!(plt, rs, mean_data[t], yerr=std_data[t], label="t="*string(times[t]), markerstrokecolor=color,
                color=color, alpha=0.7, marker=:circle, line=:solid)
        end
    end
    plot!(plt, xlabel=xlabel, ylabel=ylabel, legend=:topright)
    title!(plt, title_str)
    #display(plt)

    savefig(plt,save_name)
end





function plot_spatial_correlation_Li(plt, times, rs, mean_data, std_data, Li, L, inputf; ylabel="")#, title_str="", save_name="")
    
    markers = [:circle, :square, :diamond, :star5, :xcross, :utriangle, :hexagon]
    
    t_min = length(times)
    for t in 1:t_min
        if ((t-1) % max(1, t_min ÷ 2) == 0 && t!=1) || t==2
            color = cgrad(:rainbow)[t/t_min]
            plot!(plt, rs, mean_data[t], yerr=std_data[t], label="t=$(times[t]), L=$(L)", markerstrokecolor=color,
                color=color, alpha=0.7, marker=(markers[mod1(Li, length(markers))], 6), line=line_style(inputf), alfa=0.7)
        end
    end
    plot!(plt, ylabel=ylabel, legend=:topright)
end

function plot_spatial_correlation_one_t(plt, i, t, rs, mean_data, std_data, Li, L, inputf; ylabel="")#, title_str="", save_name="")
    
    markers = [:circle, :square, :diamond, :star5, :xcross, :utriangle, :hexagon]

    common_label=inputf == 2 ? "t=$(t), L=$(L)" : nothing

    color = cgrad(:rainbow)[i/length(mean_data)]
    plot!(plt, rs, mean_data[i], yerr=std_data[i], label=common_label, markerstrokecolor=color,
        color=color, alpha=0.7, marker=(markers[mod1(Li, length(markers))], 6), line=line_style(inputf), alfa=0.7,
        ylabel=ylabel, sp=inputf)
    
end

function plot_spatial_correlation_all(plt, i, t, rs, mean_data, Li, L_tot, inputf)#, title_str="", save_name="")
    
    markers = [:circle, :square, :diamond, :star5, :xcross, :utriangle, :hexagon]

    color = cgrad(:rainbow)[Li/L_tot]
    plot!(plt, rs, mean_data[i], markerstrokecolor=color,
        color=color, alpha=0.7, marker=(markers[mod1(Li, length(markers))], 2), line=line_style(inputf), alfa=0.7,
        sp=inputf)
    
end


# Function to find the breaking point and fit the power-law region
function find_breaking_point_and_fit(windows, Q_vs_W)
    # Take logs for analysis
    logW = log10.(windows[end:-1:1])
    logQ = log10.(Q_vs_W[end:-1:1])
    
    n_points = length(logW)
    
    # Method 1: Find where derivative (slope) exceeds a threshold
    # Calculate running slopes over windows of 3 points
    slopes = []
    break_candidates = []
    
    for i in 1:n_points-2
        # Simple linear fit over 3 consecutive points
        x_seg = logW[i:i+2]
        y_seg = logQ[i:i+2]
        
        # Calculate slope using linear regression
        X = hcat(ones(3), x_seg)
        β = (X' * X) \ (X' * y_seg)
        slope = β[2]
        push!(slopes, slope)
    end
    
    # Find where slope becomes consistently negative and decreasing
    # Looking for the transition from plateau (slope near 0) to linear (slope > threshold)
    threshold = 0.2  # Minimum slope to consider as power-law behavior
    
    break_idx = nothing
    for i in eachindex(slopes)
        if slopes[i] < -threshold
            # Check if next few slopes also maintain positive trend
            if i+2 <= length(slopes) && all(slopes[i:i+2] .< -threshold)
                break_idx = i + 1  # +1 to account for the sliding window offset
                break
            end
        end
    end
    
    # If no clear break found, use alternative method
    if break_idx === nothing
        # Alternative: Use piecewise linear fit to find the optimal break point
        best_r2 = -Inf
        
        for split_idx in 3:n_points-3
            # Fit plateau region (should be flat)
            plateau_fit = linear_fit(logW[1:split_idx], logQ[1:split_idx])
            plateau_slope = plateau_fit[2]
            
            # Fit power-law region (should be linear)
            power_fit = linear_fit(logW[split_idx:end], logQ[split_idx:end])
            power_slope = power_fit[2]
            power_r2 = calculate_r2(logW[split_idx:end], logQ[split_idx:end], power_fit)
            
            # Score this split
            if abs(plateau_slope) < 0.1 && power_slope > 0.2 && power_r2 > best_r2
                best_r2 = power_r2
                break_idx = split_idx
            end
        end
    end
    
    # If still no break found, use a default (e.g., 20% of data)
    if break_idx === nothing
        break_idx = floor(Int, 0.2 * n_points)
        @warn "Could not find clear breaking point, using index $break_idx as fallback"
    end
    
    # Fit the power-law region (from break_idx to end)

    fit_logW = logW[break_idx:end]
    fit_logQ = logQ[break_idx:end]    
    
    # Linear fit for the power-law region
    X_fit = hcat(ones(length(fit_logW)), fit_logW)
    β_fit = (X_fit' * X_fit) \ (X_fit' * fit_logQ)
    intercept, slope = β_fit
    
    # Calculate R² for the fit
    r2 = calculate_r2(fit_logW, fit_logQ, β_fit)
    
    # Get the breaking point in original units
    break_W = 10^logW[break_idx]
    break_Q = 10^logQ[break_idx]

    
    return (break_idx=break_idx, break_W=break_W, break_Q=break_Q, 
            slope=slope, intercept=intercept, r2=r2, 
            fit_logW=fit_logW[end:-1:1], fit_logQ=intercept.+slope.*fit_logW[end:-1:1])
end

# Helper function for linear fit #***esto es un fit ???? ver read_outputs_new_space_correl x las dudas
function linear_fit(x, y)
    X = hcat(ones(length(x)), x)
    return (X' * X) \ (X' * y)
end

# Helper function to calculate R²
function calculate_r2(x, y, β)
    y_pred = β[1] .+ β[2] .* x
    ss_res = sum((y - y_pred).^2)
    ss_tot = sum((y .- Statistics.mean(y)).^2)
    return 1 - ss_res/ss_tot
end




