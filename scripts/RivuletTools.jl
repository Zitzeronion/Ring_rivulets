module RivuletTools

using CSV, JLD2, FileIO, Plots, Images, DataFrames, ImageSegmentation, Random, FFTW

"""
	read_data()

Read data from a well defined path based on parameters

# Arguments
- R : Radius of the (x,y)-plane
- r : Radius of the (x,z) or (y,z)-plane
- kbt : Energy that drives thermal fluctuations
- nm : Parameters of the disjoining pressure, only 32 available
- θ : Contact angle of the substrate
- day : Day the simulation ended
- month : Month the simulation ended in
- hour : Hour at which the simulation finished
- minute : Minute at which the simulation ended
- arrested : Rivulet limited by contact angle field

# Example
```julia-repl
julia>  data = RivuletTools.read_data(R=160, r=20, kbT=0.0, month=11, day=27, hour=14, minute=51, θ=20 ,nm=32, arrested=true)
```
"""
function read_data(;R=50, r=80, kbT=0.0, nm=93, θ=20, year=2023, month=10, day=26, hour=7, minute=5, arrested=false, gamma="", slip=0)
	dpath = joinpath("/home/zitz", "Swalbe.jl/data/Rivulets")
	file_name = "$(dpath)/height_R_$(R)_r_$(r)_ang_$(θ)_kbt_$(kbT)_nm_3-2_runDate_$(year)$(month)$(day)$(hour)$(minute).jld2"
	if arrested
		file_name = "$(dpath)/arrested_height_R_$(R)_r_$(r)_ang_$(θ)_kbt_$(kbT)_nm_3-2_runDate_$(year)$(month)$(day)$(hour)$(minute).jld2"
	elseif gamma != ""
		file_name = "$(dpath)/$(gamma)height_R_$(R)_r_$(r)_ang_$(θ)_kbt_$(kbT)_nm_3-2_runDate_$(year)$(month)$(day)$(hour)$(minute).jld2"
	elseif slip != 0
		file_name = "$(dpath)/slip_$(slip)_height_R_$(R)_r_$(r)_ang_$(θ)_kbt_$(kbT)_nm_3-2_runDate_$(year)$(month)$(day)$(hour)$(minute).jld2"
	end

	if isfile(file_name) 
		data = load(file_name)
	else 
		data = 0
		@error("The file:\n$(file_name)\ndoes not exists, check data or typos")
	end
	return data
end

"""
	heatmap_data(data; time)

Creates a heatmap of the height of some simulation `data` at given `time`.

# Example
```julia-repl
julia> data = RivuletTools.read_data(R=160, r=20, kbT=0.0, month=11, day=27, hour=14, minute=51, θ=20 ,nm=32, arrested=true)

julia> heatmap_data(data, t=875000)
```
"""
function heatmap_data(data; t=25000,  just_data=false)
	h = reshape(data["h_$(t)"], 512, 512)
	if just_data
		return h
	end
	heatmap(h, 
		aspect_ratio=1, 
		c=:viridis, 
		# colorbar_title=" \nheight" # Actually the good way, but Pluto cuts for whatever reason
		colorbar_title="height" # Now numbers and labels are too close
		)
end


"""
	plot_slice(data)

Plots a cut through the rivulet

# Example

```julia-repl
julia> R = 180; rr = 40; θ = 30;

julia> data_example = read_data( 	# Simulation data
		R=R, 
		r=rr, 
		kbT=0.0, 
		month=11, 
		day=5, 
		hour=1, 
		minute=28, 
		θ=θ, 
		nm=32, 
		arrested=false)

julia> plot_slice(data_example, t=25000)
```
"""
function plot_slice(data; t=25000, window=(false, 80, 110))
	h = reshape(data["h_$(t)"], 512, 512)
	if window[1]
		plot(h[256, window[2]:window[3]], label="t=$(t)Δt", xlabel="x/[Δx]", ylabel="height")
	else
		plot(h[256, :], label="t=$(t)Δt", xlabel="x/[Δx]", ylabel="height")
	end
	# println(maximum(h))
end

"""
	do_gif(data, filename::String)

Creates an animation of the heightfield for the available time steps.

# Example 
```julia-repl
julia> h = read_data(R=160, r=40, kbT=0.0, month=12, day=20, hour=11, minute=33, θ=20, nm=32)  # Link to data 

julia> do_gif(h, filename, timeMax=maxtimestep)  # Use that data and supply a filename for the gif
```
"""
function do_gif(data, filename::String; timeMax=5000000)
	p = heatmap_data(data, t=25000)
	anim = Animation()
	for x in 50000:25000:timeMax
		plot(heatmap_data(data, t=x))
		frame(anim)
	end
	gif(anim, "../assets/$(filename).gif")
end

"""
	do_gif_slice(data, filename::String)

Creates an animation of a slice of the heightfield for the available time steps.

# Example
```julia-repl
julia> data = read_data(R=180, r=40, kbT=0.0, month=11, day=3, hour=23, minute=34, θ=40 ,nm=32, arrested=false)

julia> do_gif_slice(data, filename, timeMax=2500000)
```
"""
function do_gif_slice(data, filename::String; timeMax=5000000)
	h0 = reshape(data["h_25000"], 512, 512)
	p = plot(h0[256, :], label="t=$(25000)Δt", xlabel="x/[Δx]", ylabel="height")
	anim = Animation()
	for t in 50000:25000:timeMax
		h = reshape(data["h_$(t)"], 512, 512)
		plot(h[256, :], label="t=$(t)Δt", xlabel="x/[Δx]", ylabel="height")
		frame(anim)
	end
	gif(anim, "../assets/$(filename).gif")
end

"""
	measure_diameter(data)

Measures the diameter of the fluid torus at a single time step.

# Example
```julia-repl
julia> h = read_data(R=160, r=40, kbT=0.0, month=12, day=20, hour=11, minute=33, θ=20, nm=32)  # Link to data 

julia> R, rr, outerR = measure_diameter(h) 
``` 
"""
function measure_diameter(data; t=25000, δ=0.055)
	distance = 0
	inner = 0
	allr = 0
	if typeof(data) == Dict{String, Any}
		h_data_slice = reshape(data["h_$(t)"], 512, 512)[256, :]
		riv = findall(h_data_slice .> δ)
		if length(riv) > 1
			allr = riv[end] - riv[begin]
			for j in eachindex(riv[begin+1:end-1])
				if riv[j+1] - riv[j] > 1
					distance = riv[j+1] - riv[j]
					inner = riv[j] - riv[begin]
				end
			end
		end
	else
		h_data_slice = data[256, :]
		riv = findall(h_data_slice .> δ)
		if length(riv) > 1
			allr = riv[end] - riv[begin]
			for j in eachindex(riv[begin+1:end-1])
				if riv[j+1] - riv[j] > 1
					distance = riv[j+1] - riv[j]
					inner = riv[j] - riv[begin]
				end
			end
		end
	end
	return distance, inner, allr
end

"""
	measure_Δh(data)

Measures the absolute height difference on the computational domain at a given time.

# Example
```julia-repl
julia> h = read_data(R=160, r=40, kbT=0.0, month=12, day=20, hour=11, minute=33, θ=20, nm=32)  # Link to data 

julia> dH = measure_Δh(h)
```
"""
function measure_Δh(data; t=25000)
	Δh = 0 
	
	h = reshape(data["h_$(t)"], 512, 512)
	riv = findall(h .> 0.056)
	
	hmin = minimum(h[riv], init=0)
	hmax = maximum(h[riv], init=0)
	Δh = hmax - hmin
	return Δh
end

"""
	measure_clusters(data; t=25000)

Measures the number of clusters using `Images.jl`. 

At the beginning of the simulation the rivulet is connected and the image analysis should return that there is just a single cluster.
If the rivulet retracts into a single droplet we assume to observe one cluster throughout the whole simulation.
However, if the rivulet ruptures and forms droplets we expect that number deviate from one.

# Example
```julia-repl
julia> h = read_data(R=160, r=40, kbT=0.0, month=12, day=20, hour=11, minute=33, θ=20, nm=32)  # Link to data 

julia> clusters = measure_cluster(h)
```
"""
function measure_cluster(data; t=25000)
	clusters = Int64(0)
	h = reshape(data["h_$(t)"], 512,512)
	threshold = 0.06
	bw = Gray.(h) .< threshold
	dist = 1 .- distance_transform(feature_transform(bw))
	markers = label_components(dist .< -5)
	segments = watershed(dist, markers)
	if length(segments.segment_labels) >= 1
		clusters = segments.segment_labels[end]
	else 
		clusters = 0
	end
	# newH = map(i->get_random_color(i), labels_map(segments)) .* (1 .-bw)
	return clusters
end

"""
    torus(lx, ly, r₁, R₂, θ, center, hmin)

Generates a cut torus with contact angle `θ`, (`x`,`y`) radius `R₂` and (`x`,`z`) radius `r₁` centered at `center`.

# Arguments

- `lx::Int`: Size of the domain in x-direction
- `ly::Int`: Size of the domain in y-direction
- `r₁::AbstractFloat`: Radius in (x,z)-plane
- `R₂::AbstractFloat`: Radius in (x,y)-plane
- `θ::AbstractFloat`: contact angle in multiples of `π`
- `center::Tuple{Int, Int}`: Center position of the torus  
- `hmin::AbstractFloat`: small value above 0.0

# Examples

```jldoctest
julia> using Swalbe, Test

julia> rad = 45; R = 80; θ = 1/9; center = (128, 128);

julia> height = Swalbe.torus(256, 256, rad, R, θ, center);

julia> @test maximum(height) ≈ rad * (1 - cospi(θ)) # Simple geometry
Test Passed

julia> argmax(height) # On the outer ring!
CartesianIndex(128, 48)

```
"""
function torus(lx, ly, r₁, R₂, θ, center, hmin = 0.05; noise=0.0)
	h = zeros(lx,ly)
	for i in eachindex(h[:,1]), j in eachindex(h[1,:])
		coord = sqrt((i-center[1])^2 + (j-center[2])^2)
		half = (r₁)^2 - (coord - R₂)^2
		if half <= 0.0
			h[i,j] = hmin
		else
			h[i,j] = sqrt(half)
		end
	end
    # Second loop to have a well defined contact angle
    for i in eachindex(h[:,1]), j in eachindex(h[1,:])
        # Cut the half sphere to the desired contact angle θ
        correction = h[i,j] - r₁ * cospi(θ)
        if correction < hmin
            h[i,j] = hmin
        else
            h[i,j] = correction + randn()*noise
        end
    end
	return h
end

"""
	compute_droplet(h, θ; δ=0.052)

Computes the equivalent base radius of a spherical cap for a given initial condition.

# Returns

- `baserad::Float`: Size of the baseradius of the droplet.
- `vol::Float`: Volume of the liquid rivulet.
- `baserad::Float`: Cap height of the droplet given rivulet volume and contact angle.

# Example
```julia-repl
julia> rad = 45; R = 80; θ = 1/9; center = (128, 128);

julia> height = torus(256, 256, rad, R, θ, center);

julia> r, V, h_cap = compute_droplet(h, θ; δ=0.0505)
"""
function compute_droplet(h, θ; δ=0.0505)
	riv = findall(h .> δ)
	vol = sum(h[riv])
	rad = cbrt(3vol/(π * (2+cospi(θ)) * (1-cospi(θ))^2))
	baserad = rad * sinpi(θ)
	capheight = rad * (1 - cospi(θ))

	return baserad, vol, capheight
end

# End module
end