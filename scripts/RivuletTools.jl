module RivuletTools

using CSV, JLD2, FileIO, Plots, Images, DataFrames, ImageSegmentation, Random, FFTW, CategoricalArrays, StatsPlots

# Raw data
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
- gamma: Different surface tension simulations
- slip: Simulations with different slip length
- gradient: Simulations with wettability gradient

# Example
```julia-repl
julia>  data = RivuletTools.read_data(R=160, r=20, kbT=0.0, month=11, day=27, hour=14, minute=51, θ=20 ,nm=32, arrested=true)
```
"""
function read_data(;R=50, 
		r=80, 
		kbT=0.0, 
		nm=(3,2), 
		θ=20, 
		year=2023, 
		month=10, 
		day=26, 
		hour=7, 
		minute=5, 
		arrested=false, 
		gamma="", 
		slip=0,
		gradient=(false, 10, 40))
	dpath = joinpath("/home/zitz", "Swalbe.jl/data/Rivulets")
	file_name = "$(dpath)/height_R_$(R)_r_$(r)_ang_$(θ)_kbt_$(kbT)_nm_$(nm[1])-$(nm[2])_runDate_$(year)$(month)$(day)$(hour)$(minute).jld2"
	if arrested
		file_name = "$(dpath)/arrested_height_R_$(R)_r_$(r)_ang_$(θ)_kbt_$(kbT)_nm_$(nm[1])-$(nm[2])_runDate_$(year)$(month)$(day)$(hour)$(minute).jld2"
	elseif gamma != ""
		file_name = "$(dpath)/$(gamma)height_R_$(R)_r_$(r)_ang_$(θ)_kbt_$(kbT)_nm_$(nm[1])-$(nm[2])_runDate_$(year)$(month)$(day)$(hour)$(minute).jld2"
	elseif slip != 0
		file_name = "$(dpath)/slip_$(slip)_height_R_$(R)_r_$(r)_ang_$(θ)_kbt_$(kbT)_nm_$(nm[1])-$(nm[2])_runDate_$(year)$(month)$(day)$(hour)$(minute).jld2"
	elseif gradient[1]
		file_name = "$(dpath)/wet_grad_lin_$(gradient[2])$(gradient[3])_height_R_$(R)_r_$(r)_ang_$(θ)_kbt_$(kbT)_nm_$(nm[1])-$(nm[2])_runDate_$(year)$(month)$(day)$(hour)$(minute).jld2"
	end

	if isfile(file_name) 
		data = load(file_name)
	else 
		data = 0
		@error("The file:\n$(file_name)\ndoes not exists, check data or typos")
	end
	return data
end

# DataFrames and CSVs
"""
	csv2df(file_name)

Read a `.csv` to a dataframe.

# Example
```julia-repl
julia> file = "myfile" # Assuming "myfile" is a csv file in the local directory

julia> df = csv2df(file)
```
"""
function csv2df(file_name)
	# path = "/net/euler/zitz/Swalbe.jl/data/DataFrames/$(file_name).csv"
	path2 = "/net/euler/zitz/Ring_rivulets/data/$(file_name).csv"
	measurements = CSV.read(path2, DataFrame)
	return measurements
end

# Anything related to plotting
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
		xlims=(1, 512),
		ylims=(1, 512),
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
	anim = Animation()
	for x in 25000:25000:timeMax
		plot(heatmap_data(data, t=x))
		frame(anim)
	end
	gif(anim, "../assets/$(filename).gif")
end

"""
	do_ringgif(data, filename::String; timeMax=5000000)

Generates a gif for a circular cut along the rivulet.

# Example
```julia-repl
juli> do_ringgif(data[25], "ringGif")
```
"""
function do_ringgif(data, filename::String; timeMax=2500000)
	ringanim = Animation()
	initial_data = t0_data()
	h_drop = initial_data[(initial_data.angle .== data[9]) .& (initial_data.R0 .== data[1]) .& (initial_data.rr0 .== data[2]), :].hdrop[1]
	for t in 25000:25000:timeMax
		hCirc, Radius = RivuletTools.getRingCurve(data, t)
		Radelements = length(hCirc)
		normX = 0:2π/(Radelements-1):2π 
		plot(normX, 
			hCirc ./ h_drop, 
			label="R₀=$(data[1]) r₀=$(data[2])", 
			xlabel="x\\(\\phi\\)", 
			ylabel="h/h_d", 
			xticks = ([0:π/2:2*π;], ["0", "\\pi/2", "\\pi", "3\\pi/2", "2\\pi"]),
			title="Radius: $(Radius) time: $(t)",
			grid=false,
			ylims=(0, 1),
			xlims=(0, 2π),
			)
		frame(ringanim)
	end
	gif(ringanim, "../assets/$(filename).gif")
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
	data2gif(data; prefix)

Generates video files in gif formate named after parameters and `prefix` for the supplied set of `data`.
"""
function data2gif(data; prefix="")
	kbtDict = Dict(0.0 => "kbt_off", 1.0e-6 => "kbt_on")
	maxtimestep = 2500000
	for i in eachindex(data)
		filename = "../assets/$(prefix)ang_$(data[i][8])_R_$(data[i][1])_rr_$(data[i][2])_$(kbtDict[data[i][3]]).gif"
		if isfile(filename)
			println("There is alread a file called $(prefix)ang_$(data[i][8])_R_$(data[i][1])_rr_$(data[i][2])_$(kbtDict[data[i][3]]).gif  in the assets folder")
		else
			h = read_data(R=data[i][1], 
				r=data[i][2], 
				kbT=data[i][3], 
				year=data[i][4], 
				month=data[i][5], 
				day=data[i][6], 
				hour=data[i][7], 
				minute=data[i][8], 
				θ=data[i][9], 
				nm=32)
			println("R=$(data[i][1]) with rr=$(data[i][2]) and kbt=$(data[i][3])")
			if data[i][3] == 0.0
				do_gif(h, "$(prefix)ang_$(data[i][9])_R_$(data[i][1])_rr_$(data[i][2])_$(kbtDict[data[i][3]])", timeMax=maxtimestep)
			elseif data[i][3] == 1.0e-6
				do_gif(h, "$(prefix)ang_$(data[i][9])_R_$(data[i][1])_rr_$(data[i][2])_$(kbtDict[data[i][3]])", timeMax=maxtimestep)
			end
		end
	end
end

"""
	renderGifs()

Similar to `data2gif()` with hard coded `prefix` and all `data` sets.
"""
function renderGifs(; verbose=false)
	for input in ([data, ""], [data_arrested, "arr_"], [data_gamma05, "gamma05_"], [data_gamma20, "gamma20_"])
		kbtDict = Dict(0.0 => "kbt_off", 1.0e-6 => "kbt_on")
		arrDict = Dict("arr_" => true, "" => false, "gamma05_" => false, "gamma20_" => false)
		for i in eachindex(input[1])
			path2Swalbe = "../../Swalbe.jl/assets/"
			filename = "$(path2Swalbe)$(input[2])ang_$(input[1][i][9])_R_$(input[1][i][1])_rr_$(input[1][i][2])_$(kbtDict[input[1][i][3]]).gif"
			if isfile(filename)
				if verbose
					println("There is alread a file called $(input[2])ang_$(input[1][i][9])_R_$(input[1][i][1])_rr_$(input[1][i][2])_$(kbtDict[input[1][i][3]]).gif  in the assets folder")
				end
			else
				if input[2] == "arr_"
					h = read_data(R=input[1][i][1], 
						r=input[1][i][2], 
						kbT=input[1][i][3], 
						year=input[1][i][4], 
						month=input[1][i][5], 
						day=input[1][i][6], 
						hour=input[1][i][7], 
						minute=input[1][i][8], 
						θ=input[1][i][9], 
						nm=32, 
						arrested=arrDict[input[2]], 
						gamma="")
				else
					h = read_data(R=input[1][i][1], 
						r=input[1][i][2], 
						kbT=input[1][i][3], 
						year=input[1][i][4], 
						month=input[1][i][5], 
						day=input[1][i][6], 
						hour=input[1][i][7], 
						minute=input[1][i][8], 
						θ=input[1][i][9], 
						nm=32, 
						arrested=arrDict[input[2]], 
						gamma=input[2])
				end
				if verbose
					println("R=$(input[1][i][1]) with rr=$(input[1][i][2]) and kbt=$(input[1][i][3])")
				end
				filename = "$(path2Swalbe)$(input[2])ang_$(input[1][i][9])_R_$(input[1][i][1])_rr_$(input[1][i][2])_$(kbtDict[input[1][i][3]])"
				do_gif(h, filename, timeMax=2500000)
			end
		end
	end
end

"""
	render_slip(; verbose=false)

Similar to `data2gif()` and `renderGifs()` but only for different slip length simulations.
"""
function render_slip(; verbose=false)
	# For easier naming of output
	kbtDict = Dict(0.0 => "kbt_off", 1.0e-6 => "kbt_on")
	# Loop through the slip data
	for i in eachindex(data_slip)
		# Save it with the other files
		path2Swalbe = "../../Swalbe.jl/assets/"
		# File name for the gif
		filename = "$(path2Swalbe)slip_$(data_slip[i][10])_ang_$(data_slip[i][9])_R_$(data_slip[i][1])_rr_$(data_slip[i][2])_$(kbtDict[data_slip[i][3]]).gif"
		# Check if it is already existing
		if isfile(filename)
			if verbose
				println("This file alread exists")
			end
		else
			# Otherwise read the data
			h = read_data(R=data_slip[i][1], r=data_slip[i][2], kbT=data_slip[i][3], year=data_slip[i][4], month=data_slip[i][5], day=data_slip[i][6], hour=data_slip[i][7], minute=data_slip[i][8], θ=data_slip[i][9], nm=32, slip=data_slip[i][10])
			if verbose
				println("Reading file with parameters R=$(data_slip[i][1]), rr=$(data_slip[i][2]) and slip=$(data_slip[i][10])")
			end
			do_gif(h, "$(path2Swalbe)slip_$(data_slip[i][10])_ang_$(data_slip[i][9])_R_$(data_slip[i][1])_rr_$(data_slip[i][2])_$(kbtDict[data_slip[i][3]])", timeMax=2500000)
		end
	end
end

"""
	get_random_color(seed)

Picks a random color from the RGB space.
"""
function get_random_color(seed)
    Random.seed!(seed)
    rand(RGB{N0f8})
end

"""
	segment_image(h_data, t)

Plots the result of the image segmentation, thus colorates all found clusters.
"""
function segment_image(h_data, t)
	hTry = reshape(h_data["h_$(t)"], 512,512)
	bw = Gray.(hTry) .< 0.06
	dist = 1 .- distance_transform(feature_transform(bw))
	markers = label_components(dist .< -8)
	segments = watershed(dist, markers)
	println(segments)
	newH = map(i->get_random_color(i), labels_map(segments)) .* (1 .-bw)
end

# Measurements and data analysis
function dropletFrame(; saveme=false)
	measureDF = CSV.read("../data/dynamics_uniform_and_patterned.csv", DataFrame)
	initial_data = t0_data()
	dfLSA = DataFrame()
	maxDropsData = Int[]
	angles = Int[]
	Rs = Int[]
	rrs = Int[]
	widths = Int[]
	heights = Float64[]
	psi0 = Float64[]
	pattern = String[]
	for angle in [10, 20, 30, 40]
		for R in [150, 160, 180, 200]
			for rr in [20, 30, 40, 60, 80, 100]
				ndrops = maximum(measureDF[(measureDF.R .== R) .& (measureDF.rr .== rr) .& (measureDF.theta .== angle) .& (measureDF.kbt .== 0), :].clusters, init=0)
				pat = maximum(measureDF[(measureDF.R .== R) .& (measureDF.rr .== rr) .& (measureDF.theta .== angle) .& (measureDF.kbt .== 0), :].substrate, init="")
				push!(pattern, pat)
				
				push!(maxDropsData, ndrops)
				push!(angles, angle)
				push!(Rs, R)
				push!(rrs, rr)
				
				psi = initial_data[(initial_data.R0 .== R) .& (initial_data.rr0 .== rr) .& (initial_data.angle .== angle), :].psi0[1]
				push!(psi0, psi)
				
				wRing = 2*initial_data[(initial_data.R0 .== R) .& (initial_data.rr0 .== rr) .& (initial_data.angle .== angle), :].realrr[1]
				push!(widths, wRing)

				hRing = initial_data[(initial_data.R0 .== R) .& (initial_data.rr0 .== rr) .& (initial_data.angle .== angle), :].maxh0[1]
				push!(heights, hRing)
			end
		end
	end
	cpattern = categorical(pattern)
	dfLSA.ndrops = maxDropsData
	dfLSA.theta = angles
	dfLSA.R = Rs
	dfLSA.rr = rrs
	dfLSA.width = widths
	dfLSA.height = heights
	dfLSA.psi0 = psi0
	dfLSA.substrate = cpattern

	dfLSAclean = dfLSA[dfLSA.ndrops .> 0, :]
	if saveme
		CSV.write("../data/maxdroplets.csv", dfLSAclean)
	end
	return dfLSAclean
end


"""
	ringCurve(data; R=(false, 180))

Returns the height field along a circular cut.
"""
function getRingCurve(data, tcut::Int; CircRad=(false, 180), center=(256,256), arr=false, grad=(false, 10, 40))
	# Create a distance array, needed later
	dd = distanceArray()
	# Contact angle dict
	thetaDict = Dict(10 => 1/18, 20 => 1/9, 30 => 1/6, 40 => 2/9)
	# Read in the data that we use for the cut
	dataCut = read_data(R=data[1], 
						r=data[2], 
						kbT=data[3],
						year=data[4], 
						month=data[5], 
						day=data[6], 
						hour=data[7], 
						minute=data[8], 
						θ=data[9],
						arrested=arr, 
						gradient=grad,
						nm=(3,2))
	# Return the height field of that data at time `tcut`
	if tcut == 0
		if data[9] > π/2
			theta = thetaDict[data[9]]
			hh = torus(512, 512, data[2], data[1], theta, (256,256), noise=0.01)
		else
			hh = torus(512, 512, data[2], data[1], data[9], (256,256))
		end
	else
		hh = heatmap_data(dataCut, t=tcut, just_data=true)
	end
	# Extract the coordinates of the maximum, so that we can compute a distance
	maxHeight = findmax(hh)
	if CircRad[1]
		distCut = CircRad[2]
	else
		distCut = round(Int, sqrt((maxHeight[2][1] - center[1])^2 + (maxHeight[2][2] - center[2])^2))
	end
	# No need to compute when there is a single droplet
	if distCut == 0
		return hh[center[1], center[2]], distCut
	end
	# Now some coding to get the job done
	# Get the cartesian indices of all points for a given distance.
	cutMax = findall(dd .== distCut)
	# The result should be a simple connected curve
	# A naive solution for this problem is to separte the point finding into quadrants
	firstquad = CartesianIndex{2}[]
	secondquad = CartesianIndex{2}[]
	thirdquad = CartesianIndex{2}[]
	fourthquad = CartesianIndex{2}[]
	# Based on distance of the maximum we find upper and lower limits for the quadrants
	lxMax = maximum(cutMax)[1]
	lyMax = maximum(cutMax)[2]
	lxMin = minimum(cutMax)[1]
	lyMin = minimum(cutMax)[2]
	# Then we iterate through the quadrants and save the indices which are on the cut
	# First qudrant
	for quad in [(firstquad, 256:-1:lxMin, lyMax:-1:256), 
				(secondquad, lxMin:256, 256:-1:lyMin), 
				(thirdquad, 256:lxMax, lyMin:256), 
				(fourthquad, lxMax:-1:256, 256:lyMax)]
		for j in quad[2]
			for k in quad[3]
				if CartesianIndex(j, k) in cutMax
					push!(quad[1], CartesianIndex(j, k))
				else 
					continue
				end
			end
		end
	end
	# We glue them together and have a nice connected one dimensional representation of the height along a circle
	curve = vcat(firstquad, secondquad[2:end], thirdquad[2:end], fourthquad[2:end-1])
	# The first and last points of consecutive quadrants overlap that is why we use [2:end] and [2:end-1]
	return hh[curve], distCut
end

"""
	ringOverTime()

Measures the circular cuts for a whole simulation and returns a `dict[time => hCut]`

# Example
```julia-repl
julia> hrings = ringOverTime(data[44])   
``` 
"""
function ringOverTime(data; arr=false, grad=(false, 10, 40), tEnd=2500000, curves=true, savedataframe=true)
	dpath = joinpath("/home/zitz", "Ring_rivulets/data/")
	ringData = DataFrame()
	hRing = Dict()
	hmax = Float64[]
	hmin = Float64[]
	hdelta = Float64[]
	Rt = Int64[]
	for t in 0:25000:tEnd
		heightRing = getRingCurve(data, t, arr=arr, grad=grad)
		hx = maximum(heightRing[1])
		hi = minimum(heightRing[1])
		hRing["t_$(t)"] = heightRing[1]
		push!(hmax, hx)
		push!(hmin, hi)
		push!(hdelta, hx - hi)
		push!(Rt, heightRing[2])
	end
	ringData.time = 0:25000:tEnd
	ringData.R_t = Rt
	ringData.hmax = hmax
	ringData.hmin = hmin
	ringData.deltaH = hdelta	
	ringData.R0 = fill(data[1], length(hmax))	
	ringData.rr0 = fill(data[2], length(hmax))	
	ringData.theta = fill(data[9], length(hmax))
	# Add substrate flavor
	if arr
		ringData.substrate = fill("pattern", length(hmax))
	elseif grad[1]
		ringData.substrate = fill("grad_$(grad[2])$(grad[3])", length(hmax))		
	else
		ringData.substrate = fill("uniform", length(hmax))
	end
	# Save CSV 
	if savedataframe
		CSV.write("$(dpath)ring_sim_R_$(data[1])_rr_$(data[2])_theta_$(data[9]).csv", ringData)
	else
		if curves
			# Return height data and dataframe
			return hRing, ringData
		end
		# Return dataframe only
		return ringData
	end
end

function combine_ring_data()
	dpath = joinpath("/home/zitz", "Ring_rivulets/data/")
	df = DataFrame()
	for i in [(data, false), (data_arrested, true)]
		for j in i[1]
			if j[3] == 0.0
				somedf = ringOverTime(j, arr=i[2], curves=false, savedataframe=false)
				println("Analysing simulation R=$(j[1]) r=$(j[2]) theta=$(j[9])")
				df = vcat(df, somedf)
			end
		end
	end
	CSV.write("$(dpath)ring_all_sims_nokBT.csv", df)
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
		clusters = segments.segment_labels[end] - 1
	else 
		clusters = 0
	end
	# newH = map(i->get_random_color(i), labels_map(segments)) .* (1 .-bw)
	return clusters
end

"""
	distanceArray(; X=512, Y=512, center=(256,256))

Computes the Chebyshev distances away from `center` and returns a `X` time `Y` matrix with distances

# Example
```julia-repl
julia> d = distanceArray
```
"""
function distanceArray(; X=512, Y=512, center=(256,256))
	distances = zeros(X,Y)
	for x in 1:X
		for y in 1:Y
			distances[x,y] = Int(round(sqrt((x-center[1])^2 + (y-center[2])^2))) + 1
		end
	end
	return distances
end

"""
	simpleRadialAverage(array)

Simple radial averagering with Chebyshev distance.
"""
function simpleRadialAverage(array; abssqrt=false)
	Lx = size(array,1)
	Ly = size(array,2)
	dummy = zeros(Lx,Ly)
	if abssqrt
		dummy = abs.(array .* array)
	else
		dummy .= array
	end
	d = distanceArray(X=Lx, Y=Ly, center=(Lx÷2, Ly÷2))
	averageRadialChebyshev = zeros(Lx÷2)
	for i in 1:Lx÷2
		toAverage = dummy[d .== i]
		averageRadialChebyshev[i] = sum(toAverage) / length(toAverage)
	end
	return averageRadialChebyshev
end

"""
	measure_data(data, label::String, remeasure::Bool, pat::Bool, gam::String)

Generates a dataframe of dynamic measurements for a set of `data` and writes it to a `.csv` or reads it from a `.csv`.
"""
function measure_data(data, label::String, remeasure::Bool, pat::Bool, gam::String)
	measurements = DataFrame()
	dpath = joinpath("/home/zitz", "Ring_rivulets/data/")
	if remeasure
		measurements = DataFrame()
		for i in eachindex(data)
			# println(i)
			someFrame = DataFrame()
			h = read_data(R=data[i][1], 
						r=data[i][2], 
						kbT=data[i][3],
						year=data[i][4], 
						month=data[i][5], 
						day=data[i][6], 
						hour=data[i][7], 
						minute=data[i][8], 
						θ=data[i][9] ,
						nm=32, 
						arrested=pat, 
						gamma=gam
						)
			R = Float64[]
			rr = Float64[]
			beta = Float64[]
			allr = Float64[]
			deltaH = Float64[]
			clusters = Int64[]
			for j in 0:25000:2500000
				if j > 0
					R_measure = measure_diameter(h, t=j)
					# Push the radii not the diameters
					push!(R, R_measure[1]/2)
					push!(rr, R_measure[2]/2)
					push!(allr, R_measure[3]/2)
					push!(beta, (R_measure[2]/2)/(R_measure[1]/2 + R_measure[2]/2))
					push!(deltaH, measure_Δh(h, t=j))
					push!(clusters, measure_cluster(h, t=j))
				else 
					h0 = torus(512, 512, data[i][1], data[i][2], data[i][9], (256,256))
					R_measure = measure_diameter(h0)
					push!(R, R_measure[1]/2)
					push!(rr, R_measure[2]/2)
					push!(allr, R_measure[3]/2)
					push!(beta, (R_measure[2]/2)/(R_measure[1]/2 + R_measure[2]/2))
					push!(deltaH, maximum(h0) - 0.056)
					push!(clusters, 1)
				end
			end
			someFrame.major = R
			someFrame.minor = rr
			someFrame.outerR = allr
			someFrame.beta = beta
			someFrame.dH = deltaH
			someFrame.clusters = clusters
			someFrame.R = fill(data[i][1],101)
			someFrame.rr = fill(data[i][2],101)
			someFrame.kbt = fill(data[i][3],101)
			someFrame.theta = fill(data[i][9],101)
			someFrame.time = 0:25000:2500000
			measurements = vcat(measurements, someFrame)
			println("done with $(i) of $(length(data))")
		end
		CSV.write("$(dpath)$(label).csv", measurements)
	else
		println("No data analysis is performed because remasure is set false")
	end
end

#  Initial conditions
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
	t0_data()

Computes different parameters from the initial condition, as such the true radii, liquid volume and many more.

# Returns
- `initial_data::Dataframe` : A dataframe with multiple case specific datapoints.
"""
function t0_data()
	initial_data = DataFrame()
	angs = Float64[] 	# initial condition angle
	Rs = Float64[] 		# initial condition major radius
	rrs = Float64[] 	# initial condition minor radius
	m1 = Float64[] 		# measured outer radius
	m2 = Float64[] 		# measured minor radius
	m3 = Float64[] 		# measured outer-minor radius 
	psi0 = Float64[] 	# measured width over r(h_max) González, Diez, Kondic
	nmax = Float64[] 	# number of droplets predicted by González, Diez, Kondic
	beta = Float64[] 	# measured relation minor/major
	maxh = Float64[] 	# maximal rivulet height at t=0
	vols = Float64[] 	# liquid volume of the rivulet
	dropR = Float64[] 	# radius of a droplet containing all liquid
	droph = Float64[] 	# maximal height of a droplet containing all liquid
	charL = Float64[] 	# see paper: "Capillary instabilities in solid thin films: Lines"
	Ohs = Float64[] 	# Ohnesorge number
	# Loop through initial conditions
	for angle in [2/9, 1/6, 1/9, 1/18]
		for R in [80, 120, 150, 160, 180, 200]
			for rr in [20, 30, 40, 60, 80, 100]
				# Create initial condition
				h = RivuletTools.torus(512, 512, rr, R, angle, (256, 256))
				# measure relevant parameter
				drop_radius, drop_vol, drop_h = compute_droplet(h, angle)
				geometry = measure_diameter(h)
				push!(m1, geometry[1]/2)
				push!(m2, geometry[2]/2)
				push!(m3, geometry[3]/2)
				push!(psi0, (geometry[3] - geometry[1])/(geometry[1] + geometry[2]))
				push!(nmax, π/(2((geometry[3] - geometry[1])/(geometry[1] + geometry[2]))))
				push!(beta, (geometry[2]/2)/R)
				push!(angs, angle)
				push!(Ohs, compute_Oh(geometry[2]/2))
				push!(Rs, R)
				push!(rrs, rr)
				push!(maxh, maximum(h))
				push!(vols, drop_vol)
				push!(dropR, drop_radius)
				push!(droph, drop_h)
				push!(charL, sqrt((rr*rr*(angle*π - sinpi(angle)*cospi(angle)))/π))
			end
		end
	end
	initial_data.angle = round.(rad2deg.(angs .* π)) 
	initial_data.R0 = Rs
	initial_data.rr0 = rrs
	initial_data.realR = m3
	initial_data.realrr = m2
	initial_data.realOR = m1
	initial_data.psi0 = psi0
	initial_data.nmax = nmax
	initial_data.beta = beta
	initial_data.maxh0 = maxh
	initial_data.vol = vols
	initial_data.rdrop = dropR
	initial_data.hdrop = droph
	initial_data.charL = charL
	initial_data.r0rf = m3 .- dropR
	initial_data.t0 = round.(t₀.(m2))
	initial_data.tic = round.(tic.(m2))
	initial_data.tau = round.(tau_rim.(droph, angs))
	initial_data.ts = round.(t_s.(droph, angs))
	initial_data.t00 = round.(t_00.(m2, angs))
	initial_data.Oh = Ohs 

	return initial_data
end

# Spectral analysis and fft things
"""
	data2fft(;whichdata=data, dataset=25, time=25000, quater=false, output=false)

Computes the `fft` of a height field an
"""
function data2fft(;whichdata=data, dataset=25, time=25000, quater=false, output=false)
	input = read_data(R=whichdata[dataset][1], 
						r=whichdata[dataset][2], 
						kbT=whichdata[dataset][3], 
						year=whichdata[dataset][4], 
						month=whichdata[dataset][5], 
						day=whichdata[dataset][6], 
						hour=whichdata[dataset][7], 
						minute=whichdata[dataset][8], 
						θ=whichdata[dataset][9], 
						nm=(3,2))
	heightField = heatmap_data(input, t=time, just_data=true)
	L = size(heightField)[1]
	maxH = maximum(heightField)
	spectrumH= fftshift(fft(heightField ./ maxH))
	shifted_k = fftshift(fftfreq(L)*L)
	k_pi = shifted_k .* 2π/L
	if quater
		p = heatmap(k_pi[256:end], k_pi[256:end], log.(abs.(spectrumH[256:end, 256:end] .* spectrumH[256:end, 256:end])) .+ 1, 
			aspect_ratio=1, 
			xlims=(0,π), 
			ylims=(0,π)
		)
	else
		p = heatmap(k_pi, k_pi, log.(abs.(spectrumH .* spectrumH)) .+ 1, 
			aspect_ratio=1, 
			xlims=(-π,π), 
			ylims=(-π, π),
			# xlabel="x\\(\\phi\\)", xticks = ([0:π/2:2*π;], ["0", "\\pi/2", "\\pi", "3\\pi/2", "2\\pi"])
			#clim=(0.1, 1000) # Limits for heatmap
		)
	end
	if output
		return heightField, spectrumH
	end
	p
end

function height2fft(data, t; output = false)
	initial_data = t0_data()
	h_drop = initial_data[(initial_data.angle .== data[9]) .& (initial_data.R0 .== data[1]) .& (initial_data.rr0 .== data[2]), :].hdrop[1]
	height1D, _ = getRingCurve(data, t)
	hNorm = height1D ./ h_drop
	L = length(height1D)
	spectrumH= fftshift(fft(hNorm))
	shifted_k = fftshift(fftfreq(L)*L)
	k_pi = shifted_k .* 2π/L
	
	p = plot(k_pi, log.(abs.(spectrumH .* spectrumH)) .+ 1, 
			# aspect_ratio=1, 
			xlims=(0,π), 
			xlabel = "q",
			ylabel = "log(S(q))"
			# xlabel="x\\(\\phi\\)", xticks = ([0:π/2:2*π;], ["0", "\\pi/2", "\\pi", "3\\pi/2", "2\\pi"])
			#ylims=(-π, π),
			#clim=(0.1, 1000) # Limits for heatmap
		)
	
	if output
		return height1D, spectrumH
	end
	p
end

# Hydrodynamic scales
"""
	compute_Oh(R; μ=1/6, ρ=1, γ=0.01)

Computes the Ohnesorge number based on the minor radius.

# Definition

``Oh = \\frac{\\mu}{\\sqrt{\\rho r\\gamma}}``

# Example
```julia-repl
julia> R = 180; μ = 1/6; γ = 0.01;

julia> compute_Oh(R, μ=μ, γ=γ)
```
"""
function compute_Oh(R; μ=1/6, ρ=1, γ=0.01)
	return μ/sqrt(ρ*R*γ)
end

"""
	t₀(r, μ, γ)

Computes the visco-capillary time scale base on minor radius.

# Definition

``t_0 = \\frac{\\mu r}{\\gamma}``

# Example
```julia-repl
julia> r = 40; μ = 1/6; γ = 0.01;

julia> t₀(r, μ=μ, γ=γ)
```
"""
function t₀(r; μ=1/6, γ=0.01)
	return (r * μ)/γ
end

"""
	tic(r, ρ, γ)

Computes the inertia capillary time scale base on minor radius, density and surface tension

# Definition

``t_{ic} = \\sqrt{\\frac{\\rho r^3}{\\gamma}}``

# Example
```julia-repl
julia> r = 40; γ = 0.01;

julia> tic(r, γ=γ)
```
"""
function tic(r; ρ=1, γ=0.01)
	return sqrt((ρ * r^3)/γ)
end

"""
	tau_rim(h0, θ; μ=1/6, γ=0.01)

Capillary time scale of rim rectraction

# Definition

``\\tau_{r} = \\frac{9(R_0 - R_f))\\mu}{\\gamma\\theta^3}``

# Example
```julia-repl
julia> h0 = 10; θ=π/9; μ = 1/6; γ = 0.01;

julia> tau_rim(h0, θ, μ=μ, γ=γ)
```

# Literature
- [A. Edwards et al.](https://www.science.org/doi/full/10.1126/sciadv.1600183)
"""
function tau_rim(h0, θ; μ=1/6, γ=0.01)
	return 9h0*μ/(γ*(θ*π)^3)
end

"""
	t_s(h, θ; μ=1/6, γ=0.01, hs=0.1, M=0.5)

Capillary time scale of thin rivulet (Kondic)

# Definition

``\\tau_{r} = \\frac{3\\mu h_{\\star}}{\\gamma}\\left[\\frac{M}{1-\\cos(\\theta)}\\right]^2``

# Example
```julia-repl
julia> h = 10; θ=π/9; μ = 1/6; γ = 0.01; M=1/2;

julia> t_s(h, θ, μ=μ, γ=γ, M=M)
```

# Literature
- [J. A. Diez, A.G. González, L. Kondic](https://pubs.aip.org/pof/article/21/8/082105/256873/On-the-breakup-of-fluid-rivulets)
"""
function t_s(h, θ; μ=1/6, γ=0.01, hs=0.1, M=0.5)
	# return (3μ*hs/γ)*(M/(1 - cospi(θ)))^2
	return (3μ*h/γ)*(M/(1 - cospi(θ)))^2
end

"""
	t_00(h, θ; μ=1/6, γ=0.01, hs=0.1)

Capillary time scale for film dewetting

# Definition

``q_0^2 = h_{\\star}^{-2}(1-\\cos(\\theta))\\left(2\\frac{h_{\\star}^2}{h_0^3} - 3\\frac{h_{\\star}^3}{h_0^4}\\right)``

``t_{00} = \\frac{3\\mu}{\\gamma h_{\\star}^3 q_0^4}``

# Example
```julia-repl
julia> h = 10; θ=π/9; μ = 1/6; γ = 0.01; hs=0.1;

julia> t_00(h, θ, μ=μ, γ=γ, hs=hs)
```

# Literature
- [K. Mecke, M. Rauscher](https://iopscience.iop.org/article/10.1088/0953-8984/17/45/042/pdf)
"""
function t_00(h, θ; μ=1/6, γ=0.01, hs=0.1)
	q₀² = (1 - cospi(θ))/hs^2 * (2*hs^2/h^3 - 3*hs^3/h^4)
	
	return 3μ/(γ*h^3*q₀²*q₀²)
end

# The link to the actual simulations
data = [
	(80,  40, 0.0,    2023, 11, 1, 19, 6,  20), 	# 1
	(120, 40, 0.0,    2023, 11, 2, 6,  43, 20), 	# 2
	(120, 80, 0.0,    2023, 11, 2, 9,  37, 20), 	# 3
	(150, 20, 0.0,    2023, 11, 2, 13, 0,  20), 	# 4
	(150, 40, 0.0,    2023, 11, 2, 14, 26, 20), 	# 5
	(150, 80, 0.0,    2023, 11, 2, 15, 54, 20), 	# 6
	(180, 20, 0.0,    2023, 11, 2, 17, 23, 20), 	# 7 little smaller little larger
	(180, 40, 0.0,    2023, 11, 2, 18, 49, 20), 	# 8
	(180, 80, 0.0,    2023, 11, 2, 20, 16, 20), 	# 9
	(200, 20, 0.0,    2023, 11, 2, 21, 43, 20), 	# 10
	(200, 40, 0.0,    2023, 11, 2, 23, 8,  20), 	# 11
	(200, 80, 0.0,    2023, 11, 3, 0,  34, 20), 	# 12
	(150, 20, 1.0e-6, 2023, 11, 3, 1,  59, 20), 	# 13
	(150, 40, 1.0e-6, 2023, 11, 3, 3,  25, 20), 	# 14 
	(150, 80, 1.0e-6, 2023, 11, 3, 4,  50, 20), 	# 15
	(180, 20, 1.0e-6, 2023, 11, 3, 6,  16, 20), 	# 16
	(180, 40, 1.0e-6, 2023, 11, 3, 7,  41, 20), 	# 17
	(180, 80, 1.0e-6, 2023, 11, 3, 9,  7,  20), 	# 18
	(200, 20, 1.0e-6, 2023, 11, 3, 10, 32, 20), 	# 19
	(200, 40, 1.0e-6, 2023, 11, 3, 15, 17, 20), 	# 20 
	(200, 80, 1.0e-6, 2023, 11, 3, 16, 44, 20), 	# 21
	(150, 20, 0.0,    2023, 11, 3, 17, 49, 40), 	# 22
	(150, 40, 0.0,    2023, 11, 3, 19, 13, 40), 	# 23
	(150, 80, 0.0,    2023, 11, 3, 20, 39, 40), 	# 24
	(180, 20, 0.0,    2023, 11, 3, 22, 6,  40), 	# 25
	(180, 40, 0.0,    2023, 11, 3, 23, 34, 40), 	# 26
	(180, 80, 0.0,    2023, 11, 4, 1,  1,  40), 	# 27
	(200, 20, 0.0,    2023, 11, 4, 2,  27, 40), 	# 28
	(200, 40, 0.0,    2023, 11, 4, 3,  54, 40), 	# 29
	(200, 80, 0.0,    2023, 11, 4, 5,  20, 40), 	# 30
	(150, 20, 1.0e-6, 2023, 11, 4, 6,  46, 40), 	# 31  
	(150, 40, 1.0e-6, 2023, 11, 4, 8,  11, 40), 	# 32
	(150, 80, 1.0e-6, 2023, 11, 4, 9,  37, 40), 	# 33
	(180, 20, 1.0e-6, 2023, 11, 4, 11, 3,  40), 	# 34
	(180, 40, 1.0e-6, 2023, 11, 4, 12, 30, 40), 	# 35
	(180, 80, 1.0e-6, 2023, 11, 4, 13, 58, 40), 	# 36
	(200, 20, 1.0e-6, 2023, 11, 4, 15, 26, 40), 	# 37
	(200, 40, 1.0e-6, 2023, 11, 4, 16, 54, 40), 	# 38
	(200, 80, 1.0e-6, 2023, 11, 4, 18, 21, 40), 	# 39
	(150, 20, 0.0,    2023, 11, 4, 19, 46, 30), 	# 40
	(150, 40, 0.0,    2023, 11, 4, 21, 11, 30), 	# 41
	(150, 80, 0.0,    2023, 11, 4, 22, 36, 30), 	# 42
	(180, 20, 0.0,    2023, 11, 5, 0,  2,  30), 	# 43
	(180, 40, 0.0,    2023, 11, 5, 1,  28, 30), 	# 44
	(180, 80, 0.0,    2023, 11, 5, 2,  56, 30), 	# 45
	(200, 20, 0.0,    2023, 11, 5, 4,  24, 30), 	# 46
	(200, 40, 0.0,    2023, 11, 5, 5,  51, 30), 	# 47
	(200, 80, 0.0,    2023, 11, 5, 7,  19, 30), 	# 48
	(150, 20, 1.0e-6, 2023, 11, 5, 8,  47, 30), 	# 49
	(150, 40, 1.0e-6, 2023, 11, 5, 10, 14, 30), 	# 50
	(150, 80, 1.0e-6, 2023, 11, 5, 11, 42, 30), 	# 51
	(180, 20, 1.0e-6, 2023, 11, 5, 13, 10, 30), 	# 52
	(180, 40, 1.0e-6, 2023, 11, 5, 14, 38, 30), 	# 53
	(180, 80, 1.0e-6, 2023, 11, 5, 16, 7,  30), 	# 54
	(200, 20, 1.0e-6, 2023, 11, 5, 17, 35, 30), 	# 55
	(200, 40, 1.0e-6, 2023, 11, 5, 19, 3,  30), 	# 56
	(200, 80, 1.0e-6, 2023, 11, 5, 20, 30, 30), 	# 57
	(150, 20, 0.0,    2023, 11, 5, 21, 58, 10), 	# 58
	(150, 40, 0.0,    2023, 11, 5, 23, 26, 10), 	# 59
	(150, 80, 0.0,    2023, 11, 6, 0,  54, 10), 	# 60
	(180, 20, 0.0,    2023, 11, 6, 2,  22, 10), 	# 61
	(180, 40, 0.0,    2023, 11, 6, 3,  50, 10), 	# 62
	(180, 80, 0.0,    2023, 11, 6, 5,  17, 10), 	# 63
	(200, 20, 0.0,    2023, 11, 6, 6,  45, 10), 	# 64
	(200, 40, 0.0,    2023, 11, 6, 8,  13, 10), 	# 65
	(200, 80, 0.0,    2023, 11, 6, 9,  40, 10), 	# 66
	(150, 20, 1.0e-6, 2023, 11, 6, 11, 7,  10), 	# 67
	(150, 40, 1.0e-6, 2023, 11, 6, 12, 32, 10), 	# 68
	(150, 80, 1.0e-6, 2023, 11, 6, 13, 58, 10), 	# 69
	(180, 20, 1.0e-6, 2023, 11, 6, 15, 24, 10), 	# 70
	(180, 40, 1.0e-6, 2023, 11, 6, 16, 51, 10), 	# 71
	(180, 80, 1.0e-6, 2023, 11, 6, 18, 17, 10), 	# 72
	(200, 20, 1.0e-6, 2023, 11, 6, 19, 42, 10), 	# 73
	(200, 40, 1.0e-6, 2023, 11, 6, 21, 7,  10), 	# 74
	(200, 80, 1.0e-6, 2023, 11, 6, 22, 32, 10), 	# 75
]

data_arrested = [
	(160, 20, 0.0,    2023, 11,24, 13, 24, 40), 	#
	(160, 30, 0.0,    2023, 11,24, 14, 47, 40), 	#
	(160, 40, 0.0,    2023, 11,24, 16,  9, 40), 	#
	(180, 20, 0.0,    2023, 11,24, 17, 32, 40), 	#
	(180, 30, 0.0,    2023, 11,24, 18, 55, 40), 	#
	(180, 40, 0.0,    2023, 11,24, 20, 18, 40), 	#
	(200, 20, 0.0,    2023, 11,24, 21, 41, 40), 	#
	(200, 30, 0.0,    2023, 11,24, 23,  5, 40), 	#
	(200, 40, 0.0,    2023, 11,25,  0, 28, 40), 	#
	(160, 20, 1.0e-6, 2023, 11,25,  1, 51, 40), 	#
	(160, 30, 1.0e-6, 2023, 11,25,  3, 14, 40), 	#
	(160, 40, 1.0e-6, 2023, 11,25,  4, 37, 40), 	#
	(180, 20, 1.0e-6, 2023, 11,25,  6,  0, 40), 	#
	(180, 30, 1.0e-6, 2023, 11,25,  7, 23, 40), 	#
	(180, 40, 1.0e-6, 2023, 11,25,  8, 47, 40), 	#
	(200, 20, 1.0e-6, 2023, 11,25, 10, 10, 40), 	#
	(200, 30, 1.0e-6, 2023, 11,25, 11, 33, 40), 	#
	(200, 40, 1.0e-6, 2023, 11,25, 12, 56, 40), 	#
	(160, 20, 0.0,    2023, 11,25, 14, 19, 30), 	#
	(160, 30, 0.0,    2023, 11,25, 15, 42, 30), 	#
	(160, 40, 0.0,    2023, 11,25, 17,  4, 30), 	#
	(160, 60, 0.0,    2023, 12,23,  0, 32, 30), 	#
	(160, 80, 0.0,    2023, 12,23,  1, 53, 30), 	#
	(160, 100,0.0,    2023, 12,23,  3, 13, 30), 	#
	(180, 20, 0.0,    2023, 11,25, 18, 27, 30), 	#
	(180, 30, 0.0,    2023, 11,25, 19, 48, 30), 	#
	(180, 40, 0.0,    2023, 11,25, 21,  7, 30), 	#
	(180, 60, 0.0,    2023, 12,23,  4, 34, 30), 	#
	(180, 80, 0.0,    2023, 12,23,  5, 55, 30), 	#
	(180, 100,0.0,    2023, 12,23,  7, 15, 30), 	#
	(200, 20, 0.0,    2023, 11,25, 22, 23, 30), 	#
	(200, 30, 0.0,    2023, 11,25, 23, 40, 30), 	#
	(200, 40, 0.0,    2023, 11,26,  1,  2, 30), 	#
	(200, 60, 0.0,    2023, 12,23,  8, 36, 30), 	#
	(200, 80, 0.0,    2023, 12,23,  9, 57, 30), 	#
	(200, 100,0.0,    2023, 12,23, 11, 19, 30), 	#
	(160, 20, 1.0e-6, 2023, 11,26,  2, 24, 30), 	#
	(160, 30, 1.0e-6, 2023, 11,26,  3, 45, 30), 	#
	(160, 40, 1.0e-6, 2023, 11,26,  5,  7, 30), 	#
	(180, 20, 1.0e-6, 2023, 11,26,  6, 28, 30), 	#
	(180, 30, 1.0e-6, 2023, 11,26,  7, 50, 30), 	#
	(180, 40, 1.0e-6, 2023, 11,26,  9, 12, 30), 	#
	(200, 20, 1.0e-6, 2023, 11,26, 10, 33, 30), 	#
	(200, 30, 1.0e-6, 2023, 11,26, 11, 55, 30), 	#
	(200, 40, 1.0e-6, 2023, 11,26, 13, 16, 30), 	#
	(160, 20, 0.0,    2023, 11,27, 14, 51, 20), 	#
	(160, 30, 0.0,    2023, 11,27, 16, 12, 20), 	#
	(160, 40, 0.0,    2023, 11,27, 17, 33, 20), 	#
	(160, 60, 0.0,    2023, 12,22, 12, 25, 20), 	#
	(160, 80, 0.0,    2023, 12,22, 13, 44, 20), 	#
	(160, 100,0.0,    2023, 12,22, 15,  5, 20), 	#
	(180, 20, 0.0,    2023, 11,27, 18, 54, 20), 	#
	(180, 30, 0.0,    2023, 11,27, 20, 15, 20), 	#
	(180, 40, 0.0,    2023, 11,27, 21, 36, 20), 	#
	(180, 60, 0.0,    2023, 12,22, 16, 25, 20), 	#
	(180, 80, 0.0,    2023, 12,22, 17, 47, 20), 	#
	(180,100, 0.0,    2023, 12,22, 19,  8, 20), 	#
	(200, 20, 0.0,    2023, 11,27, 22, 57, 20), 	#
	(200, 30, 0.0,    2023, 11,28,  0, 18, 20), 	#
	(200, 40, 0.0,    2023, 11,28,  1, 39, 20), 	#
	(200, 60, 0.0,    2023, 12,22, 20, 29, 20), 	#
	(200, 80, 0.0,    2023, 12,22, 21, 50, 20), 	#
	(200,100, 0.0,    2023, 12,22, 23, 11, 20), 	#
	(160, 20, 1.0e-6, 2023, 11,28,  3,  0, 20), 	#
	(160, 30, 1.0e-6, 2023, 11,28,  4, 22, 20), 	#
	(160, 40, 1.0e-6, 2023, 11,28,  5, 42, 20), 	#
	(180, 20, 1.0e-6, 2023, 11,28,  7,  3, 20), 	#
	(180, 30, 1.0e-6, 2023, 11,28,  8, 24, 20), 	#
	(180, 40, 1.0e-6, 2023, 11,28,  9, 45, 20), 	#
	(200, 20, 1.0e-6, 2023, 11,28, 11,  6, 20), 	#
	(200, 30, 1.0e-6, 2023, 11,28, 12, 27, 20), 	#
	(200, 40, 1.0e-6, 2023, 11,28, 13, 47, 20), 	#
	(160, 20, 0.0,    2023, 11,26, 14, 38, 10), 	#
	(160, 30, 0.0,    2023, 11,26, 15, 59, 10), 	#
	(160, 40, 0.0,    2023, 11,26, 17, 21, 10), 	#
	(160, 60, 0.0,    2023, 12,23, 12, 40, 10), 	#
	(160, 80, 0.0,    2023, 12,23, 14,  1, 10), 	#
	(160, 100,0.0,    2023, 12,23, 15, 22, 10), 	#
	(180, 20, 0.0,    2023, 11,26, 18, 42, 10), 	#
	(180, 30, 0.0,    2023, 11,26, 20,  3, 10), 	#
	(180, 40, 0.0,    2023, 11,26, 21, 24, 10), 	#
	(180, 60, 0.0,    2023, 12,23, 16, 42, 10), 	#
	(180, 80, 0.0,    2023, 12,23, 18,  2, 10), 	#
	(180, 100,0.0,    2023, 12,23, 19, 21, 10), 	#
	(200, 20, 0.0,    2023, 11,26, 22, 45, 10), 	#
	(200, 30, 0.0,    2023, 11,27,  0,  5, 10), 	#
	(200, 40, 0.0,    2023, 11,27,  1, 26, 10), 	#
	(200, 60, 0.0,    2023, 12,23, 20, 35, 10), 	#
	(200, 80, 0.0,    2023, 12,23, 21, 51, 10), 	#
	(200, 100,0.0,    2023, 12,23, 23, 10, 10), 	#
	(160, 20, 1.0e-6, 2023, 11,27,  2, 47, 10), 	#
	(160, 30, 1.0e-6, 2023, 11,27,  4,  8, 10), 	#
	(160, 40, 1.0e-6, 2023, 11,27,  5, 25, 10), 	#
	(180, 20, 1.0e-6, 2023, 11,27,  6, 39, 10), 	#
	(180, 30, 1.0e-6, 2023, 11,27,  7, 56, 10), 	#
	(180, 40, 1.0e-6, 2023, 11,27,  9, 17, 10), 	#
	(200, 20, 1.0e-6, 2023, 11,27, 10, 38, 10), 	#
	(200, 30, 1.0e-6, 2023, 11,27, 11, 59, 10), 	#
	(200, 40, 1.0e-6, 2023, 11,27, 13, 20, 10), 	#
]

data_gamma05 = [
	(160, 20, 0.0, 2023, 12, 7, 17,  8, 20), 	#
	(160, 30, 0.0, 2023, 12, 7, 18, 22, 20), 	#
	(160, 40, 0.0, 2023, 12, 7, 19, 38, 20), 	#
	(180, 20, 0.0, 2023, 12, 7, 20, 58, 20), 	#
	(180, 30, 0.0, 2023, 12, 7, 22, 18, 20), 	#
	(180, 40, 0.0, 2023, 12, 7, 23, 38, 20), 	#
	(200, 20, 0.0, 2023, 12, 8,  0, 58, 20), 	#
	(200, 30, 0.0, 2023, 12, 8,  2, 18, 20), 	#
	(200, 40, 0.0, 2023, 12, 8,  3, 39, 20), 	#
	(160, 20, 0.0, 2023, 12, 8, 17,  0, 40), 	#
	(160, 30, 0.0, 2023, 12, 8, 18, 20, 40), 	#
	(160, 40, 0.0, 2023, 12, 8, 19, 40, 40), 	#
	(180, 20, 0.0, 2023, 12, 8, 21,  0, 40), 	#
	(180, 30, 0.0, 2023, 12, 8, 22, 20, 40), 	#
	(180, 40, 0.0, 2023, 12, 8, 23, 39, 40), 	#
	(200, 20, 0.0, 2023, 12, 9,  0, 55, 40), 	#
	(200, 30, 0.0, 2023, 12, 9,  2, 10, 40), 	#
	(200, 40, 0.0, 2023, 12, 9,  3, 26, 40), 	#
	(160, 20, 0.0, 2023, 12, 9, 16, 46, 30), 	#
	(160, 30, 0.0, 2023, 12, 9, 18,  6, 30), 	#
	(160, 40, 0.0, 2023, 12, 9, 19, 26, 30), 	#
	(180, 20, 0.0, 2023, 12, 9, 20, 46, 30), 	#
	(180, 30, 0.0, 2023, 12, 9, 22,  6, 30), 	#
	(180, 40, 0.0, 2023, 12, 9, 23, 26, 30), 	#
	(200, 20, 0.0, 2023, 12,10,  0, 46, 30), 	#
	(200, 30, 0.0, 2023, 12,10,  2,  6, 30), 	#
	(200, 40, 0.0, 2023, 12,10,  3, 26, 30), 	#
	(160, 20, 0.0, 2023, 12,10, 16, 31, 10), 	#
	(160, 30, 0.0, 2023, 12,10, 17, 52, 10), 	#
	(160, 40, 0.0, 2023, 12,10, 19, 12, 10), 	#
	(180, 20, 0.0, 2023, 12,10, 20, 32, 10), 	#
	(180, 30, 0.0, 2023, 12,10, 21, 53, 10), 	#
	(180, 40, 0.0, 2023, 12,10, 23, 13, 10), 	#
	(200, 20, 0.0, 2023, 12,11,  0, 34, 10), 	#
	(200, 30, 0.0, 2023, 12,11,  1, 54, 10), 	#
	(200, 40, 0.0, 2023, 12,11,  3, 14, 10), 	#
	]

data_gamma20 = [
	(160, 20, 0.0, 2023, 12, 8,  4, 59, 20), 	#
	(160, 30, 0.0, 2023, 12, 8,  6, 19, 20), 	#
	(160, 40, 0.0, 2023, 12, 8,  7, 39, 20), 	#
	(180, 20, 0.0, 2023, 12, 8,  8, 59, 20), 	#
	(180, 30, 0.0, 2023, 12, 8, 10, 19, 20), 	#
	(180, 40, 0.0, 2023, 12, 8, 11, 39, 20), 	#
	(200, 20, 0.0, 2023, 12, 8, 12, 59, 20), 	#
	(200, 30, 0.0, 2023, 12, 8, 14, 20, 20), 	#
	(200, 40, 0.0, 2023, 12, 8, 15, 40, 20), 	#
	(160, 20, 0.0, 2023, 12, 9,  4, 46, 40), 	#
	(160, 30, 0.0, 2023, 12, 9,  6,  6, 40), 	#
	(160, 40, 0.0, 2023, 12, 9,  7, 26, 40), 	#
	(180, 20, 0.0, 2023, 12, 9,  8, 46, 40), 	#
	(180, 30, 0.0, 2023, 12, 9, 10,  6, 40), 	#
	(180, 40, 0.0, 2023, 12, 9, 11, 26, 40), 	#
	(200, 20, 0.0, 2023, 12, 9, 12, 46, 40), 	#
	(200, 30, 0.0, 2023, 12, 9, 14,  6, 40), 	#
	(200, 40, 0.0, 2023, 12, 9, 15, 26, 40), 	#
	(160, 20, 0.0, 2023, 12,10,  4, 46, 30), 	#
	(160, 30, 0.0, 2023, 12,10,  6,  6, 30), 	#
	(160, 40, 0.0, 2023, 12,10,  7, 26, 30), 	#
	(180, 20, 0.0, 2023, 12,10,  8, 42, 30), 	#
	(180, 30, 0.0, 2023, 12,10,  9, 55, 30), 	#
	(180, 40, 0.0, 2023, 12,10, 11, 11, 30), 	#
	(200, 20, 0.0, 2023, 12,10, 12, 31, 30), 	#
	(200, 30, 0.0, 2023, 12,10, 13, 51, 30), 	#
	(200, 40, 0.0, 2023, 12,10, 15, 11, 30), 	#
	(160, 20, 0.0, 2023, 12,11,  4, 34, 10), 	#
	(160, 30, 0.0, 2023, 12,11,  5, 54, 10), 	#
	(160, 40, 0.0, 2023, 12,11,  7, 14, 10), 	#
	(180, 20, 0.0, 2023, 12,11,  8, 34, 10), 	#
	(180, 30, 0.0, 2023, 12,11,  9, 55, 10), 	#
	(180, 40, 0.0, 2023, 12,11, 11, 15, 10), 	#
	(200, 20, 0.0, 2023, 12,11, 12, 35, 10), 	#
	(200, 30, 0.0, 2023, 12,11, 13, 55, 10), 	#
	(200, 40, 0.0, 2023, 12,11, 15, 14, 10), 	#
]

data_slip= [
	# R, rr, kbt, year, month, day, hour, min, theta, slip
	(180, 20, 0.0, 2024, 1, 25, 11, 40, 20, 5), 	# 1
	(180, 40, 0.0, 2024, 1, 25, 12, 57, 20, 5), 	# 2
	(180, 20, 0.0, 2024, 1, 25, 14, 16, 20, 25), 	# 3
	(180, 40, 0.0, 2024, 1, 25, 15, 36, 20, 25), 	# 4
	(180, 20, 0.0, 2024, 1, 25, 16, 55, 30, 5), 	# 5
	(180, 40, 0.0, 2024, 1, 25, 18, 14, 30, 5), 	# 6
	(180, 20, 0.0, 2024, 1, 25, 19, 34, 30, 25), 	# 7
	(180, 40, 0.0, 2024, 1, 25, 20, 53, 30, 25), 	# 8 
	(180, 20, 0.0, 2024, 1, 25, 22, 13, 40, 5), 	# 9
	(180, 40, 0.0, 2024, 1, 25, 23, 32, 40, 5), 	# 10
	(180, 20, 0.0, 2024, 1, 26,  0, 51, 40, 25), 	# 11
	(180, 40, 0.0, 2024, 1, 26,  2, 11, 40, 25), 	# 12
]

# End module
end