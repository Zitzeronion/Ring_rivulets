### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ f268582b-0756-41cf-910d-7a57b698451d
using FileIO, PlutoUI, Plots, DataFrames, CSV, JLD2, Images, ImageSegmentation, Random, FFTW, LaTeXStrings, CategoricalArrays, StatsPlots

# ╔═╡ f075f19a-81b6-47b7-9104-57d2e51e7241
begin
	# Functions are stored in the local module!
	include("RivuletTools.jl")
	import .RivuletTools
end

# ╔═╡ 569bbd2c-7fae-4e26-afe5-3f4d06f7d505
TableOfContents()

# ╔═╡ 94b9bdb0-73ee-11ee-10e9-e93688ea4523
md"# Rivulet analysis

In the following notebook we study the dynamics of a ring shaped rivulet on a solid substrate.
From thin film literature we know that rivulets can have all kinds of instabilities.
Instabilities that lead to breakup and droplet fragmentation, see for example [Diez et al.](https://pubs.aip.org/pof/article/21/8/082105/256873/On-the-breakup-of-fluid-rivulets)
In the above paper the discussion on curved rivulets is quite limited.

[Mehrabian and Feng](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/capillary-breakup-of-a-liquid-torus/9B0F535D234DB9221EAB5D9E0797AA7D) invested significant time in studing a curved rivulet. 
Their results showed that a ring shaped rivulet does have in fact an overall flow that towards the center. 
Thus the rivulet is retracting to a droplet."

# ╔═╡ 1b26468c-b4f7-4252-b891-4f95bc04c869
@recipe function f(::Type{Val{:samplemarkers}}, x, y, z; step = 10)
    n = length(y)
    sx, sy = x[1:step:n], y[1:step:n]
    # add an empty series with the correct type for legend markers
    @series begin
        seriestype := :path
        markershape --> :auto
        x := [Inf]
        y := [Inf]
    end
    # add a series for the line
    @series begin
        primary := false # no legend entry
        markershape := :none # ensure no markers
        seriestype := :path
        seriescolor := get(plotattributes, :seriescolor, :auto)
        x := x
        y := y
    end
    # return  a series for the sampled markers
    primary := false
    seriestype := :scatter
    markershape --> :auto
    x := sx
    y := sy
end

# ╔═╡ 6d3c1725-75fa-412e-9b30-8f8df4e7874b
md"# Functions

At this point the data has been created and is ready to be analyzed.
But what is it that we actually look for?
At this stage we don't know where the paper will eventually lead to, but we can ask the data a series of questions.

Below we define a few functions that will help us extracting relevant findings."

# ╔═╡ 2df8c833-7ca7-4d7a-ade5-0df083a013a1
begin
	h_some = 
	RivuletTools.plot_slice(RivuletTools.read_data(R=180, r=40, kbT=0.0, month=11, day=5, hour=1, minute=28, θ=30, arrested=false), t=25000)
	
end

# ╔═╡ 04e344a3-3d5b-449e-9222-481df24015c7
# (160, 20, 0.0,    11,27, 14, 51, 20
begin
	hja = RivuletTools.read_data(R=160, r=20, kbT=0.0, month=11, day=27, hour=14, minute=51, θ=20, arrested=true)
	RivuletTools.measure_cluster(hja, t=2500000)
end

# ╔═╡ 974c334e-38fb-436e-842b-bb016854d136
RivuletTools.heatmap_data(hja, t=875000)

# ╔═╡ 5aad6dcc-8400-4a7a-a2bc-69105b32e09f
md" ### Time scales

All simulations are based on lattice Boltzmann units (l.b.u.) which is an arbitrary unit. 
To put this results into perspective we need to indentify relevant time and length scales to non-dimensionalize our units.
While this seem straightforward for the length scale, the minor radius of the rivulet, for the time scale it is more challenging.

The following few functions introduce possible time scales
- The visco-capillary time scale $t_0$
``t_0 = \frac{\mu r}{\gamma}``
- The inertia-capillary time scale $t_{ic}$
`` t_{ic} = \sqrt{\frac{\rho r^3}{\gamma}} ``
- Rim retraction time scale $\tau_{rim}$
``\tau_{rim} = \frac{9(R_0 - R_f)\mu}{\gamma\theta^3}``
- Capillary time scale for microscopic rivulets $t_s$
``t_s = \frac{3\mu h_{\ast}}{\gamma}\left[\frac{M}{1-\cos\theta}\right]^2``
- Spinodial dewetting time $\tau$
``\tau = \frac{3\mu}{\gamma h_{\ast}^3q_0^4} \\ q_0 = \frac{1}{2\gamma}\partial_h \Pi(h)|_{h=h_0}`` 

"

# ╔═╡ 351d01de-7225-40e0-b650-0ef40b1a1cf7
md" ### Initial condition

Computational fluid dynamics (CFD) relys heavily on the choice of initial and boundary conditions. 
For the boundary conditions we put our trust in the good old biperiodic lattice.
The initial condition is a velocity free donut as shown in the image below.

All our simulations are started with the function `torus()` described below.
The relevant parameters here are the larger radius `R₂` and the minor radius `r₁` as well as the contact angle `θ`.
We solve

`` (\sqrt{x^2 + y^2} - R)^2 + z^2 = r^2 ``

for $z$ instead of $r$ such that we have a representation for the thickness.
In a second step we cut from this structure, which is a half a donut, to make it fit to the chosen contact angle `θ`. 
"

# ╔═╡ 7cf4ce88-33de-472c-99c6-5b3ae258f3d1
md" With its visual representation shown below."

# ╔═╡ 0361d281-4a64-4792-812f-7eb9d268d2ae
# ╠═╡ disabled = true
#=╠═╡
a,v = compute_droplet(RivuletTools.torus(512, 512, 20, 180, 1/9, (256,256)), 1/9)
  ╠═╡ =#

# ╔═╡ 60ce933e-4335-4190-a7b0-5c86d0326a35
md"### Profile initial conditions

We assume that the simulations will be very sensitive to the ratio of the two radii
```math
\beta = \frac{r}{R}.
```

Another point of view can be that the radii are not the relevant quantity but the curvatures 
```math
	\kappa_0 = \frac{1}{r}, \\

	\kappa_1 = \frac{1}{R},
```
where the two $\kappa$'s correspond to the two radii.
One problem could in fact be that all our simulations have the same $\kappa_0$.
Therefore let us have a look at the initial conditions:
"

# ╔═╡ a9601fe8-6321-4ad7-a950-c7354290aed6
begin
	r = 40
	θ = π/9
	h = RivuletTools.torus(512, 512, 40, 180, 1/9, (256,256))
	w = 31 - 3 
	thetaC = 2*asin(13.68/(2*r))
	Ar = θ*r^2- 0.5*r*w*cos(θ)
	Ar2 = (r^2/2)*(thetaC - sin(thetaC))
	Ar3 = (2/3)*w*maximum(h[256, 61:91])
	As = sum(h[256, 61:91])
	println("Computed: $(Ar) Simulated: $(As)")
	println("Computed 2: $(Ar2) Simulated: $(As)")
	println("Computed 3: $(Ar3) Simulated: $(As)")
	println(sqrt(maximum(h[256, 61:91])*(2*r-maximum(h[256, 61:91]))))
	println(thetaC, " ",  θ, " ", 2*r*sin(θ))
	plot(h[256, 61:91], minorticks=true, aspect_ratio=1)
	
end

# ╔═╡ 1ebb5c9b-df33-4c48-8240-bbff2a06520c
begin
	h2 = RivuletTools.torus(512, 512, 20, 180, 2/9, (256,256))
	slice = h2[256, :]
	sliceC = slice[slice .> 0.0501]
	A = sum(sliceC)
end

# ╔═╡ 70da13b0-6111-4d5d-a6f5-49fcc0499738
md" ### Data analysis 

The data analysis is usually the most effort.
Here are some functions that should help with data analysis."

# ╔═╡ 4fb1d7ad-47f2-4adf-a2ba-0ecc0fc8eeb0
md" # The Data

Below we create three dataframes that hosts a load of information.
Arguably too much information to have a clear picture what we want to say.
However, with the steps outlined below we hope to clear up some of the confusion.

The three dataframes host:
- Initial geometrical data: `initial_data`
- Simulation data: `data`
- Simulation data on patterned substrates: `data_arrested`
- Simulation data with different surface tension: `data_gammaX`
- Simulation data with different slip length: `data_slip`

### Initial conditions
"

# ╔═╡ 41aee571-9016-4759-859a-c99eb143a410
begin
	initial_data = RivuletTools.t0_data()
	CSV.write("../data/initial_conditions.csv", initial_data)
	initial_data
end

# ╔═╡ 13ce2bea-889f-4727-a126-71a5006a86ab
md"A single simulation creates one data file, which is about 500mb in size. 
The data file contains the temporal evolution of the height field. 
Every 25000Δt we make a snapshot of the system until we reach the end of the time loop at 2500000Δt.
The file that contains this series is uniquely labeled with date and parameters for the initial condition (`RivuletTools.torus()`).
Every entry in the `data` and `data_arrested` array points towards one such files.

### Unpatterned Substrate
"

# ╔═╡ 0efbcab7-4a58-400a-84af-f1a6703c7ec9
data = RivuletTools.data

# ╔═╡ 8b2c77d3-f743-4840-a9d9-9308e05be28d
begin
	set = 8
	data0 = RivuletTools.read_data(R=data[set][1], r=data[set][2], kbT=data[set][3], year=data[set][4], month=data[set][5], day=data[set][6], hour=data[set][7], minute=data[set][8], θ=data[set][9])
	# println(typeof(dataH), " ", typeof(dataH) == Dict{String, Any})
	figureE = RivuletTools.heatmap_data(data0, t=25000)
	println("R: $(data[set][1]), rr: $(data[set][2]), theta: $(data[set][9]), kbt: $(data[set][3])")
	figureE
end

# ╔═╡ a1ee76ab-bf27-466e-8324-01ecde09931c
md"
### Patterned Substrate
"

# ╔═╡ 74590e82-5e2b-4a88-afde-3b2992d9b001
data_arrested = RivuletTools.data_arrested

# ╔═╡ dc5ab038-569e-4603-95af-0549c6e4ee76
md"
### Surface tension variation

For even more data we varied the surface tension γ and reduced it by a factor of 2 as well as increased it by a factor of 2.

- data_gamma05 half surface tension
- data_gamma20 double surface tension
"

# ╔═╡ 90bd4975-315d-4a55-9af4-5b40ceda4eb3
data_gamma05 = RivuletTools.data_gamma05

# ╔═╡ 93e8f4ee-7558-4178-8f06-96a422528c48
data_gamma20 = RivuletTools.data_gamma20

# ╔═╡ 05715ba9-fdd3-43c8-b6fc-ee32d225cdf0
md"
### Slip variation

Very recently we also added another set of simulations.
This time we varied the slip length, which is an effective parameter for the velocity boundary condition.
For all other simulations we used a slip length $\delta = 1$, to check if there is an influence at all we now have:
- δ = 0.5
- δ = 2

thus a similar approach as in the surface tension case.
"

# ╔═╡ 24fde296-5a6f-4a92-bf16-855df4c99227
data_slip= RivuletTools.data_slip

# ╔═╡ 42094521-c209-4c82-8226-1a0b9b1c0d85
data_grad = RivuletTools.data_gradient

# ╔═╡ b3be394c-5997-4494-ad40-ced2f10fd364
# RivuletTools.renderGifs()

# ╔═╡ d56dc1d3-992a-4be9-87bd-56dae4c41d9c
RivuletTools.do_gif(RivuletTools.read_data(R=180, r=80, kbT=0.0, nm=(3,2), θ=20, year=2024, month=3, day=6, hour=18, minute=32, arrested=false, gamma="", slip=0, gradient=(true, 30, 20)), "grad_30_20_R_180_r_80_mT_5M_kbt_off", timeMax=5000000)

# ╔═╡ 0f204a06-71b2-438a-bb49-4af8ebda0001
md" # Results

Here comes the actual research part.
So far we have set up quite a lot of tools to generate and analyse the data.
But we haven't answered the question of what we expect to find or what should happen yet.
We know though that not every rivulet is stable and we know that some of them breakup and form droplets.
But how many of them do that and is there a systematic behind it?
This and some more questions will be adressed in the forth coming cells.

## Real space
So what does the simulations produce?
Below is an image of the state at the end of the time loop.
Clearly the rivulet has broken into droplets.
"

# ╔═╡ d5152b67-bc1d-4cc0-b73e-90d79dbadcb4
begin
	nmset = 10
	ts = 1000000
	dataH = RivuletTools.read_data(R=data[nmset][1], r=data[nmset][2], kbT=data[nmset][3], month=data[nmset][5], day=data[nmset][6], hour=data[nmset][7], minute=data[nmset][8], θ=data[nmset][9])
	RivuletTools.heatmap_data(dataH, t=ts)
end

# ╔═╡ ab5b4c7c-ae24-4aae-a528-1dc427a7f1f1
md"While a picture says more than a thousand words, it's not always to best way to describe data.
For example the number of droplets and the wavelength assozitated with that number.

That is why we use the `Image.jl` library to extract image features.
Luckily one of the features is the number of clusters which equivalent to the number of droplets.
In the image below we do this analysis for a single state and colorate each feature with a random color."

# ╔═╡ c945050f-3ddc-4d0e-80dd-af909c3f4ab5
RivuletTools.segment_image(dataH, ts)

# ╔═╡ 7e2fd675-28ba-4412-9c56-4b40b3380576
md"
We measure 27 droplets and if we were to count the different colored discs we also would find 27 discs.
This is an rather straightforward way to see if a rivulet is stable or if it breaks during the simulation.
"

# ╔═╡ 89045ff9-bfb2-43e7-865b-235181cdf9f7
md" ### Dynamics in real space

The easist way to identify what is going on, is to just see whats going on.
This is why we generate a `.gif` file for each individual simulation.
We use the function `data2gif()` to do this.

#### The quartet of possibilities

In the next three cells we show what the rivulet can do by loading the gifs of some simulations.
Here we are not strictly comparing apples with apples, instead we mix our data.
Thus we include data from both the patterned substrate as well as the uniform substrate.
Considering these two setups we observe the following processes:

1. Retract
"

# ╔═╡ 03bf6a75-a98c-4641-9939-2336c78e1be7
begin
	# slice_gif = RivuletTools.read_data(R=180, r=40, kbT=0.0, month=11, day=3, hour=23, minute=34, θ=40 ,nm=32, arrested=false)
	# (180, 40, 0.0,    11, 3, 23, 34, 40
	# RivuletTools.do_gif_slice(slice_gif, "slice_R180_r40_t40"; timeMax=2500000)
end

# ╔═╡ a58ec747-09cb-4cba-a9f0-4de683c80052
LocalResource("../assets/ang_40_R_180_rr_40_kbt_off.gif", :width => 600)

# ╔═╡ c66bac82-feaf-4e77-ab1b-ea7a2a5cf6c7
md"
2. Retract into breakup
"

# ╔═╡ b385a6b2-e2b0-4179-ac81-21a8600f86cf
LocalResource("../assets/ang_40_R_180_rr_20_kbt_off.gif", :width => 600)

# ╔═╡ 7f8b5fe8-f5d1-46eb-a30e-8f0a6e9707bc
md"
3. Breakup without retraction (patterned substrate)
"

# ╔═╡ 2e7b7b97-f4b9-4ef8-b360-e086ffc0a025
LocalResource("../assets/arr_ang_30_R_180_rr_40_kbt_off.gif", :width => 600)

# ╔═╡ dc37fa99-ceb5-40cb-846a-6cdf9d33c2f3
md"
4. Don't breakup without retraction (patterned substrate)

Which we don't show, however there are quite a few of those very stable simulations where essentially nothing happens.
In the animations you see the height field which color coded and the range is dynamically adjusted.

For a better grip on the dynamics we take all our simulations and perform some simple measures. 
We are, for example, interested in the growth of the instability on the rivulet. 
That is why we measure the maximum height difference along the rivulet at every time step.
On the other hand we want to know if the rivulet has ruptured, thus we use the image analysis introduced in `measure_clusters()`. 
Often stability can be adressed using a so-called **linear-stability analysis**.
Some papers have been found in the mean time!

I still have to find a way to compute that however what I readily can do is to analyse the spectra.

For now we stay in real space and try to get one dimensional circuluar cuts of the rivulet.
The first part to do is to create a distance matrix, which can be accessed with `RivuletTools.distanceArray()`.
This matrix contains radial distances from the center using Chebyshev distances.
We are especially interested in the evolution of the center of the rivulet, where we assume for the moment that the center of the rivulet also contains the thickest part of the rivulet.
We can then use this information to get a circular cut.

It is in fact not as easy as I hoped to, because the distance matrix is not ordered.
However with an inefficient trick we get the circular cut to be simple connected and have a somewhat smooth curve as depicted below.
The details of this can be found in `RivuletTools.getRingCurve()`.
"

# ╔═╡ 7e8e31b9-15fd-4d34-b447-4440ea805811
hCirc, Radius = RivuletTools.getRingCurve(data[25], 150000)

# ╔═╡ 044b6cab-06cf-405e-864c-3e040faa602d
radial_plot = plot(0:2π/(length(hCirc)-1):2π, hCirc, label="Cut at R=$(Radius)Δx", l=(1.5, :solid), ylabel="height", grid=false, xlabel="x\\(\\phi\\)", xticks = ([0:π/2:2*π;], ["0", "\\pi/2", "\\pi", "3\\pi/2", "2\\pi"]))

# ╔═╡ c5949c51-d3f5-40b1-9415-7c40ae596b1b
savefig(radial_plot, "../assets/height_line.png")

# ╔═╡ 277d0ee8-bb02-4c5f-a6cb-9c8c01795b65
tforPlot = [25000, 250000, 2500000]

# ╔═╡ 4fddfc1c-4943-40df-981b-8609c49fa6bb
begin
	#simpleRep = plot()
	nset = 28
	dataHring = RivuletTools.read_data(R=data[nset][1], r=data[nset][2], kbT=data[nset][3], year=data[nset][4], month=data[nset][5], day=data[nset][6], hour=data[nset][7], minute=data[nset][8], θ=data[nset][9])	
	tss = 250000
	hmaxD = initial_data[(initial_data.R0 .== data[nset][1]) .& (initial_data.rr0 .== data[nset][2]) .& (initial_data.angle .== data[nset][9]), :hdrop]
	taum = initial_data[(initial_data.R0 .== data[nset][1]) .& (initial_data.rr0 .== data[nset][2]) .& (initial_data.angle .== data[nset][9]), :tauMax]
	h1d, Rad = RivuletTools.getRingCurve(data[nset], tss)
	hfield = RivuletTools.heatmap_data(dataHring, t=tss, just_data=true)
	plot(heatmap(hfield, aspect_ratio=1, c=:viridis, xlim=(1,512), ylim=(1,512)), plot(0:2π/(length(h1d)-1):2π, h1d, label="Cut at R=$(Rad)Δx", l=(1.5, :solid), ylabel="h", grid=false, xlabel="x\\(\\phi\\)", xticks = ([0:π/2:2*π;], ["0", "\\pi/2", "\\pi", "3\\pi/2", "2\\pi"])))
	# simpleRep
end

# ╔═╡ e01d84fc-cb9e-495c-a346-c88d5d9428fc
xscalesheat = collect(1:1:512) ./ hmaxD

# ╔═╡ ab3579f2-21b8-4a95-998b-ff940c9c3677
forShowHM = heatmap(xscalesheat, xscalesheat, hfield ./ hmaxD, aspect_ratio=1, c=:viridis, xlim=(minimum(xscalesheat),maximum(xscalesheat)), ylim=(minimum(xscalesheat),maximum(xscalesheat)),
xlabel=L"x/H_D",
ylabel=L"y/H_D",
colorbar_title=L"h(\mathbf{x})/H_D",
)

# ╔═╡ c104b355-2a89-4d8c-8b38-8ad504cbde18
savefig(forShowHM, "../assets/heatmap_R180_r30_a40_t25000.pdf")

# ╔═╡ e3c4daa5-69b7-465a-9e34-7c1a050a5f29
begin
	lineCutPlot = plot(0:2π/(length(h1d)-1):2π, 
		h1d ./ hmaxD, 
		label="R(t)=$(round(Rad / hmaxD[1], digits=2))/H_D, t=$(round(250000 / taum[1], digits=2))", 
		l=(2.5, :solid), 
		ylabel="h\\(\\xi\\)", 
		ylims = (0, 0.8),
		xlims = (0, 2π),
		grid=false, xlabel="x\\(\\phi\\)", 
		xticks = ([0:π/2:2*π;], ["0", "\\pi/2", "\\pi", "3\\pi/2", "2\\pi"]),
		legendfontsize = 10,
		guidefont = (16, :black),
		tickfont = (12, :black),
	)
	for t in [(850000, :dash), (1100000, :dashdot), (1600000, :dashdotdot)]
		h1dt, Radt = RivuletTools.getRingCurve(data[nset], t[1])
		plot!(0:2π/(length(h1dt)-1):2π, 
			h1dt ./ hmaxD, 
			label="R(t)=$(round(Radt / hmaxD[1], digits=2))/H_D, t=$(round(t[1] / taum[1], digits=2))",
			l=(2.5, t[2]), 
		)
	end
	lineCutPlot
end

# ╔═╡ c2354103-b70f-4371-9d71-cc7f8ec69b3d
savefig(lineCutPlot, "../assets/linecut_R180_r30_a40.pdf")

# ╔═╡ 01d77625-dc7e-4550-a9fc-ab2961606f4d
psim = initial_data[(initial_data.R0 .== data[nset][1]) .& (initial_data.rr0 .== data[nset][2]) .& (initial_data.angle .== data[nset][9]), :hdrop]

# ╔═╡ cfb78bf5-9f06-4557-a777-c34d908f0e67
md"
Similar to above we can turn the time series of the profiles into an animation.
We have normalized the y-axis with the equal volume single droplet height
```math
	h_d = r_d(1-\cos(\theta)),
```
where $r_d$ is the radius of the sphere from which the cap is cut and given by
```math
	r_d = \sqrt[3]{\frac{3V}{\pi(2+\cos(\theta))(1-cos(\theta))^2}},
```
and $V$ is the volume of the rivulet at $t=0$.
The x-axis is normalized using 
```math
	x(\phi) = \frac{2\pi x_i}{N-1},
```
where $N$ is the number of elements $x_i$ that are contained by the 
circular cut.
"

# ╔═╡ cb4a302c-fb04-4362-95d5-7680d8fb2983
RivuletTools.do_ringgif(data[25], "firstRingAnimation")

# ╔═╡ 30220b96-8154-4ca2-a016-9258860323a5
md"
### Growth rates

One rather straight question we can address using the height data is that of growth rates.
Or put it slightly differentely how does the quantity
```math
	\Delta h = h_{max} - h_{min}
```
evolve with time.
This measure has however the downside that $h_{min}$ is more or less constant because it is set by the disjoining pressure $\Pi(h)$ which reads
```math
	\Pi(h) = Kf(h),
```
with 
```math
	K = \frac{2(1-\cos(\theta))}{h_{\ast}},
```
and
```math
	f(h) = \left(\frac{h_{\ast}}{h}\right)^3 - \left(\frac{h_{\ast}}{h}\right)^2, 
```
with $h_{\ast}$ being the precursor film height that is assumed to be $h_{\ast} \ll h$.
We therefore have 
```math
	h_{min} \propto h_{\ast}
```
independent of time.

This data is already precompiled and can be found in `all_data_rivulets.csv`.
To get this data we run the function `RivuletTools.measure_data()` that collects data on $R(t)$, $r(t)$ and $\Delta h(t)$
"

# ╔═╡ df519afa-309a-4633-860d-2fe40a384fa9
for to_analyse in [(data, "dynamics_uniform", false), (data_arrested, "dynamics_patterned", true)]
	run_me = false
	RivuletTools.measure_data(to_analyse[1], to_analyse[2], run_me, to_analyse[3], "", (false, 30, 40))
end

# ╔═╡ 3128d6eb-375d-4770-8215-6ed7e3ac5b5a
growthDF = CSV.read("../data/ring_all_sims_nokBT.csv", DataFrame)

# ╔═╡ 103063b6-c5a9-4c5d-829b-4587813bfaf4
begin
	incond = (180, 20, 40, "pattern", "uniform")
	subdata = growthDF[(growthDF.R0 .== incond[1]) .& (growthDF.rr0 .== incond[2]) .& (growthDF.theta .== incond[3]) .& (growthDF.substrate .== incond[4]), :]
	psi0 = initial_data[(initial_data.R0 .== incond[1]) .& (initial_data.rr0 .== incond[2]) .& (initial_data.angle .== incond[3]), :].psi0[1]	
	T0 = initial_data[(initial_data.R0 .== incond[1]) .& (initial_data.rr0 .== incond[2]) .& (initial_data.angle .== incond[3]), :].tauMax[1]	
	H0 = initial_data[(initial_data.R0 .== incond[1]) .& (initial_data.rr0 .== incond[2]) .& (initial_data.angle .== incond[3]), :].hdrop[1]	
	sigma0 = initial_data[(initial_data.R0 .== incond[1]) .& (initial_data.rr0 .== incond[2]) .& (initial_data.angle .== incond[3]), :].sigmaMax[1]	
	growth_plot = plot(subdata.time[2:end] ./ T0, subdata.deltaH[2:end] ./ H0, 
		label="band",
		xlabel = L"t/\tau_m",
		ylabel = L"\Delta h / h_d",
		# xaxis=:log10,
		yaxis=:log10,
		# title = latexstring("\$\\psi_0 = {$(round(psi0, digits=3))}\$"),
		grid = false,
		legendfontsize = 12,
		guidefont = (16, :black),
		tickfont = (12, :black),
		minorticks = true,
		legend = :bottomright,
		w = 2,
		ylims = (0.003, 1.1),
		xlims = (0.0, 12)
		)
	subdata2 = growthDF[(growthDF.R0 .== incond[1]) .& (growthDF.rr0 .== incond[2]) .& (growthDF.theta .== incond[3]) .& (growthDF.substrate .== incond[5]), :]
	plot!(subdata2.time[2:end] ./ T0, subdata2.deltaH[2:end] ./ H0, 
		label="unifrom",
		l = (2, :dash))
	plot!(subdata2.time[1:end] ./ T0, (0.18 .* exp.(sigma0 .* subdata2.time[1:end]) .- 0.01) ./ H0, 
		label="Exponential fit",
		l = (2,  :dashdot, :black))

	plot!(subdata2.time[1:end] ./ T0, (0.001 .* exp.(0.000008 .* subdata2.time[1:end]) .+ 0.14) ./ H0, 
		label="",
		l = (2,  :dashdot, :black))

	# println(sigma0)
end

# ╔═╡ c7590419-c3d0-41f7-8777-227fcc7b1ba8
md"
### Linear Stability Analysis (LSA)

The starting point for this analysis is the equation that we numerically approximate
```math
	\partial_t h + \nabla (M(h)\nabla p) = 0,
```
where $M(h)$ is a mobility given by 
```math
	M(h) = \frac{2h^2+6h\delta +3\delta}{6\mu},
```
and the pressure 
```math
	p = \Delta h - \Pi(h).
```

For a complet and thorough derivation of this LSA we point towards the paper by [Gonzalez, Diez, Kondic](https://www.cambridge.org/core/product/identifier/S0022112012006076/type/journal_article).
They go into every detail that one can think of.

When we insert all the terms into the first equation and multiply with $6\mu$ we have
```math
	6\mu\partial_t h + \nabla[(2h^2+6h\delta +3\delta)\nabla(\Delta h - \Pi(h))]
```
Let
```math
	h = h_0(r,\phi) + \epsilon h_1(r, \phi, t),
```
with $\epsilon \ll 1$ and insert this in the above equation then we get at $O(1)$
```math
	\nabla[M(h_0)\nabla(\Delta h_0 - \Pi(h_0))] = 0.
```
At first order $O(\epsilon)$ we find the equality $\partial_t h_1 +\mathcal{L}_1 h_1 = 0$ where
```math
	\mathcal{L}_1 h_1 = \nabla[M(h_0)\nabla(\Delta h_1 - \Pi(h_0)) + 3h_1(2h_0^2 + 4h_0\delta + \delta)\nabla (\Delta h_0 - \Pi(h_0))]
```
"

# ╔═╡ c1b3e29b-51b6-4bbd-8793-ece13bfb5a70
measureDF = CSV.read("../data/dynamics_uniform_and_patterned.csv", DataFrame)

# ╔═╡ f12c1925-cf29-41e5-9499-87efe1a96528
md"
A result of the LSA done by [Gonzalez, Diez, Kondic](https://www.cambridge.org/core/product/identifier/S0022112012006076/type/journal_article) is an maximal unstable wavenumber and thus a prediction for the number of droplet $n_{max}$ after rivulet breakup.

This number can be approximated using
```math
	n_{max,app} = \frac{\pi}{2\psi_0},
```
where $\psi_0 = w/R$ with $w$ being the width of the rivulet and $R$ is the radius of the position of the maximum.
To compute $\psi_0$ we simply look it up in `t0_data()`. 
Now let's have a look at $n_{max,app}$
"

# ╔═╡ a41cab07-f8f1-4b24-bd15-5fd29651fe36
plot(collect(0.001:0.001:1), π ./ (2 .* collect(0.001:0.001:1)), 
	xlabel = L"\psi_0",
	ylabel = L"n_{max}",
	label = L"LSA",
	xlims= (0, 1),
	ylims = (0, 30),
	minorticks = true,
	grid = false)

# ╔═╡ 695bd289-a4b3-4154-ab49-6229de133164
md"
Clearly the number of droplets scales very visible with the geometric initial conditions of the rivulet.
In a next step we can our data from both the uniform and patterned substrates."

# ╔═╡ 538694ea-a0b1-4f09-b7b4-4af64d7ec10b
dfLSAclean = RivuletTools.dropletFrame()

# ╔═╡ a0239801-eb9a-4648-b164-15013fc3c445
md"
The above dataframe has some minor issues, first and foremost the 10 degree contact angle data is flawed.
`ImageSegmentation` has a problem with the small droplet heights at small contact angles.
However the droplets are clearly visible to the eye when plotted.

That is why we edit the dataframe to be in agreement with visiual inspection.
"

# ╔═╡ 068f3e73-ed23-4210-b739-ccaadc9f2ba8
dfLSAcleancorrected = CSV.read("../data/maxdroplets-corrected.csv", DataFrame)

# ╔═╡ 5c8e3e0c-3344-431e-a370-625b467e2ea9
begin
	@df dfLSAcleancorrected scatter(
    	:psi0,
    	:ndrops,
    	group = :substrate,
    	# title = "",
    	xlabel = "ψ₀",
    	ylabel = "n max",
		legendfontsize = 12,
		guidefont = (16, :black),
		tickfont = (12, :black),
    	m = (0.5, [:circle :star], 12),
    	# bg = RGB(0.2, 0.2, 0.2)
	)
	
	psis = collect(0.001:0.001:1)
	LSA_drops = plot!(psis, π ./ (2 .* psis), 
	# xlabel = L"\psi_0",
	# ylabel = L"n_{max}",
	label = "Eq. (18)",
	xlims= (0, 0.805),
	ylims = (0, 30),
	minorticks = true,
	l = (:black, 2),
	grid = false)
	
	# plot!(psis, π ./ (2 .* psis) .- exp.(-psis), 
	# xlabel = L"\psi_0",
	# ylabel = L"n_{max}",
	# label = "LSA - better",
	# l = (:black, :dash, 2),
	# )
	savefig(LSA_drops, "../assets/LSA_droplets.pdf")
	LSA_drops
end

# ╔═╡ b9cf9e47-d8da-4416-b67d-7c806595ceb9
begin
	@df dfLSAcleancorrected scatter(
    	:psi0,
    	:ndrops,
    	group = :theta,
    	# title = "",
    	xlabel = L"\psi_0",
    	ylabel = L"n_{max}",
    	m = (0.5, [:circle :star :hex :ut], 12),
    	# bg = RGB(0.2, 0.2, 0.2)
	)

	plot!(psis, π ./ (2 .* psis), 
	# xlabel = L"\psi_0",
	# ylabel = L"n_{max}",
	label = "LSA",
	xlims= (0, 1.05),
	ylims = (0, 30),
	minorticks = true,
	l = (:black, 2),
	grid = false)
end

# ╔═╡ 5d722b85-4e5b-4dc3-9b8d-eb3b6cf33941
md"
## Wettability gradient

Test me
"

# ╔═╡ 8bdffe16-02af-46bd-a568-0e238cc76b6c
RivuletTools.do_gif(RivuletTools.read_data(R=180, r=20, kbT=0.0, year=2024, month=2, day=22, hour=13, minute=5, θ=40, gradient=(true, 10, 40), nm=(3,2)), "gradient_first", timeMax=2500000)

# ╔═╡ a48e9529-bc20-4b46-a861-d6ae09e41d2c
forGradientDF = CSV.read("../data/ring_all_sims_nokBT.csv", DataFrame)

# ╔═╡ 4e19b5fa-984c-4dd0-a956-6371b53f58f6


# ╔═╡ 8886f394-ff90-4fbd-8dec-874f6a4ded83
dropset = 44

# ╔═╡ d9556b99-ac13-4278-8fc2-085728a2cfa9
begin
	tend = 2500000
	arrhere = false
	if arrhere
		checkH = RivuletTools.read_data(R=data_arrested[dropset][1], r=data_arrested[dropset][2], kbT=data_arrested[dropset][3], month=data_arrested[dropset][5], day=data_arrested[dropset][6], hour=data_arrested[dropset][7], minute=data_arrested[dropset][8], θ=data_arrested[dropset][9], nm=(3,2), arrested=true)
		println("Dataset: $(data_arrested[dropset])")
		println("width: $(2*initial_data[(initial_data.R0 .== data_arrested[dropset][1]) .& (initial_data.rr0 .== data_arrested[dropset][2]) .& (initial_data.angle .== data_arrested[dropset][9]), :].realrr[1]), h: $(initial_data[(initial_data.R0 .== data_arrested[dropset][1]) .& (initial_data.rr0 .== data_arrested[dropset][2]) .& (initial_data.angle .== data_arrested[dropset][9]), :].maxh0[1])")
	else
		checkH = RivuletTools.read_data(R=data[dropset][1], r=data[dropset][2], kbT=data[dropset][3], month=data[dropset][5], day=data[dropset][6], hour=data[dropset][7], minute=data[dropset][8], θ=data[dropset][9], nm=(3,2))
		println("Dataset: $(data[dropset])")
		println("width: $(2*initial_data[(initial_data.R0 .== data[dropset][1]) .& (initial_data.rr0 .== data[dropset][2]) .& (initial_data.angle .== data[dropset][9]), :].realrr[1]), h: $(initial_data[(initial_data.R0 .== data[dropset][1]) .& (initial_data.rr0 .== data[dropset][2]) .& (initial_data.angle .== data[dropset][9]), :].maxh0[1])")
	end
	RivuletTools.heatmap_data(checkH, t=tend)
end

# ╔═╡ 543dea77-0aea-49e6-811a-5a87218d6632
RivuletTools.segment_image(checkH, tend)

# ╔═╡ 0489cd43-a451-435e-9024-7b9ff3432761
begin
	time1d = 500000

	ll1, spec1 = RivuletTools.height2fft(data[26], time1d, output=true)
	ll2, spec2 = RivuletTools.height2fft(data[25], time1d, output=true)
	shifted_k1 = fftshift(fftfreq(length(ll1))*length(ll1))
	shifted_k2 = fftshift(fftfreq(length(ll2))*length(ll2))
	k_pi1 = shifted_k1 .* 2π/length(ll1)
	k_pi2 = shifted_k2 .* 2π/length(ll2)
	plot(k_pi1, log.(abs.(spec1 .* spec1)) .+ 1, 
			# aspect_ratio=1, 
			xlims=(0,π), 
			xlabel = "q",
			ylabel = "log(S(q))",
			label="R=$(data[26][1]) r=$(data[26][2])",
			#clim=(0.1, 1000) # Limits for heatmap
		)
	plot!(k_pi2, log.(abs.(spec2 .* spec2)) .+ 1,
		label="R=$(data[25][1]) r=$(data[25][2])",
		xlims=(0.01, pi),
		xscale = :log10,
		# ylims=(-10, 10),
		)
end

# ╔═╡ 1c6b09bb-9809-411d-8ddd-2095256d0601
md"
In the following we work with the hight data only, we don't measure but simply transform it with a **FFT**.
This way we can compute dominate wavelengths. 
We also get a dispertion relation, showing (hopefully) growth and damping of different wave modes.
This information should differ between the arrested rivulets and the contracting rivulets, because 
- the major and minor radii are not constant and 
- the wave modes are directly correlated with both them, at least I assume so.

So let's get started and compute some spectra.
First let's introduce the fft method.

## Reciprocal space
Using `Image.jl` is in fact only one way to look at this issue, because we can also **Fourier transform** the data and analyse the spectrum in reciprocal space.
Long waves have small wave numbers and short wavelengths have large wave numbers.
This is why we loaded `FFTW.jl` at the top of the notebook to compute the **FFT** (fast Fourier transform) of the height field $h(\mathbf{x},t)$, where $\mathbf{x} = (x,y)$.

We therefore choose a single simulation for the unpatterned data set.
"

# ╔═╡ 2a66eee4-be06-43a2-a9be-fc2e0c4a0f32
fftset = 25

# ╔═╡ a86d30c5-03f0-4e11-ba25-1f08ba0998b7
md"
Next we choose a single time step where we want to compute the **FFT** of the height field.
"

# ╔═╡ ea5f58f9-5394-4536-9458-0484e85fdc85
time_here = 250000

# ╔═╡ cca0ed73-b1e7-4895-8537-294ccb3f9e26
md" 
The actual height field is just a 512 by 512 matrix with height values at every lattice point, as shown below.
"

# ╔═╡ 289205ad-0bd3-473c-b076-fab42e1643c3
begin
	fft_try = RivuletTools.read_data(R=data[fftset][1], r=data[fftset][2], kbT=data[fftset][3], year=data[fftset][4], month=data[fftset][5], day=data[fftset][6], hour=data[fftset][7], minute=data[fftset][8], θ=data[fftset][9], nm=(3,2))
	fft_data = RivuletTools.heatmap_data(fft_try, t=time_here, just_data=true)
	shifted_k = fftshift(fftfreq(512)*512)
end

# ╔═╡ fc37abf6-6acd-4b11-bdab-ddee379d8d72
md" 
The above matrix is a single time step of one of our simulations.
In fact, a simulation where the rivulet breaks while retracting.
This height field, `fft_data` can then be used to compute a spectrum as shown below, where we interpret the height data as grayscale.
"

# ╔═╡ 29f4022e-28dc-4ef8-8e83-11d466437813
spectrumH= fftshift(fft(fft_data ./ maximum(fft_data)))

# ╔═╡ a1f13b96-fbd3-40ab-aa3f-9af14d55ed55
md"
### Spectra

`spectrumH` is a two dimensional *FFT* of the height which is shown in the plot three cells above.
The *FFT* itself has all the information we need to know for a dispertion relation.
Radial averaging of this data should correspond to the growth of a dominate wave number and to the damping of wave numbers which are too large to be resolved. 
"

# ╔═╡ ef66b620-cfa3-47b4-bc2b-6cc77427764f
RivuletTools.data2fft(whichdata=RivuletTools.data, dataset=fftset, time=250000, quater=false, output=false)

# ╔═╡ 010e2c3b-f66a-411b-b819-3d37448c4087
md"
The height field that produces the fft above is shown in the cell below.
We clearly see that this annulus has some funky undulations and this funky undulations are actually well captured by the fft signal.

If we would take a annulus that is just retracting we would see an isotropic annulus pattern with distance between them based on the real space annulus radius. 
"

# ╔═╡ 8d0b7517-1be8-41c4-8a4b-716bcad169fb
RivuletTools.heatmap(fft_data, c=:viridis, xlims=(1,512), ylims=(1, 512), aspect_ratio=1)

# ╔═╡ 19deee02-5fb7-400c-a853-74bd44a8deaf
md"The *FFT* is as far as I know symmetric and does only contain information for wave lengths up to $L/2$, which in our case is $256\Delta x$.
That is why a quater of the above image should be enough for further analysis."

# ╔═╡ 608b2a67-b34b-4440-9282-3f225e5714be
RivuletTools.data2fft(whichdata=RivuletTools.data, dataset=fftset, time=250000, quater=true, output=false)

# ╔═╡ e370f645-d408-4d7f-8c48-0f9e05522b5f
md"
### Power Spectral density

Text here
"

# ╔═╡ 72270f78-01c5-4ce5-aac4-901fd9a5143f
hmmm = RivuletTools.simpleRadialAverage(spectrumH, abssqrt=true) .+ 1

# ╔═╡ 783e9476-a0a8-4427-b929-9f7c0faa6b59
plot(0:π/255:π, 
	hmmm[1:256], 
	yscale=:log10, 
	xlims=(0, π)
)

# ╔═╡ a59cefd2-27c4-4332-943e-1fbd79ae2481
begin
	someset = 26
	# Retraction
	fft_anim1 = RivuletTools.read_data(R=data[someset][1], r=data[someset][2], kbT=data[someset][3], year=data[someset][4], month=data[someset][5], day=data[someset][6], hour=data[someset][7], minute=data[someset][8], θ=data[someset][9], nm=(3,2))
	# Breakup
	fft_anim2 = RivuletTools.read_data(R=data[someset-1][1], r=data[someset-1][2], kbT=data[someset-1][3], year=data[someset-1][4], month=data[someset-1][5], day=data[someset-1][6], hour=data[someset-1][7], minute=data[someset-1][8], θ=data[someset-1][9], nm=(3,2))
	anim = Animation()
	dataEnd1 = RivuletTools.heatmap_data(fft_anim1, t=2500000, just_data=true)
	dataEnd2 = RivuletTools.heatmap_data(fft_anim2, t=2500000, just_data=true)
	max1 = maximum(dataEnd1)
	max2 = maximum(dataEnd2)
	for t in 25000:25000:2500000
		fftdata1 = RivuletTools.heatmap_data(fft_anim1, t=t, just_data=true) ./ max1
		fftdata2 = RivuletTools.heatmap_data(fft_anim2, t=t, just_data=true) ./ max2
		spec1= fftshift(fft(fftdata1))
		spec2= fftshift(fft(fftdata2))
		# specH[256, 256] = 1.0
		averagedPSD1 = RivuletTools.simpleRadialAverage(spec1, abssqrt=true)
		averagedPSD2 = RivuletTools.simpleRadialAverage(spec2, abssqrt=true)
		plot(0:π/255:π, 			# x-axis from 0 to pi 
			log.(averagedPSD1) .+ 1,  	# y-axis averaged FFT
			title="t=$(t)Δt", 		# Titel
			label="Retraction",  	# Label
			xlabel="q/[Δx⁻¹]",  	# x-label
			xticks = ([0:π/4:π;], ["0", "\\pi/4", "\\pi/2", "3\\pi/4", "\\pi"]),
			ylabel="log(S(q))", 			# y-label
			# yscale=:log10, 			# y-axis scaling
			# xscale=:log10, 		# x-axis scaling
			grid=false, 			# No grid
			minorticks=true, 		# Minorticks for log
			# xlims = (1, 10)
    	)
		plot!(0:π/255:π, 
			log.(averagedPSD2) .+ 1, 
			label="Breakup", 
			xlims = (0, π)
    	)
		frame(anim)
	end
	gif(anim, "../assets/spectrum_difference.gif")
end

# ╔═╡ 38345378-66ee-42c1-b37f-6691119ecc60
md"
Below we just read two `.csv`s because we precompiled the data already.
There is data on  
- Uniform substrate
- Patterend substrate

(However to redo the analysis simply change `run_me` to `true` in the cell below.)

# Lets get some graphs tomorrow!
"

# ╔═╡ 3273792c-41fb-4225-a4f7-2f1c9d58be4a
for to_analyse in [(data_gamma05, "gamma05_uniform", "gamma05_"), (data_gamma20, "gamma20_uniform", "gamma20_")]
	run_me = false
	RivuletTools.measure_data(to_analyse[1], to_analyse[2], run_me, false, to_analyse[3], (false, 30, 40))
end

# ╔═╡ 144e23e9-ce3d-4ed6-be2c-dff1fed39e59
md"For convenience we collect all the data we have into a single dataframe, which for whatever reason is called `all_df`.
Doing so we use the function `combined_df` that either reads and combines the data or just reads an existing dataframe."

# ╔═╡ f7521761-e9f1-43df-a95c-57aec7c83011
"""
	combined_df(name::String; remeasure=false)

Combines the data of the two different substrates.
"""
function combined_df(name::String; remeasure=false)
	if remeasure
		combined = DataFrame()
		for csvs in ["dynamics_uniform", "dynamics_patterned"]
			df = csv2df(csvs)
			if csvs == "dynamics_uniform"
				df.substrate = fill(:uniform, length(df.R))
			elseif csvs == "dynamics_patterned"
				df.substrate = fill(:patterned, length(df.R))
			end
			combined = vcat(combined,df)
		end
		CSV.write("data/$(name).csv", combined)
		return combined
	else
		all_df = RivuletTools.csv2df(name)
		return all_df
	end
end

# ╔═╡ 77697cb7-40fa-4ed4-9008-8d78cfa0c247
md"
In `all_df` we have evaluated if a rivulet has ruptured, when it ruptured, how many droplets it produced and which kind of substrate we started the simulation on.
There is still a lot of data because we still consider every time step.

We might come back to `all_df` but for the first simple information we want to extract if the rivulets ruptured and when they ruptured.
"

# ╔═╡ 7f60e96f-9a5e-41f5-a388-f531585e15b0
all_df = combined_df("data_all_rivulets")

# ╔═╡ 96dea7cc-4742-460f-a18e-ae22f0c92033


# ╔═╡ e4bb69eb-e608-4f50-9c42-678a74b99192
all_df.gamma .= 0.01

# ╔═╡ 010e992a-f35a-4a7a-946e-f796aba41a32


# ╔═╡ cdbe66ff-d643-4b7a-a446-85c284c668ba
all_df

# ╔═╡ 6b34bdad-2518-43f5-9fb0-d28a99a411fe
md"
With `breakup_detection()` we reduce every simulation to a single row in a dataframe.
The only information we want to extract in this row is

- Did the rivulet rupture?
- When did it rupture?
- How many droplets are produced?

The answer to the first and third question can be answered with the same information.
We just ask how many clusters we have at the end of the simulation, if that number is larger than one we set `df.rupture = true`.
Similar we set `df.drops = data.clusters[tmax]` where `tmax` is the last time simulation time step.

To answer when the rivulet has ruptured we use the function [`findfirst()`](https://docs.julialang.org/en/v1/base/arrays/) and check where `data.clusters > 1` for the first time.
"

# ╔═╡ 627c3c50-b22f-4e95-a755-26f2197fff92
"""
	breakup_detection(df::DataFrame, label::String)

Scans the time dependent data and extracts if the rivulet breaks up into droplets and many more features. Saves the result to a csv named with `label`.
"""
function breakup_detection(df::DataFrame, label::String; remeasure=false)
	if remeasure
		droplets = DataFrame()
		frag = Bool[]
		Rs = Int64[]
		rrs = Int64[]
		rupturetime = Int64[]
		beta0 = Float64[]
		betaR = Float64[]
		kbts = Float64[]
		ndrops = Int64[]
		angles = Float64[]
		wavelengths = Float64[]
		substrate = String[]
		existing = String[]
		for rr in [20, 30, 40, 80]
			for R in [150, 160, 180, 200]
				for θ in [1/18, 1/9, 1/6, 2/9]
					for sub in ["uniform", "patterned"]
						for kbt in [0.0, 1e-6]
							sim = df[(df.theta .== round(rad2deg(θ*π))) .& (df.R .== R) .& (df.rr .== rr) .& (df.substrate .== sub) .& (df.kbt .== kbt), :]
							push!(Rs, R)
							push!(rrs, rr)
							push!(kbts, kbt)
							push!(angles, round(rad2deg(θ*π)))
							push!(substrate, sub)
							# Check if there is actual data
							if length(sim.clusters) > 0
								# Check if the rivulet ruptured
								if sim.clusters[end] > 1
									rt = findfirst(x -> x > 1, sim.clusters)
									push!(rupturetime, sim.time[rt])
									push!(frag, true)
									push!(ndrops, sim.clusters[end])
									push!(betaR, sim.beta[rt])
									push!(wavelengths, 2π*R/sim.clusters[end])
									push!(beta0, sim.beta[begin])
								else
									push!(frag, false)
									push!(rupturetime, 3000000)
									push!(betaR, sim.beta[end])
									push!(ndrops, 0)
									push!(wavelengths, 2π*R+1)
									push!(beta0, sim.beta[begin])
								end
								push!(existing, "Yes")
							else
								push!(frag, false)
								push!(rupturetime, 0)
								push!(betaR, 0)
								push!(ndrops, 0)
								push!(wavelengths, 0)
								push!(beta0, 0)
								push!(existing, "No")
							end
						end
					end
				end
			end
		end
		droplets.R = Rs
		droplets.rr = rrs
		droplets.theta = angles
		droplets.kbt = kbts
		droplets.substrate = substrate
		droplets.rupture = frag
		droplets.rupturetime = rupturetime
		droplets.drops = ndrops
		droplets.lambda = wavelengths
		droplets.beta_start = beta0
		droplets.beta_rup = betaR
		droplets.exists = existing
	
		CSV.write("data/$(label).csv", droplets)
		return droplets
	else
		# droplets = CSV.read("/net/euler/zitz/Swalbe.jl/data/DataFrames/$(label).csv", DataFrame)
		droplets = CSV.read("../data/$(label).csv", DataFrame)
		return droplets
	end
end

# ╔═╡ bebbf9b7-0d4a-44d0-baa1-aba99b9c59ef
md"
In `simpleBreakup` we collected the result of `breakup_detection`, therefore have a rupture flag, a rupture time and the number of droplets as well as a wavelength. 

The keen observer quickly finds a mismatch between `all_df` and `simpleBreakup` in terms of dimensions.
Ever simulation creates a 100 height field snapshots in the 2.500.000 time steps.
When reduced to a single rupture or not rupture question, there should be only 147 rows.
Not all data is equal and simulations on the patterend substrates use slightly different initial conditions.
Which is why we simply add ghost data and add a column called `df.exists` which has the states Yes or No.

Those rows where `df.exists = No` can be ignored, it was just easier to loop through the data like that.
"

# ╔═╡ b1953875-92ca-42d1-a22e-2f393141ddbe
simpleBreakup = breakup_detection(all_df, "stability")

# ╔═╡ 6a1ce8de-49b4-4a97-aa32-1cd20ded4b04
md"
Now we can ask which simulations ruptured and find this data in `rupture_only`.
"

# ╔═╡ adf3dbe5-ade0-4949-9507-10b5c8164ddd
rupture_only = simpleBreakup[(simpleBreakup.rupture .== true) .& (simpleBreakup.exists .== "Yes") .& (simpleBreakup.kbt .== 0.0), [:R, :rr, :theta, :substrate, :rupturetime, :drops, :lambda, :beta_start]]

# ╔═╡ b8a96a22-4950-4300-8c28-2c8aedf6b66b
md"
Similarly we can ask which existing simulations do not rupture and save that to `stable_only`.
"

# ╔═╡ 081c009c-0871-458e-9081-365d5102fefd
stable_only = simpleBreakup[(simpleBreakup.rupture .== false) .& (simpleBreakup.exists .== "Yes") .& (simpleBreakup.kbt .== 0.0), [:R, :rr, :theta, :substrate, :rupturetime, :drops, :lambda, :beta_start]]

# ╔═╡ 14d1a55a-43b0-4857-8dd9-b10c86f8a123
md"
## Actual research

So far we have discussed how we generated the data and which initial conditions we are using.
We showed that there can be four different outcomes and wrote a few functions to get insights into the data.
But we still lack a real outcome as well as a how our results relate to other studies.

Therefore we have to start some more general questions.
One of them may be which geometries break and when?

### Stability

In the following plots we measure for each simulation if the rivulet breaks up or does nothing (pattterned) or contracts into a single droplet.
"

# ╔═╡ 512e1060-eee5-4374-966c-02d7fb62f303
begin
	scatter_unstable = scatter(rupture_only[rupture_only.substrate .== "uniform", :theta], 
	rupture_only[rupture_only.substrate .== "uniform", :beta_start],
	label = "unifrom",
	xlabel = "θ/[°]",
	ylabel = "β",
	m = (:circle, 8, 0.6),
	title = "Unstable rivulets",
	)
	scatter!(rupture_only[rupture_only.substrate .== "patterned", :theta], 
	rupture_only[rupture_only.substrate .== "patterned", :beta_start],
	label = "patterned",
	m = (:rect, 8, 0.6),
	)
	# savefig(scatter_unstable, "/net/euler/zitz/Swalbe.jl/assets/unstable_scatter.png")
end

# ╔═╡ 61474944-c347-448a-beb9-aa2e4ef6331e
begin
	scatter_stable = scatter(stable_only[stable_only.substrate .== "uniform", :theta], 
	stable_only[stable_only.substrate .== "uniform", :beta_start],
	label = "uniform",
	xlabel = "θ/[°]",
	ylabel = "β",
	m = (:star5, 8, 0.6, palette(:tab10)[3]),
	title = "Stable rivulets",
	)
	scatter!(stable_only[stable_only.substrate .== "patterned", :theta], 
	stable_only[stable_only.substrate .== "patterned", :beta_start],
	label = "patterned",
	m = (:diamond, 8, 0.6, palette(:tab10)[4]),
	)
	# savefig(scatter_stable, "/net/euler/zitz/Swalbe.jl/assets/stable_scatter.png")
end

# ╔═╡ af255535-e903-4eef-8629-13d836f1f145
md"
Another way to display the data is to plot it for each substrate.

Johan: Time scale of pinching is related to minor radius, such as Rayleigh-Plateau. Coalescence is driven by outer radii mismatch. If the time scales of the instability are larger than the retraction on the patterned substrate then it should relate to the uniform substrate. 

Maybe run some simulations where I set all the velocities to 0 at some reoccuring interval `t_velDump`.
"

# ╔═╡ 13257859-e48e-4aa3-a3a3-a4ecf4c8dd1f
begin
	scatter_unstable_uni = scatter(stable_only[stable_only.substrate .== "uniform", :theta], 
	stable_only[stable_only.substrate .== "uniform", :beta_start],
	label = "Retracting",
	xlabel = "θ/[°]",
	ylabel = "β",
	title = "Uniform substrate",
	legendfontsize = 12,			# legend font size
    tickfontsize = 12,	# tick font and size
    guidefontsize = 13,	# label font and size
	m = (:circle, 8, 0.6),
	)
	scatter!(rupture_only[rupture_only.substrate .== "uniform", :theta], 
	rupture_only[rupture_only.substrate .== "uniform", :beta_start],
	label = "Fragmenting",
	m = (:rect, 8, 0.6),
	)
	# savefig(scatter_unstable_uni, "../assets/uniform_scatter.png")
end

# ╔═╡ 9e40d379-50e9-4d33-a53c-6c5089059de5
begin
	scatter_pat = scatter(stable_only[stable_only.substrate .== "patterned", :theta], 
	stable_only[stable_only.substrate .== "patterned", :beta_start],
	label = "Stable",
	xlabel = "θ/[°]",
	ylabel = "β",
	legendfontsize = 12,			# legend font size
    tickfontsize = 12,	# tick font and size
    guidefontsize = 13,	# label font and size
	title = "Patterned substrate",
	m = (:circle, 8, 0.6),
	)
	scatter!(rupture_only[rupture_only.substrate .== "patterned", :theta], 
	rupture_only[rupture_only.substrate .== "patterned", :beta_start],
	label = "Fragmenting",
	m = (:rect, 8, 0.6),
	)
	# savefig(scatter_pat, "../assets/pattern_scatter.png")
end

# ╔═╡ df08506c-f66d-430a-b235-4c9dfb80d414
# ╠═╡ disabled = true
#=╠═╡
plot(df_sub.time./10^6, df_sub.clusters, xlabel="time/10⁶ [a.u.]", ylabel="# clusters",
l = (3, :solid))
  ╠═╡ =#

# ╔═╡ ecba3acb-6bc1-4722-9cee-a388a2442fae
# ╠═╡ disabled = true
#=╠═╡
h20040 = measurements[(measurements.R .== 200) .& (measurements.rr .== 40) .& (measurements.kbt .== 0.0), :]
  ╠═╡ =#

# ╔═╡ c848d2cf-5d36-4437-b53a-e278150e75ef
#=╠═╡
begin
	p2 = plot(timescale ./ t0, 
		h20040[h20040.theta .== 20, :dH] ./ h0, 
		label="θ=20°", 
		l=(3, :solid), 
		xlabel="t/t_c", 
		ylabel="Δh/h₀",
		st = :samplemarkers,
		step = 5, 	
		title = "t_c = rμ/γ",
		marker = (8, :circle, 0.6),		
		# yaxis=:log,
		# xaxis=:log,
		# xlims=(0, 750),
		legendfontsize = 12,			# legend font size
        tickfontsize = 12,	# tick font and size
        guidefontsize = 13,	# label font and size
		)
	for ang in enumerate([30.0, 40.0])
		param = initial_data[(initial_data.R0 .== 200) .& (initial_data.rr0 .== 40) .& (initial_data.angle .== ang[2]), [:t0, :maxh0]]
		plot!(
		timescale ./ param.t0[1], 
		h20040[h20040.theta .== ang[2], :dH] ./ param.maxh0[1], 
		label="θ=$(ceil(Int, ang[2]))°", 
		l=(3, linesty1[ang[1]+1]), 
		# xlabel="t/τ", 
		# ylabel="Δh/h₀",
		st = :samplemarkers,
		step = 5, 						
		marker = (8, markers1[ang[1]+1], 0.6),		
		# yaxis=:log,
		# xaxis=:log,
		)
	end
	savefig(p2, "../assets/delth_t0scaling.png")
	p2
	
end
  ╠═╡ =#

# ╔═╡ 1007c151-b6ab-4f81-8421-4746dc3b67f4
#=╠═╡
begin
	p3 = plot(timescale ./ initial_data[(initial_data.R0 .== 200) .& (initial_data.rr0 .== 40) .& (initial_data.angle .== 20), :tau][1], 
		h20040[h20040.theta .== 20, :dH] ./ h0, 
		label="θ=20°", 
		l=(3, :solid), 
		xlabel="t/τ", 
		ylabel="Δh/h₀",
		st = :samplemarkers,
		step = 5, 						
		marker = (8, :circle, 0.6),		
		# yaxis=:log,
		# xaxis=:log,
		grid = false,
		title = "τ = 9μhₑ/(γθ³)",
		legendfontsize = 12,			# legend font size
        tickfontsize = 12,	# tick font and size
        guidefontsize = 13,	# label font and size
		# xlims = (0.0, 25),
		# ylims = (0.7, 1.5)
		)
	for ang in enumerate([30.0, 40.0])
		param = initial_data[(initial_data.R0 .== 200) .& (initial_data.rr0 .== 40) .& (initial_data.angle .== ang[2]), [:tau, :maxh0]]
		plot!(
		timescale ./ param.tau[1], 
		h20040[h20040.theta .== ang[2], :dH] ./ param.maxh0[1], 
		label="θ=$(ceil(Int, ang[2]))°", 
		l=(3, linesty1[ang[1]+1]), 
		# xlabel="t/τ", 
		# ylabel="Δh/h₀",
		st = :samplemarkers,
		step = 5, 						
		marker = (8, markers1[ang[1]+1], 0.6),		
		# yaxis=:log,
		# xaxis=:log,
		)
	end
	savefig(p3, "../assets/delth_taurimscaling.png")
	p3
	
end
  ╠═╡ =#

# ╔═╡ a32e4478-585f-431e-b4f0-bb6b92010cde
#=╠═╡
begin
	p4 = plot(timescale ./ initial_data[(initial_data.R0 .== 200) .& (initial_data.rr0 .== 40) .& (initial_data.angle .== 20), :t00][1], 
		h20040[h20040.theta .== 20, :dH] ./ h0, 
		label="θ=20°", 
		l=(3, :solid), 
		xlabel="t/t₀", 
		ylabel="Δh/h₀",
		st = :samplemarkers,
		step = 5, 						
		marker = (8, :circle, 0.6),		
		# yaxis=:log,
		# xaxis=:log,
		grid = false,
		title = "t₀ = 3μ/(γr³q₀⁴)",
		legendfontsize = 12,			# legend font size
        tickfontsize = 12,	# tick font and size
        guidefontsize = 13,	# label font and size
		# xlims = (0.0, 0.5),
		# ylims = (0.7, 1.5)
		)
	for ang in enumerate([30.0, 40.0])
		param = initial_data[(initial_data.R0 .== 200) .& (initial_data.rr0 .== 40) .& (initial_data.angle .== ang[2]), [:t00, :maxh0]]
		plot!(
		timescale ./ param.t00[1], 
		h20040[h20040.theta .== ang[2], :dH] ./ param.maxh0[1], 
		label="θ=$(ceil(Int, ang[2]))°", 
		l=(3, linesty1[ang[1]+1]), 
		# xlabel="t/τ", 
		# ylabel="Δh/h₀",
		st = :samplemarkers,
		step = 5, 						
		marker = (8, markers1[ang[1]+1], 0.6),		
		# yaxis=:log,
		# xaxis=:log,
		)
	end
	savefig(p4, "../assets/delth_filmscaling.png")
	p4
	
end
  ╠═╡ =#

# ╔═╡ 5ed1e95e-bf40-4c96-8434-ac83e54e9362
#=╠═╡
begin
	p5 = plot(timescale ./ initial_data[(initial_data.R0 .== 200) .& (initial_data.rr0 .== 40) .& (initial_data.angle .== 20), :ts][1], 
		h20040[h20040.theta .== 20, :dH] ./ h0, 
		label="θ=20°", 
		l=(3, :solid), 
		xlabel="t/tₛ", 
		ylabel="Δh/h₀",
		st = :samplemarkers,
		step = 5, 						
		marker = (8, :circle, 0.6),		
		# yaxis=:log,
		# xaxis=:log,
		title = "tₛ = 3μh/γ[M/(1-cos(θ))]²",
		grid = false,
		legendfontsize = 12,			# legend font size
        tickfontsize = 12,	# tick font and size
        guidefontsize = 13,	# label font and size
		# xlims = (0.0, 750),
		# ylims = (0.7, 1.5)
		)
	for ang in enumerate([30.0, 40.0])
		param = initial_data[(initial_data.R0 .== 200) .& (initial_data.rr0 .== 40) .& (initial_data.angle .== ang[2]), [:ts, :maxh0]]
		plot!(
		timescale ./ param.ts[1], 
		h20040[h20040.theta .== ang[2], :dH] ./ param.maxh0[1], 
		label="θ=$(ceil(Int, ang[2]))°", 
		l=(3, linesty1[ang[1]+1]), 
		# xlabel="t/τ", 
		# ylabel="Δh/h₀",
		st = :samplemarkers,
		step = 5, 						
		marker = (8, markers1[ang[1]+1], 0.6),		
		# yaxis=:log,
		# xaxis=:log,
		)
	end
	savefig(p5, "../assets/delth_filmscaling.png")
	p5
	
end
  ╠═╡ =#

# ╔═╡ 87c627a8-2c54-44ff-aa66-d9b5c379f646
#=╠═╡
begin
	markers1 = [:circle, :ut, :s]
	linesty1 = [:solid, :dash, :dashdot]
	timescale = collect(25000:25000:2500000)
	t0 = initial_data[(initial_data.R0 .== 200) .& (initial_data.rr0 .== 40) .& (initial_data.angle .== 20), :t0][1]
	h0 = initial_data[(initial_data.R0 .== 200) .& (initial_data.rr0 .== 40) .& (initial_data.angle .== 20), :maxh0][1]
	p = plot(timescale, 
		h20040[h20040.theta .== 20, :dH], 
		label="θ=20°", 
		l=(3, :solid), 
		xlabel="t", 
		ylabel="Δh",
		st = :samplemarkers,
		step = 5, 	
		title = "Unscaled",
		marker = (8, :circle, 0.6),		
		# yaxis=:log,
		#xaxis=:log,
		# ylims = (0, 5),
		legendfontsize = 12,			# legend font size
        tickfontsize = 12,	# tick font and size
        guidefontsize = 13,	# label font and size
		)
	for ang in enumerate([30.0, 40.0])
		plot!(
		timescale, 
		h20040[h20040.theta .== ang[2], :dH], 
		label="θ=$(ceil(Int, ang[2]))°", 
		l=(3, linesty1[ang[1]+1]), 
		# xlabel="t/τ", 
		# ylabel="Δh/h₀",
		st = :samplemarkers,
		step = 5, 						
		marker = (8, markers1[ang[1]+1], 0.6),		
		# yaxis=:log,
		# xaxis=:log,
		)
	end
	savefig(p, "../assets/delth_base.png")
	p
	
end
  ╠═╡ =#

# ╔═╡ 3eaf9941-d510-4a91-99bf-2084bbe3ea40
#=╠═╡
begin
	p = plot()
	for ang in [1/9, 1/6, 2/9]
		for R in [180]
			for rr in [20, 40, 80]
				h = RivuletTools.torus(512, 512, rr, R, ang, (256,256))
				plot!(h[256, :], 
					xlims=(0, 256),
					ylims=(0, 20),
					lw = 2,
					aspect_ratio = 5,
					xlabel = "x/[Δx]",
					ylabel = "height",
					palette = :Paired_10,
					label="R=$(R)-r=$(rr)-θ=$(Int(round(rad2deg((ang*π)))))",
				)
			end
		end
	end
	savefig(p, "../assets/initial_conditions_R180.png")
	p
end
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
ImageSegmentation = "80713f31-8817-5129-9cf8-209ff8fb23e1"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.10.11"
CategoricalArrays = "~0.10.8"
DataFrames = "~1.6.1"
FFTW = "~1.8.0"
FileIO = "~1.16.1"
ImageSegmentation = "~1.8.2"
Images = "~0.26.0"
JLD2 = "~0.4.38"
LaTeXStrings = "~1.3.1"
Plots = "~1.39.0"
PlutoUI = "~0.7.53"
StatsPlots = "~0.15.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.5"
manifest_format = "2.0"
project_hash = "3f7a11b0b835f62bbe6a0183f8514e7fd21fd89b"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "02f731463748db57cc2ebfbd9fbc9ce8280d3433"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "16267cf279190ca7c1b30d020758ced95db89cd0"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.5.1"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "601f7e7b3d36f18790e2caf83a882d88e9b71ff1"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.4"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "44dbf560808d49041989b8a96cae4cffbeb7966a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.11"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"

    [deps.CategoricalArrays.extensions]
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStructTypesExt = "StructTypes"

    [deps.CategoricalArrays.weakdeps]
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SentinelArrays = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
    StructTypes = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e0af648f0692ec1691b5d094b8724ba1346281cf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.18.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "05f9816a77231b07e634ab8715ba50e5249d6f76"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "8a62af3e248a8c4bad6b32cbbe663ae02275e32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "8cfa272e8bdedfa88b6aefbbca7c19f1befac519"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "f9d7112bfff8a19a3a4ea4e03a8e6a91fe8456bf"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "5225c965635d8c21168e32a12954675e7bea1151"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.10"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7c302d7a5fec5214eb8a5a4c466dcf7a51fcf169"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.107"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "5b93957f6dcd33fc343044af3d48c215be2562f1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.9.3"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "8e2d86e06ceb4580110d9e716be26658effc5bfd"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "da121cbdc95b065da07fbb93638367737969693f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.8+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "899050ace26649433ef1af25bc17a815b3db52b7"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.9.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5eab648309e2e060198b45820af1a37182de3cce"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HistogramThresholding]]
deps = ["ImageBase", "LinearAlgebra", "MappedArrays"]
git-tree-sha1 = "7194dfbb2f8d945abdaf68fa9480a965d6661e69"
uuid = "2c695a8d-9458-5d45-9878-1b8a99cf7853"
version = "0.3.1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "eb8fed28f4994600e29beef49744639d985a04b2"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.16"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageBinarization]]
deps = ["HistogramThresholding", "ImageCore", "LinearAlgebra", "Polynomials", "Reexport", "Statistics"]
git-tree-sha1 = "f5356e7203c4a9954962e3757c08033f2efe578a"
uuid = "cbc4b850-ae4b-5111-9e64-df94c024a13d"
version = "0.3.0"

[[deps.ImageContrastAdjustment]]
deps = ["ImageBase", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "eb3d4365a10e3f3ecb3b115e9d12db131d28a386"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.12"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "fc5d1d3443a124fde6e92d0260cd9e064eba69f8"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.1"

[[deps.ImageCorners]]
deps = ["ImageCore", "ImageFiltering", "PrecompileTools", "StaticArrays", "StatsBase"]
git-tree-sha1 = "24c52de051293745a9bad7d73497708954562b79"
uuid = "89d5987c-236e-4e32-acd0-25bd6bd87b70"
version = "0.1.3"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "08b0e6354b21ef5dd5e49026028e41831401aca8"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.17"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "PrecompileTools", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "432ae2b430a18c58eb7eca9ef8d0f2db90bc749c"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.8"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "bca20b2f5d00c4fbc192c3212da8fa79f4688009"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.7"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils"]
git-tree-sha1 = "b0b765ff0b4c3ee20ce6740d843be8dfce48487c"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.3.0"

[[deps.ImageMagick_jll]]
deps = ["JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1c0a2295cca535fabaf2029062912591e9b61987"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.10-12+3"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "6f0a801136cb9c229aebea0df296cdcd471dbcd1"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.5"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "PrecompileTools", "Statistics"]
git-tree-sha1 = "783b70725ed326340adf225be4889906c96b8fd1"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.7"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "3ff0ca203501c3eedde3c6fa7fd76b703c336b5f"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.8.2"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "7ec124670cbce8f9f0267ba703396960337e54b5"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.10.0"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageBinarization", "ImageContrastAdjustment", "ImageCore", "ImageCorners", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "d438268ed7a665f8322572be0dabda83634d5f45"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.26.0"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "ea8031dea4aff6bd41f1df8f2fdfb25b33626381"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.4"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "be8e690c3973443bec584db3346ddc904d4884eb"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ad37c091f7d7daf900963171600d7c1c5c3ede32"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalSets]]
deps = ["Dates", "Random"]
git-tree-sha1 = "3d8866c029dd6b16e69e0d4a939c4dfcb98fac47"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.8"
weakdeps = ["Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "PrecompileTools", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "9bbb5130d3b4fa52846546bca4791ecbdfb52730"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.38"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "9fb0b890adab1c0a4a475d4210d51f228bfc250d"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.6"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "d65930fa2bc96b07d7691c652d701dcbe7d9cf0b"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "fee018a29b60733876eb557804b5b109dd3dd8a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.8"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "62edfee3211981241b57ff1cedf4d74d79519277"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.15"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "0f5648fbae0d015e3abe5867bca2b362f67a5894"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.166"

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.LoopVectorization.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "eb006abbd7041c28e0d16260e50a24f8f9104913"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "1130dbe1d5276cb656f6e1094ce97466ed700e5a"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2ac17d29c523ce1cd38e27785a7d23024853a4bb"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.10"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a12e56c72edee3ce6b96667745e6cbbe5498f200"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "5ded86ccaf0647349231ed6c0822c10886d4a1ee"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "db8ec28846dbf846228a32de5a6912c63e2052e3"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.53"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase"]
git-tree-sha1 = "3aa2bb4982e575acd7583f01531f241af077b163"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.2.13"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "6842ce83a836fbbc0cfeca0b5a4de1a4dcbdb8d1"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.8"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "00099623ffee15972c16111bcf84c58a0051257c"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.9.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "da095158bdc8eaccb7890f9884048555ab771019"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays"]
git-tree-sha1 = "0783924e4a332493f72490253ba4e668aeba1d73"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.6.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "3aac6d68c5e57449f5b9b865c9ba50ac2970c4cf"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.42"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "0e7508ff27ba32f26cd459474ca2ede1bc10991f"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "4b33e0e081a825dbfaf314decf58fa47e53d6acb"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.4.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "f295e0a1da4ca425659c57441bcb59abb035a4bc"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.8"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "03fec6800a986d191f64f5c0996b59ed526eda25"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.4.1"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "0adf069a2a490c47273727e029371b31d44b72b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.5"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3b1dcbf62e469a67f6733ae493401e53d92ff543"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "34cc045dd0aaa59b8bbe86c644679bc57f1d5bd0"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.8"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "1fbeaaca45801b4ba17c251dd8603ef24801dd84"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.2"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "a72d22c7e13fe2de562feda8645aa134712a87ee"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.17.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "b182207d4af54ac64cbc71797765068fdeff475d"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.64"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "24b81b59bd35b3c42ab84fa589086e19be919916"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.11.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "47cf33e62e138b920039e8ff9f9841aafe1b733e"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.35.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─569bbd2c-7fae-4e26-afe5-3f4d06f7d505
# ╟─94b9bdb0-73ee-11ee-10e9-e93688ea4523
# ╠═f268582b-0756-41cf-910d-7a57b698451d
# ╠═f075f19a-81b6-47b7-9104-57d2e51e7241
# ╟─1b26468c-b4f7-4252-b891-4f95bc04c869
# ╟─6d3c1725-75fa-412e-9b30-8f8df4e7874b
# ╟─2df8c833-7ca7-4d7a-ade5-0df083a013a1
# ╟─04e344a3-3d5b-449e-9222-481df24015c7
# ╠═974c334e-38fb-436e-842b-bb016854d136
# ╟─5aad6dcc-8400-4a7a-a2bc-69105b32e09f
# ╟─351d01de-7225-40e0-b650-0ef40b1a1cf7
# ╟─7cf4ce88-33de-472c-99c6-5b3ae258f3d1
# ╠═8b2c77d3-f743-4840-a9d9-9308e05be28d
# ╟─0361d281-4a64-4792-812f-7eb9d268d2ae
# ╟─60ce933e-4335-4190-a7b0-5c86d0326a35
# ╠═a9601fe8-6321-4ad7-a950-c7354290aed6
# ╠═3eaf9941-d510-4a91-99bf-2084bbe3ea40
# ╠═1ebb5c9b-df33-4c48-8240-bbff2a06520c
# ╟─70da13b0-6111-4d5d-a6f5-49fcc0499738
# ╟─4fb1d7ad-47f2-4adf-a2ba-0ecc0fc8eeb0
# ╠═41aee571-9016-4759-859a-c99eb143a410
# ╟─13ce2bea-889f-4727-a126-71a5006a86ab
# ╠═0efbcab7-4a58-400a-84af-f1a6703c7ec9
# ╟─a1ee76ab-bf27-466e-8324-01ecde09931c
# ╠═74590e82-5e2b-4a88-afde-3b2992d9b001
# ╟─dc5ab038-569e-4603-95af-0549c6e4ee76
# ╠═90bd4975-315d-4a55-9af4-5b40ceda4eb3
# ╠═93e8f4ee-7558-4178-8f06-96a422528c48
# ╟─05715ba9-fdd3-43c8-b6fc-ee32d225cdf0
# ╠═24fde296-5a6f-4a92-bf16-855df4c99227
# ╠═42094521-c209-4c82-8226-1a0b9b1c0d85
# ╠═b3be394c-5997-4494-ad40-ced2f10fd364
# ╠═d56dc1d3-992a-4be9-87bd-56dae4c41d9c
# ╟─0f204a06-71b2-438a-bb49-4af8ebda0001
# ╠═d5152b67-bc1d-4cc0-b73e-90d79dbadcb4
# ╟─ab5b4c7c-ae24-4aae-a528-1dc427a7f1f1
# ╠═c945050f-3ddc-4d0e-80dd-af909c3f4ab5
# ╟─7e2fd675-28ba-4412-9c56-4b40b3380576
# ╟─89045ff9-bfb2-43e7-865b-235181cdf9f7
# ╠═03bf6a75-a98c-4641-9939-2336c78e1be7
# ╟─a58ec747-09cb-4cba-a9f0-4de683c80052
# ╟─c66bac82-feaf-4e77-ab1b-ea7a2a5cf6c7
# ╟─b385a6b2-e2b0-4179-ac81-21a8600f86cf
# ╟─7f8b5fe8-f5d1-46eb-a30e-8f0a6e9707bc
# ╟─2e7b7b97-f4b9-4ef8-b360-e086ffc0a025
# ╟─dc37fa99-ceb5-40cb-846a-6cdf9d33c2f3
# ╠═7e8e31b9-15fd-4d34-b447-4440ea805811
# ╠═044b6cab-06cf-405e-864c-3e040faa602d
# ╠═c5949c51-d3f5-40b1-9415-7c40ae596b1b
# ╠═277d0ee8-bb02-4c5f-a6cb-9c8c01795b65
# ╠═4fddfc1c-4943-40df-981b-8609c49fa6bb
# ╠═e01d84fc-cb9e-495c-a346-c88d5d9428fc
# ╠═ab3579f2-21b8-4a95-998b-ff940c9c3677
# ╠═c104b355-2a89-4d8c-8b38-8ad504cbde18
# ╠═e3c4daa5-69b7-465a-9e34-7c1a050a5f29
# ╠═c2354103-b70f-4371-9d71-cc7f8ec69b3d
# ╠═01d77625-dc7e-4550-a9fc-ab2961606f4d
# ╟─cfb78bf5-9f06-4557-a777-c34d908f0e67
# ╠═cb4a302c-fb04-4362-95d5-7680d8fb2983
# ╟─30220b96-8154-4ca2-a016-9258860323a5
# ╠═df519afa-309a-4633-860d-2fe40a384fa9
# ╠═3128d6eb-375d-4770-8215-6ed7e3ac5b5a
# ╠═103063b6-c5a9-4c5d-829b-4587813bfaf4
# ╟─c7590419-c3d0-41f7-8777-227fcc7b1ba8
# ╠═c1b3e29b-51b6-4bbd-8793-ece13bfb5a70
# ╟─f12c1925-cf29-41e5-9499-87efe1a96528
# ╠═a41cab07-f8f1-4b24-bd15-5fd29651fe36
# ╟─695bd289-a4b3-4154-ab49-6229de133164
# ╠═538694ea-a0b1-4f09-b7b4-4af64d7ec10b
# ╟─a0239801-eb9a-4648-b164-15013fc3c445
# ╠═068f3e73-ed23-4210-b739-ccaadc9f2ba8
# ╠═5c8e3e0c-3344-431e-a370-625b467e2ea9
# ╠═b9cf9e47-d8da-4416-b67d-7c806595ceb9
# ╠═5d722b85-4e5b-4dc3-9b8d-eb3b6cf33941
# ╠═8bdffe16-02af-46bd-a568-0e238cc76b6c
# ╠═a48e9529-bc20-4b46-a861-d6ae09e41d2c
# ╠═4e19b5fa-984c-4dd0-a956-6371b53f58f6
# ╠═8886f394-ff90-4fbd-8dec-874f6a4ded83
# ╟─d9556b99-ac13-4278-8fc2-085728a2cfa9
# ╠═543dea77-0aea-49e6-811a-5a87218d6632
# ╠═0489cd43-a451-435e-9024-7b9ff3432761
# ╟─1c6b09bb-9809-411d-8ddd-2095256d0601
# ╠═2a66eee4-be06-43a2-a9be-fc2e0c4a0f32
# ╟─a86d30c5-03f0-4e11-ba25-1f08ba0998b7
# ╠═ea5f58f9-5394-4536-9458-0484e85fdc85
# ╟─cca0ed73-b1e7-4895-8537-294ccb3f9e26
# ╠═289205ad-0bd3-473c-b076-fab42e1643c3
# ╟─fc37abf6-6acd-4b11-bdab-ddee379d8d72
# ╠═29f4022e-28dc-4ef8-8e83-11d466437813
# ╟─a1f13b96-fbd3-40ab-aa3f-9af14d55ed55
# ╠═ef66b620-cfa3-47b4-bc2b-6cc77427764f
# ╟─010e2c3b-f66a-411b-b819-3d37448c4087
# ╠═8d0b7517-1be8-41c4-8a4b-716bcad169fb
# ╟─19deee02-5fb7-400c-a853-74bd44a8deaf
# ╠═608b2a67-b34b-4440-9282-3f225e5714be
# ╟─e370f645-d408-4d7f-8c48-0f9e05522b5f
# ╠═72270f78-01c5-4ce5-aac4-901fd9a5143f
# ╠═783e9476-a0a8-4427-b929-9f7c0faa6b59
# ╠═a59cefd2-27c4-4332-943e-1fbd79ae2481
# ╟─38345378-66ee-42c1-b37f-6691119ecc60
# ╠═3273792c-41fb-4225-a4f7-2f1c9d58be4a
# ╟─144e23e9-ce3d-4ed6-be2c-dff1fed39e59
# ╠═f7521761-e9f1-43df-a95c-57aec7c83011
# ╟─77697cb7-40fa-4ed4-9008-8d78cfa0c247
# ╠═7f60e96f-9a5e-41f5-a388-f531585e15b0
# ╠═96dea7cc-4742-460f-a18e-ae22f0c92033
# ╠═e4bb69eb-e608-4f50-9c42-678a74b99192
# ╠═010e992a-f35a-4a7a-946e-f796aba41a32
# ╠═cdbe66ff-d643-4b7a-a446-85c284c668ba
# ╟─6b34bdad-2518-43f5-9fb0-d28a99a411fe
# ╟─627c3c50-b22f-4e95-a755-26f2197fff92
# ╟─bebbf9b7-0d4a-44d0-baa1-aba99b9c59ef
# ╠═b1953875-92ca-42d1-a22e-2f393141ddbe
# ╟─6a1ce8de-49b4-4a97-aa32-1cd20ded4b04
# ╠═adf3dbe5-ade0-4949-9507-10b5c8164ddd
# ╟─b8a96a22-4950-4300-8c28-2c8aedf6b66b
# ╠═081c009c-0871-458e-9081-365d5102fefd
# ╟─14d1a55a-43b0-4857-8dd9-b10c86f8a123
# ╠═512e1060-eee5-4374-966c-02d7fb62f303
# ╠═61474944-c347-448a-beb9-aa2e4ef6331e
# ╠═af255535-e903-4eef-8629-13d836f1f145
# ╠═13257859-e48e-4aa3-a3a3-a4ecf4c8dd1f
# ╠═9e40d379-50e9-4d33-a53c-6c5089059de5
# ╠═df08506c-f66d-430a-b235-4c9dfb80d414
# ╠═ecba3acb-6bc1-4722-9cee-a388a2442fae
# ╠═87c627a8-2c54-44ff-aa66-d9b5c379f646
# ╟─c848d2cf-5d36-4437-b53a-e278150e75ef
# ╟─1007c151-b6ab-4f81-8421-4746dc3b67f4
# ╟─a32e4478-585f-431e-b4f0-bb6b92010cde
# ╠═5ed1e95e-bf40-4c96-8434-ac83e54e9362
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
