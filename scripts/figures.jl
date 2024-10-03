# This script is intended to generate pngs of the actual simulation setup. 
# These pngs are than fed to blender as to create rendered scense of the initial condition. 
using Colors, Images, FileIO

"""
    torus(lx, ly, r₁, R₂, θ, center, hmin = 0.05; noise=0.0)

Creates a ring-rivulet like fluid state
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
    pattern(arrested, gradient, Lx, Ly, R, rr, angle)

Creates a wettability pattern
"""
function pattern(arrested, gradient, Lx, Ly, R, rr)
    theta = zeros(Lx, Ly)
    # Band structure
    if arrested[1]
        mask = torus(Lx, Ly, rr, R, arrested[2] + 1/36, (Lx÷2, Ly÷2))
        for i in eachindex(mask)
            if mask[i] > 0.0505
                theta[i] = arrested[2]
            else
                theta[i] = 1
            end
        end
    # Wettability gradient that radial (de/in)creases the contact angle
    elseif gradient[1]
        dist = zeros(Lx, Ly)
        for i in 1:Lx
            for j in 1:Ly
                dist[i,j] = round(Int, sqrt((i - Lx÷2)^2 + (j - Ly÷2)^2))
            end
        end
        # All values will be multiplied with pi inside the pressure calculation
        theta .= (gradient[3]-gradient[2])/R .* dist .+ gradient[2]
        theta[dist .> R] .= gradient[3]
    end
    return theta
end 


function fluid_state_image()
    ring = torus(512,512,80,150,1/9,(256,256))
    rnorm = ring ./ maximum(ring)
    img_ring = Gray.(rnorm)
    save("ring.png", img_ring)
end

function band_pattern_image()
    arr = (true, 1/9)
    contactangles = pattern(arr, (false, 1/9, 1), 512, 512, 150, 80)
    cnorm = contactangles ./ maximum(contactangles)
    img_band = Gray.(cnorm)
    save("band_pattern.png", img_band)
end

function gradient_image(which)
    if which == "negative"
        gradients_neg = (true, 1/9, 1)
        contactangles = pattern((false, 1), gradients_neg, 512, 512, 150, 80)
        cnorm = contactangles ./ maximum(contactangles)
        img_neggrad = Gray.(cnorm)
        save("grad_neg.png", img_neggrad)
    elseif which == "positive"
        gradients_pos = (true, 1, 1/9)
        contactangles = pattern((false, 1), gradients_pos, 512, 512, 150, 80)
        cnorm = contactangles ./ maximum(contactangles)
        img_posgrad = Gray.(cnorm)
        save("grad_pos.png", img_posgrad)
    end
end