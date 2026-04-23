function getpreset(preset::Symbol)
    if preset == :med
        # frequencies
        f0 = 5e9        # central frequency
        B = 1e9         # bandwidth
        df = 40e6       # frequency resolution
        λ = 3e8/f0

        # recievers
        r = zeros(3, 31)
        r[1, :] .= -14.0
        r[2, :] .= 0 .+ λ .* (-15 : 15)

        # scatterers
        ξ = zeros(3, 1500)
        ξ[1,:] .= -12.0 .+ 10.0*rand(Float64, size(ξ, 2))
        ξ[2,:] .= -5.0 .+ 10.0*rand(Float64, size(ξ, 2))

        # scattering strength
        τ = fill(0.3, size(ξ, 2))

        # imaging window
        iwx = 3e-1*λ .* (-90 : 90)
        iwy = 2e-1*λ .* (-90 : 90)
        z = zeros(3, length(iwx) * length(iwy))
        for (i, (y, x)) in enumerate(Iterators.product(iwy, iwx))
            z[1,i] = x
            z[2,i] = y
        end
    elseif preset == :tiny
        # frequencies
        f0 = 5e9        # central frequency
        B = 0.0         # bandwidth
        df = 40e6       # frequency resolution
        λ = 3e8/f0

        # recievers
        r = zeros(3, 31)
        r[1, :] .= -14.0
        r[2, :] .= 0 .+ λ .* (-15 : 15)

        # scatterers
        ξ = zeros(3, 1500)
        ξ[1,:] .= -12.0 .+ 10.0*rand(Float64, size(ξ, 2))
        ξ[2,:] .= -5.0 .+ 10.0*rand(Float64, size(ξ, 2))

        # scattering strength
        τ = fill(0.3, size(ξ, 2))

        # imaging window
        iwx = 2.0*λ .* (-6 : 6)
        iwy = 2.0/3.0*λ .* (-6 : 6)
        z = zeros(3, length(iwx) * length(iwy))
        for (i, (y, x)) in enumerate(Iterators.product(iwy, iwx))
            z[1,i] = x
            z[2,i] = y
        end
    end

    return f0, B, df, λ, r, ξ, τ, iwx, iwy, z
end