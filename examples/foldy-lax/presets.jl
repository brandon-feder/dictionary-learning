function preset1a()
    f0 = 5e9
    frq = f0 .+ 4e7 .* (-12 : 12)
    λ = 3e8/f0 # central wavelength

    # recievers
    rec = zeros(2, 31)
    rec[1, :] .= -14.0
    rec[2, :] .= λ .* (-15 : 15)

    # scatterers
    sct = zeros(2, 0)

    # scattering strength
    τ = zeros(0)

    # imaging window
    iwx = 0.166 * λ .* (-90 : 90)
    iwy = 0.05 * λ .* (-90 : 90)
    src = zeros(2, length(iwx) * length(iwy))
    for (i, (y, x)) in enumerate(Iterators.product(iwy, iwx))
        src[1,i] = x
        src[2,i] = y
    end

    return frq, iwx, iwy, src, sct, rec, τ
end

function preset1b()
    f0 = 5e9
    frq = f0 .+ 4e7 .* (-12 : 12)
    λ = 3e8/f0 # central wavelength

    # recievers
    rec = zeros(2, 31)
    rec[1, :] .= -14.0
    rec[2, :] .= λ .* (-15 : 15)

    # scatterers
    sctx = LinRange(-12.0, -2.0, 30)
    scty = LinRange(-5.0, 5.0, 30)
    sct = zeros(2, length(sctx)*length(scty))
    sct .= Iterators.product(sctx, scty) |> collect |> vec |> stack
    sct .+= 0.1*randn(2, size(sct,2))

    # scattering strength
    τ = fill(0.6, size(sct, 2))

    # imaging window
    iwx = 0.166 * λ .* (-90 : 90)
    iwy = 0.05 * λ .* (-90 : 90)
    src = zeros(2, length(iwx) * length(iwy))
    for (i, (y, x)) in enumerate(Iterators.product(iwy, iwx))
        src[1,i] = x
        src[2,i] = y
    end

    return frq, iwx, iwy, src, sct, rec, τ
end

function preset2()
    f0 = 5e9
    frq = f0 .+ 4e7 .* (-12 : 12)
    λ = 3e8/f0 # central wavelength

    # recievers
    rec = zeros(2, 31)
    rec[1, :] .= -14.0
    rec[2, :] .= λ .* (-15 : 15)

    # scatterers
    sctx = LinRange(-12.0, -2.0, 30)
    scty = LinRange(-5.0, 5.0, 30)
    sct = zeros(2, length(sctx)*length(scty))
    sct .= Iterators.product(sctx, scty) |> collect |> vec |> stack
    sct .+= 0.1*randn(2, size(sct,2))

    # scattering strength
    τ = fill(0.6, size(sct, 2))

    # imaging window
    iwx = 1.66 * λ .* (-9 : 9)
    iwy = 0.5 * λ .* (-9 : 9)
    src = zeros(2, length(iwx) * length(iwy))
    for (i, (y, x)) in enumerate(Iterators.product(iwy, iwx))
        src[1,i] = x
        src[2,i] = y
    end

    return frq, iwx, iwy, src, sct, rec, τ
end