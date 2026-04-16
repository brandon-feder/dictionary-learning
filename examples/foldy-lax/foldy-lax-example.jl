using LinearAlgebra
using Base.Iterators
import DictionaryLearning.FoldyLax as FL
using Plots
using Base.Threads
using CUDA, Adapt

include("../debug.jl")

"""
    compute_psf(G)

Computes the Point-Spread Function (PSF) for the central pixel given 
the sensing matrix G (receivers × pixels × frequencies).
"""
function compute_psf(G::AbstractArray{Complex{T}, 3}) where T
    n_rx, n_pixels, n_freq = size(G)
    
    # 1. Identify the index of the central pixel
    mid_k = div(n_pixels, 2) + 1
    
    # 2. Extract the "true" signal that would be recorded 
    # if only the central pixel had a source of unit amplitude.
    # Shape: (n_rx, n_freq)
    d_true = G[:, mid_k, :]
    
    # 3. Apply the Kirchhoff Migration (KM) imaging operator[cite: 742].
    # For each pixel 'i', we compute the correlation with the target signal:
    # I(i) = | Σ_ω Σ_r conj(G[r, i, ω]) * d_true[r, ω] |
    # This is equivalent to G' * d_true summed over frequencies.
    
    # Efficient broadcasting on GPU to compute the complex image
    # We sum over receivers (dim 1) and frequencies (dim 3)
    psf_complex = dropdims(sum(conj(G) .* reshape(d_true, n_rx, 1, n_freq), dims=(1, 3)), dims=(1, 3))
    
    # Return the magnitude (normalized)
    psf = abs.(psf_complex)
    return psf ./ maximum(psf)
end

function main()
    BLAS.set_num_threads(nthreads())
    println("$INF # threads: $(nthreads())")
    println("$INF # BLAS threads: $(BLAS.get_num_threads())")

    # ================ SETTINGS =================

    p = 30 # number of frequencies
    f0s = range(4e9, 6e9, length=p) # frequencies
    λs = 3e8 ./ f0s # wavelengths
    wns = 2π ./ λs # wave numbers
    
    back = CUDABackend() # which backend to use

    println("$INF min freq. = $(minimum(f0s))")
    println("$INF max freq. = $(maximum(f0s))")
    println("$INF p = $p")

    # ============= SETUP GEOMETRY =============
    
    # recievers
    n = 10 # number of recievers
    r = zeros(3, n)
    r[2, :] = range(-2.0, 2.0, length=n)

    println("$INF n = $n")

    # scatterers
    m = 1000 # how many scatterers
    ξ = zeros(3, m)
    ξ[1,:] .= 2.0 .+ 4.0*rand(Float64, m)
    ξ[2,:] .= -3.0 .+ 6.0*rand(Float64, m)
    τ = fill(0.1, m)
    
    println("$INF m = $m")

    # imaging window
    iwx = range(4.5, 5.5, length=100)
    iwy = range(-0.2, 0.2, length=100)
    kx, ky = length(iwx), length(iwy)
    k = kx*ky
    z = zeros(3, k)
    for (i, (y, x)) in enumerate(Iterators.product(iwy, iwx))
        z[1,i] = x
        z[2,i] = y
    end

    println("$INF k = $k")


    # ============= SOLVE =============
    r = adapt(back, r)
    ξ = adapt(back, ξ)
    τ = adapt(back, τ)
    z = adapt(back, z)
    wns = adapt(back, wns)
    Mscat = adapt(back, Array{ComplexF64}(undef, m, m, p))
    Gscat = adapt(back, Array{ComplexF64}(undef, n, k, p))
    Ghom = adapt(back, Array{ComplexF64}(undef, n, k, p))

    nws = 2*m*k + 2*n*m + 2*m^2 + max(3*m*k, m*n, n*k)
    work = adapt(back, Vector{Float64}(undef, nws))

    # get scatterer system
    println("$STS computing scatterer system")
    elap = @elapsed begin
        FL.compM!(Mscat, ξ, wns, work, τ)
    end
    println("$TAB$INF elapsed: $elap")

    # prefactorize `Mscat` in-place
    Mscat_fac = [lu!(view(Mscat, :, :, t)) for t in 1:p]

    # get sensing matrix with scatterers
    println("$STS computing sensing matrix w/t scatterers")
    elap = @elapsed begin
        FL.compG!(Gscat, Mscat, z, ξ, r, τ, wns, work)
    end
    println("$TAB$INF elapsed: $elap")

    # get homogenous sensing matrix
    println("$STS computing homogenous sensing matrix")
    elap = @elapsed begin
        FL.compGHom!(Ghom, z, r, wns, work)
    end
    println("$TAB$INF elapsed: $elap")

    # Compute PSF for scattering case
    psf_scat = compute_psf(Gscat)
    
    # Compute PSF for homogeneous case
    psf_hom = compute_psf(Ghom)

    # Reshape for plotting (using kx, ky from your setup)
    psf_scat_img = reshape(Array(psf_scat), ky, kx)
    psf_hom_img = reshape(Array(psf_hom), ky, kx)

    p1 = heatmap(iwx, iwy, psf_scat_img, title="PSF (Scattering)", xlabel="Range", ylabel="Cross-range")
    p2 = heatmap(iwx, iwy, psf_hom_img, title="PSF (Homogeneous)", xlabel="Range", ylabel="Cross-range")
    plot(p1, p2, layout=(1,2), size=(900, 400))
    savefig("/home/bfeder/DictionaryLearning/temp/flplot.png")
end

main()