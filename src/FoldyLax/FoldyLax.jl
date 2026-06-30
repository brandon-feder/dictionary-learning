@doc raw"""
    struct FoldyLaxStruct{T <: Real}

# Properties
* `G::StridedMatrix{T}`: Array of size `(p*n, k)` containing the matrix ``\mathcal{G}`` described above.

Stores parameters for and results of computation. Importantly,
    after calling `foldylax!`, the property `G` will contain
    the matrix ``\mathcal{G}`` described above.

    function FoldyLaxStruct(src, sct, rec, frq, τ) where T

# Arguments
* `src::StridedMatrix{T}`: Array of size `(d, k)` where each column is the coordinate of a source.
* `sct::StridedMatrix{T}`: Array of size `(d, m)` where each column is the coordinate of a scatterer
* `rec::StridedMatrix{T}`: Array of size `(d, n)` where each column is the coordinate of a reciever
* `frq::StridedVector{T}`: Vector of length `p` where each element specifying the frequencies
* `τ::StridedMatrix{T}`: Vector of length `m` where each element stores the scattering strength of the corresponding scatterer
"""
struct FoldyLaxStruct{T <: Real}
    back::Backend
    m::Int # how many scatterers
    n::Int # how many recievers
    k::Int # how many sources
    p::Int # number of frequencies
    d::Int # dimension of problem
    src::StridedMatrix{T} # sources
    sct::StridedMatrix{T} # scatterers
    rec::StridedMatrix{T} # recievers
    frq::StridedVector{T} # frequencies
    τ::StridedVector{T} # scatterer strengths
    G::StridedMatrix{Complex{T}} # Green's matrix
    Ghom::StridedMatrix{Complex{T}} # Homogenous part of Green's matrix

    function FoldyLaxStruct(
        back::Backend, m::Int, n::Int, k::Int, p::Int,
        d::Int, src::StridedMatrix{T}, sct::StridedMatrix{T}, 
        rec::StridedMatrix{T}, frq::StridedVector{T}, 
        τ::StridedVector{T}, G::StridedMatrix{Complex{T}},
        Ghom::StridedMatrix{Complex{T}}
    ) where T
        @assert get_backend(src) == back
        @assert backsagree(src, sct, rec, frq, τ, G)
        @assert size(src) == (d, k)
        @assert size(sct) == (d, m)
        @assert size(rec) == (d, n)
        @assert length(τ) == m
        @assert length(frq) == p
        @assert size(G) == (p*n, k)
        @assert size(Ghom) == (p*n, k)

        return new{T}(back, m, n, k, p, d, src, sct, rec, frq, τ, G, Ghom)
    end
end

@doc raw"""
    struct FoldyLaxWorkStruct{T <: Real}

Stores scratchspace required by `foldylax!`. This can be reused by subsequency calls `foldylax!` so long the geometry specified in the corresponding `FoldyLaxStruct{T}` has the same dimensions.

    function FoldyLaxWorkStruct(fls)

Create an instance of `FoldyLaxWorkStruct` that is compatible with `fls`

# Arguments
* `fls::FoldyLaxStruct{T}`
"""
struct FoldyLaxWorkStruct{T <: Real}
    Mξξ::StridedArray{Complex{T}}
    Mξz::StridedArray{Complex{T}}
    Mrz::StridedArray{Complex{T}}
    Mrξadj::StridedArray{Complex{T}}
    Mξξfac::Ref{StridedVector{<:Factorization}}
    work::StridedArray{Complex{T}}

    function FoldyLaxWorkStruct(
        Mξξ::StridedArray{Complex{T}}, Mξz::StridedArray{Complex{T}},
        Mrz::StridedArray{Complex{T}}, Mrξadj::StridedArray{Complex{T}},
        Mξξfac::Ref{StridedVector{<:Factorization}}, work::StridedArray{Complex{T}}
    ) where T
        m = size(Mξξ, 1)
        n, k, p = size(Mrz)
        @assert backsagree(Mξξ, Mξz, Mrz, Mrξadj, work)
        @assert size(Mξξ) == (m, m, p)
        @assert size(Mξz) == (m, k, p)
        @assert size(Mrξadj) == (m, n, p)
        @assert length(work) >= max(m*m, m*k, n*k, n*m)

        return new{T}(Mξξ, Mξz, Mrz, Mrξadj, Mξξfac, work)
    end
end

function FoldyLaxStruct(
    src::StridedMatrix{T}, sct::StridedMatrix{T}, 
    rec::StridedMatrix{T}, frq::StridedVector{T},
    τ::StridedVector{T}
) where T
    back = get_backend(src)
    d, k = size(src)
    m = size(sct, 2)
    n = size(rec, 2)
    p = length(frq)

    G = adapt(back, Matrix{Complex{T}}(undef, p*n, k))
    Ghom = adapt(back, Matrix{Complex{T}}(undef, p*n, k))

    return FoldyLaxStruct(back, m, n, k, p, d, src, sct, rec, frq, τ, G, Ghom)
end

function FoldyLaxWorkStruct(
    fls::FoldyLaxStruct{T}
) where T <: Real
    back, m, n, k, p = fls.back, fls.m, fls.n, fls.k, fls.p

    Mξξ = adapt(back, Array{Complex{T}}(undef, m, m, p))
    Mξz = adapt(back, Array{Complex{T}}(undef, m, k, p))
    Mrz = adapt(back, Array{Complex{T}}(undef, n, k, p))
    Mrξadj = adapt(back, Array{Complex{T}}(undef, m, n, p))
    Mξξfac = adapt(back, Ref{StridedVector{<:Factorization}}())
    work = adapt(back, Vector{Complex{T}}(undef, max(m*m, m*k, n*k, n*m)))

    return FoldyLaxWorkStruct(Mξξ, Mξz, Mrz, Mrξadj, Mξξfac, work)
end

@doc raw"""
    function foldylax!(fls, flws) where T

Overwrites `fls.G` with the matrix ``\mathcal{G}`` described above.

# Arguments
* `fls::FoldyLaxStruct{T}`
* `flws::FoldyLaxWorkStruct{T}`
"""
function foldylax!(
    fls::FoldyLaxStruct{T}, flws::FoldyLaxWorkStruct{T}
) where T
    function dsts!(
        ::Val{N}, W::StridedMatrix{Complex{T}},
        A::StridedArray{T}, B::StridedArray{T},
    ) where {N,T<:Real}
        map!(W, CartesianIndices(W)) do idx
            i, j = Tuple(idx)
            s = zero(T)
            @inbounds @simd for k in 1:N
                s += (A[k,i] - B[k,j])^2
            end
            sqrt(s)
        end

        return W
    end

    (;m, n, k, p, d, src, sct, rec, frq, τ, G, Ghom) = fls
    (;Mξξ, Mξz, Mrz, Mrξadj, Mξξfac, work) = flws

    wns = reshape(frq .* (2π / 3e8), 1, 1, p)
    wst = WorkStackTrack(work)
    
    # solve for Mrz
    withwork(wst, n, k) do W
        dsts!(Val(d), W, rec, src)
        Mrz .= reshape(W, n, k, 1) .* wns
        Mrz .= exp.(1im .* Mrz) ./ (4π .* W)
    end

    # deal with edge case
    if m == 0
        for s in 1:p
            G_ = view(G, s:p:p*n, :)
            Ghom_ = view(Ghom, s:p:p*n, :)
            G_ .= view(Mrz, :, :, s)
            Ghom_ .= view(Mrz, :, :, s)
        end
        return fls
    end 

    τ_ = reshape(τ, 1, m, 1)
    τ_adj = permutedims(τ_, (2,1,3))

    # solve for Mξξ; set diagonals to -1
    withwork(wst, m, m) do W
        dsts!(Val(d), W, sct, sct)
        Mξξ .= reshape(W, m, m, 1) .* wns
        Mξξ .= exp.(1im .* Mξξ) ./ (4π .* W) .* τ_
        view(reshape(Mξξ, m^2, :), 1:(m+1):m*m, :) .= -1
    end

    # solve for Mξz
    withwork(wst, m, k) do W
        dsts!(Val(d), W, sct, src)
        Mξz .= reshape(W, m, k, 1) .* wns
        Mξz .= exp.(1im .* Mξz) ./ (4π .* W)
    end

    # solve for Mrξadj, adjoint of Mrξ
    withwork(wst, m, n) do W
        dsts!(Val(d), W, sct, rec)
        Mrξadj .= W .* wns
        Mrξadj .= exp.(1im .* Mrξadj) ./ (4π .* W) .* τ_adj
    end
    
    # compute factorizations
    Mξξfac[] = [qr!(view(Mξξ, :, :, s)) for s in 1:p]

    for s in 1:p
        # get current slice
        Mξz_ = view(Mξz, :, :, s)
        Mξξ_ = Mξξfac[][s]
        Mrz_ = view(Mrz, :, :, s)
        Mrξadj_ = view(Mrξadj, :, :, s)
        G_ = view(G, s:p:p*n, :)
        Ghom_ = view(Ghom, s:p:p*n, :)

        ldiv!(adjoint(Mξξ_), Mrξadj_)
        Mrξadj_ = adjoint(Mrξadj_)

        G_ .= Mrz_ .- Mrξadj_ * Mξz_
        Ghom_ .= Mrz_
    end

    return fls
end

function foldylax_update!(
    fls::FoldyLaxStruct{T}, flws::FoldyLaxWorkStruct{T}
) where T
    function dsts!(
        ::Val{N}, W::StridedMatrix{Complex{T}},
        A::StridedArray{T}, B::StridedArray{T},
    ) where {N,T<:Real}
        map!(W, CartesianIndices(W)) do idx
            i, j = Tuple(idx)
            s = zero(T)
            @inbounds @simd for k in 1:N
                s += (A[k,i] - B[k,j])^2
            end
            sqrt(s)
        end

        return W
    end

    (;m, n, k, p, d, src, rec, sct, frq, τ, G, Ghom) = fls
    (;Mξξ, Mξz, Mrz, Mrξadj, Mξξfac, work) = flws

    wns = reshape(frq .* (2π / 3e8), 1, 1, p)
    wst = WorkStackTrack(work)
    
    # solve for Mrz
    withwork(wst, n, k) do W
        dsts!(Val(d), W, rec, src)
        Mrz .= reshape(W, n, k, 1) .* wns
        Mrz .= exp.(1im .* Mrz) ./ (4π .* W)
    end

    # deal with edge case
    if m == 0
        for s in 1:p
            G_ = view(G, s:p:p*n, :)
            Ghom_ = view(Ghom, s:p:p*n, :)
            G_ .= view(Mrz, :, :, s)
            Ghom_ .= view(Mrz, :, :, s)
        end
        return fls
    end 

    τ_ = reshape(τ, 1, m, 1)
    τ_adj = permutedims(τ_, (2,1,3))

    # solve for Mξz
    withwork(wst, m, k) do W
        dsts!(Val(d), W, sct, src)
        Mξz .= reshape(W, m, k, 1) .* wns
        Mξz .= exp.(1im .* Mξz) ./ (4π .* W)
    end

    # solve for Mrξadj, adjoint of Mrξ
    withwork(wst, m, n) do W
        dsts!(Val(d), W, sct, rec)
        Mrξadj .= W .* wns
        Mrξadj .= exp.(1im .* Mrξadj) ./ (4π .* W) .* τ_adj
    end

    for s in 1:p
        # get current slice
        Mξz_ = view(Mξz, :, :, s)
        Mξξ_ = Mξξfac[][s]
        Mrz_ = view(Mrz, :, :, s)
        Mrξadj_ = view(Mrξadj, :, :, s)
        G_ = view(G, s:p:p*n, :)
        Ghom_ = view(Ghom, s:p:p*n, :)

        ldiv!(adjoint(Mξξ_), Mrξadj_)
        Mrξadj_ = adjoint(Mrξadj_)

        G_ .= Mrz_ .- Mrξadj_ * Mξz_
        Ghom_ .= Mrz_
    end

    return fls
end