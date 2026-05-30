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

    function FoldyLaxStruct(
        back::Backend, m::Int, n::Int, k::Int, p::Int,
        d::Int, src::StridedMatrix{T}, sct::StridedMatrix{T}, 
        rec::StridedMatrix{T}, frq::StridedVector{T}, 
        τ::StridedVector{T}, G::StridedMatrix{Complex{T}}
    ) where T
        @assert get_backend(src) == back
        @assert backsagree(src, sct, rec, frq, τ, G)
        @assert size(src) == (d, k)
        @assert size(sct) == (d, m)
        @assert size(rec) == (d, n)
        @assert length(τ) == m
        @assert length(frq) == p
        @assert size(G) == (p*n, k)

        return new{T}(back, m, n, k, p, d, src, sct, rec, frq, τ, G)
    end
end

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
) where T <: Real
    back = get_backend(src)
    d, k = size(src)
    m = size(sct, 2)
    n = size(rec, 2)
    p = length(frq)

    G = adapt(back, Matrix{Complex{T}}(undef, p*n, k))

    return FoldyLaxStruct(back, m, n, k, p, d, src, sct, rec, frq, τ, G)
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