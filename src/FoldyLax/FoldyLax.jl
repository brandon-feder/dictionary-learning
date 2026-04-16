module FoldyLax

using LinearAlgebra
using FLoops
using KernelAbstractions, Adapt, GPUArrays
using Base.Broadcast: broadcasted, materialize

include("../Utils.jl")

"""
Efficiently fill the matrix `M` such that `M[i,j] = g0(A[i] - B[j])*τ[j]`
or simply `M[i,j] = g0(A[i] - B[j])` if `τ` is `nothing`.

# Arguments
`M::AbstractMatrix` - `m×n` complex matrix which will be overwritten with linear system
`ξ::AbstractMatrix` - `3×m` matrix where each column is the coordinate
    of a scatterer
`τ`::AbstractVector - length `m` vector containg scatterer strengths 
`wns::Real` - The wave numbers
`work::AbstractArray` - at least `m x n` work space
"""
function compM!(
    M::AbstractArray{Complex{T}}, A::AbstractMatrix{T}, 
    B::AbstractMatrix{T}, wns::AbstractVector{T}, 
    work::AbstractArray{T}, 
    τ::Union{AbstractVector{T}, Nothing}=nothing
) where T <: Union{Float32, Float64}
    m, n, p = size(M)
    
    # make sure every is correct size
    @assert size(M) == (m, n, p)
    @assert size(A) == (3, m)
    @assert size(B) == (3, n)
    @assert length(wns) == p
    @assert length(work) >= m*n
    if isnothing(τ)
        @assert backsagree(M, A, B, wns, work)
    else
        @assert length(τ) == n
        @assert backsagree(M, A, B, wns, work, τ)
    end

    # get work space
    W = reshape(view(work, 1:m*n), m, n)

    # compute norms of all pairs
    map!(W, CartesianIndices(W)) do idx
        @inbounds begin
            i, j = idx[1], idx[2]
            d1 = A[1, i] - B[1, j]
            d2 = A[2, i] - B[2, j]
            d3 = A[3, i] - B[3, j]
        end
        return sqrt(d1^2 + d2^2 + d3^2)
    end

    # compute free space propogation
    M .= reshape(W, m, n, 1) .* reshape(wns, 1, 1, p)
    M .= exp.(1im .* M) ./ reshape((4π .* W), m, n)

    if !isnothing(τ)
        # scale by strength
        M .*= reshape(τ, 1, n, 1)
    end
end

function compM!(
    M::AbstractArray{Complex{T}}, A::AbstractMatrix{T}, 
    wns::AbstractVector{T}, work::AbstractArray{T},
    τ::Union{AbstractVector{T}, Nothing}=nothing
) where T <: Union{Float32, Float64}
    compM!(M, A, A, wns, work, τ)
    
    # correct diagonals
    m = size(A, 2)
    Mflat = reshape(M, m^2, :)
    @views Mflat[1:(m+1):m*m, :] .= -1
end

function compG!(
    G::AbstractArray{Complex{T}}, Mξξ_facs, 
    z::AbstractMatrix{T}, ξ::AbstractMatrix{T},  r::AbstractMatrix{T},
    τ::AbstractVector{T}, wns::AbstractVector{T}, work::AbstractArray{T}
) where T <: Union{Float32, Float64}
    n = size(r, 2)
    k = size(z, 2)
    m = size(ξ, 2)
    p = length(wns)

    # check arguments
    # @assert size(M) == (m, m, p)
    @assert size(G) == (n, k, p)
    @assert size(ξ) == (3, m)
    @assert length(τ) == m
    @assert length(wns) == p
    @assert length(work) >= 2*m*k + 2*n*m + 
        2*m^2 + max(3*m*k, m*n, n*k)

    # make sure backends agree
    @assert backsagree(G, z, ξ, r, τ, wns, work)

    # manage work space
    wst = WorkStackTrack{T}(work)

    for t in 1:p
        Ge = reinterpret(ComplexF64, pop!(wst, 2*m, k))
        Mrξ = reinterpret(ComplexF64, pop!(wst, 2*n, m))
        Mξz = reinterpret(ComplexF64, pop!(wst, 2*m, k))
        # Mξξqr = reinterpret(ComplexF64, pop!(wst, 2*m, m))

        # fill Mξz
        work2 = pop!(wst, m, k)
        compM!(reshape(Mξz, size(Mξz)..., 1), ξ, z, view(wns, t:t), work2)
        push!(wst, m, k)

        # copyto!(Mξξqr, view(M, :, :, t))
        # Mξξqr .*= -1

        ldiv!(Ge, Mξξ_facs[t], Mξz)

        # return space for Mξξqr
        # push!(wst, 2*m, m)

        # return workspace for Mξz
        push!(wst, 2*m, k)

        work2 = pop!(wst, n, m)
        compM!(reshape(Mrξ, size(Mrξ)..., 1), r, ξ, view(wns, t:t), work2)
        push!(wst, n, m)

        # compute total field
        work2 = pop!(wst, n, k)
        compM!(view(G, :, :, t:t), r, z, view(wns, t:t), work2)
        push!(wst, n, k)

        view(G, :, :, t:t) .+= Mrξ * Ge

        # return workspace for Mrξ
        push!(wst, 2*n, m)

        # return workspace for Ge
        push!(wst, 2*m, k)
    end
end

function compGHom!(
    G::AbstractArray{Complex{T}}, 
    z::AbstractMatrix{T}, r::AbstractMatrix{T},
    wns::AbstractVector{T}, work::AbstractArray{T}
) where T <: Union{Float32, Float64}
    n = size(r, 2)
    k = size(z, 2)
    p = length(wns)

    # check arguments
    @assert size(G) == (n, k, p)
    @assert length(wns) == p
    @assert length(work) >= n*k

    # make sure backends agree
    @assert backsagree(G, z, r, wns, work)

    for t in 1:p
        compM!(view(G, :, :, t:t), r, z, view(wns, t:t), work)
    end
end

export compM!, compG!, compGHom!

end