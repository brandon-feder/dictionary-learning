module FoldyLax

using LinearAlgebra
using FLoops
using KernelAbstractions, Adapt, GPUArrays
using Base.Broadcast: broadcasted, materialize

include("../Utils.jl")

@doc raw"""
    function compM!(M, A, B, wns, work, τ=nothing)

Computes ``M`` so that 

```math
e^T_i M_{:,:,s} e_j = e_s^Tf(A_i, B_j) \tau_j
``` 

if ``\tau`` is provided or simply

```math
e^T_i M_{:,:,s} e_j = e_s^Tf(A_i, B_j)
``` 

otherwise. The wavenumbers defining ``f`` are given 
in `wns`.

# Arguments
* `M::AbstractArray{Complex{T}}`: Array of size `(m, n, p)` which will be overwritten with the system.
* `A::AbstractMatrix{Complex{T}}`:  Array of size `(3, m)` representing coordinates.
* `B::AbstractMatrix{Complex{T}}`: Array of size `(3, n)` representing coordinates.
* `wns::AbstractVector{Complex{T}}`: Vector of length `p` containing wave numbers.
* `work::AbstractArray{Complex{T}}`: Array of length at least `m*n`. Used as scratch space
    and is overwritten.
* `τ::Union{AbstractVector{T}, Nothing}=nothing`: Optional argument specifing the scaling of the second
    dimension of `M`. If `nothing` assumed to containg all `1`.


    function compM!(M, A, wns, work, τ=nothing)


Same as `compM!(M, A, A, wns, work, τ)` except the diagonals are overwritten by `-1`.


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


@doc raw"""
    function compG!(G, Mξξfac, Mrξ, Mξz,  Mrz, wns)

Computes ``G`` so that 
```math
G_{:,:,s} = M^{[rz]}_{:,:,s} - M^{[rξ]}_{:,:,s} \left(M^{[ξξ]}_{:,:,s}
\right)^{-1}M^{[ξz]}_{:,s}
```
for ``s = 1, \cdots, p``

# Arguments
* `G::AbstractArray{Complex{T}}`: Array of size `(n, k, p)` which will be overwritten with the Green's tensor. Will be overwritten.
* `Mξξfac::AbstractVector`: Vector of length `p` containg factorizations of ``M^{[\xi\xi]}_s`` for ``s = 1, \cdots, p``. The factorizations should be supported by `ldiv!`.
* `Mrξ::AbstractArray{Complex{T}}`: Array of size `(n, m, p)` containing each ``M^{[r\xi]}_s`` along the last dimension. Will be overwritten.
* `Mξz::AbstractArray{Complex{T}}`: Array of size `(m, k, p)` containing each ``M^{[\xi z]}_s`` along the last dimension.
* `Mrz::AbstractArray{Complex{T}}`: Array of size `(n, k, p)` containing each ``M^{[rz]}_s`` along the last dimension.
* `wns::AbstractVector{T}`: Vector of length `p` containing wavenumbers.
"""
function compG!(
    G::AbstractArray{Complex{T}}, Mξξfac,
    Mrξ::AbstractArray{Complex{T}}, Mξz::AbstractArray{Complex{T}},
    Mrz::AbstractArray{Complex{T}}, wns::AbstractVector{T}
) where T <: Union{Float32, Float64}
    n, k, p = size(G)
    m = size(Mrξ, 2)

    # check arguments
    @assert size(G) == (n, k, p)
    @assert length(Mξξfac) == p
    @assert all(M -> size(M) == (m,m), Mξξfac)
    @assert size(Mrξ) == (n, m, p)
    @assert size(Mξz) == (m, k, p)
    @assert size(Mrz) == (n, k, p)
    @assert length(wns) == p

    # make sure backends agree
    @assert backsagree(G, Mrξ, Mξz, Mrz, wns)

    # solve LS problems
    for s in 1:p
        # get current slice
        Mξz_ = view(Mξz, :, :, s)
        Mξξ_ = Mξξfac[s]
        Mrz_ = view(Mrz, :, :, s)
        Mrξ_ = view(Mrξ, :, :, s)
        G_ = view(G, :, :, s)

        ldiv!(Mξz_, Mξξ_, Mξz_)
        G_ .= Mrz_ .- Mrξ_ * Mξz_
    end
end

export compM!, compG!

end