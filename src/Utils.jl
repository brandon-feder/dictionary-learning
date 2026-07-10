function eachldim(A::AbstractArray)
    """
    Returns an iterator over the last dimension of `A`
    """
    return eachslice(A, dims=(ndims(A),))
end

macro blasst(f)
    quote
        nbt = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        $(esc(f))
        BLAS.set_num_threads(nbt)
        @assert BLAS.get_num_threads() == nbt
    end
end

function backsagree(arrs::Vararg{AbstractArray})
    back = get_backend(arrs[1])
    for arr in arrs
        if get_backend(arr) != back
            return false
        end
    end
    return true
end

mutable struct WorkStackTrack{T}
    work::AbstractArray{T}
    ws::Int
end

function WorkStackTrack(work::AbstractArray{T}) where T
    return WorkStackTrack{T}(work, 0)
end

function Base.pop!(wst::WorkStackTrack{T}, dims::Vararg{Int}) where T
    n = prod(dims)
    @assert length(wst.work) >= wst.ws + n
    wspace = view(wst.work, wst.ws+1:wst.ws+n)
    wst.ws += n
    return reshape(wspace, dims)
end

function Base.push!(wst::WorkStackTrack{T}, dims::Vararg{Int}) where T
    n = prod(dims)
    @assert wst.ws - n >= 0
    wst.ws -= n
    return wst
end

function Base.push!(wst::WorkStackTrack{T}, arr::AbstractArray{T}) where T
    return push!(wst, length(arr))
end

function withwork(f::Function, wst::WorkStackTrack, dims::Vararg{Int})
    ws = pop!(wst, dims...)
    res = f(ws)
    push!(wst, dims...)
    return res
end

@doc raw"""
    function gen_sparse_samples(T, k, n, s)

Construct a sparse matrix `X` of size `(k, n)` such that each column contains `s` non-zero elements. The values of these non-zero elements is
chosen randomly on the real or complex sphere as specified by `T`.

# Arguments
* `T::Union{Type{Float32}, Type{Float64}, Type{ComplexF32}, Type{ComplexF64}}`: Element type of `X`
* `k::Int`: First dimension of `X`
* `n::Int`: Second dimension of `X`
* `s::Int`: Sparsity of each column of `X`
"""
function gen_sparse_samples(
    T::Union{Type{Float32}, Type{Float64}, Type{ComplexF32}, Type{ComplexF64}}, 
    k::Int, n::Int, s::Int
)
    # construct COO format of SparseMatrix
    C = repeat(1:n, inner=[s])
    R = Vector{Int64}(undef, s*n)

    if T <: Real
        V = rand(T, s*n)
        V .= 2 .* V .- 1
    else
        V = rand(T, s*n)
        V .= 2 .* V .- (1.0 + 1.0im)
    end

    V ./= abs.(V)

    # choose support in each column
    @floop ThreadedEx() for j in 1:n
        @init begin
            rng = MersenneTwister()
            perm = collect(1:k)
        end

        # choose support
        shuffle!(rng, perm)
        sup = view(perm, 1:s)
        copyto!(view(R, s*(j-1)+1:s*j), sup)
    end

    # create sparse matrix
    X = sparse(R, C, V, k, n)

    return X
end

@doc raw"""
    function subdist(A, B, nrm=:opnorm)

Computes the subspace distance between the column-space
of `A` and `B` under the norm specified by `:opnorm`. It
is assumed that the columns of `A` and `B` form an 
orthonormal basis.

# Arguments
* `A::StridedMatrix`
* `B::StridedMatrix`
* `nrm::Symbol`: Either `:opnorm` or `:fnorm` for operator
        norm and Frobenious norms respectivelly.
"""
function subdist(
    A::StridedMatrix{T}, B::StridedMatrix{T}, nrm::Symbol=:spa
) where T
    @assert size(A) == size(B)
    @assert nrm ∈ [:fnorm, :spa]

    if nrm == :fnorm
        return norm(A - B*adjoint(B)*A)
    else
        return acos(opnorm(A'*B))
    end
end

function batched_opnorm!(
    nrms::StridedVector{T1}, M::StridedArray{T2}
) where T1 where T2
    k = length(nrms)
    @assert ndims(M) == 3
    @assert size(M, 3) == k
    
    for i in 1:k
        nrms[i] = opnorm(view(M, :, :, i))
    end
end

# function batched_norm!(
#     nrms::StridedVector{T1}, M::StridedArray{T2}
# ) where T1 where T2
#     k = length(nrms)
#     @assert ndims(M) == 3
#     @assert size(M, 3) == k

#     sum!(abs2, reshape(nrms, 1, 1, :), M)
#     nrms .= sqrt.(nrms)
# end

# function batched_opnorm!(
#     nrms::CuVector{T1}, M::CuArray{T2}; niters::Int=20
# ) where T1 where T2
#     m, n, k = size(M)
#     @assert length(nrms) == k

#     v = CUDA.randn(real(T2), n, k)

#     u = CuMatrix{T2}(undef, m, k)
#     σs = reshape(nrms, 1, :)

#     for i in 1:niters
#         v ./= sqrt.(sum(abs2, v, dims=1)) .+ eps(real(T2))
#         CUBLAS.gemv_strided_batched!(
#             'N', one(T2), M, v, zero(T2), u
#         )
#         σs .= sqrt.(sum(abs2, u, dims=1))
#         u ./= σs .+ eps(real(T2))
#         CUBLAS.gemv_strided_batched!(
#             'C', one(T2), M, u, zero(T2), v
#         )
#     end
# end

function max_coh(
    A::StridedMatrix{T}, B::StridedMatrix{T}
) where T
    @assert get_backend(A) == get_backend(B)
    X = A'*B
    normA = sqrt.(sum(abs2, A, dims=1))
    normB = sqrt.(sum(abs2, B, dims=1))
    X ./= reshape(normA, :, 1)
    X ./= reshape(normB, 1, :)
    return maximum(abs, X)
end

function max_offdiag_coh(
    A::StridedMatrix{T}, B::StridedMatrix{T}
) where T
    @assert get_backend(A) == get_backend(B)
    @assert size(A, 2) == size(B, 2)
    k = size(A, 2)

    X = A'*B
    normA = sqrt.(sum(abs2, A, dims=1))
    normB = sqrt.(sum(abs2, B, dims=1))
    X ./= reshape(normA, :, 1)
    X ./= reshape(normB, 1, :)
    X[1:k+1:k^2] .= 0.0
    return maximum(abs, X)
end

function max_offdiag_coh(
    A::StridedMatrix{T}
) where T
    return max_offdiag_coh(A, A)
end

@doc raw"""
    function get_free_mem(back)

Returns the amount of free memory on the device
specified by `back`. Useful for allocating work space.

# Arguments
* `back::StridedMatrix`
"""
function get_free_mem(back::Backend)
    GC.gc()
    if back == CUDABackend()
        CUDA.reclaim()
        return CUDA.memory_info()[1]
    elseif back == CPU()
        return Sys.free_memory()
    else
        throw(ErrorException("backend $back not supported"))
    end
end

struct ComplexAngularDist <: Distances.Metric end

function (dist::ComplexAngularDist)(x, y)
    nx = norm(x)
    ny = norm(y)
    c = abs(dot(x, y)) / (nx * ny)
    return acos(clamp(c, -1.0, 1.0))
end

struct ComplexAngularDist_Normalized <: Distances.Metric end

function (dist::ComplexAngularDist_Normalized)(x, y)
    c = abs(dot(x, y))
    return acos(clamp(c, -1.0, 1.0))
end

struct ComplexAngularDist_Faked <: Distances.Metric end

function (dist::ComplexAngularDist_Faked)(x, y)
    nx = norm(x)
    ny = norm(y)
    R = dot(x, y)
    I = zero(eltype(x))
    @inbounds @simd for i in 1:2:(length(x) - 1)
        I += x[i] * y[i+1] - x[i+1] * y[i]
    end
    c = sqrt(R^2 + I^2) / (nx * ny)

    return acos(clamp(c, -1.0, 1.0))
end

struct ComplexAngularDist_Faked_Normalized <: Distances.Metric end

function (dist::ComplexAngularDist_Faked_Normalized)(x, y)
    R = dot(x, y)
    I = zero(eltype(x))
    @inbounds @simd for i in 1:2:(length(x) - 1)
        I += x[i] * y[i+1] - x[i+1] * y[i]
    end
    c = sqrt(R^2 + I^2)

    return acos(clamp(c, -1.0, 1.0))
end