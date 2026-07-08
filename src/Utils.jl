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
    function align_dict(A, B)

Computes the permutation of columns of `B`
and unit-magnitude scaling of those columns
so that the described matrix ``\tilde B`` minimizes
``diag(\lVert A'\tilde B) - I``.

# Arguments
* `A::StridedMatrix{T}`
* `B::StridedMatrix{T}`

# Returns
* `perm::Vector{Int}`: Vector of length `k` describing
        a permutation of the columns of `B`.
* `rot::Vector{T}`: Vector of length `k` describing
        how to scale the columns of `B`.

See the implementation of `align_dict!` for how to compute
    ``\tilde B`` from ``B``.

"""
function align_dict(
    A::StridedMatrix{T}, B::StridedMatrix{T}
) where T
    back = get_backend(A)
    k = size(A, 2)

    @assert get_backend(A) == back
    @assert backsagree(A, B)
    @assert size(A) == size(B)
    @assert size(A, 2) == k

    X = A'B

    # compute cost matrix
    normX = max.(sqrt.(sum(abs2, X, dims=1)), eps(real(T)))
    cost = -(abs.(X) ./ normX)
    
    # solve the assignment problem
    assignment, _ = hungarian(cost)
    perm = assignment
    
    # calculate optimal scaling
    scale = Vector{T}(undef, size(A, 2))
    X_cpu = adapt(CPU(), X)
    normXsq_cpu = vec(adapt(CPU(), sum(abs2, X, dims=1)))
    
    for i in 1:size(A, 2)
        j = perm[i]
        scale[i] = conj(X_cpu[i,j]) / max(normXsq_cpu[j], eps(real(T)))
    end

    return perm, scale
end

@doc raw"""
    function align_dict!(A, B)

Same as `align_dict` except `B` is transformed
in-place as well.
"""
function align_dict!(
    A::StridedMatrix{T}, B::StridedMatrix{T}
) where T
    back = get_backend(B)
    perm, rot = align_dict(A, B)
    B .= view(B, :, perm)
    B .*= reshape(adapt(back, rot), 1, :)
    return perm, rot
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

function get_white(
    Y::StridedMatrix{T}, k::Int, s::Int
) where T
    m, n = size(Y)
    Yh = collect(Y')
    qrYadj = qr(Yh)
    R = copy(qrYadj.R')

    svdR = svd(R)
    U = svdR.U
    svls = svdR.S * sqrt(k/(n*s))

    Winv = U * Diagonal(svls) * U'

    if k < m
        svls[k+1:end] .= 0.0
        svls[1:k] .^= -1
    else
        svls .^= -1
    end

    W = U * Diagonal(svls) * U'

    return W, Winv
end

function get_white(D::StridedMatrix{T}) where T
    return get_white(D, size(D, 2), 1)
end

function get_white_fast(
    Y::StridedMatrix{T}, k::Int, s::Int
) where T
    m, n = size(Y)
    A = Y*Y'

    eigA = eigen(Hermitian(A))
    U = eigA.vectors
    evls = eigA.values

    evls_rev = view(evls, length(evls):-1:1)
    evls_rev[min(m,k)+1:end] .= 0.0
    evls_rev_ = view(evls_rev, 1:min(m,k))

    evls_rev_[1:min(m,k)] .= sqrt.(evls_rev_) * sqrt(k/(n*s))

    Winv = U * Diagonal(evls) * U'

    evls_rev_ .^= -1

    W = U * Diagonal(evls) * U'

    return W, Winv
end

function get_white_fast(D::StridedMatrix{T}) where T
    return get_white_fast(D, size(D, 2), 1)
end

function align_dict_blocks(
    A::StridedMatrix{T}, B::StridedMatrix{T}, p::Int;
    max_iter::Int = 100
) where T
    @assert size(A) == size(B)
    mp, k = size(A)
    @assert mp % p == 0
    m = div(mp, p)
    
    # isolate px1 blocks
    A_flat = reshape(A, p, m * k)
    B_flat = reshape(B, p, m * k)

    # compute inner products
    X_all = A_flat' * B_flat

    # cosine similarirty
    normA = sqrt.(sum(abs2, A_flat, dims=1))
    normB = sqrt.(sum(abs2, B_flat, dims=1))
    
    # all similarities
    Sim_all = abs.(X_all) ./ max.(normA' .* normB, eps(real(T)))
    Sim = reshape(Sim_all, m, k, m, k)

    # will store permutations
    pi_r = collect(1:m)
    pi_c = collect(1:k)
    
    converged = false
    for iter in 1:max_iter
        pi_r_old = copy(pi_r)
        pi_c_old = copy(pi_c)

        # fix pi_r, optimize pi_c
        cost_c = zeros(real(T), k, k)
        for j2 in 1:k, j1 in 1:k
            score = zero(real(T))
            for i1 in 1:m
                score += Sim[i1, j1, pi_r[i1], j2]
            end
            cost_c[j1, j2] = -score
        end
        pi_c, _ = hungarian(cost_c)

        # fix pi_c, optimize pi_r
        cost_r = zeros(real(T), m, m)
        for i2 in 1:m, i1 in 1:m
            score = zero(real(T))
            for j1 in 1:k
                score += Sim[i1, j1, i2, pi_c[j1]]
            end
            cost_r[i1, i2] = -score
        end
        pi_r, _ = hungarian(cost_r)

        # check for convergence
        if pi_r == pi_r_old && pi_c == pi_c_old
            converged = true
            break
        end
    end

    # calculate optimal scaling
    scale = Matrix{T}(undef, m, k)
    normBsq = reshape(sum(abs2, B_flat, dims=1), m, k)
    for j in 1:k
        for i in 1:m
            i2 = pi_r[i]
            j2 = pi_c[j]
            
            # reconstruct flattening indecies
            idxA = i + (j - 1) * m
            idxB = i2 + (j2 - 1) * m
            
            # scaling
            cross_term = conj(X_all[idxA, idxB])
            scale[i, j] = cross_term / max(normBsq[i2, j2], eps(real(T)))
        end
    end

    return pi_r, pi_c, scale
end

function apply_block_alignment(
    A::StridedMatrix{T}, 
    pi_r::Vector{Int}, 
    pi_c::Vector{Int}, 
    scale::Matrix{T}, 
    p::Int
) where T
    mp, k = size(A)
    m = div(mp, p)
    
    # Pre-allocate output
    A_aligned = similar(A)
    
    # Iterate through each block (i, j)
    for j in 1:k
        j_orig = pi_c[j]
        for i in 1:m
            i_orig = pi_r[i]
            
            # Identify the row indices for this p x 1 block
            rows_range = ((i-1)*p + 1):(i*p)
            rows_orig = ((i_orig-1)*p + 1):(i_orig*p)
            
            # Apply permutation and scaling
            A_aligned[rows_range, j] = scale[i, j] .* A[rows_orig, j_orig]
        end
    end
    
    return A_aligned
end