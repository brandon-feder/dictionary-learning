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

function WorkStackTrack{T}(work::AbstractArray{T}) where T
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

function gendata(k::Int, n::Int, s::Int)
    """
    Generate an `k x n` matrix consisting of 
    columns with sparsity `s` and with non-zero
    entries taking values 1 or -1.
    """

    # construct COO format of SparseMatrix
    C = repeat(1:n, inner=[s])
    R = Vector{Int64}(undef, s*n)
    V = rand((-1.0, 1.0), s*n)

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

function gendict(m::Int, k::Int)
    """
    Generate an `M x K` dictionary consisting of 
    columns which are distributed about the M-sphere
    uniformly at random.
    """

    D = randn(m, k)
    normalize!.(eachcol(D))
    return D
end
