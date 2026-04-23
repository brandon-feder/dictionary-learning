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