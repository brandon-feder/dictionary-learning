function compΣ!(
    Σ::AbstractArray{T}, D::AbstractArray{T}, X::AbstractSparseMatrix{T}, 
    work::AbstractArray{T}
) where T
    m, k = size(D)
    n = size(X, 2)

    # check arguments
    @assert size(Σ) == (m, m)
    @assert size(D) == (m, k)
    @assert size(X) == (k, n)
    @assert length(work) >= m*k
    @assert backsagree(Σ, D, X, work)

    # used to manage workspace
    wst = WorkStackTrack{T}(work)

    # allocate work space
    A = pop!(wst, m, k)

    # compute Σ
    mul!(A, D, 1/n * X*X')
    mul!(Σ, A, D')
end

function compG!(
    G::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractSparseMatrix{T},
    sindx::AbstractVector{Int}, work::AbstractArray{T}
) where T
    """
    Requires `k^2 + kt` workspace
    """

    m, k = size(D)
    n, t = size(G)

    @assert size(G) == (n, t)
    @assert size(D) == (m, k)
    @assert size(X) == (k, n)
    @assert length(work) >= k^2 + k*t
    @assert backsagree(G, D, X, work)

    # used to manage workspace
    wst = WorkStackTrack{T}(work)

    # allocate work space
    A = pop!(wst, k, k)
    B = pop!(wst, k, t)

    # compute G
    mul!(A, D', D)
    @allowscalar mul!(B, A, view(X, :, sindx))
    mul!(G, X', B)
    G .= G.^2 ./ n
end

function compV!_cuda(V, YT, n, m)
    k = (blockIdx().x-1) * blockDim().x + threadIdx().x
    h = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if k <= n && h <= m*m
        i = (h-1) % m + 1
        j = (h-1) ÷ m + 1
        @inbounds V[k, h] = YT[k, i] * YT[k, j]
    end

    return
end

function compV!(
    V::AbstractMatrix{T}, YT::AbstractMatrix{T}
) where T
    n, m = size(YT)
    back = get_backend(V)

    # check arguments
    @assert size(V) == (n, m^2)
    @assert size(YT) == (n, m)
    @assert backsagree(V, YT)

    if back == CPU()
        @blasst @floop ThreadedEx() for h in 1:m^2
            uh, vh = divrem(h-1, m) .+ 1
            u = view(YT, :, uh)
            v = view(YT, :, vh)
            Vh = view(V, :, h)
            Vh .= u .* v
        end
    elseif back == CUDABackend()
        threads = (16, 16)
        blocks = (cld(n, 16), cld(m*m, 16))

        @cuda threads=threads blocks=blocks compV!_cuda(V, YT, n, m)
    else
        throw(ArgumentError("Unsupported backend $back"))
    end
end

function compZ!(
    Z::AbstractArray{T}, G::AbstractMatrix{T}, V::AbstractMatrix{T}
) where T
    n, m2 = size(V)
    t = size(Z, 3)

    # check arguments
    @assert size(V) == (n, m2)
    @assert size(G) == (n, t)
    @assert size(Z, 3) == t
    @assert size(Z, 1) * size(Z, 2) == m2
    @assert backsagree(Z, G, V)

    Z = reshape(Z, m2, :)

    mul!(Z, V', G)
end

function compS!(
    S::AbstractArray{T}, Z::AbstractArray{T}, Σ::AbstractMatrix{T}, s::Int
) where T
    Σnrmsq = norm(Σ)^2

    back = get_backend(S)
    m = size(Σ,1)
    t = size(S, 3)

    @assert size(S) == (m, s, t)
    @assert size(Z) == (m, m, t)
    @assert size(Σ) == (m, m)
    @assert backsagree(S, Z, Σ)

    # recover subspaces
    @blasst @floop ThreadedEx() for (Si, Zi) in zip(eachldim(S), eachldim(Z))
        @init begin
            tau = adapt(back, Vector{T}(undef, s))
        end
        
        # subspace projection
        Zi .-= dot(Zi, Σ) * Σ ./ Σnrmsq

        # compute leading s eigenvectors
        # ToDo: Replace with TruncatedSVD/Kyrlov Subspace Methods
        # eig = eigen(Zi)
            # compute leading s eigenvectors
        eig = eigen(Hermitian(Zi))
        eigvindx = sortperm(adapt(CPU(), eig.values), by=λ->-λ)
        Si .= view(eig.vectors, :, eigvindx[1:s])

        # orthonormalizeze
        LAPACK.geqrf!(Si, tau)
        LAPACK.orgqr!(Si, tau)
    end
end

function truesubs!(
    S::AbstractArray{T}, D::AbstractMatrix{T}, Xt::AbstractSparseMatrix{T}, s::Int
) where T
    t = size(S, 3)
    m, k = size(D)
    back = get_backend(S)

    @assert size(S) == (m, s, t)
    @assert size(D) == (m, k)
    @assert size(Xt) == (k, t)
    @assert backsagree(S, D)
    @assert get_backend(Xt) == CPU()

    @blasst @floop ThreadedEx() for (Si, xi) in
            zip(eachldim(S), eachldim(Xt))
        @init tau = adapt(back, Vector{T}(undef, s))

        # find support of vector
        sup, _ = findnz(xi)

        # get basis for subspace
        copyto!(Si, view(D, :, sup))

        # orthonormalize
        LAPACK.geqrf!(Si, tau)
        LAPACK.orgqr!(Si, tau)
    end
end

function subdists!(
    sd::Vector{T1}, A::AbstractArray{T2}, B::AbstractArray{T2}
) where T1 where T2
    @blasst @floop ThreadedEx() for (j, (X, Y)) in 
            enumerate(zip(eachldim(A), eachldim(B)))
        sd[j] = norm(X - Y*Y'*X)
    end
end

function nsupshared!(
    ss::Vector{Int}, A::AbstractArray{T}, B::AbstractArray{T}
) where T
    for (i, (Ai, Bi)) in 
            enumerate(zip(eachldim(A), eachldim(B)))
        indAi = findnz(Ai)[1]
        indBi = findnz(Bi)[1]
        ss[i] = length(intersect(indAi, indBi))
    end
end