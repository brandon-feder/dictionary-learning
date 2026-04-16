function compΣ!(
    Σ::Matrix{T}, D::AbstractMatrix{T}, X::AbstractSparseMatrix{T}, 
    work::AbstractArray{T}
) where T
    """
    Requires `mk` workspace.
    """

    # used to manage workspace
    wst = WorkStackTrack{T}(work)

    # allocate work space
    m, k = size(D)
    A = pop!(wst, m, k)

    # compute Σ
    n = size(X, 2)
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

    # used to manage workspace
    wst = WorkStackTrack{T}(work)

    # allocate work space
    t = length(sindx)
    k = size(D, 2)
    A = pop!(wst, k, k)
    B = pop!(wst, k, t)

    # compute G
    mul!(A, D', D)
    mul!(B, A, view(X, :, sindx))
    mul!(G, X', B)
    G .^= 2
end

function compV!(
    V::AbstractMatrix{T}, YT::AbstractMatrix{T}
) where T
    m = size(YT, 2)

    @blasst @floop ThreadedEx() for h in 1:m^2
        uh, vh = divrem(h-1, m) .+ 1
        u = view(YT, :, uh)
        v = view(YT, :, vh)
        Vh = view(V, :, h)
        Vh .= u .* v
    end
end

function compZ!(
    Z::AbstractArray{T}, G::AbstractMatrix{T}, V::AbstractMatrix{T}
) where T
    n, m2 = size(V)
    Z = reshape(Z, m2, :)
    mul!(Z, V', G ./ n)
end

function compS!(
    S::AbstractArray{T}, X::AbstractArray{T}, Σ::AbstractMatrix{T}, s::Int
) where T
    Σnrmsq = norm(Σ)^2

    # recover subspaces
    @blasst @floop ThreadedEx() for (Si, Xi) in zip(eachldim(S), eachldim(X))
        @init begin
            tau = Vector{T}(undef, s)
        end
        
        # subspace projection
        Xi .-= dot(Xi, Σ) * Σ ./ Σnrmsq

        # compute leading s eigenvectors
        eig = eigen(Symmetric(Xi), sortby = λ -> -λ)
        Si .= view(eig.vectors, :, 1:s)

        # orthonormalizeze
        LAPACK.geqrf!(Si, tau)
        LAPACK.orgqr!(Si, tau)
    end
end

function truesubs!(
    S::AbstractArray{T}, D::AbstractMatrix{T}, X::AbstractSparseMatrix{T}, s::Int
) where T
    @blasst @floop ThreadedEx() for (Si, xi) in
            zip(eachldim(S), eachldim(X))
        @init tau = Vector{T}(undef, s)

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
    sd::Vector{T}, A::AbstractArray{T}, B::AbstractArray{T}
) where T
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