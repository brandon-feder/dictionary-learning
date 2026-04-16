
function recdict!(
    D::AbstractMatrix{T}, S::AbstractArray{T}, simps::Simplices
) where T
    m, _ = size(D)

    @blasst @floop ThreadedEx() for (i,di) in enumerate(eachcol(D))
        # get i-th simplex
        simp = simps.simps[i]

        @init begin
            # per-thread scratch space
            M = Matrix{T}(undef, m, m)
        end

        # reset M
        fill!(M, 0.0)

        # compute M
        for i in simp
            BLAS.syrk!('U', 'N', 1.0, view(S, :, :, i), 1.0, M)
        end

        # set di as largest eigenvector
        eig = eigen(Symmetric(M), sortby = λ -> -λ)
        di .= view(eig.vectors, :, 1)
    end
end