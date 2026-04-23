
function recdict!(
    D::AbstractMatrix{T}, S::AbstractArray{T}, simps::Simplices
) where T
    m, k = size(D)
    back = get_backend(D)

    @assert size(D) == (m, k)
    @assert backsagree(D, S)

    
    @floop ThreadedEx() for i in 1:k
        @init begin
            M = adapt(back, Matrix{T}(undef, m, m))
        end

        # get i-th simplex
        simp = simps.simps[i]

        # reset M
        fill!(M, 0.0)

        # compute M
        for i in simp
            if back == CPU()
                if T <: Real
                    BLAS.syrk!('U', 'N', one(T), view(S, :, :, i), one(T), M)
                else
                    BLAS.herk!('U', 'N', one(T), view(S, :, :, i), one(T), M)
                end
            elseif back == CUDABackend()
                if T <: Real
                    CUBLAS.syrk!('U', 'N', one(T), view(S, :, :, i), one(T), M)
                else
                    CUBLAS.herk!('U', 'N', one(T), view(S, :, :, i), one(T), M)
                end
            end
        end

        # ToDo: Replace with TruncatedSVD/Kyrlov Subspace Methods
        eig = eigen(Hermitian(M))
        eigvindx = sortperm(adapt(CPU(), eig.values), by=λ->-λ)
        view(D, :, i) .= view(eig.vectors, :, eigvindx[1])
    end
end