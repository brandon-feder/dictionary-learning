function align_dict_blocks(
    A::StridedMatrix{T}, B::StridedMatrix{T}, p::Int=size(A, 1);
    max_iter::Int = 100
) where T
    @assert get_backend(A) == CPU()
    @assert get_backend(B) == CPU()
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
    p::Int=size(A, 1)
) where T
    @assert get_backend(A) == CPU()
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