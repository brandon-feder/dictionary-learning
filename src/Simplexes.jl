mutable struct Simplices
    simps::DefaultDict{Int, Set{Int}}
    membs::DefaultDict{Int, Set{Int}}
end

function Simplices()
    return Simplices(
        DefaultDict{Int, Set{Int}}(()->Set{Int}()),
        DefaultDict{Int, Set{Int}}(()->Set{Int}())
    )
end

function Base.push!(simp::Simplices, ni::Int, si::AbstractVector{Int})
    for i in si
        push!(simp.simps[i], ni)
        push!(simp.membs[ni], si...)
    end
end