module DictionaryLearning

using LinearAlgebra
using SparseArrays
using Random
using StatsBase
using Printf
using Base.Threads
using Base.Iterators
using ProgressMeter
using Folds
using FLoops
using LoopVectorization
using Graphs
using Combinatorics
using Clustering
using KrylovKit
using DataStructures
using CUDA, KernelAbstractions, GPUArrays, Adapt

# overloaded in `Utils.jl`
import Base: push!, pop!

include("Utils.jl")
include("Simplexes.jl")
include("SubspaceRecovery.jl")
include("Clustering.jl")
include("DictionaryRecovery.jl")
include("FoldyLax/FoldyLax.jl")

export compΣ!, compG!, compV!, compZ!, compS!,
    truesubs!, subdists!, nsupshared!, truesimps, recdict!, formgraph,
    truetruegraph, nedgedif, Simplices, cluster, randsvd
end