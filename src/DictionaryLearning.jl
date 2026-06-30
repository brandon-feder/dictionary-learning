module DictionaryLearning

using LinearAlgebra
using SparseArrays
using StaticArrays
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
using NNlib
using Hungarian

# overloaded in `Utils.jl`
import Base: push!, pop!

include("Utils.jl")
include("FoldyLax/FoldyLax.jl")
include("GELMA/GELMA.jl")
include("MOD/Structs.jl")
include("MOD/Sample.jl")
include("MOD/Cluster.jl")
include("SSDL/SubRec.jl")
include("SSDL/SubDist.jl")
# include("SSDL/SubCluster.jl")

export FoldyLaxStruct, FoldyLaxWorkStruct, foldylax!, foldylax_update!, subdist,
    gen_sparse_samples, MODSampleStruct, MODSampleWorkStruct, mod_sample!, 
    mod_cluster!,
    GELMAStruct, GELMAWorkStruct, gelma!,
    align_dict, align_dict!, max_coh, max_offdiag_coh, 
    get_free_mem,
    SSDLFakeSubRecStruct, SSDLSubRecStruct, ssdl_subrec!, ssdl_subrec_true!,
    SSDLSubDistStruct, SSDLTrueSubDistStruct, SSDLSubDistWorkStruct, sub_dist!,
    SSDLSubClusterStruct, SSDLSubClusterWorkStruct, ssdl_sub_cluster!, get_white
end
