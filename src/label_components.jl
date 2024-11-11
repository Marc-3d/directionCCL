#=
   Copied verbatim from https://github.com/JuliaImages/Images.jl/blob/master/src/connected.jl
=#

import Base.push!  # for DisjointMinSets
using Base.Cartesian
import Unroll.@unroll

"""
   Code taken from ImageComponentAnalysis.jl, 'label_components.jl'. I wanted to 
   modify the source code to implement CCL clustering on vector fields based on 
   direction similarity between adjacent vectors. 

   The input to this 'direction-based CLL' is pressumed to be one of: 
     - 2D   vector field ( U::Array{T,2} & V::Array{T,2} )
     - 3D   vector field ( U::Array{T,3} & V::Array{T,3} & W::Array{T,3} )
     - 2D+t vector field ( U::Array{T,3} & V::Array{T,3} )
     - 3D+t vector fiedl ( U::Array{T,4} & V::Array{T,4} & W::Array{T,4} )

   A) One way to accommodate all possible inputs would be to pass the compoents in 
   a tuple, which can be described with the type ::NTuple{NC,AbstractArray{T,ND}}, 
   where NC stands for "number of components" of the vectorfield and ND refers
   the "number of dimensions" of the vector field. The inputs can be integral arrays,
   instead of the original VF components, allowing to compute multiscale dot products.

   B) Another way would be to express it as a Matrix of size (LxNC), with one column 
   for each component, and with L rows, one for each position in the vector field. 
   This representation requires that we provide the original stride of the matrices, 
   before flattening them into L rows. This approach has the advantage that dot 
   products are easily expressed as VF[k,:] .* VF[k-offset_d,:]. However, this 
   approach is not compatible with integral arrays.  

   C) Another way is to combine the arrays into a ND+1 array, where the first (or last)
   dimension contains the vector field components. For instance, a 3D array of size
   (2,h,w) can store a 2D vector field given by U and V. Dot products can be expressed
   as VF[:,coords1...] .* VF[:,coords2...], with Cartesian coordinates, not linear
   indices. This approach supports using integral arrays.

   I guess we have to check which approach is faster. 
"""
function label_components( VF, 
                           dot_th=0.5,
                           connectivity = 1:ndims(VF[1])
                         )
   return label_components!( zeros(Int, size(VF[1])), 
                             VF, 
                             dot_th, 
                             connectivity )
end

global label_components!
let _label_components_cache = Dict{Tuple{Int, Vector{Int}}, Function}()


#### 4-connectivity in 2d, 6-connectivity in 3d, etc.
function label_components!( Albl::AbstractArray{Int},
                            VF::NTuple{NC,AbstractArray{T,ND}},
                            dot_th::T = T(0),
                            region::Union{Dims, AbstractVector{Int}}=1:ndims(VF[1]) 
                          ) where {NC,ND,T<:AbstractFloat}
    N = length(VF[1]); 

    uregion = unique(region)
    if isempty(uregion)
        # Each pixel is its own component
        k = 0
        for i = 1:N
            if sum_( VF, i ) != T(0.0)
                # Vectors with mangitude 0 are ignored
                k += 1
                Albl[i] = k
            end
        end
        return Albl
    end

    # We're going to compile a version specifically for the chosen region. 
    # This should make it very fast.

    key = (ndims(VF[1]), uregion)

    if !haskey(_label_components_cache, key)
        # Need to generate the function
        N   = length(uregion)
        ND_ = ndims(A)
        iregion = [Symbol("i_", d) for d in uregion]

        f! = eval(quote
            local lc!
            function lc!( Albl::AbstractArray{Int}, 
                          sets, 
                          VF::NTuple{NC,AbstractArray{T,ND}},
                          dot_th::T = T(0)
                        ) where {NC,ND,T<:AbstractFloat}

                offsets = strides(VF[1])
                @nexprs $ND d->(offsets_d = offsets[d])
                k = 0
                @nloops $ND i VF[1] begin
                    k += 1
                    label = typemax(Int)
                    if sum_( VF, k ) != T(0.0)
                        @nexprs $ND d->begin
                            if $iregion[d] > 1  # is this neighbor in-bounds?
                                dot = dot_( VF, k, k-offsets_d )
                                if dot > dot_th  # if the two have similar angles...
                                    newlabel = Albl[k-offsets_d]
                                    if label != typemax(Int) && label != newlabel
                                        label = union!(sets, label, newlabel)  # ...merge labels...
                                    else
                                        label = newlabel  # ...and assign the smaller to current pixel
                                    end
                                end
                            end
                        end
                        if label == typemax(Int)
                            label = push!(sets)   # there were no neighbors, create a new label
                        end
                        Albl[k] = label
                    end
                end
                Albl
            end
        end)
        _label_components_cache[key] = f!
    else
        f! = _label_components_cache[key]
    end

    sets = DisjointMinSets()
    eval(:($f!($Albl, $sets, ($U,$V), $dot_th)))

    # Now parse sets to find the labels
    newlabel = minlabel(sets)
    for i = 1:length(U)
        if (U[i]+V[i]) != 0.0
            Albl[i] = newlabel[find_root!(sets, Albl[i])]
        end
    end
    Albl
end


end # let

# Copied directly from DataStructures.jl, but specialized
# to always make the parent be the smallest label
struct DisjointMinSets
    parents::Vector{Int}

    DisjointMinSets(n::Integer) = new([1:n;])
end
DisjointMinSets() = DisjointMinSets(0)

function find_root!(sets::DisjointMinSets, m::Integer)
    p = sets.parents[m]   # don't use @inbounds here, it might not be safe
    @inbounds if sets.parents[p] != p
        sets.parents[m] = p = find_root_unsafe!(sets, p)
    end
    p
end

# an unsafe variant of the above
function find_root_unsafe!(sets::DisjointMinSets, m::Int)
    @inbounds p = sets.parents[m]
    @inbounds if sets.parents[p] != p
        sets.parents[m] = p = find_root_unsafe!(sets, p)
    end
    p
end

function union!(sets::DisjointMinSets, m::Integer, n::Integer)
    mp = find_root!(sets, m)
    np = find_root!(sets, n)
    if mp < np
        sets.parents[np] = mp
        return mp
    elseif np < mp
        sets.parents[mp] = np
        return np
    end
    mp
end

function push!(sets::DisjointMinSets)
    m = length(sets.parents) + 1
    push!(sets.parents, m)
    m
end

function minlabel(sets::DisjointMinSets)
    out = Vector{Int}(undef, length(sets.parents))
    k = 0
    for i = 1:length(sets.parents)
        if sets.parents[i] == i
            k += 1
        end
        out[i] = k
    end
    out
end

#### testing dot products 

# A) WINNER
# @btime directionCCL.dot_( $inp, $idx1, $idx2 ) (on my miniPC... seems unbeatable)
#   4.308 ns (0 allocations: 0 bytes)
function dot_( inp::NTuple{2,AbstractArray{T,ND}}, idx1=1, idx2=100 ) where {ND,T}
    dot = T(0)
    @unroll for i in 1:2
       dot += inp[i][idx1]*inp[i][idx2]
    end
    return dot
end

function dot_( inp::NTuple{3,AbstractArray{T,ND}}, idx1=1, idx2=100 ) where {ND,T}
    dot = T(0)
    @unroll for i in 1:3
       dot += inp[i][idx1]*inp[i][idx2]
    end
    return dot
end

# Very similar code we can also be used to compute "the sum of vector components", 
# which allows us to discard vectors with mangitude == 0 (since all components must
# be 0). 
function sum_( inp::NTuple{2,AbstractArray{T,ND}}, idx1=1 ) where {ND,T}
    sum = T(0)
    @unroll for i in 1:2
       sum += inp[i][idx1]
    end
    return sum
end

function sum_( inp::NTuple{3,AbstractArray{T,ND}}, idx1=1 ) where {ND,T}
    sum = T(0)
    @unroll for i in 1:3
       sum += inp[i][idx1]
    end
    return sum
end


# B)
# @btime directionCCL.dot_( $VF, $idx1, $idx2 )
#   97.821 ns (3 allocations: 192 bytes)
function dot_( inp::AbstractArray{T,NCD}, idx1::Dims{ND}, idx2::Dims{ND} ) where {T,NCD,ND}
    return sum( inp[:,idx1...] .* inp[:,idx2...] )
end


