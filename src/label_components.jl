#=
   Copied verbatim from https://github.com/JuliaImages/Images.jl/blob/master/src/connected.jl
=#

# TODO: add MarcIntegralArrays as a dependecy
include("R:/users/pereyram/Gits/MarcIntegralArrays/src/MarcIntegralArrays.jl"); 


"""
   Code taken from ImageComponentAnalysis.jl, 'label_components.jl'. I wanted to 
   modify the source code to implement CCL clustering on vector fields based on 
   direction similarity between adjacent vectors. 

   The input to this 'direction-based CLL' is pressumed to be one of: 
     - 2D   vector field ( U::Array{T,2} & V::Array{T,2} )
     - 3D   vector field ( U::Array{T,3} & V::Array{T,3} & W::Array{T,3} )
     - 2D+t vector field ( U::Array{T,3} & V::Array{T,3} )
     - 3D+t vector field ( U::Array{T,4} & V::Array{T,4} & W::Array{T,4} )

   A) One way to accommodate all possible inputs would be to wrap the vector components
   within a tuple, which can be described with the type ::NTuple{NC,AbstractArray{T,ND}}, 
   where NC stands for "number of components" of the vectorfield and ND refers
   the "number of dimensions" of the vector field. The inputs can be integral arrays,
   instead of the original VF components, allowing to compute multiscale dot products.

   B) Another way is to combine the arrays into a ND+1 array, where the first (or last)
   dimension contains the vector field components. For instance, a 3D array of size
   (2,h,w) can store the two components (U and V) of a 2D vector field. Dot products can
   be expressed as VF[:,coords1...] .* VF[:,coords2...], with Cartesian coordinates, not
   linear indices. This approach supports using integral arrays.

   Below, I tested which approach is faster, with A) being much faster with loop unrolling. 
"""
function label_components( VF::NTuple{NC,AbstractArray{T,ND}}, 
                           dot_th::T=T(0.5),
                           connectivity = 1:ndims(VF[1])
                         ) where {NC,ND,T}
   return label_components!( zeros(Int, size(VF[1])), 
                             VF, 
                             dot_th, 
                             connectivity )
end

global label_components!
# let _label_components_cache = Dict{Tuple{Int, Vector{Int}}, Function}()
_label_components_cache = Dict{Tuple{Int, Vector{Int}}, Function}()

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
        ND_ = ndims(VF[1])
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
                U = VF[1]
                @nloops $ND i U  begin
                    k += 1
                    label = typemax(Int)
                    if sum_( VF, k ) != T(0.0)
                        @nexprs $ND d->begin
                            if $iregion[d] > 1 && Albl[k-offsets_d] > 0 # is this neighbor in-bounds?
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
    eval(:($f!($Albl, $sets, $VF, $dot_th)))

    # Now parse sets to find the labels
    newlabel = minlabel(sets)
    for i = 1:length(VF[1])
        if sum_( VF, i ) != 0.0
            Albl[i] = newlabel[find_root!(sets, Albl[i])]
        end
    end
    Albl
end


# end # let



###############################


#### MULTISCALE IMPLEMENTATION WITH INTEGRAL ARRAYS
# NOTE: I don't think I can use the @macro notation
# that i copy-pasted from ImageComponentAnalysis.jl,
# since I need to use cartesian coordinates for the
# integral sums.
function label_components!( Albl::AbstractArray{Int},
                            VF::NTuple{NC,MarcIntegralArrays.IntegralArray{T,ND}},
                            dot_th::T = T(0),
                            region::Union{Dims, AbstractVector{Int}}=1:ndims(VF[1]) 
                          ) where {NC,ND,T<:AbstractFloat}
    N = length(VF[1]); 

    uregion = unique(region)
    if isempty(uregion)
        k = 0
        for i in CartesianIndices( Albl )
            if sum_( VF, i ) != T(0.0)
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
        ND_ = ndims(VF[1])
        iregion = [Symbol("i_", d) for d in uregion]

        f! = eval(quote
            local lc!
            function lc!( Albl::AbstractArray{Int}, 
                          sets, 
                          VF::NTuple{NC,MarcIntegralArrays.IntegralArray{T,ND}},
                          dot_th::T = T(0)
                        ) where {NC,ND,T<:AbstractFloat}

                offsets = strides(VF[1])
                @nexprs $ND d->(offsets_d = offsets[d])
                k = 0
                U = VF[1]
                @nloops $ND i U  begin
                    k += 1
                    label = typemax(Int)
                    if sum_( VF, k ) != T(0.0)
                        @nexprs $ND d->begin
                            if $iregion[d] > 1 && Albl[k-offsets_d] > 0 # is this neighbor in-bounds?
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
    eval(:($f!($Albl, $sets, $VF, $dot_th)))

    # Now parse sets to find the labels
    newlabel = minlabel(sets)
    for i = 1:length(VF[1])
        if sum_( VF, i ) != 0.0
            Albl[i] = newlabel[find_root!(sets, Albl[i])]
        end
    end
    Albl
end

# A) WINNER
# @btime directionCCL.dot_( $inp, $idx1, $idx2 ) (on my miniPC... seems unbeatable)
#   4.308 ns (0 allocations: 0 bytes)
function dot_( inp::NTuple{2,MarcIntegralArrays.IntegralArray{T,ND}}, ROI1=1, ROI2=100 ) where {ND,T}
    dot = T(0)
    @unroll for i in 1:2
       dot += inp[i][ROI1]*inp[i][ROI2]
    end
    return dot
end

function dot_( inp::NTuple{3,MarcIntegralArrays.IntegralArray{T,ND}}, ROI1=1, ROI2=100 ) where {ND,T}
    dot = T(0)
    @unroll for i in 1:3
       dot += inp[i][ROI1]*inp[i][ROI2]
    end
    return dot
end

# Very similar code we can also be used to compute "the sum of vector components", 
# which allows us to discard vectors with mangitude == 0 (since all components must
# be 0). 
function sum_( inp::NTuple{2,MarcIntegralArrays.IntegralArray{T,ND}}, idx1=1 ) where {ND,T}
    sum = T(0)
    @unroll for i in 1:2
       sum += inp[i][idx1]
    end
    return sum
end

function sum_( inp::NTuple{3,MarcIntegralArrays.IntegralArray{T,ND}}, idx1=1 ) where {ND,T}
    sum = T(0)
    @unroll for i in 1:3
       sum += inp[i][idx1]
    end
    return sum
end