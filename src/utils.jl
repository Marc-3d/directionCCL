import Unroll.@unroll

@inline minmax( a, min=1, max=10 ) = a + (min-a)*(a<min) + (a>max)*(max-a)
@inline clipmin( a, min=1 ) = a + (min-a)*(a<min)
@inline clipmax( a, max=1 ) = a + (a>max)*(max-a)
@inline round_( a ) = round( Int, a )
@inline round_and_clip( a, min=1, max=10 ) = round( Int, a + (min-a)*(a<min) + (a>max)*(max-a) )



"""
 Testing dot product implementations on different formats for storing
  the input vector fields: 
    A) storing vector fields as a Tuple of arrays with one entry for 
       each component, e.g. (U,V) or (U,V,W). 
    B) storing vector fields as an ND+1 matrix, where each slice of
       the ND+1 matrix is a components of the vector field. For instance, 
       VF[1,:,:] = U, VF[2,:,:] = V. 
"""

# A) WINNER
# @btime directionCCL.dot_( $inp, $idx1, $idx2 ) (on my miniPC... seems unbeatable)
#   4.308 ns (0 allocations: 0 bytes)
function dot_( inp::NTuple{2,AbstractArray{T,ND}}, idx1::Int=1, idx2::Int=100 ) where {ND,T}
    dot = T(0)
    @unroll for i in 1:2
       dot += inp[i][idx1]*inp[i][idx2]
    end
    return dot
end

function dot_( inp::NTuple{3,AbstractArray{T,ND}}, idx1::Int=1, idx2::Int=100 ) where {ND,T}
    dot = T(0)
    @unroll for i in 1:3
       dot += inp[i][idx1]*inp[i][idx2]
    end
    return dot
end

# B)
# @btime directionCCL.dot_( $VF, $idx1, $idx2 )
#   97.821 ns (3 allocations: 192 bytes)
function dot_( inp::AbstractArray{T,NCD}, idx1::Dims{ND}, idx2::Dims{ND} ) where {T,NCD,ND}
    return sum( inp[:,idx1...] .* inp[:,idx2...] )
end


# Very similar code we can also be used to compute "the sum of vector components", 
# which allows us to discard vectors with mangitude == 0 (since all components must
# be 0). 
function sum_( inp::NTuple{2,AbstractArray{T,ND}}, idx1::Int=1 ) where {ND,T}
    sum = T(0)
    @unroll for i in 1:2
       sum += inp[i][idx1]
    end
    return sum
end

function sum_( inp::NTuple{3,AbstractArray{T,ND}}, idx1::Int=1 ) where {ND,T}
    sum = T(0)
    @unroll for i in 1:3
       sum += inp[i][idx1]
    end
    return sum
end

######### dot product

# TODO: test that this doesn't introduce any overheads
valid_index{ND} = Union{Int,Dims{ND},CartesianIndex{ND}} where {ND}

function dot_( inp::NTuple{2,AbstractArray{T,ND}}, idx1::valid_index{ND}, idx2::valid_index{ND} ) where {ND,T}
    dot = T(0)
    @unroll for i in 1:2
       dot += inp[i][idx1]*inp[i][idx2]
    end
    return dot
end

function dot_( inp::NTuple{3,AbstractArray{T,ND}}, idx1::valid_index{ND}, idx2::valid_index{ND} ) where {ND,T}
    dot = T(0)
    @unroll for i in 1:3
       dot += inp[i][idx1]*inp[i][idx2]
    end
    return dot
end

function dot_( inp::NTuple{2,AbstractArray{T,ND}}, idx1::Dims{ND}, idx2::Dims{ND} ) where {ND,T}
    dot = T(0)
    @unroll for i in 1:2
       dot += inp[i][idx1...]*inp[i][idx2...]
    end
    return dot
end

function dot_( inp::NTuple{3,AbstractArray{T,ND}}, idx1::Dims{ND}, idx2::Dims{ND} ) where {ND,T}
    dot = T(0)
    @unroll for i in 1:3
       dot += inp[i][idx1...]*inp[i][idx2...]
    end
    return dot
end

function dot_( inp::NTuple{2,AbstractArray{T,ND}}, idx1::Dims{ND}, idx2::Vector{Int} ) where {ND,T}
    dot = T(0)
    @unroll for i in 1:2
       dot += inp[i][idx1...]*inp[i][idx2...]
    end
    return dot
end

function dot_( inp::NTuple{3,AbstractArray{T,ND}}, idx1::Dims{ND}, idx2::Vector{Int} ) where {ND,T}
    dot = T(0)
    @unroll for i in 1:3
       dot += inp[i][idx1...]*inp[i][idx2...]
    end
    return dot
end

######### sum of compoments

function sum_( inp::NTuple{2,AbstractArray{T,ND}}, idx1::valid_index{ND} ) where {ND,T}
    sum = T(0)
    @unroll for i in 1:2
       sum += inp[i][idx1]
    end
    return sum
end

function sum_( inp::NTuple{3,AbstractArray{T,ND}}, idx1::valid_index{ND}) where {ND,T}
    sum = T(0)
    @unroll for i in 1:3
       sum += inp[i][idx1]
    end
    return sum
end

# inplace

function sum_unsafe!( out::Vector{T}, inp::NTuple{2,AbstractArray{T,ND}}, idx1::valid_index{ND} ) where {ND,T}
    @unroll for i in 1:2
       sum[i] += inp[i][idx1]
    end
    return nothing
end

function sum_unsafe!( out::Vector{T}, inp::NTuple{3,AbstractArray{T,ND}}, idx1::valid_index{ND}) where {ND,T}
    @unroll for i in 1:3
       sum[i] += inp[i][idx1]
    end
    return nothing
end

function sub_unsafe!( out::Vector{T}, inp::NTuple{2,AbstractArray{T,ND}}, idx1::valid_index{ND} ) where {ND,T}
    @unroll for i in 1:2
       out[i] -= inp[i][idx1]
    end
    return nothing
end

function sub_unsafe!( out::Vector{T}, inp::NTuple{3,AbstractArray{T,ND}}, idx1::valid_index{ND}) where {ND,T}
    @unroll for i in 1:3
       out[i] -= inp[i][idx1]
    end
    return nothing
end

function sub_M_unsafe!( out::Vector{T}, inp::NTuple{2,AbstractArray{T,ND}}, M::AbstractArray{T,ND}, idx1::valid_index{ND} ) where {ND,T}
    @unroll for i in 1:2
       out[i] -= inp[i][idx1]*M[idx1]
    end
    return nothing
end

function sub_M_unsafe!( out::Vector{T}, inp::NTuple{3,AbstractArray{T,ND}}, M::AbstractArray{T,ND}, idx1::valid_index{ND}) where {ND,T}
    @unroll for i in 1:3
       out[i] -= inp[i][idx1]*M[idx1]
    end
    return nothing
end

function sub_M_unsafe!( out::Vector{T}, inp::NTuple{2,AbstractArray{T,ND}}, M::T, idx1::valid_index{ND} ) where {ND,T}
    @unroll for i in 1:2
       out[i] -= inp[i][idx1]*M
    end
    return nothing
end

function sub_M_unsafe!( out::Vector{T}, inp::NTuple{3,AbstractArray{T,ND}}, M::T, idx1::valid_index{ND}) where {ND,T}
    @unroll for i in 1:3
       out[i] -= inp[i][idx1]*M
    end
    return nothing
end




function sum_M_unsafe!( out::Vector{T}, inp::NTuple{2,AbstractArray{T,ND}}, M::AbstractArray{T,ND}, idx1::valid_index{ND} ) where {ND,T}
    @unroll for i in 1:2
       out[i] += inp[i][idx1]*M[idx1]
    end
    return nothing
end

function sum_M_unsafe!( out::Vector{T}, inp::NTuple{3,AbstractArray{T,ND}}, M::AbstractArray{T,ND}, idx1::valid_index{ND}) where {ND,T}
    @unroll for i in 1:3
       out[i] += inp[i][idx1]*M[idx1]
    end
    return nothing
end

function sum_M_unsafe!( out::Vector{T}, inp::NTuple{2,AbstractArray{T,ND}}, M::T, idx1::valid_index{ND} ) where {ND,T}
    @unroll for i in 1:2
       out[i] += inp[i][idx1]*M
    end
    return nothing
end

function sum_M_unsafe!( out::Vector{T}, inp::NTuple{3,AbstractArray{T,ND}}, M::T, idx1::valid_index{ND}) where {ND,T}
    @unroll for i in 1:3
       out[i] += inp[i][idx1]*M
    end
    return nothing
end

######### magnitude squared

function magSQ_( inp::NTuple{2,AbstractArray{T,ND}}, idx1::valid_index{ND} ) where {ND,T}
    sum = T(0)
    @unroll for i in 1:2
       sum += inp[i][idx1]^2
    end
    return sum
end

function magSQ_( inp::NTuple{3,AbstractArray{T,ND}}, idx1::valid_index{ND} ) where {ND,T}
    sum = T(0)
    @unroll for i in 1:3
       sum += inp[i][idx1]^2
    end
    return sum
end

############

function _label_contours( lbls::AbstractArray{Int,N},
                          region::Union{Dims, AbstractVector{Int}} = 1:ndims(lbls), 
                        ) where {N}

    lblsize = size( lbls )
    contours = zeros( UInt8, lblsize )
    _label_contours!( contours, lbls, region )
    return contours
end

function _label_contours!( contours::AbstractArray{UInt8,2},
                           lbls::AbstractArray{Int,2},
                           region::Union{Dims, AbstractVector{Int}} = 1:2, 
                         ) 

    @assert size( contours ) == size( lbls )

    # Checking which dimensions should be analyzed
    doDim1 = 1 in region; 
    doDim2 = 2 in region; 

    # If no dimensions are analyzed, we do nothing (for now).
    if ( !doDim1 && !doDim2 )
        return contours
    end

    ROI = UnitRange.( 2, size( lbls ) ); 

    for c in CartesianIndices( ROI )

        value = UInt8( 0 )

        # for first dimension
        c_N = c + CartesianIndex( -1, 0 )
        if ( doDim1 )
            value += UInt8( lbls[c] != lbls[c_N] )
        end

        # for second dimension
        c_N = c + CartesianIndex( 0, -1 )
        if ( doDim2  )
            value += UInt8( lbls[c] != lbls[c_N] )
        end

        contours[c] = UInt8( value > 0 )
    end

    return nothing
end

function _label_contours!( contours::AbstractArray{UInt8,3},
                           lbls::AbstractArray{Int,3},
                           region::Union{Dims, AbstractVector{Int}} = 1:3, 
                         ) 

    @assert size( contours ) == size( lbls )

    # Checking which dimensions should be analyzed
    doDim1 = 1 in region; 
    doDim2 = 2 in region;
    doDim2 = 3 in region;  

    # If no dimensions are analyzed, we do nothing (for now).
    if ( !doDim1 && !doDim2 && !doDim3 )
        return contours
    end

    ROI = UnitRange.( 2, size( lbls ) ); 

    for c in CartesianIndices( ROI )

        value = UInt8(0)

        # for first dimension
        c_N = c + CartesianIndex( -1, 0, 0 )
        if ( doDim1 )
            
            value += UInt8( lbls[c] != lbls[c_N] )
        end

        # for second dimension
        c_N = c + CartesianIndex( 0, -1, 0 )
        if ( doDim2  )
            value += UInt8( lbls[c] != lbls[c_N] )
        end

        # for second dimension
        c_N = c + CartesianIndex( 0, 0, -1 )
        if ( doDim2  )
            value += UInt8( lbls[c] != lbls[c_N] )
        end

        contours[c] = UInt8( value > 0 )
    end

    return nothing
end


############# looks at all neighbours, not just previous ones. probably not optimal

function _label_contours_2( lbls::AbstractArray{Int,N},
                            region::Union{Dims, AbstractVector{Int}} = 1:ndims(lbls), 
                          ) where {N}

    lblsize = size( lbls )
    contours = zeros( UInt8, lblsize )
    _label_contours_2!( contours, lbls, region )
    return contours
end

function _label_contours_2!( contours::AbstractArray{UInt8,2},
                             lbls::AbstractArray{Int,2},
                             region::Union{Dims, AbstractVector{Int}} = 1:2, 
                           ) 

    @assert size( contours ) == size( lbls )

    # Checking which dimensions should be analyzed
    doDim1 = 1 in region; 
    doDim2 = 2 in region; 

    # If no dimensions are analyzed, we do nothing (for now).
    if ( !doDim1 && !doDim2 )
        return contours
    end

    ROI = UnitRange.( 2, size( lbls ) ); 

    for c in CartesianIndices( ROI )

        value = UInt8( 0 )

        # for first dimension
        c_N1 = c + CartesianIndex( -1, 0 )
        c_N2 = c + CartesianIndex( -1, 0 )
        if ( doDim1 )
            value += UInt8( lbls[c] != lbls[c_N1] )
            value += UInt8( lbls[c] != lbls[c_N2] )
        end

        # for second dimension
        c_N1 = c + CartesianIndex( 0, -1 )
        c_N2 = c + CartesianIndex( 0,  1 )
        if ( doDim2  )
            value += UInt8( lbls[c] != lbls[c_N1] )
            value += UInt8( lbls[c] != lbls[c_N2] )
        end

        contours[c] = UInt8( value > 0 )
    end

    return nothing
end

function _label_contours_2!( contours::AbstractArray{UInt8,3},
                             lbls::AbstractArray{Int,3},
                             region::Union{Dims, AbstractVector{Int}} = 1:3, 
                           ) 

    @assert size( contours ) == size( lbls )

    # Checking which dimensions should be analyzed
    doDim1 = 1 in region; 
    doDim2 = 2 in region;
    doDim3 = 3 in region;  

    # If no dimensions are analyzed, we do nothing (for now).
    if ( !doDim1 && !doDim2 && !doDim3 )
        return contours
    end

    ROI = UnitRange.( 2, size( lbls ) .- 1 ); 

    for c in CartesianIndices( ROI )

        value = UInt8(0)

        # for first dimension
        c_N1 = c + CartesianIndex( -1, 0, 0 )
        c_N2 = c + CartesianIndex(  1, 0, 0 )
        if ( doDim1 )
            value += UInt8( lbls[c] != lbls[c_N1] )
            #value += UInt8( lbls[c] != lbls[c_N2] )
        end

        # for second dimension
        c_N1 = c + CartesianIndex( 0, -1, 0 )
        c_N2 = c + CartesianIndex( 0,  1, 0 )
        if ( doDim2  )
            value += UInt8( lbls[c] != lbls[c_N1] )
            #value += UInt8( lbls[c] != lbls[c_N2] )
        end

        # for third dimension
        c_N1 = c + CartesianIndex( 0, 0, -1 )
        c_N2 = c + CartesianIndex( 0, 0,  1 )
        if ( doDim3  )
            value += UInt8( lbls[c] != lbls[c_N1] )
            #value += UInt8( lbls[c] != lbls[c_N2] )
        end

        contours[c] = UInt8( value > 0 )
    end

    return nothing
end