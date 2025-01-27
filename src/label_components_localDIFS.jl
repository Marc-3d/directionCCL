"""
"""
function _label_components_localDifs( 
    data::AbstractArray{T,ND},
    difs::AbstractArray{T,ND},
    stds::AbstractArray{T,ND};
    region::Union{Dims, AbstractVector{Int}} = 1:ndims(data), 
    tol::T = T(1)
) where {
    ND,
    T<:Real
}
    Albl = zeros( Int, size( data ) )
    for i in 1:length(Albl)
        Albl[i] = i 
    end
    _label_components_localDifs!( Albl, data, difs, stds, region, tol )
    return Albl
end

"""
"""
function _label_components_localDifs!( 
    Albl::AbstractArray{Int,2},
    data::AbstractArray{T,2},
    difs::AbstractArray{T,2},
    stds::AbstractArray{T,2},
    region::Union{Dims, AbstractVector{Int}} = 1:2, 
    tol::T = T(1)
) where {
    T<:Real
}
    MTHsize = size( data )
    @assert size( Albl ) == MTHsize

    # Checking which dimensions should be analyzed
    doDim1 = 1 in region; 
    doDim2 = 2 in region; 

    # If no dimensions are analyzed, we do nothing (for now).
    if ( !doDim1 && !doDim2 )
        return Albl
    end

    # This is needed before the for loop
    sets = DisjointMinSets(length(Albl))

    # This function looks at adjacent neighbors (up,left,front). Thus, the first index of each dimensions is skipped. This was also done in the "macro" version of this function.
    ROI = UnitRange.( 2, size( Albl ) );


    for c in CartesianIndices( ROI )

        # for first dimension
        c_N = c + CartesianIndex( -1, 0 )
        if ( doDim1 
        && ( ( data[ c ] - data[ c_N ] )^2 < ( difs[ c ] - stds[ c ] * tol ) ) 
        && ( Albl[ c_N ] != Albl[ c ] ) 
        )
            label = union!( sets, Albl[ c ], Albl[ c_N ] )
            Albl[ c ] = label
        end

        # for second dimension
        c_N = c + CartesianIndex( 0, -1 )
        if ( doDim2 
        && ( ( data[ c ] - data[ c_N ] )^2 < ( difs[ c ] - stds[ c ] * tol ) ) 
        && ( Albl[ c_N ] != Albl[ c ] )
        )
            label = union!( sets, Albl[ c ], Albl[ c_N ] )
            Albl[ c ] = label
        end
    end

    # Now parse sets to find the labels
    newlabel = minlabel(sets)
    for i = 1:length(Albl)
        Albl[i] = newlabel[find_root!(sets, Albl[i])]
    end

    return Albl
end


"""
"""
function _label_components_inout_difs( 
    indifs::AbstractArray{T,ND},
    outdifs::AbstractArray{T,ND};
    region::Union{Dims, AbstractVector{Int}} = 1:ndims(indifs), 
    tol::T = T(1)
) where {
    ND,
    T<:Real
}
    Albl = zeros( Int, size( indifs ) )
    for i in 1:length(Albl)
        Albl[i] = i 
    end
    _label_components_inout_difs!( Albl, indifs, outdifs, region, tol )
    return Albl
end

"""
"""
function _label_components_inout_difs!( 
    Albl::AbstractArray{Int,2},
    indifs::AbstractArray{T,2},
    outdifs::AbstractArray{T,2},
    region::Union{Dims, AbstractVector{Int}} = 1:2, 
    tol::T = T(1)
) where {
    T<:Real
}
    _size = size( indifs )
    @assert size( Albl ) == _size

    # Checking which dimensions should be analyzed
    doDim1 = 1 in region; 
    doDim2 = 2 in region; 

    # If no dimensions are analyzed, we do nothing (for now).
    if ( !doDim1 && !doDim2 )
        return Albl
    end

    # This is needed before the for loop
    sets = DisjointMinSets(length(Albl))

    # This function looks at adjacent neighbors (up,left,front). Thus, the first index of each dimensions is skipped. This was also done in the "macro" version of this function.
    ROI = UnitRange.( 2, size( Albl ) );


    for c in CartesianIndices( ROI )

        # for first dimension
        c_N = c + CartesianIndex( -1, 0 )
        if ( doDim1 
        && ( indifs[c] < outdifs[c] ) 
        && ( Albl[ c_N ] != Albl[ c ] ) 
        )
            label = union!( sets, Albl[ c ], Albl[ c_N ] )
            Albl[ c_N ] = label
            Albl[ c ] = label
        end

        # for second dimension
        c_N = c + CartesianIndex( 0, -1 )
        if ( doDim2 
        && ( indifs[c] < outdifs[c] )
        && ( Albl[ c_N ] != Albl[ c ] )
        )
            label = union!( sets, Albl[ c ], Albl[ c_N ] )
            Albl[ c ] = label
            Albl[ c_N ] = label
        end
    end

    # Now parse sets to find the labels
    newlabel = minlabel(sets)
    for i = 1:length(Albl)
        Albl[i] = newlabel[find_root!(sets, Albl[i])]
    end

    return Albl
end


"""
"""
function _label_components_inout_difs( 
    ondifs::AbstractArray{T,ND},
    indifs::AbstractArray{T,ND},
    outdifs::AbstractArray{T,ND};
    region::Union{Dims, AbstractVector{Int}} = 1:ndims(indifs), 
    tol::T = T(1)
) where {
    ND,
    T<:Real
}
    Albl = zeros( Int, size( indifs ) )
    for i in 1:length(Albl)
        Albl[i] = i 
    end
    _label_components_inout_difs!( Albl, ondifs, indifs, outdifs, region, tol )
    return Albl
end

"""
"""
function _label_components_inout_difs!( 
    Albl::AbstractArray{Int,2},
    ondifs::AbstractArray{T,2},
    indifs::AbstractArray{T,2},
    outdifs::AbstractArray{T,2},
    region::Union{Dims, AbstractVector{Int}} = 1:2, 
    tol::T = T(1)
) where {
    T<:Real
}
    _size = size( indifs )
    @assert size( Albl ) == _size

    # Checking which dimensions should be analyzed
    doDim1 = 1 in region; 
    doDim2 = 2 in region; 

    # If no dimensions are analyzed, we do nothing (for now).
    if ( !doDim1 && !doDim2 )
        return Albl
    end

    # This is needed before the for loop
    sets = DisjointMinSets(length(Albl))

    # This function looks at adjacent neighbors (up,left,front). Thus, the first index of each dimensions is skipped. This was also done in the "macro" version of this function.
    ROI = UnitRange.( 2, size( Albl ) );


    for c in CartesianIndices( ROI )

        # for first dimension
        c_N = c + CartesianIndex( -1, 0 )
        if ( doDim1 
        && ( ondifs[c] < outdifs[c] )
        && ( indifs[c] < outdifs[c] ) 
        && ( Albl[ c_N ] != Albl[ c ] ) 
        )
            label = union!( sets, Albl[ c ], Albl[ c_N ] )
            Albl[ c_N ] = label
            Albl[ c ] = label
        end

        # for second dimension
        c_N = c + CartesianIndex( 0, -1 )
        if ( doDim2 
        && ( ondifs[c] < outdifs[c] )
        && ( indifs[c] < outdifs[c] )
        && ( Albl[ c_N ] != Albl[ c ] )
        )
            label = union!( sets, Albl[ c ], Albl[ c_N ] )
            Albl[ c ] = label
            Albl[ c_N ] = label
        end
    end

    # Now parse sets to find the labels
    newlabel = minlabel(sets)
    for i = 1:length(Albl)
        Albl[i] = newlabel[find_root!(sets, Albl[i])]
    end

    return Albl
end
