function _label_components_MTH( 
    MTH::AbstractArray{T,ND};
    region::Union{Dims, AbstractVector{Int}} = 1:ndims(MTH), 
    tol::T = T(1),
    max_th::T = maximum(MTH),
) where {
    ND,
    T<:Real
}
    Albl = zeros( Int, size( MTH ) )
    for i in 1:length(Albl)
        Albl[i] = i 
    end
    _label_components_MTH!( Albl, MTH, region, tol, max_th )
    return Albl
end

function _label_components_MTH!( 
    Albl::AbstractArray{Int,2},
    MTH::AbstractArray{T,2},
    region::Union{Dims, AbstractVector{Int}} = 1:2, 
    tol::T = T(1),
    max_th::T = maximum( MTH )
) where {
    T<:Real
}
    MTHsize = size( MTH )
    @assert size( Albl ) == MTHsize

    # Checking which dimensions should be analyzed
    doDim1 = 1 in region; 
    doDim2 = 2 in region; 

    # If no dimensions are analyzed, we do nothing (for now).
    if ( !doDim1 && !doDim2 )
        return Albl
    end

    println( "adsf")

    # This is needed before the for loop
    sets = DisjointMinSets(length(Albl))

    # This function looks at adjacent neighbors (up,left,front). Thus, the
    # first index of each dimensions is skipped. This was also done in the
    # "macro" version of this function.
    ROI = UnitRange.( 2, size( Albl ) ); 

    for c in CartesianIndices( ROI )

        if MTH[ c ] > max_th
            continue
        end

        # for first dimension
        c_N = c + CartesianIndex( -1, 0 )
        if ( doDim1 && ( abs( MTH[ c_N ] - MTH[ c ] ) <= tol ) && ( Albl[ c_N ] != Albl[ c ] ) )

            label = union!( sets, Albl[ c ], Albl[ c_N ] )

            if MTH[ c ] > MTH[ c_N ]
                # if this label is threshold index is greater... merge with lower index
                Albl[ c ] = Albl[ c_N ]
            else 
                # if this label is threshold index is greater... merge with lower index
                Albl[ c_N ] = Albl[ c ]
            end
        end

        # for second dimension
        c_N = c + CartesianIndex( 0, -1 )
        if ( doDim2 && ( abs( MTH[ c_N ] - MTH[ c ] ) <= tol ) && ( Albl[ c_N ] != Albl[ c ] ) )
            
            label = union!( sets, Albl[ c ], Albl[ c_N ] )

            if MTH[ c ] > MTH[ c_N ]
                # if this label is threshold index is greater... merge with lower index
                Albl[ c ] = Albl[ c_N ]
            else 
                # if this label is threshold index is greater... merge with lower index
                Albl[ c_N ] = Albl[ c ]
            end
        end
    end

    # Now parse sets to find the labels
    newlabel = minlabel(sets)
    for i = 1:length(Albl)
        Albl[i] = newlabel[find_root!(sets, Albl[i])]
    end

    return Albl
end