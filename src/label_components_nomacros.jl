#=
   Macros allow to create code that generalizes to arbitrary dimensions by automating
   the creation of "for-loops". However, macros are not very flexible and they can be
   very hard to read and understand. 

   In this file I've rewriten label_components! for inputs of several dimensions.
   These functions include independent thresholds for each dimension.

   We support:
    2D vector fields
    3D vector fields, applicable to both 3D + 2D+t.
    TODO: 4D vector fields, specifically 3D+t
=#

function _label_components( VF::NTuple{NC,AbstractArray{T,ND}};
                            dot_th::NTuple{ND,T} = Tuple(zeros(T,ND)),
                            region::Union{Dims, AbstractVector{Int}} = 1:ndims(VF[1]), 
                            mag_th::T = T(0.0)
                          ) where {NC,ND,T<:AbstractFloat}

    VFsize = size( VF[1] )
    Albl   = zeros( Int, VFsize )
    _label_components!( Albl, VF, dot_th, region, mag_th^2 )
    return Albl
end

function _label_components!( Albl::AbstractArray{Int,2},
                             VF::NTuple{NC,AbstractArray{T,2}},
                             dot_th::NTuple{2,T} = Tuple(zeros(T,2)),
                             region::Union{Dims, AbstractVector{Int}} = 1:2, 
                             mag_th::T = T(0.0)
                           ) where {NC,T<:AbstractFloat}

    VFsize = size( VF[1] )
    @assert size( Albl ) == VFsize

    # Checking which dimensions should be analyzed
    doDim1 = 1 in region; 
    doDim2 = 2 in region; 

    # If no dimensions are analyzed, we do nothing (for now).
    if ( !doDim1 && !doDim2 )
        return Albl
    end

    # This is needed before the for loop
    sets = DisjointMinSets()

    # This function looks at adjacent neighbors (up,left,front). Thus, the
    # first index of each dimensions is skipped. This was also done in the
    # "macro" version of this function.
    ROI = UnitRange.( 2, size( Albl ) ); 

    for c in CartesianIndices( ROI )

        label = typemax(Int)

        if magSQ_( VF, c ) > mag_th # if the vector isn't too small

            # NOTE: we might be able to use a macro to deal with conditially excluding dimensions at compile time... before the for loop.

            # for first dimension
            c_N = c + CartesianIndex( -1, 0 )
            if ( doDim1 && ( Albl[ c_N ] > 0 ) )
                dot = dot_( VF, c, c_N )
                if dot > dot_th[1]  # if the two have similar angles...
                    newlabel = Albl[ c_N ]
                    if label != typemax(Int) && label != newlabel
                        label = union!(sets, label, newlabel)  # ...merge labels...
                    else
                        label = newlabel  # ...and assign the smaller to current pixel
                    end
                end
            end

            # for second dimension
            c_N = c + CartesianIndex( 0, -1 )
            if ( doDim2 && ( Albl[ c_N ] > 0 ) )
                dot = dot_( VF, c, c_N )
                if dot > dot_th[2]  # if the two have similar angles...
                    newlabel = Albl[ c_N ]
                    if label != typemax(Int) && label != newlabel
                        label = union!(sets, label, newlabel)  # ...merge labels...
                    else
                        label = newlabel  # ...and assign the smaller to current pixel
                    end
                end
            end

            # there were no neighbors, create a new label
            if label == typemax(Int)
                label = push!(sets)   
            end

            Albl[c] = label
        end
    end

    # Now parse sets to find the labels
    newlabel = minlabel(sets)
    for i = 1:length(Albl)
        if magSQ_( VF, i ) > mag_th
            Albl[i] = newlabel[find_root!(sets, Albl[i])]
        end
    end

    return Albl
end


function _label_components!( Albl::AbstractArray{Int,3},
                             VF::NTuple{NC,AbstractArray{T,3}},
                             dot_th::NTuple{3,T} = Tuple(zeros(T,3)),
                             region::Union{Dims, AbstractVector{Int}} = 1:3,
                             mag_th::T = T(0.0)
                           ) where {NC,T<:AbstractFloat}

    VFsize = size( VF[1] )
    @assert size( Albl ) == VFsize

    # Checking which dimensions should be analyzed
    doDim1 = 1 in region; 
    doDim2 = 2 in region; 
    doDim3 = 3 in region;

    # If no dimensions are analyzed, we do nothing (for now).
    if ( !doDim1 && !doDim2 && !doDim3 )
        return Albl
    end

    # This is needed before the for loop
    sets = DisjointMinSets()

    # This function looks at adjacent neighbors (up,left,front). Thus, the
    # first index of each dimensions is skipped. This was also done in the
    # "macro" version of this function.
    ROI = UnitRange.( 2, size( Albl ) ); 

    for c in CartesianIndices( ROI )

        label = typemax(Int)

        if magSQ_( VF, c ) > mag_th # if the vector isn't too small

            # NOTE: we might be able to use a macro to deal with conditially excluding dimensions at compile time... before the for loop.

            # for first dimension
            c_N = c + CartesianIndex( -1, 0, 0 )
            if ( doDim1 && ( Albl[ c_N ] > 0 ) )
                dot = dot_( VF, c, c_N )
                if dot > dot_th[1]  # if the two have similar angles...
                    newlabel = Albl[ c_N ]
                    if label != typemax(Int) && label != newlabel
                        label = union!(sets, label, newlabel)  # ...merge labels...
                    else
                        label = newlabel  # ...and assign the smaller to current pixel
                    end
                end
            end

            # for second dimension
            c_N = c + CartesianIndex( 0, -1, 0 )
            if ( doDim2 && ( Albl[ c_N ] > 0 ) )
                dot = dot_( VF, c, c_N )
                if dot > dot_th[2]  # if the two have similar angles...
                    newlabel = Albl[ c_N ]
                    if label != typemax(Int) && label != newlabel
                        label = union!(sets, label, newlabel)  # ...merge labels...
                    else
                        label = newlabel  # ...and assign the smaller to current pixel
                    end
                end
            end

            # for third dimension
            c_N = c + CartesianIndex( 0, 0, -1 )
            if ( doDim2 && ( Albl[ c_N ] > 0 ) )
                dot = dot_( VF, c, c_N )
                if dot > dot_th[3]  # if the two have similar angles...
                    newlabel = Albl[ c_N ]
                    if label != typemax(Int) && label != newlabel
                        label = union!(sets, label, newlabel)  # ...merge labels...
                    else
                        label = newlabel  # ...and assign the smaller to current pixel
                    end
                end
            end

            # there were no neighbors, create a new label
            if label == typemax(Int)
                label = push!(sets)   
            end

            Albl[c] = label
        end
    end

    # Now parse sets to find the labels
    newlabel = minlabel(sets)
    for i = 1:length(Albl)
        if magSQ_( VF, i ) > mag_th && Albl[i] > 0 
            Albl[i] = newlabel[find_root!(sets, Albl[i])]
        end
    end

    return Albl
end


#######################################



function _label_components_t( VF::NTuple{NC,AbstractArray{T,ND}};
                              dot_th::NTuple{ND,T} = Tuple(zeros(T,ND)),
                              region::Union{Dims, AbstractVector{Int}} = 1:ndims(VF[1]), 
                              mag_th::T = T(0.0)
                            ) where {NC,ND,T<:AbstractFloat}

    VFsize = size( VF[1] )
    Albl   = zeros( Int, VFsize )
    for i in 1:length(Albl)
        Albl[i] = i 
    end
    _label_components_t!( Albl, VF, dot_th, region, mag_th^2 )
    return Albl
end

function _label_components_t!( Albl::AbstractArray{Int,2},
                               VF::NTuple{NC,AbstractArray{T,2}},
                               dot_th::NTuple{2,T} = Tuple(zeros(T,2)),
                               region::Union{Dims, AbstractVector{Int}} = 1:2, 
                               mag_th::T = T(0.0)
                             ) where {NC,T<:AbstractFloat}

    VFsize = size( VF[1] )
    @assert size( Albl ) == VFsize

    # Checking which dimensions should be analyzed
    doDim1 = 1 in region; 
    doDim2 = 2 in region; 
    if ( !doDim1 && !doDim2 )
        return Albl
    end

    sets = DisjointMinSets( length(Albl) )
    ROI = UnitRange.( 2, size( Albl ) ); 

    for c in CartesianIndices( ROI )

        if magSQ_( VF, c ) > mag_th # if the vector isn't too small

            label = Albl[c]

            # for first dimension
            c_N = c + CartesianIndex( -1, 0 )
            if ( doDim1 )
                dot = dot_( VF, c, c_N )
                if dot > dot_th[1]  # if the two have similar angles...
                    newlabel = Albl[ c_N ]
                    if label != newlabel
                        label = union!(sets, label, newlabel)  # ...merge labels...
                    else
                        label = newlabel  # ...and assign the smaller to current pixel
                    end
                end
            end

            # for second dimension
            c_N = c + CartesianIndex( 0, -1 )
            if ( doDim2 )
                dot = dot_( VF, c, c_N )
                if dot > dot_th[2]  # if the two have similar angles...
                    newlabel = Albl[ c_N ]
                    if label != newlabel
                        label = union!(sets, label, newlabel)  # ...merge labels...
                    else
                        label = newlabel  # ...and assign the smaller to current pixel
                    end
                end
            end

            Albl[c] = label
        end
    end

    # Now parse sets to find the labels
    newlabel = minlabel(sets)
    for i = 1:length(Albl)
        if magSQ_( VF, i ) > mag_th
            Albl[i] = newlabel[find_root!(sets, Albl[i])]
        end
    end

    return Albl
end


function _label_components_t!( Albl::AbstractArray{Int,3},
                               VF::NTuple{NC,AbstractArray{T,3}},
                               dot_th::NTuple{3,T} = Tuple(zeros(T,3)),
                               region::Union{Dims, AbstractVector{Int}} = 1:3,
                               mag_th::T = T(0.0)
                             ) where {NC,T<:AbstractFloat}

    VFsize = size( VF[1] )
    @assert size( Albl ) == VFsize

    # Checking which dimensions should be analyzed
    doDim1 = 1 in region; 
    doDim2 = 2 in region; 
    doDim3 = 3 in region;
    if ( !doDim1 && !doDim2 && !doDim3 )
        return Albl
    end

    # This is needed before the for loop
    sets = DisjointMinSets( length(Albl))
    ROI  = UnitRange.( 2, size( Albl ) ); 

    for c in CartesianIndices( ROI )

        label = Albl[ c ]

        if magSQ_( VF, c ) > mag_th # if the vector isn't too small

            # for first dimension
            c_N = c + CartesianIndex( -1, 0, 0 )
            if ( doDim1 )
                dot = dot_( VF, c, c_N )
                if dot > dot_th[1]  # if the two have similar angles...
                    newlabel = Albl[ c_N ]
                    if label != newlabel
                        label = union!(sets, label, newlabel)  # ...merge labels...
                    else
                        label = newlabel  # ...and assign the smaller to current pixel
                    end
                end
            end

            # for second dimension
            c_N = c + CartesianIndex( 0, -1, 0 )
            if ( doDim2 )
                dot = dot_( VF, c, c_N )
                if dot > dot_th[2]  # if the two have similar angles...
                    newlabel = Albl[ c_N ]
                    if label != newlabel
                        label = union!(sets, label, newlabel)  # ...merge labels...
                    else
                        label = newlabel  # ...and assign the smaller to current pixel
                    end
                end
            end

            # for third dimension
            c_N = c + CartesianIndex( 0, 0, -1 )
            if ( doDim3 )
                dot = dot_( VF, c, c_N )
                if dot > dot_th[3]  # if the two have similar angles...
                    newlabel = Albl[ c_N ]
                    if label != newlabel
                        label = union!(sets, label, newlabel)  # ...merge labels...
                    else
                        label = newlabel  # ...and assign the smaller to current pixel
                    end
                end
            end

            Albl[c] = label
        end
    end

    # Now parse sets to find the labels
    newlabel = minlabel(sets)
    for i = 1:length(Albl)
        if magSQ_( VF, i ) > mag_th && Albl[i] > 0 
            Albl[i] = newlabel[find_root!(sets, Albl[i])]
        end
    end

    return Albl
end