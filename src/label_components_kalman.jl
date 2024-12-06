# not using macros

#=
   In space, instead of looking at previous immediate neighbours ((y,x).-(1,0) and (y,x).-(0,1)), 
   we check the neighbour after applying the (inverse) displacement (y,x).-VF[y,x,t]. In other 
   words:

   pos_t   = (y,x)
   vec_t   = VF[ pos_t..., t ]
   pos_t-1 = round( Int, pos_t .- vec_t ); 
   vec_t-1 = VF[ pos_t-1..., t-1 ]
   if angle( vec_t, vec_t-1 ) < th --> (pos_t...,t) inherits label from (pos_t-1...,t-1)

   It is assumed that the last dimension of the inputs is time:
    2D+t vector fields (U,V), where U and V are 3D matrices
    3D+t vector fields (U,V,W), where U, V, and W are 4D matrices

   In addition, the function expects to be given normalized vector fields (VF), and 
   the magnitudes (M) as separate inputs.
=#

function _label_components_kalman( VF::NTuple{NC,AbstractArray{T,ND}},
                                   M::AbstractArray{T,ND};
                                   dot_th::T = T(0.0),
                                   region::Union{Dims, AbstractVector{Int}} = 1:ndims(VF[1]), 
                                   mag_th::T = T(0.0)
                                 ) where {NC,ND,T<:AbstractFloat}

    VFsize = size( VF[1] )
    Albl   = zeros( Int, VFsize )
    _label_components_kalman!( Albl, VF, M, dot_th, mag_th )
    return Albl
end

function _label_components_kalman!( Albl::AbstractArray{Int,3},
                                    VF::NTuple{NC,AbstractArray{T,3}},
                                    M::AbstractArray{T,3},
                                    dot_th::T = T(0.0),
                                    mag_th::T = T(0.0)
                                  ) where {NC,T<:AbstractFloat}

    VFsize = size( VF[1] )
    @assert size( Albl ) == VFsize

    sets     = DisjointMinSets(); 
    prev_pos  = zeros( T, 3 ); 

    for c in CartesianIndices( Albl )
        # NOTE: "c" is a coordinate in space and time ( y, x, t )

        label = typemax(Int)

        if M[c] > mag_th # if the vector isn't too small

            # storing current spatial coordinates in prev_pos
            prev_pos[1] = c.I[1]
            prev_pos[2] = c.I[2]
            prev_pos[3] = c.I[3]-1

            # substracting the current vector (direction .* magnitude) to prev_pos
            # NOTE: we should look at the previous magnitude... no the current one.
            # sub_M_unsafe!( prev_pos, VF, T(2), c )
            sub_M_unsafe!( prev_pos, VF, M, c )

            # dot product between the vector at 
            dot = dot_( VF, c.I, round_and_clip.( prev_pos, 1, size( Albl ) ) )

            if dot > dot_th && Albl[ round_and_clip.( prev_pos, 1, size( Albl ) )... ] > 0# if the two have similar angles...
                newlabel = Albl[ round_and_clip.( prev_pos, 1, size( Albl ) )... ]
                if label != typemax(Int) && label != newlabel
                    label = union!(sets, label, newlabel)  # ...merge labels...
                else
                    label = newlabel  # ...and assign the smaller to current pixel
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
        if M[ i ] > mag_th
            Albl[i] = newlabel[find_root!(sets, Albl[i])]
        end
    end

    return Albl
end

########

function _label_components_kalman( VF_to::NTuple{NC,AbstractArray{T,ND}},
                                   VF_ot::NTuple{NC,AbstractArray{T,ND}},
                                   M_to::AbstractArray{T,ND},
                                   M_ot::AbstractArray{T,ND};
                                   dot_xy::T = T(0.0),
                                    dot_t::T = T(0.0),
                                   region::Union{Dims, AbstractVector{Int}} = 1:ndims(VF_to[1]), 
                                   mag_th::T = T(0.0)
                                 ) where {NC,ND,T<:AbstractFloat}

    VFsize = size( VF_to[1] )
    Albl   = zeros( Int, VFsize )
    _label_components_kalman!( Albl, VF_to, VF_ot, M_to, M_ot, dot_xy, dot_t, region, mag_th )
    return Albl
end

function _label_components_kalman!( Albl::AbstractArray{Int,3},
                                    VF_to::NTuple{NC,AbstractArray{T,3}},
                                    VF_ot::NTuple{NC,AbstractArray{T,3}},
                                    M_to::AbstractArray{T,3},
                                    M_ot::AbstractArray{T,3},
                                    dot_xy::T = T(0.0),
                                     dot_t::T = T(0.0),
                                     region::Union{Dims, AbstractVector{Int}} = 1:3, 
                                    mag_th::T = T(0.0)
                                  ) where {NC,T<:AbstractFloat}

    VFsize = size( VF_to[1] )
    @assert size( Albl ) == VFsize

    # Checking which dimensions should be analyzed
    doDim1 = 1 in region; 
    doDim2 = 2 in region; 
    doDim3 = 3 in region;

    sets     = DisjointMinSets(); 
    prev_pos = zeros( T, 3 ); 
    ROI      = UnitRange.( 2, size( Albl ) ); 

    for c in CartesianIndices( ROI )
        # NOTE: "c" is a coordinate in space and time ( y, x, t )

        label = typemax(Int)

        if M_to[c] > mag_th # if the vector isn't too small

            if doDim3
                # storing current spatial coordinates in prev_pos
                prev_pos[1] = c.I[1]
                prev_pos[2] = c.I[2]
                prev_pos[3] = c.I[3]-1

                # translating prev_pos spatially by VF_ot to move to the previous position
                sum_M_unsafe!( prev_pos, VF_ot, M_ot, c )

                # dot product between the current vector and the vector at the previous position
                dot = dot_( VF_to, c.I, round_and_clip.( prev_pos, 1, size( Albl ) ) )

                # label at previous position
                prev_lbl = Albl[ round_and_clip.( prev_pos, 1, size( Albl ) )... ]

                if dot > dot_t && prev_lbl > 0 # if the two have similar angles...
                    newlabel = prev_lbl
                    if label != typemax(Int) && label != newlabel
                        label = union!(sets, label, newlabel)  # ...merge labels...
                    else
                        label = newlabel  # ...and assign the smaller to current pixel
                    end
                end
            end

            if doDim1 # for first dimension
                c_N = c + CartesianIndex( -1, 0, 0 )
                if ( doDim1 && ( Albl[ c_N ] > 0 ) )
                    dot = dot_( VF_to, c, c_N )
                    if dot > dot_xy  # if the two have similar angles...
                        newlabel = Albl[ c_N ]
                        if label != typemax(Int) && label != newlabel
                            label = union!(sets, label, newlabel)  # ...merge labels...
                        else
                            label = newlabel  # ...and assign the smaller to current pixel
                        end
                    end
                end
            end

            if doDim2 # for second dimension
                c_N = c + CartesianIndex( 0, -1, 0 )
                if ( doDim2 && ( Albl[ c_N ] > 0 ) )
                    dot = dot_( VF_to, c, c_N )
                    if dot > dot_xy  # if the two have similar angles...
                        newlabel = Albl[ c_N ]
                        if label != typemax(Int) && label != newlabel
                            label = union!(sets, label, newlabel)  # ...merge labels...
                        else
                            label = newlabel  # ...and assign the smaller to current pixel
                        end
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
        if M_to[ i ] > mag_th && ( Albl[i] > 0 )
            new_lbl = newlabel[find_root!(sets, Albl[i])]
            Albl[i] = new_lbl
        end
    end

    return Albl
end



# not using macros

#=
   In space, instead of looking at previous immediate neighbours ((y,x).-(1,0) and (y,x).-(0,1)), 
   we check the neighbour after applying the (inverse) displacement (y,x).-VF[y,x,t]. In other 
   words:

   pos_t   = (y,x)
   vec_t   = VF[ pos_t..., t ]
   pos_t-1 = round( Int, pos_t .- vec_t ); 
   vec_t-1 = VF[ pos_t-1..., t-1 ]
   if angle( vec_t, vec_t-1 ) < th --> (pos_t...,t) inherits label from (pos_t-1...,t-1)

   It is assumed that the last dimension of the inputs is time:
    2D+t vector fields (U,V), where U and V are 3D matrices
    3D+t vector fields (U,V,W), where U, V, and W are 4D matrices

   In addition, the function expects to be given normalized vector fields (VF), and 
   the magnitudes (M) as separate inputs.
=#

function _label_components_kalman( VF::NTuple{NC,AbstractArray{T,ND}},
                                   M::AbstractArray{T,ND};
                                   dot_th::T = T(0.0),
                                   region::Union{Dims, AbstractVector{Int}} = 1:ndims(VF[1]), 
                                   mag_th::T = T(0.0)
                                 ) where {NC,ND,T<:AbstractFloat}

    VFsize = size( VF[1] )
    Albl   = zeros( Int, VFsize )
    _label_components_kalman!( Albl, VF, M, dot_th, mag_th )
    return Albl
end

function _label_components_kalman!( Albl::AbstractArray{Int,3},
                                    VF::NTuple{NC,AbstractArray{T,3}},
                                    M::AbstractArray{T,3},
                                    dot_th::T = T(0.0),
                                    mag_th::T = T(0.0)
                                  ) where {NC,T<:AbstractFloat}

    VFsize = size( VF[1] )
    @assert size( Albl ) == VFsize

    sets     = DisjointMinSets(); 
    prev_pos  = zeros( T, 3 ); 

    for c in CartesianIndices( Albl )
        # NOTE: "c" is a coordinate in space and time ( y, x, t )

        label = typemax(Int)

        if M[c] > mag_th # if the vector isn't too small

            # storing current spatial coordinates in prev_pos
            prev_pos[1] = c.I[1]
            prev_pos[2] = c.I[2]
            prev_pos[3] = c.I[3]-1

            # substracting the current vector (direction .* magnitude) to prev_pos
            # NOTE: we should look at the previous magnitude... no the current one.
            # sub_M_unsafe!( prev_pos, VF, T(2), c )
            sub_M_unsafe!( prev_pos, VF, M, c )

            # dot product between the vector at 
            dot = dot_( VF, c.I, round_and_clip.( prev_pos, 1, size( Albl ) ) )

            if dot > dot_th && Albl[ round_and_clip.( prev_pos, 1, size( Albl ) )... ] > 0# if the two have similar angles...
                newlabel = Albl[ round_and_clip.( prev_pos, 1, size( Albl ) )... ]
                if label != typemax(Int) && label != newlabel
                    label = union!(sets, label, newlabel)  # ...merge labels...
                else
                    label = newlabel  # ...and assign the smaller to current pixel
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
        if M[ i ] > mag_th
            Albl[i] = newlabel[find_root!(sets, Albl[i])]
        end
    end

    return Albl
end




########

function _label_components_kalman_t( VF_to::NTuple{NC,AbstractArray{T,ND}},
                                     VF_ot::NTuple{NC,AbstractArray{T,ND}},
                                     M_to::AbstractArray{T,ND},
                                     M_ot::AbstractArray{T,ND};
                                     dot_xy::T = T(0.0),
                                     dot_t::T = T(0.0),
                                     region::Union{Dims, AbstractVector{Int}} = 1:ndims(VF_to[1]), 
                                     mag_th::T = T(0.0)
                                   ) where {NC,ND,T<:AbstractFloat}

    VFsize = size( VF_to[1] )
    Albl   = zeros( Int, VFsize )
    for i in 1:length(Albl)
        Albl[i] = i 
    end
    Albl[1,:,:] .= 0
    Albl[:,1,:] .= 0
    Albl[:,:,1] .= 0
    _label_components_kalman_t!( Albl, VF_to, VF_ot, M_to, M_ot, dot_xy, dot_t, region, mag_th )
    return Albl
end

function _label_components_kalman_t!( Albl::AbstractArray{Int,3},
                                      VF_to::NTuple{NC,AbstractArray{T,3}},
                                      VF_ot::NTuple{NC,AbstractArray{T,3}},
                                      M_to::AbstractArray{T,3},
                                      M_ot::AbstractArray{T,3},
                                      dot_xy::T = T(0.0),
                                      dot_t::T = T(0.0),
                                      region::Union{Dims, AbstractVector{Int}} = 1:3, 
                                      mag_th::T = T(0.0)
                                    ) where {NC,T<:AbstractFloat}

    VFsize = size( VF_to[1] )
    @assert size( Albl ) == VFsize

    # Checking which dimensions should be analyzed
    doDim1 = 1 in region; 
    doDim2 = 2 in region; 
    doDim3 = 3 in region;

    sets     = DisjointMinSets(length(Albl)); 
    prev_pos = zeros( T, 3 ); 
    ROI      = UnitRange.( 2, size( Albl ) ); 

    for c in CartesianIndices( ROI )
        # NOTE: "c" is a coordinate in space and time ( y, x, t )

        label = Albl[c]

        if M_to[c] > mag_th # if the vector isn't too small

            if doDim3
                # storing current spatial coordinates in prev_pos
                prev_pos[1] = c.I[1]
                prev_pos[2] = c.I[2]
                prev_pos[3] = c.I[3]-1

                # translating prev_pos spatially by VF_ot to move to the previous position
                sum_M_unsafe!( prev_pos, VF_ot, M_ot, c )

                # dot product between the current vector and the vector at the previous position
                dot = dot_( VF_to, c.I, round_and_clip.( prev_pos, 1, size( Albl ) ) )

                # label at previous position
                prev_lbl = Albl[ round_and_clip.( prev_pos, 1, size( Albl ) )... ]
                
                if dot > dot_t && prev_lbl > 0 # if the two have similar angles...
                    if label != prev_lbl
                        label = union!(sets, label, prev_lbl)  # ...merge labels...
                    end
                end
            end

            if doDim1 # for first dimension
                c_N = c + CartesianIndex( -1, 0, 0 )
                if doDim1
                    dot = dot_( VF_to, c, c_N )
                    if dot > dot_xy  # if the two have similar angles...
                        newlabel = Albl[ c_N ]
                        if label != newlabel
                            label = union!(sets, label, newlabel)  # ...merge labels...
                        else
                            label = newlabel  # ...and assign the smaller to current pixel
                        end
                    end
                end
            end

            if doDim2 # for second dimension
                c_N = c + CartesianIndex( 0, -1, 0 )
                if doDim2
                    dot = dot_( VF_to, c, c_N )
                    if dot > dot_xy  # if the two have similar angles...
                        newlabel = Albl[ c_N ]
                        if label != newlabel
                            label = union!(sets, label, newlabel)  # ...merge labels...
                        else
                            label = newlabel  # ...and assign the smaller to current pixel
                        end
                    end
                end
            end

            Albl[c] = label
        end
    end

    # Now parse sets to find the labels
    newlabel = minlabel(sets)
    for i = 1:length(Albl)
        if M_to[ i ] > mag_th && Albl[i] > 0
            new_lbl = newlabel[find_root!(sets, Albl[i])]
            Albl[i] = new_lbl
        end
    end

    return Albl
end
