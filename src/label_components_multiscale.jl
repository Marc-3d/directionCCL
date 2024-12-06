#=
    If a position is collective -> inherit label from previous neighbors

    scales = ( (5,5,3), (3,3,1)   ) # a set of scales (in descending order)... one scale for each dimension
    dot_th = ( cosd(10), cosd(10) ) # one threshold for each scale

    Vectors should be normalized
=#

function _label_components_multiscale( VF::NTuple{NC,AbstractArray{T,ND}};
                                       scales::NTuple{NS,NTuple{ND,Int}} = (Tuple(ones(Int,ND)),),
                                       dot_th::NTuple{NS,T} = (T(0),),
                                       region::Union{Dims, AbstractVector{Int}} = 1:ndims(VF[1]), 
                                       mag_th::T = T(0.0)
                                     ) where {NC,ND,NS,T<:AbstractFloat}

    VFsize = size( VF[1] )
    Albl   = ones( Int, VFsize )
    VF_i   = MarcIntegralArrays.getIntegralArray.( VF ); 
    _label_components_multiscale!( Albl, VF_i, scales, dot_th, region, mag_th^2 )
    return Albl
end

function _label_components_multiscale!( Albl::AbstractArray{Int,2},
                                        VF::NTuple{NC,MarcIntegralArrays.IntegralArray{T,2}},
                                        scales::NTuple{NS,NTuple{2,Int}} = ((1,1),),
                                        dot_th::NTuple{NS,T} = (T(0),),
                                        region::Union{Dims, AbstractVector{Int}} = 1:ndims(VF[1]), 
                                        magSQ_th::T = T(0.0)
                                      ) where {NC,NS,T<:AbstractFloat}

    VFsize = size( VF[1] )
    @assert size( Albl ) == VFsize

    sets = DisjointMinSets()
    ROI  = UnitRange.( 2, size( Albl ) ); 

    for si in 1:NS

        # each scale has a different size (rad) and dot threshold
        rad = scales[si]
        dth = dot_th[si]

        for c in CartesianIndices( ROI )

            label = typemax(Int)

            if magSQ_( VF, c ) > magSQ_th

                dot = local_dot_( VF, c.I, rad )

                if dot > dth # the patch around c has similar angles... do all the merging logic
                    # Y
                    newlabel = Albl[ c + CartesianIndex( -1, 0 ) ]
                    if label != typemax(Int) && label != newlabel
                        label = union!(sets, label, newlabel) 
                    else
                        label = newlabel 
                    end
                    # X
                    newlabel = Albl[ c + CartesianIndex( 0, -1 ) ]
                    if label != typemax(Int) && label != newlabel
                        label = union!(sets, label, newlabel) 
                    else
                        label = newlabel
                    end
                end

                if label == typemax(Int)
                    label = push!(sets)   
                end

                Albl[c] = label
            end # if label

        end # cartesian

    end # scales

    # Now parse sets to find the labels
    newlabel = minlabel(sets)
    for i = 1:length(Albl)
        if magSQ_( VF, i ) > mag_th
            Albl[i] = newlabel[find_root!(sets, Albl[i])]
        end
    end

    return Albl
end


function _label_components_multiscale!( Albl::AbstractArray{Int,3},
                                        VF::NTuple{NC,MarcIntegralArrays.IntegralArray{T,3}},
                                        scales::NTuple{NS,NTuple{3,Int}} = ((1,1,1),),
                                        dot_th::NTuple{NS,T} = (T(0),),
                                        region::Union{Dims, AbstractVector{Int}} = 1:ndims(VF[1]), 
                                        magSQ_th::T = T(0.0)
                                      ) where {NC,NS,T<:AbstractFloat}

    VFsize = size( VF[1] )
    @assert size( Albl ) == ( VFsize .- 1 )

    sets = DisjointMinSets()
    ROI  = UnitRange.( 2, size( Albl ) ); 

    for si in 1:NS

        rad = scales[si]
        dth = dot_th[si]
        doDim1 = rad[1] >= 1
        doDim2 = rad[2] >= 1
        doDim3 = rad[3] >= 1

        for c in CartesianIndices( ROI )

            label = typemax(Int)

            dot = local_dot_( VF, c.I, rad )
                
            if dot > dth # if the two have similar angles... do all the merging logic
                # Y
                if doDim1
                    newlabel = Albl[ c + CartesianIndex( -1, 0, 0 ) ]
                    if label != typemax(Int) && label != newlabel
                        label = union!(sets, label, newlabel) 
                    else
                        label = newlabel 
                    end
                end
                # X
                if doDim2
                    newlabel = Albl[ c + CartesianIndex( 0, -1, 0 ) ]
                    if label != typemax(Int) && label != newlabel
                        label = union!(sets, label, newlabel) 
                    else
                        label = newlabel
                    end
                end
                # Z
                if doDim3
                    newlabel = Albl[ c + CartesianIndex( 0, 0, -1 ) ]
                    if label != typemax(Int) && label != newlabel
                        label = union!(sets, label, newlabel) 
                    else
                        label = newlabel
                    end
                end


            end

            if label == typemax(Int)
                label = push!(sets)   
            end

            Albl[c] =  label

        end # cartesian

    # Now parse sets to find the labels
    newlabel = minlabel(sets)
    for i = 1:length(Albl)
        if Albl[i] != 0 # magSQ_( VF, i ) > mag_th
            Albl[i] = newlabel[find_root!(sets, Albl[i])]
        end
    end

    end # scales



    return Albl
end


#######

function local_dot_( VF::NTuple{2,MarcIntegralArrays.IntegralArray{T,ND}}, c::Dims{ND}, rad::Dims{ND} ) where {ND,T}
    dot_ = T(0.0)
    @unroll for i in 1:2
        dot_ += MarcIntegralArrays.integralSumN( VF[i].arr, c .- rad, c )^2
    end
    return dot_
end

function local_dot_( VF::NTuple{3,MarcIntegralArrays.IntegralArray{T,ND}}, c::Dims{ND}, rad::Dims{ND} ) where {ND,T}
    dot_ = T(0.0)
    @unroll for i in 1:3
        dot_ += MarcIntegralArrays.integralSumN( VF[i].arr, c .- rad, c )^2
    end
    return dot_
end

function local_dot_unsafe( VF::NTuple{2,MarcIntegralArrays.IntegralArray{T,ND}}, TL::Dims{ND}, BR::Dims{ND} ) where {ND,T}
    dot_ = T(0.0)
    @unroll for i in 1:2
        dot_ += MarcIntegralArrays.integralSumN_unsafe( VF[i].arr, TL, BR )^2
    end
    return dot_
end

function local_dot_unsafe( VF::NTuple{3,MarcIntegralArrays.IntegralArray{T,ND}}, TL::Dims{ND}, BR::Dims{ND} ) where {ND,T}
    dot_ = T(0.0)
    @unroll for i in 1:3
        dot_ += MarcIntegralArrays.integralSumN_unsafe( VF[i].arr, TL, BR )^2
    end
    return dot_
end

function local_dot_unsafe( VF1::NTuple{2,MarcIntegralArrays.IntegralArray{T,ND}},
                           TL1::Dims{ND}, BR1::Dims{ND}, TL2::Dims{ND}, BR2::Dims{ND} ) where {ND,T}
    dot_ = T(0.0)
    @unroll for i in 1:2
        dot_ += MarcIntegralArrays.integralSumN_unsafe( VF1[i].arr, TL1, BR1 )*MarcIntegralArrays.integralSumN_unsafe( VF1[i].arr, TL2, BR2 )
    end
    return dot_
end

function local_dot_unsafe( VF1::NTuple{3,MarcIntegralArrays.IntegralArray{T,ND}},
                           TL1::Dims{ND}, BR1::Dims{ND}, TL2::Dims{ND}, BR2::Dims{ND} ) where {ND,T}
    dot_ = T(0.0)
    @unroll for i in 1:3
        dot_ += MarcIntegralArrays.integralSumN_unsafe( VF1[i].arr, TL1, BR1 )*MarcIntegralArrays.integralSumN_unsafe( VF1[i].arr, TL2, BR2 )
    end
    return dot_
end

#############################


function _label_components_multiscale_t( VF::NTuple{NC,AbstractArray{T,ND}};
                                         scales::NTuple{NS,NTuple{ND,Int}} = (Tuple(ones(Int,ND)),),
                                         dot_th::NTuple{NS,T} = (T(0),),
                                         region::Union{Dims, AbstractVector{Int}} = 1:ndims(VF[1]), 
                                         mag_th::T = T(0.0)
                                       ) where {NC,ND,NS,T<:AbstractFloat}

    # Each pixel becomes its own element
    VFsize = size( VF[1] )
    Albl   = ones( Int, VFsize )
    @inbounds @simd for i in 1:length(Albl)
        Albl[i] = i 
    end
    # Computing integral array of each vector field component
    VF_i   = MarcIntegralArrays.getIntegralArray.( VF ); 
    # Needed matrices
    U = zeros( Int8, size( Albl ) )
    V = zeros( Int8, size( Albl ) )
    W = zeros( Int8, size( Albl ) )
    done = zeros( Bool, size( Albl ) )
    # Doing the thing
    _label_components_multiscale_t!( Albl, U, V, W, done, VF_i, scales, dot_th, region, mag_th^2 )
    return Albl
end

# The thing in 2d
function _label_components_multiscale_t!( Albl::AbstractArray{Int,2},
                                        VF::NTuple{NC,MarcIntegralArrays.IntegralArray{T,2}},
                                        scales::NTuple{NS,NTuple{2,Int}} = ((1,1),),
                                        dot_th::NTuple{NS,T} = (T(0),),
                                        region::Union{Dims, AbstractVector{Int}} = 1:ndims(VF[1]), 
                                        magSQ_th::T = T(0.0)
                                      ) where {NC,NS,T<:AbstractFloat}

    VFsize = size( VF[1] )
    @assert size( Albl ) == VFsize

    sets = DisjointMinSets()
    ROI  = UnitRange.( 2, size( Albl ) ); 

    for si in 1:NS

        # each scale has a different size (rad) and dot threshold
        rad = scales[si]
        dth = dot_th[si]

        for c in CartesianIndices( ROI )

            label = typemax(Int)

            if magSQ_( VF, c ) > magSQ_th

                dot = local_dot_( VF, c.I, rad )

                if dot > dth # the patch around c has similar angles... do all the merging logic
                    # Y
                    newlabel = Albl[ c + CartesianIndex( -1, 0 ) ]
                    if label != typemax(Int) && label != newlabel
                        label = union!(sets, label, newlabel) 
                    else
                        label = newlabel 
                    end
                    # X
                    newlabel = Albl[ c + CartesianIndex( 0, -1 ) ]
                    if label != typemax(Int) && label != newlabel
                        label = union!(sets, label, newlabel) 
                    else
                        label = newlabel
                    end
                end

                if label == typemax(Int)
                    label = push!(sets)   
                end

                Albl[c] = label
            end # if label

        end # cartesian

    end # scales

    # Now parse sets to find the labels
    newlabel = minlabel(sets)
    for i = 1:length(Albl)
        if magSQ_( VF, i ) > mag_th
            Albl[i] = newlabel[find_root!(sets, Albl[i])]
        end
    end

    return Albl
end

# The thing in 3d
function _label_components_multiscale_t!( Albl::AbstractArray{Int,3},
                                          U::AbstractArray{Int8,3},
                                          V::AbstractArray{Int8,3},
                                          W::AbstractArray{Int8,3},
                                          done::AbstractArray{Bool,3},
                                          VF::NTuple{NC,MarcIntegralArrays.IntegralArray{T,3}},
                                          scales::NTuple{NS,NTuple{3,Int}} = ((1,1,1),),
                                          dot_th::NTuple{NS,T} = (T(0),),
                                          region::Union{Dims, AbstractVector{Int}} = 1:ndims(VF[1]), 
                                          magSQ_th::T = T(0.0)
                                        ) where {NC,NS,T<:AbstractFloat}

    VFsize = size( VF[1] )
    Asize  = size( Albl ); 
    @assert Asize == ( VFsize .- 1 )

    sets = DisjointMinSets( length(Albl) ); 
    ROI  = UnitRange.( 2, size( Albl ) ); 

    rects = [];

    for si in 1:NS

        rad = scales[si]
        dth = dot_th[si]

        # doDim1 = rad[1] >= 1
        # doDim2 = rad[2] >= 1
        # doDim3 = rad[3] >= 1
        # UP   = ( -rad[1]-1, 0, 0 )
        # LEFT = ( 0, -rad[2]-1, 0 )
        # FRNT = ( 0, 0, -rad[3]-1 )
        # DOWN = (  rad[1]+1, 0, 0 )
        # RIGT = ( 0,  rad[2]+1, 0 )
        # BACK = ( 0, 0,  rad[3]+1 )


        idx = 0; 
        for c in CartesianIndices( ROI )
            idx += 1; 

            if done[ c ]
                continue
            end

            label = Albl[ c ]
            TLF   = clipmin.( c.I .- rad )
            DRB   = clipmax.( c.I .+ rad, Asize )
            dot   = local_dot_unsafe( VF, TLF, DRB )
                
            if dot > dth # similar ROI

                # inherit label from c + ( u, v, w ). At the beginning (u,v,w) == 0, 
                # so each initial similar patch gets the value of the central element.
                # new_lbl = Albl[ c.I[1] + U[c], c.I[2] + V[c], c.I[3] + W[c] ]; 
                # new_lbl = check_rect_intersect( rects, TLF, DRB, Albl[c] )
                new_lbl = check_rect_intersect( VF, dth, rects, TLF, DRB, Albl[c], c.I, rad )

                # Set the surrounding ROI to the same lbl as the center
                Albl[ UnitRange.( TLF, DRB )... ] .= new_lbl
                done[ UnitRange.( TLF, DRB )... ] .= true

                # Set the outer points around the surrounding area to point towards
                # U[ clipmin.( c.I .+  UP  )... ] = 1
                # V[ clipmin.( c.I .+ LEFT )... ] = 1
                # W[ clipmin.( c.I .+ FRNT )... ] = 1 
                # U[ clipmax.( c.I .+ DOWN, Asize )... ] = -1
                # V[ clipmax.( c.I .+ RIGT, Asize )... ] = -1
                # W[ clipmax.( c.I .+ BACK, Asize )... ] = -1

                push!( rects, ( TLF, DRB, new_lbl, c.I, rad ) ); 
            end # if the two have similar angles... 
        end # cartesian
    end # scales



    return Albl
end


function  check_rect_intersect( VF, dot_th, rects, TLF::Dims{3}, DRB::Dims{3}, this_lbl, c, rad )

    ymin, xmin, zmin = TLF;
    ymax, xmax, zmax = DRB; 

    for rect in rects
        ymin_, xmin_, zmin_ = rect[1];
        ymax_, xmax_, zmax_ = rect[2]; 
        intersects = ( ( ymin_ < ymin < ymax_ ) || ( ymin_ < ymax < ymax_ ) ) &&
                     ( ( xmin_ < xmin < xmax_ ) || ( xmin_ < xmax < xmax_ ) ) &&
                     ( ( zmin_ < zmin < zmax_ ) || ( zmin_ < zmax < zmax_ ) )
        # c_ = rect[4]
        # intersects = any( abs.( ( c_ .- c ) ) .<  rad )
        if intersects
            # check if angles are similar
            dot = local_dot_unsafe( VF, TLF, DRB, rect[1], rect[2] )
            if dot > dot_th
                return rect[3]
            end
        end
    end
    return this_lbl
end
