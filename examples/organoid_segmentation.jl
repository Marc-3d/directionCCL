using ColorTypes, FileIO, PyPlot
include("R:/users/pereyram/Gits/multi_quickPIV/src/multi_quickPIV.jl");
include("R:/users/pereyram/Gits/directionCCL/src/directionCCL.jl"); 
include("R:/users/pereyram/Gits/MarcIntegralArrays/src/MarcIntegralArrays.jl"); 

#= 
 For each pixel, we wish to compute the average vector in the direction of (A) the greatest difference or (B) the highest gradient. This is done by considering a local square/rectangular ROI around each pixel, and computing a weighted sum between normalized vectors in a divergent pattern (aka pointing away from the center of the ROI) and the difference of each local pixel with respect to the central value:  

 GradientL2_X[ ROI ] = ∑(( i - c )²*x) = ∑(i²x) + ∑(c²x) - 2c∑(ix)  
 GradientL2_Y[ ROI ] = ∑(( i - c )²*y) = ∑(i²y) + ∑(c²y) - 2c∑(iy) 

where "i" indicates the intensity of each pixel withing the ROI and "c" is the value of the central pixel in the ROI. If the kernel is anti-symmetric, which is the case of the diverging pattern that I use, ∑(c²x) and ∑(c²y) cancel out.

For signed difference:

 Gradient_X[ ROI ] = ∑(( i - c )x) = ∑(ix) - ∑(cx) 
 Gradient_Y[ ROI ] = ∑(( i - c )y) = ∑(iy) - ∑(cy)

Again, ∑(cx) and ∑(cy) cancel out.
=#
function gradientL2( img::AbstractArray{T,2}, krad::Dims{2}, norm=true ) where {T}

    img2  = img .* img; 

    ky    = -krad[1]:krad[1]; 
    kx    = -krad[2]:krad[2];
    ykern = T[ y/sqrt(x^2+y^2) for y in ky, x in kx ]
    xkern = T[ x/sqrt(x^2+y^2) for y in ky, x in kx ]
    ykern[ isnan.( ykern ) ] .= 0
    xkern[ isnan.( xkern ) ] .= 0

    sumix  = multi_quickPIV.FFTCC_crop( xkern, img  ); 
    sumi2x = multi_quickPIV.FFTCC_crop( xkern, img2 ); 
    sumiy  = multi_quickPIV.FFTCC_crop( ykern, img  ); 
    sumi2y = multi_quickPIV.FFTCC_crop( ykern, img2 );

    U = sumi2y .- 2 .* img .* sumiy
    V = sumi2x .- 2 .* img .* sumix
    if norm
        M = sqrt.( U .^ 2 .+ V .^ 2 ); 
        U[ M .> 0 ] ./= M[ M .> 0 ]
        V[ M .> 0 ] ./= M[ M .> 0 ]
    end 

    return U, V
end

# NOTE: The naive implementation in 3D of the code above could consume EXTREME amounts of memory. For instance, if the input volume occupies 1Gb of RAM, simply creating the U, V and W arrays of the same size and type will already occupy 3Gb of RAM. Each intermediary convolution takes the same amount of memory as input volume, plus each call to multi_quickPIV.FFTCC_crop allocates temporary variables of similar sizes as the input. Therefore, it is very important to allocate the minimum memory required, and to operate in-place as much as possible. These optimizations are implemented at the bottom of this file. 
function gradientL2( vol::AbstractArray{T,3}, krad::Dims{3}, norm=true ) where {T}
    return gradientL2_opt( vol, krad, norm ); 
end

function gradient( img::AbstractArray{T,N}, krad::Dims{N}, norm=true ) where {T,N}

    ky    = -krad[1]:krad[1]; 
    kx    = -krad[2]:krad[2];
    ykern = T[ y/sqrt(x^2+y^2) for y in ky, x in kx ]
    xkern = T[ x/sqrt(x^2+y^2) for y in ky, x in kx ]
    ykern[ isnan.( ykern ) ] .= 0
    xkern[ isnan.( xkern ) ] .= 0

    sumix  = multi_quickPIV.FFTCC_crop( xkern, img  ); 
    sumiy  = multi_quickPIV.FFTCC_crop( ykern, img  ); 

    U = sumiy
    V = sumix
    if norm
        M = sqrt.( U .^ 2 .+ v .^ 2 ); 
        U[ M .> 0 ] ./= M[ M .> 0 ]
        V[ M .> 0 ] ./= M[ M .> 0 ]
    end 

    return U, V
end

function directionCCL_on_gradient( filename; typ=Float64, dot_th=cosd(10), s=(4,4), krad=(10,10), mode="L2" )
    
    # LOADING IMAGE AS GRAYSCALE
	img  = typ.( Gray.( load( filename ) ) );

    # COMPUTING GRADIENTS (IN ONE OF TWO WAYS)
    if mode == "L2"
        U, V = gradientL2( img, krad )
    else
        U, V = gradient( img, krad ); 
    end

    # SUBSUMPLING VECTOR FIELD (ONLY USEFULL FOR VISUALIZATION)
	srange = StepRange.( 1, s, size(U) );  
	xx = [ x for y in srange[1], x in srange[2] ]; 
	yy = [ y for y in srange[1], x in srange[2] ]; 
	U_ = U[ srange... ]; 
	V_ = V[ srange... ]; 

        # DIRECTION-BASED CCL ON THE GRADIENTS
	Albls  = directionCCL.label_components( (U,V), dot_th );
	Albls_ = directionCCL.label_components( (U_,V_), dot_th );
         
    output = Dict( "img" => img, "UV" => (U,V), "UV_" => (U_,V_), "lbls" => Albls );  
    return output
end


pwd_ = @__DIR__
filename1 = pwd_*"/organoids.png"
filename2 = pwd_*"/embryo_1.png"
filename3 = pwd_*"/embryo_2.png"

out1 = directionCCL_on_gradient( filename1 );
out2 = directionCCL_on_gradient( filename2 ); 
out3 = directionCCL_on_gradient( filename3 ); 


nothing



#######################

function gradientL2_opt( img::AbstractArray{T,2}, krad::Dims{2}, norm=true ) where {T}

    # tmp data for PIV
    tmp_data = multi_quickPIV.allocate_tmp_data( multi_quickPIV.FFTCC(), 2 .* krad .+ 1, size( img ), precision=sizeof( T ) * 8, unpadded=false, good_pad=true ); 

    ky    = -krad[1]:krad[1]; 
    kx    = -krad[2]:krad[2];
    ykern = T[ y/sqrt(x^2+y^2) for y in ky, x in kx ]
    xkern = T[ x/sqrt(x^2+y^2) for y in ky, x in kx ]
    ykern[ krad[1]+1, krad[2]+1 ] = 0;
    xkern[ krad[1]+1, krad[2]+1 ] = 0;

    U = zeros( T, size( img ) ); 
    V = zeros( T, size( img ) ); 

    gradientL2_opt!( img, ykern, xkern, U, V, tmp_data, norm )

    multi_quickPIV.destroy_fftw_plans( multi_quickPIV.FFTCC(), tmp_data )
    tmp_data = nothing

    return U, V
end

# in-place implementation for reusability of tmp_data, kernels, U and V
function gradientL2_opt!( img::AbstractArray{T,2}, 
                          ykern::AbstractArray{T,2}, 
                          xkern::AbstractArray{T,2}, 
                          U::AbstractArray{T,2}, 
                          V::AbstractArray{T,2},
                          tmp_data, norm=true ) where {T}

    FFTCC_crop_add!( T(-2), U, ykern, img, tmp_data )   # U = -2 .* sumiy
    @inbounds U .*= img;                                # U = -2 .* img .* sumiy
    FFTCC_crop_SQ_add!( T(1), U, ykern, img, tmp_data ) # U = sumi2y .- 2 .* img .* sumiy

    FFTCC_crop_add!( T(-2), V, xkern, img, tmp_data )   # V = -2 .* sumix
    @inbounds V .*= img;                                # V = -2 .* img .* sumix
    FFTCC_crop_SQ_add!( T(1), V, xkern, img, tmp_data ) # V = sumi2x .- 2 .* img .* sumix

    if norm
        @inbounds for i in 1:length(U)
            M = sqrt( U[i]^2 + V[i]^2); 
            U[i] /= M*T(M>0) + T(M==0)
            V[i] /= M*T(M>0) + T(M==0)
        end
    end 

    return nothing
end

function gradientL2_opt( vol::AbstractArray{T,3}, krad::Dims{3}, norm=true ) where {T}

    # vol2  = vol .* vol; 
    ksize = 2 .* krad .+ 1; 
    vsize = size( vol ); 
    prec  = sizeof( T ) * 8; 
    tmp_data = multi_quickPIV.allocate_tmp_data( multi_quickPIV.FFTCC(), ksize, vsize, precision=prec, unpadded=false, good_pad=true ); 

    ky    = -krad[1]:krad[1]; 
    kx    = -krad[2]:krad[2];
    kz    = -krad[3]:krad[3]; 
    #Mkern = T[ sqrt(x^2+y^2+z^2) for y in ky, x in kx, z in kz ]
    ykern = T[ y/sqrt(x^2+y^2+z^2) for y in ky, x in kx, z in kz ]
    xkern = T[ x/sqrt(x^2+y^2+z^2) for y in ky, x in kx, z in kz ]
    zkern = T[ z/sqrt(x^2+y^2+z^2) for y in ky, x in kx, z in kz ]
    ykern[ isnan.( ykern ) ] .= 0
    xkern[ isnan.( xkern ) ] .= 0
    zkern[ isnan.( zkern ) ] .= 0

    U = zeros( T, size( vol ) ); 
    FFTCC_crop_add!( T(-2), U, ykern, vol, tmp_data )   # U = -2 .* sumiy
    @inbounds U .*= vol;                                # U = -2 .* vol .* sumiy
    FFTCC_crop_SQ_add!( T(1), U, ykern, vol, tmp_data ) # U = sumi2y .- 2 .* vol .* sumiy

    V = zeros( T, size( vol ) ); 
    FFTCC_crop_add!( T(-2), V, xkern, vol, tmp_data )   # V = -2 .* sumix
    @inbounds V .*= vol;                                # V = -2 .* vol .* sumix
    FFTCC_crop_SQ_add!( T(1), V, xkern, vol, tmp_data ) # V = sumi2x .- 2 .* vol .* sumix

    W = zeros( T, size( vol ) ); 
    FFTCC_crop_add!( T(-2), W, zkern, vol, tmp_data )   # W = -2 .* sumiz
    @inbounds W .*= vol;                                # W = -2 .* vol .* sumiz
    FFTCC_crop_SQ_add!( T(1), W, zkern, vol, tmp_data ) # W = sumi2z .- 2 .* vol .* sumiz

    if norm
        @inbounds for i in 1:length(U)
            M = sqrt( U[i]^2 + V[i]^2 + W[i]^2 ); 
            if M > 0
                U[i] /= M
                V[i] /= M
                W[i] /= M
            end
        end
    end 

    multi_quickPIV.destroy_fftw_plans( multi_quickPIV.FFTCC(), tmp_data )
    tmp_data = nothing

    return U, V, W
end

function putdata!( out::AbstractArray{T,N}, data::AbstractArray{T,N} ) where {T<:Real,N}
    @inbounds out[UnitRange.(1,size(data))...] .= data
    return nothing
end

@inline r2c_pad( tmp_data ) = size(tmp_data[1]) .- tmp_data[end]; 


function FFTCC_crop_add!( f::T, out::AbstractArray{T,N}, F::AbstractArray{T,N}, G::AbstractArray{T,N}, tmp_data ) where {T<:Real,N}

    @assert size(out) == size(G);
    
    isize = size( F ); 
    ssize = size( G ); 

    tmp_data[1] .= eltype(tmp_data[1])(0.0)
    tmp_data[2] .= eltype(tmp_data[2])(0.0)
    putdata!( tmp_data[1], F ); 
    putdata!( tmp_data[2], G );
    
    multi_quickPIV._FFTCC!( tmp_data... )

    r2c_pad_ = r2c_pad( tmp_data )
    Base.circshift!(  view( tmp_data[2], UnitRange.( 1, size(tmp_data[1]) .- r2c_pad_ )... ),
                      view( tmp_data[1], UnitRange.( 1, size(tmp_data[1]) .- r2c_pad_ )... ),
                      div.( isize, 2 ) ); 
        
    out .+= f .* tmp_data[2][ UnitRange.( 1, ssize )... ]
    
    return nothing
end
    
function FFTCC_crop_SQ_add!( f::T, out::AbstractArray{T,N}, F::AbstractArray{T,N}, G::AbstractArray{T,N}, tmp_data ) where {T<:Real,N}
    
    @assert size(out) == size(G);

    isize = size( F ); 
    ssize = size( G ); 
    
    tmp_data[1] .= eltype(tmp_data[1])(0.0)
    tmp_data[2] .= eltype(tmp_data[2])(0.0)
    putdata!( tmp_data[1], F ); 
    putdata!( tmp_data[2], G );
    @inbounds tmp_data[2][UnitRange.(1,ssize)...] .*= tmp_data[2][UnitRange.(1,ssize)...];
    
    multi_quickPIV._FFTCC!( tmp_data... )

    r2c_pad_ = r2c_pad( tmp_data )
    Base.circshift!(  view( tmp_data[2], UnitRange.( 1, size(tmp_data[1]) .- r2c_pad_ )... ),
                      view( tmp_data[1], UnitRange.( 1, size(tmp_data[1]) .- r2c_pad_ )... ),
                      div.( isize, 2 ) ); 
        
    out .+= f .* tmp_data[2][ UnitRange.( 1, ssize )... ]
    
    return nothing
end