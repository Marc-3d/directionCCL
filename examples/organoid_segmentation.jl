using multi_quickPIV, ColorTypes, directionCCL, FileIO, PyPlot

#= 
 For each pixel, we wish to compute the average vector in the direction of (A) the greatest difference or (B) the highest gradient. This is done by considering a local square/rectangular ROI around each pixel, and computing a weighted sum between normalized vectors in a divergent pattern (aka pointing away from the center of the ROI) and the difference of each local pixel with respect to the central value:  

 ∑(( i - v )²*x) = ∑(i²x) + ∑(v²x) - 2v∑(ix)  
 ∑(( i - v )²*y) = ∑(i²y) + ∑(v²y) - 2v∑(iy) 

 I think that ∑(v²x) and ∑(v²y) cancel out, when computing it over the whole kernel, since the kernels are anti-symmetric.

For signed difference:

 ∑(( i - v )x) = ∑(ix) - ∑(vx) 
 ∑(( i - v )y) = ∑(iy) - ∑(vy)

Again, ∑(vx) and ∑(vy) cancel out.
=#

function compute_L2_gradient( img::AbstractArray{T,N}, krad::Dims{N}, norm=true ) where {T,N}

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

function compute_gradient( img::AbstractArray{T,N}, krad::Dims{N}, norm=true ) where {T,N}

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
            U, V = compute_L2_gradient( img, krad )
        else
            U, V = compute_gradient( img, krad ); 
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
