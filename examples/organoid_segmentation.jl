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
function directionCCL_on_gradient( filename; typ=Float64, dot_th=cosd(10), s=(4,4), krad=(10,10), mode="L2" )

	img  = typ.( Gray.( load( filename ) ) );
	img2 = img .* img; 

	ksize = 2 .* krad .+ 1;
        ky    = -krad[1]:krad[1]; 
        kx    = -krad[2]:krad[2];
	ykern = typ[ y/sqrt(x^2+y^2) for y in ky, x in kx ]
	xkern = typ[ x/sqrt(x^2+y^2) for y in ky, x in kx ]
	ykern[ isnan.( ykern ) ] .= 0
	xkern[ isnan.( xkern ) ] .= 0

	xcorr  = multi_quickPIV.FFTCC_crop( xkern, img  ); 
	xcorr2 = multi_quickPIV.FFTCC_crop( xkern, img2 ); 
	ycorr  = multi_quickPIV.FFTCC_crop( ykern, img  ); 
	ycorr2 = multi_quickPIV.FFTCC_crop( ykern, img2 );

	if mode == "L2"
	   # L2
	   U = xcorr2 .- 2 .* xcorr .* img; 
	   V = ycorr2 .- 2 .* ycorr .* img; 
	else
	   # L
	   U = xcorr
	   V = ycorr
	end

	# normalizing the vector field 
	M = sqrt.( U .^ 2 .+ V .^ 2 ); 
	U[ M .> 0 ] ./= M[ M .> 0 ]
	V[ M .> 0 ] ./= M[ M .> 0 ]

	srange = StepRange.( 1, s, size(U) );  
	xx = [ x for y in srange[1], x in srange[2] ]; 
	yy = [ y for y in srange[1], x in srange[2] ]; 
	U_ = U[ srange... ]; 
	V_ = V[ srange... ]; 

	Albls  = zeros( Int, size( U  ) ); 
	Albls_ = zeros( Int, size( U_ ) ); 
	directionCCL.lc!( Albls, U , V , dot_th );
	directionCCL.lc!( Albls, U_, V_, dot_th );
         
        return Albls
end


pwd_ = @__DIR__
filename = pwd_*"/organoids.png" 

Albls = directionCCL_on_gradient( filename )

nothing
