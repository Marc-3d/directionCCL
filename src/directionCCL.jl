module directionCCL

using Base.Cartesian
using MarcIntegralArrays

include("utils.jl")
include("disjoint_sets.jl")
include("label_components.jl")
include("label_components_nomacros.jl")
include("label_components_multiscale.jl")
include("label_components_kalman.jl")

# this one isn't for vector fields, but for thresholded images at increasing thresholds
include("label_components_multithreshold.jl")

# anotherone that isn't for vector fields, but images
include("label_components_localDIFS.jl")


greet() = print("Hello World!")

end # module directionCCL
