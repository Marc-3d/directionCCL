module directionCCL

using Base.Cartesian
using MarcIntegralArrays

# MiA = MarcIntegralArrays # shorter name... not allowed

include("utils.jl")
include("disjoint_sets.jl")
include("label_components.jl")
include("label_components_nomacros.jl")
include("label_components_multiscale.jl")
include("label_components_kalman.jl")

greet() = print("Hello World!")

end # module directionCCL
