module MVSVD

using ArrayViews

export als, splitData, rmse,Parameter,ALSOptions, ccdpp

include("src/types.jl")
include("src/utils.jl")
include("src/als.jl")

end

