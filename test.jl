include("./setup.jl")
using .Const, LinearAlgebra, InteractiveUtils, Distributed

@simd for ix in Const.dimB+1:Const.dimB+Const.dimS
    ixnext = Const.dimB + (ix - Const.dimB) % Const.dimS + 1
    println(ix, ixnext)
end

