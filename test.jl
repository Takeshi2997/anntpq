include("./setup.jl")
using .Const, LinearAlgebra, InteractiveUtils

B = [0  0  0  0
     0  1 -1  0
     0 -1  1  0
     0  0  0  0] ./ 2.0
for ix in Const.dimB + 1:2:Const.dimB+Const.dimS-1
    ixnext = Const.dimB + (ix - Const.dimB) % Const.dimS + 1
    println(ix, "\t", ixnext)
end
