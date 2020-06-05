include("./setup.jl")
using .Const, LinearAlgebra, InteractiveUtils, Distributed


B = [1  0  0  0
     0 -1  2  0
     0  2 -1  0
     0  0  0  1]
for ix in Const.dimB+1:Const.dimB+Const.dimS
    ixnext = Const.dimB + (ix - Const.dimB) % Const.dimS + 1
    println(ix, "\t", ixnext)
end
