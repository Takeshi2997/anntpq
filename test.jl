include("./setup.jl")
using .Const, LinearAlgebra, Plots


ψ = randn(Float32, 1000)

histogram(real.(ψ); norm=true, alpha=0.3, label="probability density")
savefig("histogram.png")
