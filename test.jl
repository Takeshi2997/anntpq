include("./setup.jl")
using .Const

abstract type Parameters end

mutable struct Middle <: Parameters
    W::Array{Complex{Float32}, 2}
    b::Array{Complex{Float32}, 1}
end

mutable struct Output <: Parameters
    W::Array{Complex{Float32}, 2}
    b::Array{Complex{Float32}, 1}
    a::Array{Complex{Float32}, 1}
end

o  = Vector{Parameters}(undef, Const.layers_num)
oe = Vector{Parameters}(undef, Const.layers_num)

function initO()

    for i in 1:Const.layers_num-1
        W = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i])
        b = zeros(Complex{Float32}, Const.layer[i+1])
        global o[i]  = Middle(W, b)
        global oe[i] = Middle(W, b)
    end
    W = zeros(Complex{Float32}, Const.layer[end], Const.layer[end-1])
    b = zeros(Complex{Float32}, Const.layer[end])
    a = zeros(Complex{Float32}, Const.layer[1])
    global o[end]  = Output(W, b, a)
    global oe[end] = Output(W, b, a)
end

