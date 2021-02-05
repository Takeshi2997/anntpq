module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote, CUDA
using Flux: @functor
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

# Initialize Variables

o   = Array{Array}(undef, Const.layers_num, 2)
oe  = Array{Array}(undef, Const.layers_num, 2)
oo  = Array{Array}(undef, Const.layers_num, 2, 2)
const I = [Diagonal(CUDA.ones(Float32, Const.layer[i+1] * (Const.layer[i] + 1))) for i in 1:Const.layers_num]

function initO()
    for i in 1:Const.layers_num
        W  = zeros(Complex{Float32}, Const.layer[i+1] * (Const.layer[i] + 1))
        S  = transpose(W) .* W
        global o[i, 1]  = W
        global o[i, 2]  = W
        global oe[i, 1] = W
        global oe[i, 2] = W
        global oo[i, 1, 1] = S
        global oo[i, 1, 2] = S
        global oo[i, 2, 1] = S
        global oo[i, 2, 2] = S
    end
end

# Define Network

struct Layer{F,S<:AbstractArray}
    W1::S
    W2::S
    σ::F
end
function Layer(in::Integer, out::Integer, σ = identity;
               initW = Flux.glorot_uniform)
    return Layer(initW(out, in+1), initW(out, in+1), σ)
end
@functor Layer
function (m::Layer)(x::AbstractArray)
    W1, W2, σ = m.W1, m.W2, m.σ
    z = vcat(x, 1)
    σ.(W1*z) .+ σ.(W2*z)
end

mutable struct Network
    f::Flux.Chain
    p::Zygote.Params
end

function Network()
    layers = Vector(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layers[i] = Layer(Const.layer[i], Const.layer[i+1], tanh)
    end
    layers[end] = Layer(Const.layer[end-1], Const.layer[end])
    f = Chain([layers[i] for i in 1:Const.layers_num]...)
    p = Flux.params(f)
    Network(f, p)
end

network = Network()

# Network Utility

function save(filename)
    f = getfield(network, :f)
    @save filename f
end

function load(filename)
    @load filename f
    p = Flux.params(f)
    Flux.loadparams!(network.f, p)
end

function init()
    parameters = Vector{Array}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W1 = Flux.glorot_uniform(Const.layer[i+1], Const.layer[i]+1) 
        W2 = Flux.glorot_uniform(Const.layer[i+1], Const.layer[i]+1) 
        parameters[i] = [W1, W2]
    end
    paramset = [param for param in parameters]
    p = Flux.params(paramset...)
    Flux.loadparams!(network.f, p)
end

# Learning Method

function forward(x::Vector{Float32})
    out = network.f(x)
    return out[1] + im * out[2]
end

loss(x::Vector{Float32}) = real(forward(x))

function backward(x::Vector{Float32}, e::Complex{Float32})
    gs = gradient(() -> loss(x), network.p)
    for i in 1:Const.layers_num
        dw1 = gs[network.f[i].W1]
        dw2 = gs[network.f[i].W2]
        dw1 = reshape(dw1, length(dw1))
        dw2 = reshape(dw2, length(dw2))
        o[i, 1]  += dw1
        o[i, 2]  += dw2
        oe[i, 1] += dw1 .* e
        oe[i, 2] += dw2 .* e
        oo[i, 1, 1] += transpose(dw1) .* conj.(dw1)
        oo[i, 1, 2] += transpose(dw1) .* conj.(dw2)
        oo[i, 2, 1] += oo[i, 1, 2]'
        oo[i, 2, 2] += transpose(dw2) .* conj.(dw2)
    end
end

opt(lr::Float32) = Descent(lr)

function update(energy::Float32, ϵ::Float32, lr::Float32)
    α = 1f0 / Const.iters_num
    for i in 1:Const.layers_num
        o[i, 1]  ./= Const.iters_num
        o[i, 2]  ./= Const.iters_num
        oe[i, 1] ./= Const.iters_num 
        oe[i, 2] ./= Const.iters_num 
        oo[i, 1, 1] ./= Const.iters_num 
        oo[i, 1, 2] ./= Const.iters_num 
        oo[i, 2, 1] ./= Const.iters_num 
        oo[i, 2, 2] ./= Const.iters_num 
    end
    for i in 1:Const.layers_num
        R1  = CuArray(2f0 .* real.(o[i, 1] .- (ϵ - energy) * o[i, 1]))
        R2  = CuArray(2f0 .* real.(o[i, 2] .- (ϵ - energy) * o[i, 2]))
        S11 = CuArray(2f0 .* real.(oo[i, 1, 1] - transpose(o[i, 1]) .* conj.(o[i, 1])))
        S12 = CuArray(2f0 .* real.(oo[i, 1, 2] - transpose(o[i, 1]) .* conj.(o[i, 2])))
        S21 = CuArray(2f0 .* real.(oo[i, 2, 1] - transpose(o[i, 2]) .* conj.(o[i, 1])))
        S22 = CuArray(2f0 .* real.(oo[i, 2, 2] - transpose(o[i, 2]) .* conj.(o[i, 2])))
        ΔW1 = reshape((S11 .+ Const.η .* I[i])\R1 - (S21 .+ Const.η .* I[i])\R2, 
                      (Const.layer[i+1], Const.layer[i]+1)) |> cpu
        ΔW2 = reshape((S22 .+ Const.η .* I[i])\R2 - (S12 .+ Const.η .* I[i])\R1, 
                      (Const.layer[i+1], Const.layer[i]+1)) |> cpu
        update!(opt(lr), network.f[i].W1, ΔW1)
        update!(opt(lr), network.f[i].W2, ΔW2)
    end
end

end
