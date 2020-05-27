module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

mutable struct Network

    f::Flux.Chain
    g::Flux.Dense
    p::Zygote.Params
    q::Zygote.Params
end

function Network()

    func(x::Float32) = tanh(x)
    layer1 = Dense(Const.layer[1], Const.layer[2], func)
    layer2 = Dense(Const.layer[2], Const.layer[3], func)
    layer3 = Dense(Const.layer[3], Const.layer[4], func)
    f = Chain(layer1, layer2, layer3)
    p = params(f)
    W = randn(Complex{Float32}, Const.layer[end], Const.layer[end-1])
    b = zeros(Complex{Float32}, Const.layer[end])
    g = Dense(W, b)
    q = params([W, b])
    Network(f, g, p, q)
end

network = Network()

function save(filename)

    f = getfield(network, :f)
    g = getfield(network, :g)
    @save filename f g
end

function load(filename)

    @load filename f g
    p = params(f)
    q = params(g)
    Flux.loadparams!(network.f, p)
    Flux.loadparams!(network.g, q)
end

function init()

    W1 = randn(Float32, Const.layer[2], Const.layer[1]) * sqrt(1.0f0 / Const.layer[1])
    W2 = randn(Float32, Const.layer[3], Const.layer[2]) * sqrt(1.0f0 / Const.layer[2])
    W3 = randn(Float32, Const.layer[4], Const.layer[3]) * sqrt(1.0f0 / Const.layer[3])
    b1 = zeros(Float32, Const.layer[2])
    b2 = zeros(Float32, Const.layer[3])
    b3 = zeros(Float32, Const.layer[4])
    p  = params([W1, b1], [W2, b2], [W3, b3])
    W  = Array(Diagonal(randn(Complex{Float32}, Const.layer[end], Const.layer[end-1])))
    b  = zeros(Complex{Float32}, Const.layer[end])
    q  = params([W, b])
    Flux.loadparams!(network.f, p)
    Flux.loadparams!(network.g, q)
end

function forward(x::Vector{Float32})

    out = network.f(x)
    return network.g(out)
end

loss(x::Vector{Float32}, h::Vector{Float32}) = real(dot(h, forward(x)))

function backward(x::Vector{Float32}, h::Vector{Float32}, e::Complex{Float32})

    gs = gradient(() -> loss(x, h), network.p)
    for i in 1:Const.layers_num-1
        dw = gs[network.f[i].W]
        db = gs[network.f[i].b]
        o[i].W  += dw
        oe[i].W += dw * e
        o[i].b  += db
        oe[i].b += db * e
    end
    dw = transpose(network.f(x)) .* h
    o[end].W  += dw
    oe[end].W += dw * e
end

opt(lr::Float32) = ADAM(lr, (0.9, 0.999))

function update(energy::Float32, ϵ::Float32, lr::Float32)

    for i in 1:Const.layers_num-1
        ΔW = 4.0f0 * (energy - ϵ) * [(energy - ϵ)^2 - Const.η^2 > 0.0f0] .* 
        real.(oe[i].W .- energy * o[i].W) / Const.iters_num
        Δb = 4.0f0 * (energy - ϵ) * [(energy - ϵ)^2 - Const.η^2 > 0.0f0] .* 
        real.(oe[i].b .- energy * o[i].b) / Const.iters_num
        update!(opt(lr), network.f[i].W, ΔW)
        update!(opt(lr), network.f[i].b, Δb)
    end
    ΔW = 4.0f0 * (energy - ϵ) * [(energy - ϵ)^2 - Const.η^2 > 0.0f0] .* 
    (oe[end].W .- energy * o[end].W) / Const.iters_num
    update!(opt(lr), network.g.W, ΔW)
end

mutable struct Parameters

    W::Array{Complex{Float32}, 2}
    b::Array{Complex{Float32}, 1}
end

o  = Vector{Parameters}(undef, Const.layers_num)
oe = Vector{Parameters}(undef, Const.layers_num)

function initO()

    for i in 1:Const.layers_num
        global o[i]  = Parameters(zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i]), 
                                  zeros(Complex{Float32}, Const.layer[i+1]))
        global oe[i] = Parameters(zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i]), 
                                  zeros(Complex{Float32}, Const.layer[i+1]))
    end
end

end
