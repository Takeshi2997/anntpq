module ANN
include("./setup.jl")
include("./legendreTF.jl")
using .Const, .LegendreTF, LinearAlgebra, Flux, Zygote
using Flux: @functor
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

mutable struct Parameters

    W::Array
    b::Array
end

o  = Vector{Parameters}(undef, Const.layers_num)
oe = Vector{Parameters}(undef, Const.layers_num)

function initO()

    for i in 1:Const.layers_num
        W = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i])
        b = zeros(Complex{Float32}, Const.layer[i+1])
        global o[i]  = Parameters(W, b)
        global oe[i] = Parameters(W, b)
    end
end

∂S  = Vector{Parameters}(undef, Const.layers_num)
S∂T = Vector{Parameters}(undef, Const.layers_num)
∂T  = Vector{Parameters}(undef, Const.layers_num)

function initS()

    for i in 1:Const.layers_num
        W = zeros(Float32, Const.layer[i+1], Const.layer[i])
        b = zeros(Float32, Const.layer[i+1])
        global ∂S[i]  = Parameters(W, b)
        global S∂T[i] = Parameters(W, b)
        global ∂T[i]  = Parameters(W, b)
    end
end

mutable struct Network

    f::Flux.Chain
    p::Zygote.Params
end

function Network()

    layer = Vector{Any}(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layer[i] = Dense(Const.layer[i], Const.layer[i+1], tanh)
    end
    layer[end] = Dense(Const.layer[end-1], Const.layer[end])
    f = Chain([layer[i] for i in 1:Const.layers_num]...)
    p = params(f)
    Network(f, p)
end

network = Network()

function save(filename)

    f = getfield(network, :f)
    @save filename f
end

function load(filename)

    @load filename f
    p = params(f)
    Flux.loadparams!(network.f, p)
end

function init()

    parameters = Vector{Array}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = randn(Float32, Const.layer[i+1], Const.layer[i]) ./ Float32(sqrt(Const.layer[i]))
        b = zeros(Float32, Const.layer[i+1])
        parameters[i] = [W, b]
    end
    paramset = [param for param in parameters]
    p = params(paramset...)
    Flux.loadparams!(network.f, p)
end

function forward(x::Vector{Float32})

    out = network.f(x)
    return out[1] + im * out[2]
end

loss(x::Vector{Float32}) = real(forward(x))
init_loss(x::Vector{Float32}) = network.f(x)[1]

function init_backward(x::Vector{Float32}, y::Vector{Float32}, s::Float32)

    x′ = vcat((@views x[1:Const.dimB]), (@views y[Const.dimB+1:end]))
    y′ = vcat((@views y[1:Const.dimB]), (@views x[Const.dimB+1:end]))

    gsx  = gradient(() -> init_loss(x),  network.p)
    gsy  = gradient(() -> init_loss(y),  network.p)
    gsx′ = gradient(() -> init_loss(x′), network.p)
    gsy′ = gradient(() -> init_loss(y′), network.p)
    for i in 1:Const.layers_num-1
        dw1 = gsx[network.f[i].W]  .+ gsy[network.f[i].W]
        dw2 = gsx′[network.f[i].W] .+ gsy′[network.f[i].W]
        db1 = gsx[network.f[i].b]  .+ gsy[network.f[i].b]
        db2 = gsx′[network.f[i].b] .+ gsy′[network.f[i].b]
        ∂S[i].W  += dw1 .- dw2
        S∂T[i].W += s .* dw1
        ∂T[i].W  += dw1
        ∂S[i].b  += db1 .- db2
        S∂T[i].b += s .* db1
        ∂T[i].b  += db1
    end
    dw1 = gsx[network.f[end].W]  .+ gsy[network.f[end].W]
    dw2 = gsx′[network.f[end].W] .+ gsy′[network.f[end].W]
    ∂S[end].W  += dw1 .- dw2
    S∂T[end].W += s .* dw1
    ∂T[end].W  += dw1
end

function init_update(S::Float32, lr::Float32)

    α = -1f0 / Const.iters_num
    for i in 1:Const.layers_num-1
        ΔW = α .* (∂S[i].W .+ S∂T[i].W - S .* ∂T[i].W)
        Δb = α .* (∂S[i].b .+ S∂T[i].b - S .* ∂T[i].b)
        update!(opt(lr), network.f[i].W, ΔW)
        update!(opt(lr), network.f[i].b, Δb)
    end
    ΔW = α .* (∂S[end].W .+ S∂T[end].W - S .* ∂T[end].W)
    update!(opt(lr), network.f[end].W, ΔW)
end

function backward(x::Vector{Float32}, e::Complex{Float32})

    gs = gradient(() -> loss(x), network.p)
    for i in 1:Const.layers_num-1
        dw = gs[network.f[i].W]
        db = gs[network.f[i].b]
        o[i].W  += dw
        oe[i].W += dw * e
        o[i].b  += db
        oe[i].b += db * e
    end
    dw = gs[network.f[end].W]
    o[end].W  += dw
    oe[end].W += dw * e
end

opt(lr::Float32) = ADAM(lr, (0.9f0, 0.999f0))

function update(energy::Float32, ϵ::Float32, lr::Float32)

    x = (energy - ϵ)
    α = 2f0 * x / Const.iters_num
    for i in 1:Const.layers_num-1
        ΔW = α .* 2f0 .* real.(oe[i].W .- energy * o[i].W)
        Δb = α .* 2f0 .* real.(oe[i].b .- energy * o[i].b)
        update!(opt(lr), network.f[i].W, ΔW)
        update!(opt(lr), network.f[i].b, Δb)
    end
    ΔW = α .* 2f0 * real(oe[end].W .- energy * o[end].W)
    update!(opt(lr), network.f[end].W, ΔW)
end

end
