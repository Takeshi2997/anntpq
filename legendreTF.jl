module LegendreTF
using QuadGK, NLsolve

πf0 = Float32(π)
u   = 0f0

function nls(func, params...; ini = [0.0])
    if typeof(ini) <: Number
        return nlsolve((vout,vin)->vout[1]=func(vin[1],params...), [ini]).zero[1]
    else
        return nlsolve((vout,vin)->vout .= func(vin,params...), ini).zero
    end
end

function initu(u1::Float32)
    global u = u1
end

function ds(t)
    gm = x -> tanh(abs(cos(x)) / t) / πf0 * abs(cos(x))
    g = quadgk(gm, 0f0, πf0)[1]
    return (u + g)
end

function energy(α, t)
    um = x -> -tanh(abs(α - cos(x)) / t) / πf0 * abs(α - cos(x))
    return quadgk(um, 0f0, πf0)[1]
end

function calc_temperature(u1)
    initu(u1)
    t = nls(ds, ini=0.1f0)
    return 1/t
end

end
