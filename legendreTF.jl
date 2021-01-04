module LegendreTF
using QuadGK

function f(t)
    fm = x -> -log(cosh((abs(cos(x)) / t))) / π * t
    return quadgk(fm, 0f0, πf0)[1]
end

function df(t)
    dfm = x -> -log(cosh((abs(cos(x)) / t))) / π + 
    tanh((abs(cos(x)) / t)) * abs(cos(x)) / t / π
    return quadgk(dfm, 0f0, π)[1]
end

function s(u, t)
    return (u - f(t)) / t
end

function ds(u, t)
    return -(u - f(t)) / t^2 - df(t) / t
end

function calc_temperature(u)
    sout = 0.0
    t = 5.0
    tm = 0.0
    tv = 0.0
    for n in 1:1000
        dt = ds(u, t)
        lr_t = 0.5 * sqrt(1.0 - 0.999^n) / (1.0 - 0.9^n)
        tm += (1.0 - 0.9) * (dt - tm)
        tv += (1.0 - 0.999) * (dt.^2 - tv)
        t  -= lr_t * tm ./ (sqrt.(tv) .+ 10.0^(-7))
        sout = s(u, t)
    end
    return 1.0 / t
end

end
