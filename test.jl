using Gen, GenViz
import Random

function make_data_set(n)
    Random.seed!(1)
    prob_outlier = 0.5
    true_inlier_noise = 0.5
    true_outlier_noise = 5.0
    true_slope = -1
    true_intercept = 2
    xs = collect(range(-5, stop=5, length=n))
    ys = Float64[]
    for (i, x) in enumerate(xs)
        if rand() < prob_outlier
            y = randn() * true_outlier_noise
        else
            y = true_slope * x + true_intercept + randn() * true_inlier_noise
        end
        push!(ys, y)
    end
    (xs, ys)
end

(xs, ys) = make_data_set(200)

server = VizServer(8000)
v = Viz(server, joinpath(@__DIR__, "vue/dist"), [xs, ys])
sleep(0.5)
openInBrowser(v)
sleep(3)

