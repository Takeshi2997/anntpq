module Const

# System Size
const dimS = 16
const dimB = 80

# System Param
const t = 1f0
const J = 1f0
const λ = 1f0

# Repeat Number
const burnintime = 10
const iters_num = 1000
const it_num = 100
const iϵmax = 4

# Learning Rate
const lr = 1f-3

# Network Params
const layer1 = [dimB, 80, 80, 80, 2]
const layer2 = [dimS, 32, 32, 32, 2]
const layers_num = length(layer1) - 1
const batchsize = 64

end
