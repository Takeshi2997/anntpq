module Const

# System Size
const dimS = 24
const dimB = 64

# System Param
const t = 1f0
const J = 1f0
const λ = 1f0

# Repeat Number
const burnintime = 10
const iters_num = 200
const it_num = 100
const iϵmax = 4

# Learning Rate
const lr = 1f-5

# Network Params
const layer = [dimS, 64]
const layers_num = length(layer) - 1
const batchsize = 64

end
