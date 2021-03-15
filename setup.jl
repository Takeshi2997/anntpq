module Const

# System Size
const dimS = 16
const dimB = 80

# System Param
const t = 1.0f0
const J = 1.0f0

# Repeat Number
const burnintime = 10
const iters_num = 500
const it_num = 100
const iϵmax = 4

# Learning Rate
const lr = 1f-3

# Network Params
const layer1 = [dimB,      48, 48, 2]
const layer2 = [dimS,      48, 48, 2]
const layer3 = [dimB+dimS, 48, 48, 2]
const layers_num = length(layer3) - 1
const batchsize = 128
const λ = 0.1f0

end
