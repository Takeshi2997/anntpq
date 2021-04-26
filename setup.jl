module Const

# System Size
const dimS = 8
const dimB = 40

# System Param
const t = 1f0
const J = 1f0
const λ = 1f0

# Repeat Number
const burnintime = 10
const iters_num = 200
const it_num = 100
const iϵmax = 1

# Network Params
const layer = [dimB+dimS, 12, 12, 2]
const layers_num = length(layer) - 1
const networkdim = sum([layer[i+1] * (layer[i] + 1) for i in 1:layers_num])

# Learning Rate
const lr = 1f-3
const batchsize = 64

end
