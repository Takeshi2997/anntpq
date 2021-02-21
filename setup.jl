module Const

# System Size
const dimS = 8
const dimB = 72

# System Param
const t = 1.0f0
const J = 1.0f0

# Repeat Number
const burnintime = 10
const iters_num = 500
const it_num = 100
const inv_n = 20
const iϵmax = 4

# Learning Rate
const lr = 1f-3

# Network Params
const layer = [dimB+dimS, 40, 40, 40, 2]
const layers_num = length(layer) - 1
const batchsize = 80

end
