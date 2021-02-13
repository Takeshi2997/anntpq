module Const

# System Size
const dimS = 8
const dimB = 72

# System Param
const t = 1.0f0
const J = 1.0f0

# Repeat Number
const burnintime = 10
const iters_num = 1000
const it_num = 50
const inv_n = 5
const iϵmax = 3

# Network Params
const layer = [dimB+dimS, 40, 40, 40, 2]
const layers_num = length(layer) - 1
const batchsize = 128

end
