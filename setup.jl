module Const

# System Size
const dimS = 8
const dimB = 32

# System Param
const t = 1.0f0
const J = 1.0f0

# Repeat Number
const burnintime = 10
const iters_num = 1000
const it_num = 1000
const iϵmax = 4
const num = 100000
const inv_n = 100

# Network Params
const layer = [dimB+dimS, 2]
const layers_num = length(layer) - 1
const η = 0.001

# Learning Rate
const lr = 0.01f0

end
