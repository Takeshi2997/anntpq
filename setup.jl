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
const it_num = 20
const iϵmax = 1
const inv_n = 10

# Network Params
const layer = [dimB+dimS, 40, 2]
const layers_num = length(layer) - 1
const η = 1f-9

# Learning Rate
const lr = 1f-11

end
