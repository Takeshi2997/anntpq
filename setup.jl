module Const

# System Size
const dimS = 8
const dimB = 32

# System Param
const t = 1.0f0
const J = 1.0f0

# Repeat Number
const burnintime = 10
const iters_num = 500
const it_num = 1000
const iϵmax = 1

# Network Params
const layer = [dimB+dimS, 20, 20, 2]
const layers_num = length(layer) - 1

# Learning Rate
const lr = 1f-2

end
