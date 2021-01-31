module Const

# System Size
const dimS = 8
const dimB = 40

# System Param
const t = 1.0f0
const J = 1.0f0

# Repeat Number
const burnintime = 10
const iters_num = 200
const it_num = 1000
const iϵmax = 1
const num = 10000

# Network Params
const layer = [dimB+dimS, 48, 48, 48, 48, 48, 48, 48, 48, 2]
const layers_num = length(layer) - 1

# Learning Rate
const lr = 1f-3

end
