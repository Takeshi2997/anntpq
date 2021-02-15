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
const it_num = 200
const inv_n = 20
const iϵmax = 4

# Network Params
const layer = [dimB+dimS, 48, 48, 48, 2]
const layers_num = length(layer) - 1
const batchsize = 80

end
