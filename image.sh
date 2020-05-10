#!/bin/sh

x=${1:?}
y=${x%.*}

gnuplot << EOF
    set terminal png
    set output '$y.png'
    set xlabel "inverse temperature"
    set ylabel "System energy"
    set xrange [0.0:4.0]
    plot '$x' u 1:2 title "numerical solution", 'exact_energy.txt' u 1:2 w l title "exact solution"
EOF



