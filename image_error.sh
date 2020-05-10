#!/bin/sh

x=${1:?}
y=${x%.*}

gnuplot << EOF
    set terminal png
    set output 'learningerror.png'
    set xlabel "Time step"
    set ylabel "Error"
    plot '$x' u 1:2 w l title "error"
EOF



