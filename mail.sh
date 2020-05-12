#!/bin/sh

zip energy_data.zip energy_data.txt
zip exact_energy.zip exact_energy.txt
zip -r error.zip error
export LANG=ja_JP.utf8
(uuencode ./energy_data.zip energy_data.zip; uuencode ./exact_energy.zip exact_energy.zip; uuencode ./error.zip error.zip; echo "energy data S = 8, B = 56" ) | mail -s "Energy Data" "takeshi6.6260@gmail.com"

