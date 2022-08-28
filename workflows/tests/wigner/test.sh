#!/bin/bash
set -e
LOW_FREQ_THR=200
rm -f initconds_new.xyz initconds.xyz
# Create reference data
./wigner_sharc.py -f 1 -x -n 2 -L $LOW_FREQ_THR --keep_trans_rot freq_test.molden > /dev/null

./tst_wigner.py
diff initconds.xyz initconds_new.xyz
