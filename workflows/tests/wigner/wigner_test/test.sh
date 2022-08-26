#!/bin/bash
set -e
LOW_FREQ_THR=200
rm -f initconds_new.xyz initconds.xyz
# Create reference data
../wigner_sharc.py -f 1 -x -n 2 -L $LOW_FREQ_THR freq_test.molden > /dev/null

echo "Testing wigner_test.py"
../tst_wigner.py # > /dev/null
diff initconds.xyz initconds_new.xyz
