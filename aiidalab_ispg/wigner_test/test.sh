#!/bin/bash
set -e
rm -f initconds_new.xyz initconds.xyz
# Create reference data
../wigner_sharc.py -f 1 -x -n 2 freq_single.molden > /dev/null

echo "Testing wigner_position.py"
../wigner_position.py -x -n 2 freq_single.molden > /dev/null
diff initconds.xyz initconds_new.xyz

rm -f initconds_new.xyz
echo "Testing wigner_test.py"
../wigner_test.py freq_single.molden > /dev/null
diff initconds.xyz initconds_new.xyz
