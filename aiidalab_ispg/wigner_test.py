#!/usr/bin/env python3

import random
import wigner_position as wigner

seed = 16661
nsample = 2
scaling = 1.0
outfile = "initconds_new.xyz"

random.seed(seed)

def create_data_manually():
  molecule=[]
  natom = 7
  nmode = 1

  elements = ['C', "O", "O", "H", "H", "H", "H"]

  coords = []
  coords.append([-2.108693, -0.416896, 0.045299])
  coords.append([0.041355, 1.135331, -0.045883  ])
  coords.append([2.168905, -0.518350 , -0.183783  ])
  coords.append([-2.218397, -1.638147, -1.620046  ])
  coords.append([-2.139768, -1.573543, 1.763514  ])
  coords.append([-3.704471, 0.892714, 0.052831  ])
  coords.append([3.032718, -0.115496, 1.369239  ])

  frequencies = [183.7000]
  vibrations = []
  vib1 = [
   [ 0.000000,    -0.010000,     0.030000 ],
   [ 0.010000,     0.000000,    -0.020000 ],
   [ 0.030000,     0.050000,    -0.040000 ],
   [ 0.050000,    -0.120000,     0.120000 ],
   [-0.050000,     0.110000,     0.110000 ],
   [-0.010000,    -0.010000,    -0.090000 ],
   [-0.480000,    -0.710000,     0.440000 ]
  ]
  vibrations.append(vib1)

  for i in range(natom):
    symb = elements[i].lower().title()
    num = wigner.NUMBERS[symb]
    mass = wigner.get_mass(symb)
    molecule.append(wigner.ATOM(symb, num, coords[i], mass))

  modes = []
  for imode in range(nmode):
    mode = {'freq': frequencies[imode] * wigner.CM_TO_HARTREE}
    mode['move'] = vibrations[imode]
    modes.append(mode)

  return molecule, modes

molecule, modes = create_data_manually()

w = wigner.Wigner(molecule, modes, seed)
#modes = wigner.convert_orca_normal_modes(modes, molecule)
ic_list = []
for i in range(nsample):
    ic_list.append(w.get_sample())

wigner.make_dyn_file(ic_list, outfile)
