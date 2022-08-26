#!/usr/bin/env python3

# import ase
import aiidalab_atmospec_workchain.wigner as mywigner

ANG_TO_BOHR = 1.0 / 0.529177211  # 1.889725989      # conversion from Angstrom to bohr

LOW_FREQ_THR = 200


def print_xyz_file(elements, coordinates, filename):
    assert len(elements) == len(coordinates[0])
    natom = len(elements)
    string = ""
    for i, ic in enumerate(coordinates):
        string += "%i\n%i\n" % (natom, i)
        for symb, coord in zip(elements, ic):
            string += "%s" % (symb)
            for j in range(3):
                string += " %f" % (coord[j] / ANG_TO_BOHR)
            string += "\n"

    with open(filename, "w") as fl:
        fl.write(string)


seed = 16661  # This is the default in wigner_sharc.py
nsample = 2
scaling = 1.0
outfile = "initconds_new.xyz"

elements = ["C", "O", "O", "H", "H", "H", "H"]
masses = [12.0000, 15.994915, 15.994915, 1.007825, 1.007825, 1.007825, 1.007825]

coords = []
coords.append([-2.108693, -0.416896, 0.045299])
coords.append([0.041355, 1.135331, -0.045883])
coords.append([2.168905, -0.518350, -0.183783])
coords.append([-2.218397, -1.638147, -1.620046])
coords.append([-2.139768, -1.573543, 1.763514])
coords.append([-3.704471, 0.892714, 0.052831])
coords.append([3.032718, -0.115496, 1.369239])

frequencies = [183.7000, 263.8600]

vib1 = [
    [0.000000, -0.010000, 0.030000],
    [0.010000, 0.000000, -0.020000],
    [0.030000, 0.050000, -0.040000],
    [0.050000, -0.120000, 0.120000],
    [-0.050000, 0.110000, 0.110000],
    [-0.010000, -0.010000, -0.090000],
    [-0.480000, -0.710000, 0.440000],
]

vib2 = [
    [0.000000, 0.000000, -0.010000],
    [0.000000, 0.010000, 0.100000],
    [-0.010000, -0.010000, -0.060000],
    [0.370000, -0.320000, 0.200000],
    [-0.340000, 0.330000, 0.200000],
    [-0.020000, -0.020000, -0.610000],
    [0.160000, 0.060000, -0.180000],
]

vibrations = [vib1, vib2]
vibrations.append(vib1)
# ase_molecule = ase.Atoms(symbols=elements, masses=masses, positions=coords)

if __name__ == "__main__":
    w = mywigner.Wigner(
        elements,
        masses,
        coords,
        frequencies,
        vibrations,
        seed=seed,
        low_freq_thr=LOW_FREQ_THR,
    )
    ic_list = []
    for i in range(nsample):
        ic_list.append(w.get_sample())

    print_xyz_file(elements, ic_list, outfile)
