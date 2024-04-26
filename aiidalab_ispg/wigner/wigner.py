#!/usr/bin/env python3

# Specification of standalone script dependencies according to PEP 723
# /// script
# dependencies = ['ase', 'tdqm', 'cclib>=1.8']
# ///

# Script for the calculation of Wigner distributions in coordinate space
import copy
import math
import random

# some constants
CM_TO_HARTREE = (
    1.0 / 219474.6
)  # 4.556335252e-6 # conversion factor from cm-1 to Hartree
HARTREE_TO_EV = 27.211396132  # conversion factor from Hartree to eV
U_TO_AMU = 1.0 / 5.4857990943e-4  # conversion from g/mol to amu
ANG_TO_BOHR = 1.0 / 0.529177211  # 1.889725989      # conversion from Angstrom to bohr


class Wigner:
    def __init__(
        self,
        ase_molecule,
        frequencies,
        vibrations,
        seed=16661,
        low_freq_thr=10.0,
    ):
        """
        ase_molecule - ASE Atoms object, coordinates in Angstroms
        frequencies - list or normal mode frequencies in reciprocal centimiter
        units
        modes - vibrational normal mode displacements in atomic units
        seed - random number seed
        low_freq_thr - discard normal modes below this threshold (cm^-1)
        """
        self._set_random_seed(seed)

        self.ase_molecule = ase_molecule
        self.low_freq_thr = low_freq_thr * CM_TO_HARTREE

        modes = [
            {"freq": freq * CM_TO_HARTREE, "move": vib}
            for vib, freq in zip(vibrations, frequencies)
        ]
        self.modes = self._convert_orca_normal_modes(
            modes, self.ase_molecule.get_masses()
        )

    def _set_random_seed(self, seed):
        self.rnd = random.Random(seed)

    def get_ase_sample(self):
        return self._sample_initial_condition()

    def _sample_initial_condition(self):
        """This function samples a single initial condition from the
        modes and atomic coordinates by the use of a Wigner distribution.
        The first atomic dictionary in the molecule list contains also
        additional information like kinetic energy and total harmonic
        energy of the sampled initial condition.
        Method is based on L. Sun, W. L. Hase J. Chem. Phys. 133, 044313
        (2010) nonfixed energy, independent mode sampling."""
        # copy coordinates in equilibrium geometry
        positions = self.ase_molecule.get_positions() * ANG_TO_BOHR
        masses = self.ase_molecule.get_masses() * U_TO_AMU
        Epot = 0.0

        for mode in self.modes:  # for each uncoupled harmonatomlist oscillator
            random_Q, random_P = self._sample_unit_mode()
            # now transform the dimensionless coordinate into a real one
            # paper says, that freq_factor is sqrt(2*PI*freq)
            # QM programs directly give angular frequency (2*PI is not needed)
            freq_factor = math.sqrt(mode["freq"])
            # Higher frequencies give lower displacements and higher momentum.
            # Therefore scale random_Q and random_P accordingly:
            random_Q /= freq_factor
            random_P *= freq_factor
            # add potential energy of this mode to total potential energy
            Epot += 0.5 * mode["freq"] ** 2 * random_Q**2

            for i in range(len(positions)):
                for xyz in range(3):
                    # distort geometry according to normal mode movement
                    # and unweigh mass-weighted normal modes
                    positions[i][xyz] += (
                        random_Q * mode["move"][i][xyz] * math.sqrt(1.0 / masses[i])
                    )

        sample = self.ase_molecule.copy()
        sample.set_positions(positions / ANG_TO_BOHR)
        return sample

    # TODO: Get rid of these shenanigans and use random.gauss()
    def _sample_unit_mode(self):
        while True:
            # get random Q and P in the interval [-5,+5]
            # this interval is good for vibrational ground state
            # should be increased for higher states
            random_Q = self.rnd.random() * 10.0 - 5.0
            random_P = self.rnd.random() * 10.0 - 5.0
            # calculate probability for this set of P and Q with Wigner distr.
            probability = self.wigner(random_Q, random_P)
            if probability[0] > 1.0 or probability[0] < 0.0:
                msg = f"Wrong probability {probability[0]:f} detected in _sample_initial_condition()!"
                raise ValueError(msg)
            elif probability[0] > self.rnd.random():
                return random_Q, random_P

    def _convert_orca_normal_modes(self, modes, masses):
        """apply transformations to normal modes"""
        converted_modes = []
        for imode in range(len(modes)):
            freq = modes[imode]["freq"]

            if freq < self.low_freq_thr:
                continue

            norm = 0.0
            for j, mass in enumerate(masses):
                for xyz in range(3):
                    norm += modes[imode]["move"][j][xyz] ** 2 * mass / U_TO_AMU
            norm = math.sqrt(norm)
            if norm == 0.0 and freq >= self.low_freq_thr:
                raise ValueError(
                    "Displacement vector of mode %i is null vector!" % (imode + 1)
                )

            converted_mode = copy.deepcopy(modes[imode])
            for j, mass in enumerate(masses):
                for xyz in range(3):
                    converted_mode["move"][j][xyz] /= norm / math.sqrt(mass / U_TO_AMU)
            converted_modes.append(converted_mode)

        return converted_modes

    # TODO: Remove this. This doesn't make sense,
    # since the Q and P probabilities are independent!
    @staticmethod
    def wigner(Q, P):
        """This function calculates the Wigner distribution for
        a single one-dimensional harmonic oscillator.
        Q contains the dimensionless coordinate of the
        oscillator and P contains the corresponding momentum.
        The function returns a probability for this set of parameters."""
        return (math.exp(-(Q**2)) * math.exp(-(P**2)), 0.0)


# Below are functions for CLI standalone use
def parse_cmd():
    import argparse

    desc = "Program for harmonic Wigner sampling"
    prog = "harmonwig"
    parser = argparse.ArgumentParser(description=desc, prog=prog)
    parser.add_argument(
        "input_file", metavar="INPUT_FILE", help="Output file from ab initio program."
    )
    parser.add_argument(
        "-n",
        "--nsamples",
        type=int,
        default=1,
        help="Number of Wigner samples",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=42424242,
        help="Random seed",
    )
    parser.add_argument(
        "--freqthr",
        dest="low_freq_thr",
        default=0.0,
        type=float,
        help="Low-frequency threshold",
    )

    parser.add_argument(
        "--file-format",
        dest="file_fmt",
        default="auto",
        help="Format of the output file",
    )

    parser.add_argument(
        "-o",
        "--output-file",
        dest="output_fname",
        default="harmonic_samples.xyz",
        help="Output file name",
    )

    return parser.parse_args()


def error(msg: str):
    import sys

    print(f"ERROR: {msg}")
    sys.exit(1)


def read_qm_output(fname: str, fmt: str = "auto") -> dict:
    from pathlib import Path

    from cclib.io import ccread

    path = Path(fname)
    try:
        with path.open("r") as f:
            parsed_obj = ccread(f)
    except FileNotFoundError as e:
        error(str(e))

    if parsed_obj is None:
        error("Could not read ORCA output")
    d = parsed_obj.getattributes()
    return d


def validate(out: dict):
    if not out["optdone"]:
        error("Optimization did not finish!")
    req_keys = ["atomcoords", "atommasses", "vibdisps", "vibfreqs"]
    for key in req_keys:
        if key not in out:
            error("Could not find frequency data in ORCA output file")


if __name__ == "__main__":
    opts = parse_cmd()

    import ase
    from tqdm import tqdm

    out = read_qm_output(opts.input_file, fmt=opts.file_fmt)
    # We assume that the last coordinates are the optimized ones
    coords = out["atomcoords"][0]
    ase_mol = ase.Atoms(
        numbers=out["atomnos"], positions=coords, masses=out["atommasses"], pbc=False
    )

    print("Normal mode frequencies [cm^-1]:")
    print(out["vibfreqs"])
    print("Atom masses:")
    print(out["atommasses"])

    wigner = Wigner(
        ase_mol,
        out["vibfreqs"],
        out["vibdisps"],
        seed=opts.seed,
        low_freq_thr=opts.low_freq_thr,
    )

    print(f"Generating {opts.nsamples} samples to {opts.output_fname}")
    barfmt = "{l_bar}{bar}|{n_fmt}/{total_fmt}    "
    wigner_samples = []
    for _ in tqdm(range(opts.nsamples), delay=0.3, colour="green", bar_format=barfmt):
        wigner_samples.append(wigner.get_ase_sample())

    ase.io.write(
        opts.output_fname,
        images=wigner_samples,
        format="extxyz",
    )
