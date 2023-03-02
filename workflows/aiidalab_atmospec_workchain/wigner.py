#!/usr/bin/env python3

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
            # TODO: Pass in the proper sigma directly
            sigma = math.sqrt(0.5)
            random_Q = random.gauss(mu=0.0, sigma=sigma)
            random_P = random.gauss(mu=0.0, sigma=sigma)
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
