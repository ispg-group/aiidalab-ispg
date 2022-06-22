#!/usr/bin/env python3

# Script for the calculation of Wigner distributions in coordinate space
import copy
import math
import random
import sys

# some constants
CM_TO_HARTREE = (
    1.0 / 219474.6
)  # 4.556335252e-6 # conversion factor from cm-1 to Hartree
HARTREE_TO_EV = 27.211396132  # conversion factor from Hartree to eV
U_TO_AMU = 1.0 / 5.4857990943e-4  # conversion from g/mol to amu
ANG_TO_BOHR = 1.0 / 0.529177211  # 1.889725989      # conversion from Angstrom to bohr

# thresholds
LOW_FREQ = (
    10.0  # threshold in cm^-1 for ignoring rotational and translational low frequencies
)
KTR = False


# TODO: Use ASE object instead of this
class ATOM:
    def __init__(self, symb="??", coord=[0.0, 0.0, 0.0], m=0.0, veloc=[0.0, 0.0, 0.0]):
        self.symb = symb
        self.coord = coord
        self.mass = m
        self.veloc = veloc
        self.Ekin = 0.5 * self.mass * sum([self.veloc[i] ** 2 for i in range(3)])

    def __str__(self):
        s = "%2s % 5.1f " % (self.symb)
        s += "% 12.8f % 12.8f % 12.8f " % tuple(self.coord)
        s += "% 12.8f " % (self.mass / U_TO_AMU)
        s += "% 12.8f % 12.8f % 12.8f" % tuple(self.veloc)
        return s


class INITCOND:
    def __init__(self, atomlist=[], eref=0.0, epot_harm=0.0):
        self.atomlist = atomlist
        self.eref = eref
        self.Epot_harm = epot_harm
        self.natom = len(atomlist)
        self.Ekin = sum([atom.Ekin for atom in self.atomlist])
        self.nstate = 0
        self.Epot = epot_harm

    def __str__(self):
        s = "Atoms\n"
        for atom in self.atomlist:
            s += str(atom) + "\n"
        s += "Ekin      % 16.12f a.u.\n" % (self.Ekin)
        s += "Epot_harm % 16.12f a.u.\n" % (self.Epot_harm)
        s += "Epot      % 16.12f a.u.\n" % (self.Epot)
        s += "Etot_harm % 16.12f a.u.\n" % (self.Epot_harm + self.Ekin)
        s += "Etot      % 16.12f a.u.\n" % (self.Epot + self.Ekin)
        s += "\n\n"
        return s


def wigner(Q, P, mode):
    """This function calculates the Wigner distribution for
    a single one-dimensional harmonic oscillator.
    Q contains the dimensionless coordinate of the
    oscillator and P contains the corresponding momentum.
    The function returns a probability for this set of parameters."""
    return (math.exp(-(Q ** 2)) * math.exp(-(P ** 2)), 0.0)


# I think this one will not be needed
# as it is build into the ASE object.
def get_center_of_mass(molecule):
    """This function returns a list containing the center of mass
    of a molecule."""
    mass = 0.0
    for atom in molecule:
        mass += atom.mass
    com = [0.0 for xyz in range(3)]
    for atom in molecule:
        for xyz in range(3):
            com[xyz] += atom.coord[xyz] * atom.mass / mass
    return com


def restore_center_of_mass(molecule, ic):
    """This function restores the center of mass for the distorted
    geometry of an initial condition."""
    # calculate original center of mass
    com = get_center_of_mass(molecule)
    # caluclate center of mass for initial condition of molecule
    com_distorted = get_center_of_mass(ic)
    # get difference vector and restore original center of mass
    diff = [com[xyz] - com_distorted[xyz] for xyz in range(3)]
    for atom in ic:
        for xyz in range(3):
            atom.coord[xyz] += diff[xyz]


class Wigner:

    RESTORE_COM = True
    LOW_FREQ = 0.0

    def __init__(self, atom_names, masses, coordinates, frequencies, vibrations, seed):
        """atom_names - list of elements
        masses - masses in relative atomic masses
        coordinates - bohr
        frequencies - cm^-1
        modes - a.u.
        seed - random number seed, int
        """

        self.set_random_seed(seed)

        # Molecule as a list of atoms
        molecule = []
        self.natom = len(atom_names)
        for iat in range(self.natom):
            symb = atom_names[iat].lower().title()
            # TODO: Get rid of this, keep only masses
            molecule.append(ATOM(symb, coordinates[iat], masses[iat] * U_TO_AMU))
        self.molecule = molecule

        nmode = len(frequencies)
        modes = []
        for imode in range(nmode):
            mode = {"freq": frequencies[imode] * CM_TO_HARTREE}
            mode["move"] = vibrations[imode]
            modes.append(mode)
        self.modes = self._convert_orca_normal_modes(modes, molecule)

    def set_random_seed(self, seed):
        # TODO: Use our own instance of random number generator
        self.rnd = random.Random(seed)

    def get_sample(self):  # remove_com -> Bool
        # TODO: Remove COM here, based on input parameter
        # TODO: Return an ASE object, positions in angstroms
        ic = self._sample_initial_condition(self.molecule, self.modes)
        coordinates = []
        for iat in range(self.natom):
            coordinates.append(ic.atomlist[iat].coord)
        # Returning coordinates in bohrs
        return coordinates

    # TODO: Convert this to work on the ASE object
    def _sample_initial_condition(self, molecule, modes):
        """This function samples a single initial condition from the
        modes and atomic coordinates by the use of a Wigner distribution.
        The first atomic dictionary in the molecule list contains also
        additional information like kinetic energy and total harmonic
        energy of the sampled initial condition.
        Method is based on L. Sun, W. L. Hase J. Chem. Phys. 133, 044313
        (2010) nonfixed energy, independent mode sampling."""
        # copy the molecule in equilibrium geometry
        atomlist = copy.deepcopy(molecule)  # initialising initial condition object
        Epot = 0.0
        for mode in modes:  # for each uncoupled harmonatomlist oscillator
            # TODO: Get rid of these shenanigans and use random.gauss()
            while True:
                # get random Q and P in the interval [-5,+5]
                # this interval is good for vibrational ground state
                # should be increased for higher states
                random_Q = self.rnd.random() * 10.0 - 5.0
                random_P = self.rnd.random() * 10.0 - 5.0
                # calculate probability for this set of P and Q with Wigner distr.
                probability = wigner(random_Q, random_P, mode)
                if probability[0] > 1.0 or probability[0] < 0.0:
                    print("WARNING: wrong probability %f detected!" % (probability[0]))
                    sys.exit(1)
                elif probability[0] > self.rnd.random():
                    break  # coordinates accepted
            # now transform the dimensionless coordinate into a real one
            # paper says, that freq_factor is sqrt(2*PI*freq)
            # QM programs directly give angular frequency (2*PI is not needed)
            freq_factor = math.sqrt(mode["freq"])
            # Higher frequencies give lower displacements and higher momentum.
            # Therefore scale random_Q and random_P accordingly:
            random_Q /= freq_factor
            random_P *= freq_factor
            # add potential energy of this mode to total potential energy
            Epot += 0.5 * mode["freq"] ** 2 * random_Q ** 2
            for i, atom in enumerate(atomlist):  # for each atom
                for xyz in range(3):  # and each direction
                    # distort geometry according to normal mode movement
                    # and unweigh mass-weighted normal modes
                    atom.coord[xyz] += (
                        random_Q * mode["move"][i][xyz] * math.sqrt(1.0 / atom.mass)
                    )

        if self.RESTORE_COM:
            restore_center_of_mass(molecule, atomlist)

        ic = INITCOND(atomlist, 0.0, Epot)
        return ic

    def _convert_orca_normal_modes(self, modes, molecule):
        """apply transformations to normal modes"""
        converted_modes = copy.deepcopy(modes)
        for imode in range(len(modes)):
            norm = 0.0
            for j, atom in enumerate(molecule):
                for xyz in range(3):
                    norm += modes[imode]["move"][j][xyz] ** 2 * atom.mass / U_TO_AMU
            norm = math.sqrt(norm)
            if norm == 0.0 and modes[imode]["freq"] >= LOW_FREQ * CM_TO_HARTREE:
                print(
                    "WARNING: Displacement vector of mode %i is null vector. Ignoring this mode!"
                    % (imode + 1)
                )
                # This is not expected, let's stop
                sys.exit(1)

            for j, atom in enumerate(molecule):
                for xyz in range(3):
                    converted_modes[imode]["move"][j][xyz] /= norm / math.sqrt(
                        atom.mass / U_TO_AMU
                    )

        return converted_modes
