# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy>=2.0.2",
#     "scipy>=1.13.1",
# ]
# ///
"""Class for calculating UV/vis spectra using Nuclear Ensemble Approach (NEA).

Authors:
    * Daniel Hollas <daniel.hollas@bristol.ac.uk>
"""

from __future__ import annotations

from enum import Enum, unique
from typing import TypedDict

import numpy as np
from scipy import constants

# copied from utils.py
AUtoEV = 27.2114386245


@unique
class EnergyUnit(Enum):
    EV = "eV"
    CM = "cm^-1"
    NM = "nm"


@unique
class BroadeningKernel(Enum):
    GAUSS = "gaussian"
    LORENTZ = "lorentzian"


class Transition(TypedDict):
    energy: int
    osc_strength: float
    geom_index: int


class ConformerTransitions(TypedDict):
    transitions: list[Transition]
    nsample: int
    weight: float


class Spectrum:
    """NEA spectrum class

    This is where the spectrum is actually calculated.
    Constructor gets a set of excitations, characterized by excitation energy
    and oscillator strenghts.

    The spectrum is then calculated using the self.get_spectrum(),
    by specifying the type of broadening and broadening parameter.
    """

    COEFF = (
        constants.pi
        * 8.478354e-30**2  # AUtoCm
        * AUtoEV
        * 1e4
        / (2 * constants.hbar * constants.epsilon_0 * constants.c)
    )

    # TODO: We should make this dependent on the energy range
    N_SAMPLE_POINTS = 500

    def __init__(self, transitions: list[Transition], nsample: int):
        # Excitation energies in eV
        self.excitation_energies = np.array(
            [tr["energy"] for tr in transitions], dtype=float
        )
        # Oscillator strengths
        self.osc_strengths = np.array(
            [tr["osc_strength"] for tr in transitions], dtype=float
        )

        # Number of molecular geometries sampled from ground state distribution
        self.nsample = nsample

        num_exc = len(self.excitation_energies)
        num_osc = len(self.osc_strengths)
        assert num_exc == num_osc, (
            f"# excitation energies ({num_exc}) != # osc. strengths ({num_osc})"
        )
        assert nsample <= num_exc, (
            f"Number of samples ({nsample}) cannot be bigger than number of transitions ({num_exc})"
        )

    @staticmethod
    def get_energy_range_ev(excitation_energies: np.ndarray):
        """Get spectrum energy range in eV based on the minimum and maximum excitation energy"""
        en_min_ev = excitation_energies.min()
        en_max_ev = excitation_energies.max()
        assert en_min_ev > 0.0
        assert en_max_ev > 0.0
        padding_ev = 1.5
        # You're not supposed to understand this. :-)
        # Okay, so essentially we're determining the x-axis of the spectrum
        # by taking a minimum and maximum excitation energy and adding some padding.
        # However, for low-energy excitation, we want to use smaller padding, since small
        # excitation energies result in big tail when converted to nanometers.
        x_max = en_max_ev + padding_ev
        x_min = en_min_ev - padding_ev
        if x_min < 1.0:
            x_min = en_min_ev - en_min_ev / 2.0
        return x_min, x_max

    @staticmethod
    def get_energy_unit_factor(unit: EnergyUnit) -> float:
        """Returns a multiplication factor to go from eV to other energy units"""

        # TODO: Construct these factors from scipy.constants or use pint
        # https://physics.nist.gov/cgi-bin/cuu/Info/Constants/basis.html
        unit_factors = {
            EnergyUnit.EV: 1.0,
            EnergyUnit.NM: 1239.8,
            # https://physics.nist.gov/cgi-bin/cuu/Convert?exp=0&num=1&From=ev&To=minv&Action=Only+show+factor
            EnergyUnit.CM: 8065.547937,
        }
        return unit_factors[unit]

    def _calc_lorentzian_spectrum(
        self, x: np.ndarray, y: np.ndarray, tau: float
    ) -> None:
        """Calculate NEA spectrum broadened with a Lorentzian function:

        https://en.wikipedia.org/wiki/Cauchy_distribution#Probability_density_function
        """
        normalization_factor = tau / 2 / constants.pi / self.nsample
        for exc_energy, osc_strength in zip(
            self.excitation_energies, self.osc_strengths
        ):
            prefactor = normalization_factor * self.COEFF * osc_strength
            y += prefactor / ((x - exc_energy) ** 2 + (tau**2) / 4)

    def _calc_gauss_spectrum(self, x: np.ndarray, y: np.ndarray, sigma: float) -> None:
        """Calculate NEA spectrum broadened with a Gaussian function

        https://en.wikipedia.org/wiki/Normal_distribution
        """
        normalization_factor = 1 / np.sqrt(2 * constants.pi) / sigma / self.nsample
        for exc_energy, osc_strength in zip(
            self.excitation_energies, self.osc_strengths
        ):
            prefactor = normalization_factor * self.COEFF * osc_strength
            y += prefactor * np.exp(-((x - exc_energy) ** 2) / 2 / sigma**2)

    def get_spectrum(
        self,
        kernel: BroadeningKernel,
        width: float,
        x_unit: EnergyUnit,
        x_min: float | None = None,
        x_max: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if x_min is None or x_max is None:
            x_min, x_max = self.get_energy_range_ev(self.excitation_energies)

        x = np.linspace(x_min, x_max, num=self.N_SAMPLE_POINTS)
        y = np.zeros(len(x))

        if kernel is BroadeningKernel.GAUSS:
            self._calc_gauss_spectrum(x, y, width)
        elif kernel is BroadeningKernel.LORENTZ:
            self._calc_lorentzian_spectrum(x, y, width)
        else:
            msg = f"Invalid broadening kernel {kernel}"
            raise ValueError(msg)

        # Conversion factor from eV to given energy unit
        if x_unit is EnergyUnit.NM:
            x, y = self._convert_to_nanometers(x, y)
            x_stick = self.get_energy_unit_factor(x_unit) / self.excitation_energies
        else:
            x_factor = self.get_energy_unit_factor(x_unit)
            x *= x_factor
            x_stick = self.excitation_energies * x_factor

        # We also return "stick" spectrum, e.g. just the transitions themselves,
        # where osc. strengths are normalized to the maximum of the spectrum.
        y_stick = self.osc_strengths * np.max(y) / np.max(self.osc_strengths)

        return x, y, x_stick, y_stick

    def _convert_to_nanometers(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x = self.get_energy_unit_factor(EnergyUnit.NM) / x
        return x, y


# Below are functions for CLI standalone use
def parse_cmd():
    """Parse command line arguments"""
    import argparse

    desc = (
        "WIP: Program for computing UV/vis spectra based on Nuclear Ensemble Approach"
    )
    prog = "neavis"
    parser = argparse.ArgumentParser(description=desc, prog=prog)
    parser.add_argument("input_file", metavar="INPUT_FILE", help="TBD: Input file")
    parser.add_argument(
        "-n",
        "--nsamples",
        type=int,
        default=1,
        help="Number of samples (molecular geometries)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    import sys

    opts = parse_cmd()

    sys.exit("ERROR: Command line usage is not ready yet")
