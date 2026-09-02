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

import sys
from enum import Enum, unique
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from scipy import constants

if TYPE_CHECKING:
    # ty: ignore[unresolved-import]
    from aiida import orm

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
        if (max_osc_strength := np.max(self.osc_strengths)) == 0:
            y_stick = self.osc_strengths
        else:
            y_stick = self.osc_strengths * np.max(y) / max_osc_strength

        return x, y, x_stick, y_stick

    def _convert_to_nanometers(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x = self.get_energy_unit_factor(EnergyUnit.NM) / x
        return x, y


# A copy from spectrum_widget.py
def compute_total_cross_section(
    conformer_transitions,
    kernel: BroadeningKernel,
    width: float,
    energy_unit: EnergyUnit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Determine spectrum energy range based on all excitation energies
    all_exc_energies = np.array(
        [
            transitions["energy"]
            for conformer in conformer_transitions
            for transitions in conformer["transitions"]
        ]
    )

    x_min, x_max = Spectrum.get_energy_range_ev(all_exc_energies)

    total_cross_section = np.zeros(Spectrum.N_SAMPLE_POINTS)
    x_stick = np.array([])
    y_stick = np.array([])

    # Iterate over conformers, the total spectrum is a sum of
    # individual conformer spectra multiplied by a Boltzmann factor.
    for conf_id, conformer in enumerate(conformer_transitions):
        spec = Spectrum(conformer["transitions"], conformer["nsample"])
        x, y, xs, ys = spec.get_spectrum(
            kernel, width, energy_unit, x_min=x_min, x_max=x_max
        )

        y *= conformer["weight"]
        total_cross_section += y

        ys *= conformer["weight"]
        x_stick = np.concatenate((x_stick, xs))
        y_stick = np.concatenate((y_stick, ys))

    return x, total_cross_section, x_stick, y_stick


def _orca_output_to_transitions(output_dict: dict, geom_index: int) -> list[Transition]:
    EVtoCM = Spectrum.get_energy_unit_factor(EnergyUnit.CM)
    en = output_dict["excitation_energies_cm"]
    osc = output_dict["oscillator_strengths"]
    return [
        {"energy": tr[0] / EVtoCM, "osc_strength": tr[1], "geom_index": geom_index}
        for tr in zip(en, osc)
    ]


def _wigner_output_to_transitions(wigner_outputs: list) -> list[Transition]:
    transitions = []
    for i, params in enumerate(wigner_outputs):
        transitions += _orca_output_to_transitions(params, i)
    return transitions


def get_transitions_from_workchain(
    process: orm.WorkChainNode,
) -> list[ConformerTransitions]:
    """Convert process.outputs.spectrum_data into a data structure that
    is passed to the SpectrumWidget and Spectrum classes"""

    # Number of conformers
    optimized = process.inputs.optimize
    n_input_geoms = len(process.inputs.structure.get_stepids())
    # Number of Wigner geometries per conformer
    wigner_sampled = optimized and process.inputs.nwigner.value > 0
    if wigner_sampled:
        nconf = n_input_geoms
        nsample = process.inputs.nwigner.value
    elif optimized:
        nconf = n_input_geoms
        nsample = 1
    else:
        # If the input geometries were not optimized, we treat them
        # as samples, not conformers!
        nconf = 1
        nsample = n_input_geoms

    # Unfortunately, we don't have number of states as attribute in process.inputs
    nstates = None
    if bp := process.base.extras.get("builder_parameters", None):
        nstates = bp["nstates"]

    # For the case of unoptimized geometries, flatten the list
    # so that the geometries are treated as a single conformer
    spectrum_data = process.outputs.spectrum_data.get_list()
    if not optimized:
        spectrum_data = [[conf[0] for conf in spectrum_data]]

    # Use Boltzmann weighting if we optimized the molecule and have Gibbs energies
    if nconf > 1 and optimized:
        conformer_weights = process.outputs.relaxed_structures.get_array(
            "boltzmann_weights"
        )
    else:
        equal_weight = 1.0 / nconf
        conformer_weights = [equal_weight for i in range(nconf)]

    conformer_transitions: list[ConformerTransitions] = [
        ConformerTransitions(
            transitions=_wigner_output_to_transitions(conformer),
            nsample=nsample,
            weight=conformer_weights[i],
        )
        for i, conformer in enumerate(spectrum_data)
    ]

    # Make sure our data is consistent
    assert nconf == len(conformer_transitions), (
        f"{nconf=} != {len(conformer_transitions)=}"
    )
    if nstates:
        for c in conformer_transitions:
            trans: list = c["transitions"]
            nsample = c["nsample"]
            assert nsample * nstates == len(trans), (
                f"{nstates * nsample=} != {len(trans)=}: {trans=}"
            )

    return conformer_transitions


# Below are functions for CLI standalone use
def parse_cmd():
    """Parse command line arguments"""
    import argparse

    desc = (
        "WIP: Program for computing UV/vis spectra based on Nuclear Ensemble Approach"
    )
    prog = "neavis"
    parser = argparse.ArgumentParser(description=desc, prog=prog)
    parser.add_argument("--input_file", help="TBD: Input file")
    parser.add_argument(
        "-wc",
        "--workchain-id",
        type=int,
        default=None,
        help="Load data from AtmoSpec workchain",
    )
    parser.add_argument(
        "--json-output", type=str, help="Output spectral data to a json file"
    )
    parser.add_argument(
        "--kernel",
        type=BroadeningKernel,
        default=BroadeningKernel.GAUSS,
        help="Broadening kernel ('gaussian' or 'lorentzian')",
    )
    parser.add_argument(
        "--energy-unit",
        type=EnergyUnit,
        default=EnergyUnit.EV,
        help="Broadening kernel ('gaussian' or 'lorentzian')",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=0.05,
        help="Broadening width (eV)",
    )
    parser.add_argument(
        "-n",
        "--nsamples",
        type=int,
        default=1,
        help="Number of samples (molecular geometries)",
    )

    return parser.parse_args()


def load_atmospec_data(pk: int) -> list[ConformerTransitions]:
    from aiida import load_profile, orm

    load_profile()

    process = orm.load_node(pk)

    if not isinstance(process, orm.WorkChainNode):
        sys.exit(f"{pk=} does not correspond to AtmospecWorkChain, but {type(process)}")

    if process.process_type != "aiidalab_ispg.workflows.atmospec.AtmospecWorkChain":
        sys.exit(
            f"{pk=} is not a top-level AtmospecWorkChain, but '{process.process_type}'"
        )

    return get_transitions_from_workchain(process)


if __name__ == "__main__":
    import json

    import numpy as np

    opts = parse_cmd()
    conformer_transitions = []
    if opts.workchain_id is not None:
        conformer_transitions = load_atmospec_data(opts.workchain_id)

    if not conformer_transitions:
        sys.exit()

    energy, total_cross_section, _x_stick, _y_stick = compute_total_cross_section(
        conformer_transitions, opts.kernel, opts.width, opts.energy_unit
    )
    fname = f"spectrum_{opts.workchain_id}_{opts.energy_unit.value}.dat"
    print(f"Saving spectrum to file '{fname}'")

    header = (
        f"Kernel: {opts.kernel.value}  Width: {opts.width}\n"
        f"Energy ({opts.energy_unit.value})       Cross Section (cm^-1 per molecule)"
    )
    if opts.workchain_id:
        header = f"AtmoSpec WorkChain: {opts.workchain_id}\n" + header

    np.savetxt(
        fname,
        np.column_stack((energy, total_cross_section)),
        header=header,
        encoding="utf-8",
    )

    if opts.json_output:
        with open(opts.json_output, "w") as f:
            json.dump(conformer_transitions, f, indent=2)
