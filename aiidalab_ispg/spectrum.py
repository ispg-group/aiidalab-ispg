"""Widget for displaying UV/VIS spectra in interactive graphs

Authors:
    * Daniel Hollas <daniel.hollas@durham.ac.uk>
"""
import scipy
import numpy as np


class Spectrum(object):
    AUtoCm = 8.478354e-30
    COEFF = (
        scipy.constants.pi
        * AUtoCm ** 2
        * 1e4
        / (3 * scipy.constants.hbar * scipy.constants.epsilon_0 * scipy.constants.c)
    )
    # Transition Dipole to Osc. Strength
    COEFF_NEW = COEFF * 3 / 2
    # COEFF =  scipy.constants.pi * AUtoCm**2 * 1e4 * scipy.constants.hbar / (2 * scipy.constants.epsilon_0 * scipy.constants.c * scipy.constants.m_e)

    def __init__(self, transitions, nsample):
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
        self.set_defaults()

    # TODO: These parameters should be passed as arguments to get_spectrum()
    def set_defaults(self):
        # Default grid on x-axis in eV
        self.de = 0.02

    # TODO: Specialize this function for Gaussian / Lorentzian broadening
    def get_spectrum(self, x_min, x_max, x_units, y_units):
        """Returns a computer spectrum as a tuple of x and y Numpy arrays"""

        n_points = int((x_max - x_min) / self.de)
        x = np.arange(x_min, x_max, self.de)
        y = np.zeros(n_points)
        return x, y

    def get_gaussian_spectrum(self, sigma, x_units, y_units):
        """Returns Gaussian broadened spectrum"""
        # TODO: Determine x_min automatically based on transition energies
        # and x_units
        x_min = 0
        x_max = 5
        assert x_min < x_max
        x = np.arange(x_min, x_max, self.de)
        y = np.zeros(len(x))
        assert len(x) == len(y)
        normalization_factor = 1 / np.sqrt(2 * scipy.constants.pi) / sigma / nsample
        for exc_energy, osc_strength in zip(
            self.excitation_energies, self.osc_strengths
        ):
            y += (
                osc_strength
                * self.COEFF_NEW
                * np.exp(-((x - exc_energy) ** 2) / 2 / sigma ** 2)
            )
        y *= normalization_factor

        return x, y


if __name__ == "__main__":

    transition1 = {"energy": 1, "osc_strength": "0.016510951"}  # Excited energy in eV
    transition2 = {"energy": 2.0, "osc_strength": "0.0"}  # Excited energy in eV
    transitions = [transition1, transition2]
    nsample = 1
    spec = Spectrum(transitions, nsample)
    x, y = spec.get_gaussian_spectrum(0.3, "ev", "cross_section")

    from bokeh.plotting import figure
    from bokeh.io import show, output_notebook

    output_notebook()
    p = figure(
        title="Spectrum test",
        x_axis_label="E / eV",
        y_axis_label="I / cm^-2 * molecule ^ -1",
    )
    # p.line(x, y, legend_label='legend title', line_width=2)
    p.line(x, y, line_width=2)
    show(p)
