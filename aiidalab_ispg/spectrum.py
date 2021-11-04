"""Widget for displaying UV/VIS spectra in interactive graphs

Authors:
    * Daniel Hollas <daniel.hollas@durham.ac.uk>
"""
import ipywidgets as ipw
import traitlets
import scipy
import numpy as np

# import matplotlib.pyplot as plt
import bqplot.pyplot as plt


class Spectrum(object):
    AUtoCm = 8.478354e-30
    COEFF = (
        scipy.constants.pi
        * AUtoCm ** 2
        * 1e4
        / (3 * scipy.constants.hbar * scipy.constants.epsilon_0 * scipy.constants.c)
    )
    # Transition Dipole to Osc. Strength in atomic units
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

    # TODO
    def get_spectrum(self, x_min, x_max, x_units, y_units):
        """Returns a non-broadened spectrum as a tuple of x and y Numpy arrays"""
        n_points = int((x_max - x_min) / self.de)
        x = np.arange(x_min, x_max, self.de)
        y = np.zeros(n_points)
        return x, y

    # TODO: Make this function aware of units
    def _get_energy_range(self):
        x_min = max(0.0, self.excitation_energies.min() - 2.0)
        x_max = self.excitation_energies.max() + 2.0
        return x_min, x_max

    def get_gaussian_spectrum(self, sigma, x_units, y_units):
        """Returns Gaussian broadened spectrum"""
        # TODO: Should probably pass units here
        x_min, x_max = self._get_energy_range()
        x = np.arange(x_min, x_max, self.de)
        y = np.zeros(len(x))

        normalization_factor = (
            1 / np.sqrt(2 * scipy.constants.pi) / sigma / self.nsample
        )
        # TODO: Support other units
        unit_factor = self.COEFF_NEW
        for exc_energy, osc_strength in zip(
            self.excitation_energies, self.osc_strengths
        ):
            prefactor = normalization_factor * unit_factor * osc_strength
            y += prefactor * np.exp(-((x - exc_energy) ** 2) / 2 / sigma ** 2)

        return x, y

    def get_lorentzian_spectrum(self, tau, x_units, y_units):
        """Returns Gaussian broadened spectrum"""
        # TODO: Determine x_min automatically based on transition energies
        # and x_units
        x_min, x_max = self._get_energy_range()
        x = np.arange(x_min, x_max, self.de)
        y = np.zeros(len(x))

        normalization_factor = tau / 2 / scipy.constants.pi / self.nsample
        unit_factor = self.COEFF_NEW

        for exc_energy, osc_strength in zip(
            self.excitation_energies, self.osc_strengths
        ):
            prefactor = normalization_factor * unit_factor * osc_strength
            y += prefactor / ((x - exc_energy) ** 2 + (tau ** 2) / 4)

        return x, y


class SpectrumWidget(ipw.VBox):

    transitions = traitlets.List()

    def __init__(self, **kwargs):
        title = ipw.HTML(
            """<div style="padding-top: 0px; padding-bottom: 0px">
            <h4>UV/Vis Spectrum</h4></div>"""
        )

        # TODO: Remove this debugging output later
        self.output = ipw.Output()
        self.spectrum_container = ipw.Box()
        self.width_slider = ipw.FloatSlider(
            min=0.05, max=1, step=0.05, value=0.5, description="Width / eV"
        )
        self.kernel_selector = ipw.ToggleButtons(
            options=["gaussian", "lorentzian"],  # TODO: None option
            description="Broadening kernel:",
            disabled=False,
            button_style="info",  # 'success', 'info', 'warning', 'danger' or ''
            tooltips=[
                "Description of slow",
                "Description of regular",
                "Description of fast",
            ],
        )

        super().__init__(
            [
                title,
                self.kernel_selector,
                self.width_slider,
                self.spectrum_container,
                self.output,
            ],
            **kwargs,
        )

    def _plot_spectrum(self, kernel, width):
        nsample = 1
        spec = Spectrum(self.transitions, nsample)
        energy_unit = "eV"
        intensity_unit = "cm^-1"
        if kernel == "lorentzian":
            x, y = spec.get_lorentzian_spectrum(width, energy_unit, intensity_unit)
        elif kernel == "gaussian":
            x, y = spec.get_gaussian_spectrum(width, energy_unit, intensity_unit)
        else:
            with self.output:
                print("Invalid broadening type")
                return

        # Determine min max of x and y axes so that they
        # don't change when changing width
        # fig = plt.figure()
        plt.plot(x, y)
        plt.xlabel(f"Energy / {energy_unit}")
        plt.ylabel(f"Intensity / {intensity_unit}")
        plt.show()

    def _validate_transitions(self):
        for tr in self.transitions:
            if not isinstance(tr, dict) or (
                "energy" not in tr or "osc_strength" not in tr
            ):
                with self.output:
                    print("Invalid transition", tr)
                    return False

        return True

    def _show_spectrum(self):
        self.output.clear_output()

        if self._validate_transitions:
            spectrum = ipw.interactive_output(
                self._plot_spectrum,
                {"width": self.width_slider, "kernel": self.kernel_selector},
            )
            self.spectrum_container.children = [spectrum]
        else:
            # TODO: Add proper error handling
            raise KeyError

    @traitlets.observe("transitions")
    def _observe_transitions(self, change):
        self._show_spectrum()


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
