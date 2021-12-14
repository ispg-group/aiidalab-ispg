"""Widget for displaying UV/VIS spectra in an interactive graph.

Authors:
    * Daniel Hollas <daniel.hollas@durham.ac.uk>
"""
import ipywidgets as ipw
import traitlets
import scipy
from scipy import constants
import numpy as np

from aiida.orm import QueryBuilder
from aiida.plugins import DataFactory

# https://docs.bokeh.org/en/latest/docs/user_guide/jupyter.html
# https://github.com/bokeh/bokeh/blob/branch-3.0/examples/howto/server_embed/notebook_embed.ipynb
# https://github.com/bokeh/bokeh/blob/branch-3.0/examples/howto/server_embed/notebook_embed.ipynb
from bokeh.io import push_notebook, show, output_notebook
import bokeh.plotting as plt

output_notebook()
XyData = DataFactory("array.xy")


# This code was provided by a good soul on GitHub.
# https://github.com/bokeh/bokeh/issues/7023#issuecomment-839825139
class BokehFigureContext(ipw.Output):
    """Helper class for rendering Bokeh figures inside ipywidgets"""

    def __init__(self, fig):
        super().__init__()
        self._figure = fig
        self.on_displayed(lambda x: x.set_handle())

    def set_handle(self):
        self.clear_output()
        with self:
            self._handle = show(self._figure, notebook_handle=True)

    def get_handle(self):
        return self._handle

    def get_figure(self):
        return self._figure

    def update(self):
        push_notebook(handle=self._handle)


class Spectrum(object):
    AUtoCm = 8.478354e-30
    COEFF = (
        constants.pi
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

    # TODO
    def get_spectrum(self, x_min, x_max, x_units, y_units):
        """Returns a non-broadened spectrum as a tuple of x and y Numpy arrays"""
        n_points = int((x_max - x_min) / self.de)
        x = np.arange(x_min, x_max, self.de)
        y = np.zeros(n_points)
        return x, y

    # TODO: Make this function aware of units?
    def _get_energy_range(self, energy_unit):
        # NOTE: We don't include zero to prevent
        # division by zero when converting to wavelength
        x_min = max(0.01, self.excitation_energies.min() - 2.0)
        x_max = self.excitation_energies.max() + 2.0

        # conversion to nanometers is handled later
        if energy_unit.lower() == "nm":
            return x_min, x_max

        # energy_factor_unit = self._get_energy_unit_factor(energy_unit)
        # x_min *= energy_factor_unit
        # x_max *= energy_factor_unit
        return x_min, x_max

    # TODO: Define energy units as Enum in this file.
    # TODO: Put this function outside of this class
    # So that it can be useful for experimental spectrum as well
    def _get_energy_unit_factor(self, unit):

        # https://physics.nist.gov/cgi-bin/cuu/Info/Constants/basis.html
        # TODO: We should probably start from atomic units
        if unit.lower() == "ev":
            return 1.0
        # TODO: Construct these factors from scipy.constants
        elif unit.lower() == "nm":
            return 1239.8
        elif unit.lower() == "cm^-1":
            # https://physics.nist.gov/cgi-bin/cuu/Convert?exp=0&num=1&From=ev&To=minv&Action=Only+show+factor
            return 8065.547937

    def get_gaussian_spectrum(self, sigma, x_unit, y_unit):
        """Returns Gaussian broadened spectrum"""

        x_min, x_max = self._get_energy_range(x_unit)

        # Conversion factor from eV to given energy unit
        # (should probably switch to atomic units as default)
        energy_unit_factor = self._get_energy_unit_factor(x_unit)

        energies = np.copy(self.excitation_energies)
        # Conversion to wavelength in nm is done at the end instead
        # Since it's not a linear transformation
        # if x_unit.lower() != 'nm':
        #    sigma *= energy_unit_factor
        #    energies *= energy_unit_factor

        # TODO: How to determine this properly to cover a given interval?
        n_sample = 500
        x = np.linspace(x_min, x_max, num=n_sample)
        y = np.zeros(len(x))

        normalization_factor = (
            1 / np.sqrt(2 * scipy.constants.pi) / sigma / self.nsample
        )
        # TODO: Support other intensity units
        unit_factor = self.COEFF_NEW
        for exc_energy, osc_strength in zip(energies, self.osc_strengths):
            prefactor = normalization_factor * unit_factor * osc_strength
            y += prefactor * np.exp(-((x - exc_energy) ** 2) / 2 / sigma ** 2)

        if x_unit.lower() == "nm":
            x, y = self._convert_to_nanometers(x, y)
        else:
            x *= energy_unit_factor

        return x, y

    def get_lorentzian_spectrum(self, tau, x_unit, y_unit):
        """Returns Gaussian broadened spectrum"""
        # TODO: Determine x_min automatically based on transition energies
        # and x_units
        x_min, x_max = self._get_energy_range(x_unit)

        # Conversion factor from eV to given energy unit
        # (should probably switch to atomic units as default)
        energy_unit_factor = self._get_energy_unit_factor(x_unit)

        energies = np.copy(self.excitation_energies)

        # TODO: How to determine this properly to cover a given interval?
        n_sample = 500
        x = np.linspace(x_min, x_max, num=n_sample)
        y = np.zeros(len(x))

        normalization_factor = tau / 2 / scipy.constants.pi / self.nsample
        unit_factor = self.COEFF_NEW

        for exc_energy, osc_strength in zip(energies, self.osc_strengths):
            prefactor = normalization_factor * unit_factor * osc_strength
            y += prefactor / ((x - exc_energy) ** 2 + (tau ** 2) / 4)

        if x_unit.lower() == "nm":
            x, y = self._convert_to_nanometers(x, y)
        else:
            x *= energy_unit_factor

        return x, y

    def _convert_to_nanometers(self, x, y):
        x = self._get_energy_unit_factor("nm") / x
        # Filtering out ultralong wavelengths
        # TODO: Generalize this based on self.transitions
        nm_thr = 1000
        to_delete = []
        for i in range(len(x)):
            if x[i] > nm_thr:
                to_delete.append(i)
        x = np.delete(x, to_delete)
        y = np.delete(y, to_delete)
        return x, y


class SpectrumWidget(ipw.VBox):

    transitions = traitlets.List(allow_none=True)
    # We use SMILES to find matching experimental spectra
    # that are possibly stored in our DB as XyData.
    smiles = traitlets.Unicode(allow_none=True)
    experimental_spectrum = traitlets.Instance(XyData, allow_none=True)

    # For now, we do not allow different intensity units
    intensity_unit = "cm^2 per molecule"

    THEORY_SPEC_LABEL = "theory"
    EXP_SPEC_LABEL = "experiment"

    def __init__(self, **kwargs):
        title = ipw.HTML(
            """<div style="padding-top: 0px; padding-bottom: 0px">
            <h4>UV/Vis Spectrum</h4></div>"""
        )

        self.width_slider = ipw.FloatSlider(
            min=0.05, max=1, step=0.05, value=0.5, description="Width / eV"
        )

        self.kernel_selector = ipw.ToggleButtons(
            options=["gaussian", "lorentzian"],
            description="Broadening",
            disabled=False,
            button_style="info",
            tooltips=[
                "Gaussian broadening",
                "Lorentzian broadening",
            ],
        )

        self.energy_unit_selector = ipw.RadioButtons(
            # TODO: Make an enum with different energy units
            options=["eV", "nm", "cm^-1"],
            disabled=False,
            description="Energy unit",
        )

        controls = ipw.HBox(
            children=[
                ipw.VBox(children=[self.kernel_selector, self.width_slider]),
                self.energy_unit_selector,
            ]
        )

        # TODO: Add Download button for downloading raw data
        # as an CSV file
        # https://github.com/bokeh/bokeh/tree/branch-3.0/examples/app/export_csv

        # We use this for Debug output for now
        self.debug_output = ipw.Output()

        # https://docs.bokeh.org/en/latest/docs/user_guide/tools.html?highlight=tools#specifying-tools
        tools = "pan,wheel_zoom,box_zoom,reset"
        # https://docs.bokeh.org/en/latest/docs/user_guide/tools.html?highlight#hovertool
        tooltips = [("(energy, cross_section)", "($x,$y)")]
        self._init_figure(tools=tools, tooltips=tooltips)
        self.spectrum_container = ipw.Box()
        self.spectrum_container.children = [self.figure]
        self.kernel_selector.observe(self._handle_kernel_update, names="value")
        self.energy_unit_selector.observe(
            self._handle_energy_unit_update, names="value"
        )
        self.width_slider.observe(self._handle_width_update, names="value")

        super().__init__(
            [
                title,
                self.debug_output,
                controls,
                self.spectrum_container,
            ],
            **kwargs,
        )

    def _validate_transitions(self):
        # TODO: Maybe use named tuple instead of dictionary?
        # We should probably make a traitType for this and export it.
        # https://realpython.com/python-namedtuple/
        if self.transitions is None or len(self.transitions) == 0:
            return False

        for tr in self.transitions:
            if not isinstance(tr, dict) or (
                "energy" not in tr or "osc_strength" not in tr
            ):
                self.debug_print("Invalid transition", tr)
                return False
        return True

    def _handle_width_update(self, change):
        """Redraw spectra when user changes broadening width via slider"""
        width = change["new"]
        self._plot_spectrum(
            width=width,
            kernel=self.kernel_selector.value,
            energy_unit=self.energy_unit_selector.value,
        )
        self.figure.update()

    def _handle_kernel_update(self, change):
        """Redraw spectra when user changes kernel for broadening"""
        kernel = change["new"]
        self._plot_spectrum(
            width=self.width_slider.value,
            kernel=kernel,
            energy_unit=self.energy_unit_selector.value,
        )
        self.figure.update()

    def _handle_energy_unit_update(self, change):
        """Updates the spectrum when user changes energy units
        In this case, we also redraw experimental spectra, if available."""

        energy_unit = change["new"]
        xlabel = f"Energy / {energy_unit}"
        self.figure.get_figure().xaxis.axis_label = xlabel

        self._plot_spectrum(
            width=self.width_slider.value,
            kernel=self.kernel_selector.value,
            energy_unit=energy_unit,
        )
        if self.experimental_spectrum is not None:
            self._plot_experimental_spectrum(
                spectrum_node=self.experimental_spectrum, energy_unit=energy_unit
            )
        self.figure.update()

    def _plot_spectrum(self, kernel, width, energy_unit):
        if not self._validate_transitions():
            self.hide_line(self.THEORY_SPEC_LABEL)
            return
        # TODO: Remove this try/except, and figure out a better
        # way to pass number of samples maybe.
        try:
            nsample = self.transitions[-1]["geom_index"] + 1
        except KeyError:
            self.debug_print("Could not determine number of samples")
            nsample = 1

        spec = Spectrum(self.transitions, nsample)
        if kernel == "lorentzian":
            x, y = spec.get_lorentzian_spectrum(width, energy_unit, self.intensity_unit)
        elif kernel == "gaussian":
            x, y = spec.get_gaussian_spectrum(width, energy_unit, self.intensity_unit)
        else:
            self.debug_print("Invalid broadening type")
            return

        self.plot_line(x, y, self.THEORY_SPEC_LABEL)

    def debug_print(self, *args):
        with self.debug_output:
            print(*args)

    # plot_line() and hide_line() are public, we allow for additinal studff to be plotted
    def plot_line(self, x, y, label):
        """To actually display the line, calling this function must be
        followed by self.figure.update()"""
        # https://docs.bokeh.org/en/latest/docs/reference/models/renderers.html?highlight=renderers#renderergroup
        f = self.figure.get_figure()
        line = f.select_one({"name": label})
        if line is None:
            line = f.line(x, y, line_width=2, name=label)
        line.visible = True
        line.data_source.data = {"x": x, "y": y}
        # TODO: Remove this for performance
        self.figure.update()

    def hide_line(self, label):
        """Hide given line from the plot"""
        f = self.figure.get_figure()
        line = f.select_one({"name": label})
        if line is None or not line.visible:
            return
        line.visible = False
        self.figure.update()

    def _init_figure(self, *args, **kwargs):
        """Initialize Bokeh figure. Arguments are passed to bokeh.plt.figure()"""
        self.figure = BokehFigureContext(plt.figure(*args, **kwargs))
        f = self.figure.get_figure()
        f.xaxis.axis_label = f"Energy / {self.energy_unit_selector.value}"
        f.yaxis.axis_label = f"Cross section / {self.intensity_unit}"

        # Initialize lines for theoretical and possible experimental spectra
        # NOTE: Hardly earned experience: It is crucial that both lines
        # are initiated here, before the figure is first shown. Otherwise,
        # apparently updates via line.data_source are not picked up.
        x = np.array([4.0])
        y = np.array([0.0])
        # TODO: Choose inclusive colors!
        # https://doi.org/10.1038/s41467-020-19160-7
        theory_line = f.line(x, y, line_width=2, name=self.THEORY_SPEC_LABEL)
        theory_line.visible = False
        exp_line = f.line(
            x,
            y,
            line_width=2,
            line_dash="dashed",
            line_color="orange",
            name=self.EXP_SPEC_LABEL,
        )
        # Experimental spectrum only available for some molecules
        exp_line.visible = False

    def reset(self):
        with self.hold_trait_notifications():
            self.transitions = None
            self.smiles = None

        self.hide_line(self.EXP_SPEC_LABEL)
        self.hide_line(self.THEORY_SPEC_LABEL)
        self.debug_output.clear_output()

    @traitlets.observe("transitions")
    def _observe_transitions(self, change):
        self._plot_spectrum(
            width=self.width_slider.value,
            kernel=self.kernel_selector.value,
            energy_unit=self.energy_unit_selector.value,
        )
        self.figure.update()

    @traitlets.observe("smiles")
    def _observe_smiles(self, change):
        self._find_experimental_spectrum(change["new"])

    def _find_experimental_spectrum(self, smiles):
        """Find an experimental spectrum for a given SMILES
        and plot it if it is available in our DB"""
        if smiles is None or smiles == "":
            self.hide_line(self.EXP_SPEC_LABEL)
            return

        qb = QueryBuilder()
        # TODO: Should we subclass XyData specifically for UV/Vis spectra?
        # Or should we differentiate from other possible Xy nodes
        # by looking at attributes or extras? Maybe label?
        qb.append(XyData, filters={"extras.smiles": smiles})

        if qb.count() == 0:
            self.hide_line(self.EXP_SPEC_LABEL)
            return

        # for spectrum in qb.iterall():
        # TODO: For now let's just assume we have one
        # canonical experimental spectrum per compound.
        self.experimental_spectrum = qb.first()[0]
        self._plot_experimental_spectrum(
            spectrum_node=self.experimental_spectrum,
            energy_unit=self.energy_unit_selector.value,
        )
        self.figure.update()

    def _plot_experimental_spectrum(self, spectrum_node, energy_unit):
        """Render experimental spectrum that was loaded to AiiDA database manually
        param: spectrum_node: XyData node
        energy_unit: energy unit of the plotted spectra"""
        # TODO: When we're creating spectrum as XyData,
        # can we choose nicer names for x and y?
        # This would also serve as a validation.

        if (
            "x_array" not in spectrum_node.get_arraynames()
            or "y_array_0" not in spectrum_node.get_arraynames()
        ):
            return
        energy = spectrum_node.get_array("x_array")
        cross_section = spectrum_node.get_array("y_array_0")
        # TODO: Extract units
        # TODO: We really need to define units as Enum and use them
        # consistently everywhere.
        # data_energy_unit = spectrum.node.get_attribute('x_units')
        # cross_section_unit = spectrum.node.get_attribute('y_units')

        # TODO: Refactor how units are handled in this file for forks sake!
        # We should decouple changing units from spectra plotting in general
        # (e.g. do not recalculate the intensity of theoretical spectrum
        # when units are changed)!
        if energy_unit.lower() == "ev":
            energy = 1239.8 / energy
        elif energy_unit.lower() == "cm^-1":
            energy = 8065.7 * 1239.8 / energy

        self.plot_line(energy, cross_section, self.EXP_SPEC_LABEL)
