"""Widget for displaying UV/VIS spectra in an interactive graph.

Authors:
    * Daniel Hollas <daniel.hollas@durham.ac.uk>
"""
from enum import Enum, unique
import ipywidgets as ipw
import traitlets
from scipy import constants
import numpy as np

from aiida.orm import QueryBuilder
from aiida.plugins import DataFactory

# https://docs.bokeh.org/en/latest/docs/user_guide/jupyter.html
# https://github.com/bokeh/bokeh/blob/branch-3.0/examples/howto/server_embed/notebook_embed.ipynb
from bokeh.io import push_notebook, show, output_notebook
import bokeh.plotting as plt

# https://docs.bokeh.org/en/latest/docs/reference/io.html#bokeh.io.output_notebook
output_notebook(hide_banner=True, load_timeout=5000, verbose=True)
XyData = DataFactory("array.xy")


# Conversion factor from atomic units to electronvolts
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
    COEFF = (
        constants.pi
        * 8.478354e-30**2  # AUtoCm
        * AUtoEV
        * 1e4
        / (2 * constants.hbar * constants.epsilon_0 * constants.c)
    )

    def __init__(self, transitions: dict, nsample: int):
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

    def _get_energy_range_ev(self):
        """Get spectrum energy range in eV based on the minimum and maximum excitation energy"""
        # NOTE: We don't include zero to prevent
        # division by zero when converting to wavelength
        x_min = max(0.01, self.excitation_energies.min() - 2.0)
        x_max = self.excitation_energies.max() + 2.0
        return x_min, x_max

    @staticmethod
    def get_energy_unit_factor(unit: EnergyUnit):
        """Returns a multiplication factor to go from eV to other energy units"""

        # https://physics.nist.gov/cgi-bin/cuu/Info/Constants/basis.html
        if unit is EnergyUnit.EV:
            return 1.0
        # TODO: Construct these factors from scipy.constants or use pint
        elif unit is EnergyUnit.NM:
            return 1239.8
        elif unit is EnergyUnit.CM:
            # https://physics.nist.gov/cgi-bin/cuu/Convert?exp=0&num=1&From=ev&To=minv&Action=Only+show+factor
            return 8065.547937

    def calc_lorentzian_spectrum(self, x, y, tau: float):
        normalization_factor = tau / 2 / constants.pi / self.nsample
        for exc_energy, osc_strength in zip(
            self.excitation_energies, self.osc_strengths
        ):
            prefactor = normalization_factor * self.COEFF * osc_strength
            y += prefactor / ((x - exc_energy) ** 2 + (tau**2) / 4)

    def calc_gauss_spectrum(self, x, y, sigma: float):
        normalization_factor = 1 / np.sqrt(2 * constants.pi) / sigma / self.nsample
        for exc_energy, osc_strength in zip(
            self.excitation_energies, self.osc_strengths
        ):
            prefactor = normalization_factor * self.COEFF * osc_strength
            y += prefactor * np.exp(-((x - exc_energy) ** 2) / 2 / sigma**2)

    def get_spectrum(self, kernel: BroadeningKernel, width: float, x_unit: EnergyUnit):
        x_min, x_max = self._get_energy_range_ev()

        # TODO: How to determine this properly to cover a given interval?
        n_sample = 500
        x = np.linspace(x_min, x_max, num=n_sample)
        y = np.zeros(len(x))

        if kernel is BroadeningKernel.GAUSS:
            self.calc_gauss_spectrum(x, y, width)
        elif kernel is BroadeningKernel.LORENTZ:
            self.calc_lorentzian_spectrum(x, y, width)
        else:
            raise ValueError(f"Invalid broadening kernel {kernel}")

        # Conversion factor from eV to given energy unit
        if x_unit == EnergyUnit.NM:
            x, y = self._convert_to_nanometers(x, y)
        else:
            x *= self.get_energy_unit_factor(x_unit)
        return x, y

    def _convert_to_nanometers(self, x, y):
        x = self.get_energy_unit_factor(EnergyUnit.NM) / x
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
        self.width_slider = ipw.FloatSlider(
            min=0.01,
            max=0.5,
            step=0.01,
            value=0.05,
            description="Width / eV",
            continuous_update=True,
        )

        self.kernel_selector = ipw.ToggleButtons(
            options=[(kernel.value, kernel) for kernel in BroadeningKernel],
            description="Broadening",
            disabled=False,
            button_style="info",
            tooltips=[
                "Gaussian broadening",
                "Lorentzian broadening",
            ],
        )

        self.energy_unit_selector = ipw.RadioButtons(
            options=[(unit.value, unit) for unit in EnergyUnit],
            disabled=False,
            description="Energy unit",
        )

        controls = ipw.HBox(
            children=[
                ipw.VBox(children=[self.kernel_selector, self.width_slider]),
                self.energy_unit_selector,
            ]
        )

        # We use this for Debug output for now
        self.debug_output = ipw.Output()

        # https://docs.bokeh.org/en/latest/docs/user_guide/tools.html?highlight=tools#specifying-tools
        tools = "pan,wheel_zoom,box_zoom,reset,save"
        # https://docs.bokeh.org/en/latest/docs/user_guide/tools.html?highlight#hovertool
        tooltips = [("(energy, cross_section)", "($x,$y)")]
        self.figure = self._init_figure(tools=tools, tooltips=tooltips)

        self.download_btn = ipw.Button(
            description="Download spectrum",
            button_style="primary",
            tooltip="Download spectrum as CSV file",
            disabled=True,
            icon="download",
            layout=ipw.Layout(width="max-content"),
        )
        self.download_btn.on_click(self._download_spectrum)

        self.kernel_selector.observe(self._handle_kernel_update, names="value")
        self.energy_unit_selector.observe(
            self._handle_energy_unit_update, names="value"
        )
        self.width_slider.observe(self._handle_width_update, names="value")

        super().__init__(
            [
                self.debug_output,
                controls,
                self.figure,
                self.download_btn,
            ],
            **kwargs,
        )

    def _download_spectrum(self, btn):
        """Download spectrum lines as CSV file"""
        from IPython.display import Javascript, display

        filename = "spectrum.tsv"
        if self.smiles:
            filename = f"spectrum_{self.smiles}.tsv"

        payload = self._prepare_payload()
        if not payload:
            return

        js = Javascript(
            f"""
            var link = document.createElement('a')
            link.href = "data:text/csv;base64,{payload}"
            link.download = "{filename}"
            document.body.appendChild(link)
            link.click()
            document.body.removeChild(link)
            """
        )
        display(js)

    def _prepare_payload(self):
        import base64
        import csv
        from tempfile import SpooledTemporaryFile

        # TODO: Download multiple spectra if available
        line = self.figure.get_figure().select_one({"name": self.THEORY_SPEC_LABEL})
        x = line.data_source.data.get("x")
        y = line.data_source.data.get("y")

        # We're using a tab as a delimiter (TSV file) since the resulting file
        # should be readabale both by Excel and Xmgrace
        delimiter = "\t"

        fieldnames = [
            f"Energy / {self.energy_unit_selector.value.value}",
            f"Intensity / {self.intensity_unit}",
            f"{self.kernel_selector.value.value} broadening, width = {self.width_slider.value} eV",
        ]
        with SpooledTemporaryFile(mode="w+", newline="", max_size=10000000) as csvfile:
            header = delimiter.join(fieldnames)
            csvfile.write(f"# {header}\n")
            writer = csv.writer(csvfile, delimiter=delimiter)
            writer.writerows(zip(x, y))
            csvfile.seek(0)
            return base64.b64encode(csvfile.read().encode()).decode()

    def _validate_transitions(self):
        # TODO: Maybe use named tuple instead of dictionary?
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
        self._plot_spectrum(
            width=change["new"],
            kernel=self.kernel_selector.value,
            energy_unit=self.energy_unit_selector.value,
        )

    def _handle_kernel_update(self, change):
        """Redraw spectra when user changes kernel for broadening"""
        self._plot_spectrum(
            width=self.width_slider.value,
            kernel=change["new"],
            energy_unit=self.energy_unit_selector.value,
        )

    def _handle_energy_unit_update(self, change):
        """Updates the spectra when user changes energy units"""

        energy_unit = change["new"]
        xlabel = f"Energy / {energy_unit.value}"
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

    def _plot_spectrum(
        self, kernel: BroadeningKernel, width: float, energy_unit: EnergyUnit
    ):
        self.download_btn.disabled = True
        if not self._validate_transitions():
            self.hide_line(self.THEORY_SPEC_LABEL)
            return
        # TODO: Need to fix this normalization now that we have multiple conformers!
        # We should have explicit metadata about number of conformers, number of states, number of geometries
        try:
            nsample = self.transitions[-1]["geom_index"] + 1
        except KeyError:
            self.debug_print("Could not determine number of samples")
            nsample = 1

        spec = Spectrum(self.transitions, nsample)
        x, y = spec.get_spectrum(kernel, width, energy_unit)
        self.plot_line(x, y, self.THEORY_SPEC_LABEL)
        self.download_btn.disabled = False

    def debug_print(self, *args):
        with self.debug_output:
            print(*args)

    # plot_line(), hide_line() and remove_line() are public
    # so that additinal stuff can be plotted.
    def plot_line(self, x, y, label: str, **args):
        """Update existing plot line or create a new one.
        Updating existing plot lines unfortunately only work for label=theory
        and label=experiment, that are predefined in _init_figure()
        To modify a custom line, first remove it by calling remove_line(label)

        **args additional arguments are passed into Figure.line()"""
        # https://docs.bokeh.org/en/latest/docs/reference/models/renderers.html?highlight=renderers#renderergroup
        f = self.figure.get_figure()
        line = f.select_one({"name": label})
        if line is None:
            line = f.line(x, y, line_width=2, name=label, **args)
        line.visible = True
        line.data_source.data = {"x": x, "y": y}
        self.figure.update()

    def hide_line(self, label: str):
        """Hide given line from the plot"""
        f = self.figure.get_figure()
        line = f.select_one({"name": label})
        if line is None or not line.visible:
            return
        line.visible = False
        self.figure.update()

    def remove_line(self, label: str):
        # This approach is potentially britle, see:
        # https://discourse.bokeh.org/t/clearing-plot-or-removing-all-glyphs/6792/7
        # Observation: Removing and adding lines via
        # plot_line() and remove_line() works well. However, doing
        # updates on existing lines only works for lines defined in _init_figure()
        f = self.figure.get_figure()
        line = f.select_one({"name": label})
        if line is None:
            return
        f.renderers.remove(line)
        self.figure.update()

    def _init_figure(self, *args, **kwargs) -> BokehFigureContext:
        """Initialize Bokeh figure. Arguments are passed to bokeh.plt.figure()"""
        figure = BokehFigureContext(plt.figure(*args, **kwargs))
        f = figure.get_figure()
        f.xaxis.axis_label = f"Energy / {self.energy_unit_selector.value.value}"
        f.yaxis.axis_label = f"Cross section / {self.intensity_unit}"

        # Initialize line for theoretical spectrum.
        # NOTE: Hardly earned experience: For any lines added later, their updates
        # via line.data_source are not picked up for some unknown reason.
        # Thus, if they need to be updated (e.g. experimental spectrum),
        # they have to be removed (remove_line()) and added again.
        x = np.array([4.0])
        y = np.array([0.0])
        # TODO: Choose inclusive colors!
        # https://doi.org/10.1038/s41467-020-19160-7
        theory_line = f.line(x, y, line_width=2, name=self.THEORY_SPEC_LABEL)
        theory_line.visible = False
        return figure

    def reset(self):
        with self.hold_trait_notifications():
            self.transitions = None
            self.smiles = None
            self.experimental_spectrum = None

        self.download_btn.disabled = True
        self.hide_line(self.THEORY_SPEC_LABEL)
        self.remove_line(self.EXP_SPEC_LABEL)
        self.debug_output.clear_output()

    @traitlets.observe("transitions")
    def _observe_transitions(self, change):
        self._plot_spectrum(
            width=self.width_slider.value,
            kernel=self.kernel_selector.value,
            energy_unit=self.energy_unit_selector.value,
        )

    @traitlets.observe("smiles")
    def _observe_smiles(self, change):
        self._find_experimental_spectrum(change["new"])

    def _find_experimental_spectrum(self, smiles: str):
        """Find an experimental spectrum for a given SMILES
        and plot it if it is available in our DB"""
        if smiles is None or smiles == "":
            self.remove_line(self.EXP_SPEC_LABEL)
            return

        qb = QueryBuilder()
        # TODO: Should we subclass XyData specifically for UV/Vis spectra?
        # Or should we differentiate from other possible Xy nodes
        # by looking at attributes or extras? Maybe label?
        qb.append(XyData, filters={"extras.smiles": smiles})

        if qb.count() == 0:
            self.remove_line(self.EXP_SPEC_LABEL)
            return

        # TODO: For now let's just assume we have one
        # canonical experimental spectrum per compound.
        # for spectrum in qb.iterall():
        self.experimental_spectrum = qb.first()[0]
        self._plot_experimental_spectrum(
            spectrum_node=self.experimental_spectrum,
            energy_unit=self.energy_unit_selector.value,
        )

    def _plot_experimental_spectrum(
        self, spectrum_node: XyData, energy_unit: EnergyUnit
    ):
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
        # TODO: Extract units. Right now we expect energy in nanometers
        # data_energy_unit = spectrum.node.get_attribute('x_units')
        # cross_section_unit = spectrum.node.get_attribute('y_units')

        if energy_unit is EnergyUnit.EV:
            energy = Spectrum.get_energy_unit_factor(EnergyUnit.NM) / energy
        elif energy_unit is EnergyUnit.CM:
            energy = (
                Spectrum.get_energy_unit_factor(EnergyUnit.CM)
                * Spectrum.get_energy_unit_factor(EnergyUnit.NM)
                / energy
            )

        line_options = {
            "line_color": "orange",
            "line_dash": "dashed",
        }
        self.plot_line(energy, cross_section, self.EXP_SPEC_LABEL, **line_options)
