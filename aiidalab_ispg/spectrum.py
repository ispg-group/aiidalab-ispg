"""Calculating NEA UV/vis spectra and displaying them in an interactive plot.

Authors:
    * Daniel Hollas <daniel.hollas@bristol.ac.uk>
"""
from enum import Enum, unique

import ipywidgets as ipw
import traitlets
from scipy import constants
import numpy as np

from aiida.orm import load_node, QueryBuilder
from aiida.plugins import DataFactory

import bokeh.plotting as plt

from .widgets import TrajectoryDataViewer
from .utils import AUtoEV, BokehFigureContext
from .spectrum_analysis import SpectrumAnalysisWidget

XyData = DataFactory("core.array.xy")
StructureData = DataFactory("core.structure")
TrajectoryData = DataFactory("core.array.trajectory")


@unique
class EnergyUnit(Enum):
    EV = "eV"
    CM = "cm^-1"
    NM = "nm"


@unique
class BroadeningKernel(Enum):
    GAUSS = "gaussian"
    LORENTZ = "lorentzian"


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

    @staticmethod
    def get_energy_range_ev(excitation_energies):
        """Get spectrum energy range in eV based on the minimum and maximum excitation energy"""
        # NOTE: We don't include zero to prevent
        # division by zero when converting to wavelength
        x_min = max(0.01, excitation_energies.min() - 1.5)
        x_max = excitation_energies.max() + 1.5
        return x_min, x_max

    @staticmethod
    def get_energy_unit_factor(unit: EnergyUnit):
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

    def _calc_lorentzian_spectrum(self, x, y, tau: float):
        """Calculate NEA spectrum broadened with a Lorentzian function:

        https://en.wikipedia.org/wiki/Cauchy_distribution#Probability_density_function
        """
        normalization_factor = tau / 2 / constants.pi / self.nsample
        for exc_energy, osc_strength in zip(
            self.excitation_energies, self.osc_strengths
        ):
            prefactor = normalization_factor * self.COEFF * osc_strength
            y += prefactor / ((x - exc_energy) ** 2 + (tau**2) / 4)

    def _calc_gauss_spectrum(self, x, y, sigma: float):
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
        x_min=None,
        x_max=None,
    ):
        if x_min is None or x_max is None:
            x_min, x_max = self.get_energy_range_ev(self.excitation_energies)

        x = np.linspace(x_min, x_max, num=self.N_SAMPLE_POINTS)
        y = np.zeros(len(x))

        if kernel is BroadeningKernel.GAUSS or kernel == BroadeningKernel.GAUSS:
            self._calc_gauss_spectrum(x, y, width)
        elif kernel is BroadeningKernel.LORENTZ:
            self._calc_lorentzian_spectrum(x, y, width)
        else:
            raise ValueError(f"Invalid broadening kernel {kernel}")

        # Conversion factor from eV to given energy unit
        if x_unit is EnergyUnit.NM or x_unit == EnergyUnit.NM:
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

    def _convert_to_nanometers(self, x, y):
        x = self.get_energy_unit_factor(EnergyUnit.NM) / x
        # Filtering out ultralong wavelengths
        # TODO: Set this threshold adaptively based on lowest energy transition
        # TODO: Because the arrays are ordered, there's much easier way than np.delete()
        nm_thr = 1000
        to_delete = [i for i, nm in enumerate(x) if nm > nm_thr]
        x = np.delete(x, to_delete)
        y = np.delete(y, to_delete)
        return x, y


class SpectrumWidget(ipw.VBox):

    disabled = traitlets.Bool(default=True)
    conformer_transitions = traitlets.List(
        trait=traitlets.Dict, allow_none=True, default=None
    )
    conformer_structures = traitlets.Union(
        [traitlets.Instance(StructureData), traitlets.Instance(TrajectoryData)],
        allow_none=True,
    )

    selected_conformer_id = traitlets.Int(allow_none=True, default_value=None)

    spectrum_data = traitlets.List(trait=traitlets.List, allow_none=True, default=None)

    # We use SMILES to find matching experimental spectra
    # that are possibly stored in our DB as XyData.
    smiles = traitlets.Unicode(allow_none=True, default_value=None)
    experimental_spectrum_uuid = traitlets.Unicode(
        allow_none=True, default_value=None, read_only=True
    )

    # For now, we do not allow different intensity units
    intensity_unit = "cm² per molecule"

    THEORY_SPEC_LABEL = "theory"
    EXP_SPEC_LABEL = "experiment"
    STICK_SPEC_LABEL = "sticks"

    # https://docs.bokeh.org/en/latest/docs/user_guide/tools.html?highlight=tools#specifying-tools
    _TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    # https://docs.bokeh.org/en/latest/docs/user_guide/tools.html?highlight#hovertool
    _TOOLTIPS = [("(energy, cross_section)", "($x,$y)")]

    def __init__(self, **kwargs):
        self.width_slider = ipw.FloatSlider(
            min=0.01,
            max=0.5,
            step=0.01,
            value=0.05,
            description="Width (eV)",
            continuous_update=True,
            disabled=True,
        )
        self.width_slider.observe(self._handle_width_update, names="value")

        self.kernel_selector = ipw.ToggleButtons(
            options=[(kernel.value, kernel) for kernel in BroadeningKernel],
            value=BroadeningKernel.GAUSS,
            description="Broadening",
            disabled=True,
            button_style="info",
            tooltips=[
                "Gaussian broadening",
                "Lorentzian broadening",
            ],
        )
        self.kernel_selector.observe(self._handle_kernel_update, names="value")

        self.energy_unit_selector = ipw.RadioButtons(
            options=[(unit.value, unit) for unit in EnergyUnit],
            disabled=True,
            description="Energy unit",
        )
        self.energy_unit_selector.observe(
            self._handle_energy_unit_update, names="value"
        )

        self.spectrum_controls = ipw.VBox(
            children=[
                self.kernel_selector,
                self.width_slider,
                self.energy_unit_selector,
            ]
        )

        self.stick_toggle = ipw.ToggleButton(
            description="Show stick spectrum",
            tooltip="Show individual transitions as sticks in the spectrum.",
            disabled=True,
            value=False,
        )
        self.stick_toggle.observe(self._handle_stick_toggle, names="value")

        self.conformer_toggle = ipw.ToggleButton(
            description="Show conformers",
            tooltip="Show spectra of individual conformers",
            disabled=True,
            value=False,
        )
        self.conformer_toggle.observe(self._handle_conformer_toggle, names="value")

        self.debug_output = ipw.HTML()

        # https://docs.bokeh.org/en/latest/docs/examples/basic/layouts/sizing_mode.html
        figure_size = {
            "sizing_mode": "fixed",
            "height": 500,
            "width": 500,
        }
        self.figure = self._init_figure(
            tools=self._TOOLS, tooltips=self._TOOLTIPS, **figure_size
        )
        self.figure.layout = ipw.Layout(overflow="initial")

        self.download_btn = ipw.Button(
            description="Download spectrum",
            button_style="primary",
            tooltip="Download spectrum as CSV file",
            disabled=True,
            icon="download",
            layout=ipw.Layout(width="max-content"),
        )
        self.download_btn.on_click(self._download_spectrum)

        layout = ipw.Layout(justify_content="flex-start")
        self.conformer_header = ipw.HTML()
        self.conformer_header.layout.padding = "0px 0px 0px 15px"
        self.conformer_viewer = TrajectoryDataViewer(configuration_tabs=[])
        ipw.dlink(
            (self.conformer_viewer, "selected_structure_id"),
            (self, "selected_conformer_id"),
        )
        ipw.dlink(
            (self, "conformer_structures"),
            (self.conformer_viewer, "trajectory"),
        )

        self.analysis = SpectrumAnalysisWidget()
        ipw.dlink(
            (self, "conformer_transitions"),
            (self.analysis, "conformer_transitions"),
        )

        ipw.dlink(
            (self, "spectrum_data"),
            (self.analysis, "spectrum_data"),
        )

        super().__init__(
            [
                self.debug_output,
                ipw.HBox([self.download_btn, self.conformer_toggle, self.stick_toggle]),
                ipw.HBox(
                    [
                        self.figure,
                        ipw.VBox(
                            [
                                self.spectrum_controls,
                                self.conformer_header,
                                self.conformer_viewer,
                            ],
                            layout=layout,
                        ),
                    ],
                ),
                self.analysis,
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

    def _validate_transitions(self, transitions):
        # TODO: Maybe use named tuple instead of dictionary?
        # https://realpython.com/python-namedtuple/
        if transitions is None or len(transitions) == 0:
            self.debug_print("ERROR: Got empty transitions")
            return False

        for tr in transitions:
            if not isinstance(tr, dict) or (
                "energy" not in tr or "osc_strength" not in tr
            ):
                self.debug_print("ERROR: Invalid transition", tr)
                return False
        return True

    def _handle_stick_toggle(self, change):
        """Redraw show/hide stick transitions"""
        # Note: We replot the whole spectrum as sticks are currently tied
        # to the whole spectrum.
        self._plot_spectrum(
            width=self.width_slider.value,
            kernel=self.kernel_selector.value,
            energy_unit=self.energy_unit_selector.value,
        )

    def _handle_conformer_toggle(self, change):
        """Show/hide conformers and their individual spectra"""
        if not change["new"]:
            self._hide_all_conformers()
            return

        if len(self.conformer_transitions) == 1:
            return

        self._plot_spectrum(
            width=self.width_slider.value,
            kernel=self.kernel_selector.value,
            energy_unit=self.energy_unit_selector.value,
        )

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
        # print(energy_unit)
        xlabel = f"Energy / {energy_unit.value}"
        self.figure.get_figure().xaxis.axis_label = xlabel

        self._plot_spectrum(
            width=self.width_slider.value,
            kernel=self.kernel_selector.value,
            energy_unit=energy_unit,
        )
        if self.experimental_spectrum_uuid:
            node = load_node(self.experimental_spectrum_uuid)
            self.plot_experimental_spectrum(spectrum_node=node, energy_unit=energy_unit)

    def _unhighlight_conformer(self):
        self.remove_line("conformer_selected")

    def _highlight_conformer(self, conf_id: int):
        f = self.figure.get_figure()
        label = f"conformer_{conf_id}"
        if line := f.select_one({"name": label}):
            # This does not seem to work, possibly because conformers
            # are not there from the beginning
            # line.glyph.update(line_dash="solid")
            x = line.data_source.data["x"]
            y = line.data_source.data["y"]
            self.plot_line(x, y, label="conformer_selected", line_color="red")

    def _hide_all_conformers(self):
        self._unhighlight_conformer()
        f = self.figure.get_figure()
        labels = [r.name for r in f.renderers]
        for label in filter(lambda label: label.startswith("conformer_"), labels):
            # NOTE: Hiding does not seem to work
            # Removing without immediate figure update also does not work
            self.remove_line(label, update=True)

    def _plot_conformer(self, x, y, conf_id, update=True, line_dash="dashed"):
        line_options = {
            "line_color": "black",
            "line_dash": line_dash,
            "line_width": 1,
        }
        label = f"conformer_{conf_id}"
        self.plot_line(x, y, label, update=update, **line_options)

    def _plot_spectrum(
        self, kernel: BroadeningKernel, width: float, energy_unit: EnergyUnit
    ):
        # Determine spectrum energy range based on all excitation energies
        all_exc_energies = np.array(
            [
                transitions["energy"]
                for conformer in self.conformer_transitions
                for transitions in conformer["transitions"]
            ]
        )

        x_min, x_max = Spectrum.get_energy_range_ev(all_exc_energies)

        total_cross_section = np.zeros(Spectrum.N_SAMPLE_POINTS)

        x_stick = []
        y_stick = []
        # Iterate over conformers, the total spectrum is a sum of
        # individual conformer spectra multiplied by a Boltzmann factor.
        for conf_id, conformer in enumerate(self.conformer_transitions):
            spec = Spectrum(conformer["transitions"], conformer["nsample"])
            x, y, xs, ys = spec.get_spectrum(
                kernel, width, energy_unit, x_min=x_min, x_max=x_max
            )

            y *= conformer["weight"]
            total_cross_section += y

            ys *= conformer["weight"]
            x_stick = np.concatenate((x_stick, xs))
            y_stick = np.concatenate((y_stick, ys))

            # Plot spectrum of an individual conformer
            if self.conformer_toggle.value:
                self._plot_conformer(x, y, conf_id, update=False)

        self.spectrum_data = [x.tolist(), total_cross_section.tolist()]

        # Energy unit not nm needs converting for spectrum analysis
        if energy_unit != EnergyUnit.NM:
            total_cross_section_nm = np.zeros(Spectrum.N_SAMPLE_POINTS)

            for conf_id, conformer in enumerate(self.conformer_transitions):
                spec = Spectrum(conformer["transitions"], conformer["nsample"])
                x_nm, y_nm, xs_nm, ys_nm = spec.get_spectrum(
                    kernel, width, EnergyUnit.NM, x_min=x_min, x_max=x_max
                )

                y_nm *= conformer["weight"]
                total_cross_section_nm += y_nm

            self.spectrum_data = [x_nm.tolist(), total_cross_section_nm.tolist()]

        # Plot total spectrum
        self.plot_line(x, total_cross_section, self.THEORY_SPEC_LABEL, line_width=2)

        if self.conformer_toggle.value and len(self.conformer_transitions) > 1:
            self._highlight_conformer(self.selected_conformer_id)

        if self.stick_toggle.value:
            self.plot_sticks(x_stick, y_stick, self.STICK_SPEC_LABEL)
        else:
            self.remove_line(self.STICK_SPEC_LABEL)
        self.download_btn.disabled = False

    def debug_print(self, *args):
        self.debug_output.value = "<br>".join(*args)

    def plot_sticks(self, x, y, label: str, **args):
        """Plot stick spectrum"""
        f = self.figure.get_figure()
        # First remove existing sticks (for now we only support one sets of sticks)
        if sticks := f.select_one({"name": label}):
            f.renderers.remove(sticks)
        sticks = f.segment(
            x0=x,
            x1=x,
            y0=np.zeros(x.size),
            y1=y,
            line_color="black",
            line_width=1,
            name=label,
            **args,
        )
        self.figure.update()

    # plot_line(), hide_line() and remove_line() are public
    # so that additinal stuff can be plotted.
    def plot_line(self, x, y, label, update=True, **args):
        """Update existing plot line or create a new one.
        Updating existing plot lines unfortunately only work for label=theory
        and label=experiment, that are predefined in _init_figure()
        To modify a custom line, first remove it by calling remove_line(label)

        **args additional arguments are passed into Figure.line()"""
        # https://docs.bokeh.org/en/latest/docs/reference/models/renderers.html?highlight=renderers#renderergroup
        f = self.figure.get_figure()
        line = f.select_one({"name": label})
        if line is not None:
            # line.data_source.data = {"x": x, "y": y}
            self.remove_line(label)
        f.line(x, y, name=label, **args)
        if update:
            self.figure.update()

    def hide_line(self, label: str, update=True):
        """Hide given line from the plot"""
        f = self.figure.get_figure()
        line = f.select_one({"name": label})
        if line is None or not line.visible:
            return
        line.visible = False
        if update:
            self.figure.update()

    def remove_line(self, label: str, update=True):
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
        if update:
            self.figure.update()

    def _init_figure(self, *args, **kwargs) -> BokehFigureContext:
        """Initialize Bokeh figure. Arguments are passed to bokeh.plt.figure()"""
        figure = BokehFigureContext(plt.figure(*args, **kwargs))
        f = figure.get_figure()
        f.xaxis.axis_label = f"Energy ({self.energy_unit_selector.value.value})"
        f.yaxis.axis_label = f"Cross section ({self.intensity_unit})"

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

    @traitlets.observe("disabled")
    def _observe_disabled(self, change):
        disabled = change["new"]
        if disabled:
            self.download_btn.disabled = True
            self.stick_toggle.disabled = True
            self.conformer_toggle.disabled = True
            self.energy_unit_selector.disabled = True
            self.width_slider.disabled = True
            self.kernel_selector.disabled = True
        else:
            self.download_btn.disabled = False
            self.stick_toggle.disabled = False
            self.energy_unit_selector.disabled = False
            self.width_slider.disabled = False
            self.kernel_selector.disabled = False
            if len(self.conformer_transitions) > 1:
                self.conformer_toggle.disabled = False

    def reset(self):
        with self.hold_trait_notifications():
            self.conformer_transitions = None
            self.conformer_structures = None
            self.smiles = None
            self.set_trait("experimental_spectrum_uuid", None)
            self.analysis.reset()

        self.disabled = True
        self.figure.clean()
        self.debug_output.value = ""

    @traitlets.validate("conformer_transitions")
    def _validate_conformers(self, change):
        conformer_transitions = change["value"]
        if conformer_transitions is None:
            return None
        if not all(
            self._validate_transitions(c["transitions"]) for c in conformer_transitions
        ):
            raise ValueError("Invalid conformer transitions")
        return conformer_transitions

    @traitlets.validate("conformer_structures")
    def _validate_conformer_structures(self, change):
        structures = change["value"]
        if structures is None:
            return None

        if isinstance(structures, TrajectoryData):
            return structures
        elif isinstance(structures, StructureData):
            return TrajectoryData(structurelist=(structures,))
        else:
            raise ValueError(f"Unsupported type {type(structures)}")

    @traitlets.observe("selected_conformer_id")
    def _observe_selected_conformer(self, change):
        self._unhighlight_conformer()
        self._highlight_conformer(change["new"])

    @traitlets.observe("conformer_structures")
    def _observe_conformers(self, change):
        self.conformer_viewer._viewer.handle_resize()

    @traitlets.observe("conformer_transitions")
    def _observe_conformer_transitions(self, change):
        self.disabled = True
        self._hide_all_conformers()
        if change["new"] is None:
            return
        self._plot_spectrum(
            width=self.width_slider.value,
            kernel=self.kernel_selector.value,
            energy_unit=self.energy_unit_selector.value,
        )
        self.disabled = False

    @traitlets.observe("smiles")
    def _observe_smiles(self, change):
        self.find_experimental_spectrum_by_smiles(change["new"])

    @traitlets.observe("experimental_spectrum_uuid")
    def _observe_experimental_spectrum_uuid(self, change):
        if change["new"] == change["old"]:
            return
        if change["new"] is None:
            self.remove_line(self.EXP_SPEC_LABEL)
            return
        self.plot_experimental_spectrum(
            spectrum_node=load_node(change["new"]),
            energy_unit=self.energy_unit_selector.value,
        )

    def find_experimental_spectrum_by_smiles(self, smiles: str):
        """Find an experimental spectrum for a given SMILES
        and plot it if it is available in our DB"""

        self.set_trait("experimental_spectrum_uuid", None)
        if smiles is None or smiles == "":
            return

        qb = QueryBuilder()
        # TODO: Should we subclass XyData specifically for UV/Vis spectra?
        # Or should we differentiate from other possible Xy nodes
        # by looking at attributes or extras? Maybe label?
        qb.append(XyData, filters={"extras.smiles": smiles})
        if qb.count() == 0:
            return

        # TODO: For now let's just assume we have one
        # canonical experimental spectrum per compound.
        # for spectrum in qb.iterall():
        experimental_spectrum_node = qb.first()[0]
        self.set_trait("experimental_spectrum_uuid", experimental_spectrum_node.uuid)

    def plot_experimental_spectrum(
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
            "line_width": 2,
        }
        self.plot_line(energy, cross_section, self.EXP_SPEC_LABEL, **line_options)
