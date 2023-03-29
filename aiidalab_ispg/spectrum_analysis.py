"""Tools for analyzing the spectra.

Authors:
    * Daniel Hollas <daniel.hollas@bristol.ac.uk>
"""

from dataclasses import dataclass
from enum import Enum, unique

import bokeh.plotting as plt
import bokeh.palettes
from bokeh.models import ColumnDataSource, Scatter, Range1d, LogAxis, LogScale

import ipywidgets as ipw
import traitlets
import scipy
from scipy import constants
import numpy as np

from .utils import BokehFigureContext


@dataclass
class Density2D:
    # https://stackoverflow.com/questions/60876995/how-to-declare-numpy-array-of-particular-type-as-type-in-dataclass
    # TODO: Improve typing to include only 1D arrays if possible
    xi: np.ndarray
    yi: np.ndarray
    zi: np.ndarray


class SpectrumAnalysisWidget(ipw.VBox):
    """A container class for organizing various analysis widgets"""

    conformer_transitions = traitlets.List(
        trait=traitlets.Dict, allow_none=True, default=None
    )

    spectrum_data = traitlets.List(trait=traitlets.List, allow_none=True, default=None)

    disabled = traitlets.Bool(default=True)

    def __init__(self):
        title = ipw.HTML("<h3>Spectrum analysis</h3>")

        self.density_tab = DensityPlotWidget()
        ipw.dlink(
            (self, "conformer_transitions"),
            (self.density_tab, "conformer_transitions"),
        )
        ipw.dlink(
            (self, "disabled"),
            (self.density_tab, "disabled"),
        )
        ###########################################################################################################
        self.photolysis_tab = PhotolysisPlotWidget()
        ipw.dlink(
            (self, "conformer_transitions"),
            (self.photolysis_tab, "conformer_transitions"),
        )
        ipw.dlink(
            (self, "disabled"),
            (self.photolysis_tab, "disabled"),
        )
        ipw.dlink(
            (self, "spectrum_data"),
            (self.photolysis_tab, "spectrum_data"),
        )
        ###########################################################################################################
        tab_components = [self.photolysis_tab, self.density_tab]

        tab = ipw.Tab(children=tab_components)
        tab.set_title(0, "Photolysis constant")
        tab.set_title(1, "Energy x Oscillator strength")
        super().__init__(children=[title, tab])

    def reset(self):
        with self.hold_trait_notifications():
            self.disabled = True
            self.density_tab.reset()
            self.photolysis_tab.reset()


class DensityPlotWidget(ipw.VBox):
    """A widget for analyzing the correlation between excitation energies
    and oscillator strenghts, seen either as a scatter plot or a 2D density map
    """

    conformer_transitions = traitlets.List(
        trait=traitlets.Dict, allow_none=True, default=None
    )
    disabled = traitlets.Bool(default=True)

    _density: Density2D = None
    _BOKEH_LABEL = "energy-osc"

    def __init__(self):
        self.density_toggle = ipw.ToggleButtons(
            options=[
                ("Scatterplot", "SCATTER"),
                ("2D Density", "DENSITY"),
            ],
            value="SCATTER",
        )
        self.density_toggle.observe(self._observe_density_toggle, names="value")

        # https://docs.bokeh.org/en/latest/docs/user_guide/tools.html?highlight=tools#specifying-tools
        bokeh_tools = "save"
        figure_size = {
            "sizing_mode": "stretch_width",
            "height": 400,
            "max_width": 400,
        }
        self.figure = self._init_figure(tools=bokeh_tools, **figure_size)
        self.figure.layout = ipw.Layout(overflow="initial")

        super().__init__(children=[self.density_toggle, self.figure])

    def _init_figure(self, *args, **kwargs) -> BokehFigureContext:
        """Initialize Bokeh figure. Arguments are passed to bokeh.plt.figure()"""
        figure = BokehFigureContext(plt.figure(*args, **kwargs))
        f = figure.get_figure()
        f.xaxis.axis_label = f"Excitation Energy (eV)"
        f.yaxis.axis_label = f"Oscillator strength (-)"
        return figure

    @traitlets.observe("conformer_transitions")
    def _observe_conformer_transitions(self, change):
        self.disabled = True
        if change["new"] is None or len(change["new"]) == 0:
            self.reset()
            return
        self._update_density_plot(plot_type=self.density_toggle.value)
        self.disabled = False

    def _observe_density_toggle(self, change):
        self._update_density_plot(plot_type=change["new"])

    def _update_density_plot(self, plot_type: str):
        if self.conformer_transitions is None:
            return
        energies, osc_strengths = self._flatten_transitions()
        if plot_type == "SCATTER":
            self.plot_scatter(energies, osc_strengths)
        elif plot_type == "DENSITY":
            self.plot_density(energies, osc_strengths)
        else:
            raise ValueError(f"Unexpected value for toggle: {plot_type}")

    def _flatten_transitions(self):
        # Flatten transitions for all conformers.
        # In the future, we might want to plot individual conformers
        # separately in the scatter plot.
        energies = np.array(
            [
                transitions["energy"]
                for conformer in self.conformer_transitions
                for transitions in conformer["transitions"]
            ]
        )
        osc_strengths = np.array(
            [
                transitions["osc_strength"]
                for conformer in self.conformer_transitions
                for transitions in conformer["transitions"]
            ]
        )
        return energies, osc_strengths

    def plot_scatter(self, energies, osc_strengths):
        """Update existing scatter plot or create a new one."""
        self.figure.remove_renderer(self._BOKEH_LABEL, update=True)
        f = self.figure.get_figure()
        f.x_range.range_padding = f.y_range.range_padding = 0.1
        f.circle(
            energies, osc_strengths, name=self._BOKEH_LABEL
        )  # fill_color="black", size=5

        self.figure.update()

    def plot_density(self, energies, osc_strengths):
        self.figure.remove_renderer(self._BOKEH_LABEL, update=True)
        # TODO: Don't do any density estimation for small number of samples,
        # Instead just do a 2D histogram.
        min_nsample = 3
        if len(energies) < min_nsample:
            return
        nbins = 40
        if self._density is None:
            self._density = self.get_kde(energies, osc_strengths, nbins=nbins)

        xi, yi, zi = self._density.xi, self._density.yi, self._density.zi
        f = self.figure.get_figure()
        f.x_range.range_padding = f.y_range.range_padding = 0
        dw = max(energies) - min(energies)
        dh = max(osc_strengths) - min(osc_strengths)
        f.image(
            image=[zi.transpose()],
            x=min(xi[0]),
            y=min(yi[0]),
            dw=dw,
            dh=dh,
            name=self._BOKEH_LABEL,
            palette=bokeh.palettes.mpl["Magma"][256][::-1],
            level="image",
        )
        f.grid.grid_line_width = 0.5
        self.figure.update()

    # TODO: The rule-of-thumb approach to estimating the bandwidths can fail miserably,
    # see for example formaldehyde with three states.
    # We could provide a user with a scaling factor to adjust...
    # Crucially, we should not attempt to do this if we do not have enough data.
    @staticmethod
    def get_kde(x, y, nbins=20):
        """Evaluate a gaussian kernel density estimate (KDE)
        on a regular grid of nbins x nbins over data extents

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
        """
        xy = np.vstack((x, y))
        # TODO: This call may raise LinAlgError if the matrix is singular!
        k = scipy.stats.gaussian_kde(xy)
        xi, yi = np.mgrid[
            x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j
        ]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        return Density2D(xi, yi, zi.reshape(xi.shape))

    def reset(self):
        with self.hold_trait_notifications():
            self.disabled = True
            self._density = None
            self.figure.clean()
            self.density_toggle.value = "SCATTER"

    @traitlets.observe("disabled")
    def _observe_disabled(self, change):
        disabled = change["new"]
        if disabled:
            self.density_toggle.disabled = True
        else:
            self.density_toggle.disabled = False


# **********************************************************************
"""A widget providing anaylsis of the main spectrum. Namely, the
differential photolysis rate of the molecule is calculated and plotted.
The intensity of actinic flux can be selected by the user - either High,
Medium, or Low. The quantum yield can be altered by the user moving a
slider. The photolysis rate constant will also be displayed"""
# **********************************************************************
class PhotolysisPlotWidget(ipw.VBox):
    conformer_transitions = traitlets.List(
        trait=traitlets.Dict, allow_none=True, default=None
    )

    disabled = traitlets.Bool(default=True)

    spectrum_data = traitlets.List(trait=traitlets.List, allow_none=True, default=None)

    _data = None

    # **********************************************************************
    def __init__(self):
        self.flux_toggle = ipw.ToggleButtons(
            options=[("Low flux", "LOW"), ("Med flux", "MED"), ("High flux", "HIGH")],
            value="HIGH",
        )

        self.flux_toggle.observe(self._observe_flux_toggle, names="value")

        self.yield_slider = ipw.FloatSlider(
            min=0.01,
            max=1,
            step=0.01,
            value=1,
            description="Quantum yield",
            continuous_update=True,
            disabled=False,
        )
        self.yield_slider.observe(self.handle_slider_change, names="value")

        self.flux_data = self.read_in_actinic()

        self.number = ipw.HTML(
            description=r"$$Photolysis rate constant (s^{-1})$$ =",
            style={"description_width": "initial"},
            disabled=True,
        )

        bokeh_tools = "pan,wheel_zoom,box_zoom,reset,save"
        figure_size = {
            "sizing_mode": "stretch_width",
            "height": 400,
            "max_width": 400,
        }
        self.figure = self._init_figure(tools=bokeh_tools, **figure_size)
        self.figure.layout = ipw.Layout(overflow="initial")

        super().__init__(
            children=[self.flux_toggle, self.yield_slider, self.number, self.figure]
        )
        layout = ipw.Layout(justify_content="flex-start")

    # **********************************************************************
    def _init_figure(self, *args, **kwargs) -> BokehFigureContext:
        """Initialize Bokeh figure. Arguments are passed to bokeh.plt.figure()"""
        figure = BokehFigureContext(plt.figure(*args, **kwargs))
        f = figure.get_figure()
        f.xaxis.axis_label = f"λ (nm)"
        f.yaxis.axis_label = r"$$j (s^{-1} nm^{-1})$$"
        f.x_range = Range1d(280, 400)
        f.y_range = Range1d(0, 3e-5)

        f.extra_y_ranges = {"V": Range1d(start=1.0, end=1e15)}
        f.extra_y_scales = {"V": LogScale()}
        f.add_layout(
            LogAxis(
                y_range_name="V",
                axis_label=r"$$ F \hspace{1mm} (quanta \hspace{1mm} cm^{-2} \hspace{1mm} s^{-1} \hspace{1mm} nm^{-1})$$",
            ),
            "right",
        )

        return figure

    # **********************************************************************

    # **********************************************************************
    # Observing changes in widget and spectrum data, update j plot for each change

    @traitlets.observe("spectrum_data")
    def _observe_spectrum_data(self, change):
        """
        Observe changes to the spectrum data and update the J plot accordingly.

        :param change: The change object containing the old and new values of the spectrum data.
        :type change: dict
        """
        self.disabled = True
        if change["new"] is None or len(change["new"]) == 0:
            self.reset()
            return
        self._update_j_plot(
            plot_type=self.flux_toggle.value, quantumY=self.yield_slider.value
        )
        self.disabled = False

    def _observe_flux_toggle(self, change):
        """Redraw spectra when user changes flux via toggle

        :param change: The change object containing the old and new values of the flux toggle.
        :type change: dict
        """
        self._update_j_plot(plot_type=change.new, quantumY=self.yield_slider.value)

    def handle_slider_change(self, change):
        """Redraw spectra when user changes quantum yield via slider

        :param change: The change object containing the old and new values of the yield slider.
        :type change: dict
        """
        self._update_j_plot(plot_type=self.flux_toggle.value, quantumY=change.new)

    # **********************************************************************

    # **********************************************************************
    def _update_j_plot(self, plot_type: str, quantumY):
        """
        Update the J plot based on the given plot type and quantum yield

        :param plot_type: The flux of plot to generate. Can be "LOW", "MED", or "HIGH".
        :type plot_type: str
        :param quantumY: The quantum yield value to use in the calculation.
        :type quantumY: float
        :return: A tuple containing the J values and wavelengths used in the plot.
        :rtype: tuple
        """

        if self.spectrum_data is None:
            return
        if plot_type == "LOW":
            j_values = self.calculation(1, quantum_yield=quantumY)
            wavelengths = self.flux_data[0]
            self.plot_line(wavelengths, j_values, label="label")
            self.add_log_axis(wavelengths, 1, label="log")
        elif plot_type == "MED":
            j_values = self.calculation(2, quantum_yield=quantumY)
            wavelengths = self.flux_data[0]
            self.plot_line(wavelengths, j_values, label="label")
            self.add_log_axis(wavelengths, 2, label="log")

        elif plot_type == "HIGH":
            j_values = self.calculation(3, quantum_yield=quantumY)
            wavelengths = self.flux_data[0]
            self.plot_line(wavelengths, j_values, label="label")
            self.add_log_axis(wavelengths, 3, label="log")

        else:
            raise ValueError(f"Unexpected value for toggle: {plot_type}")

        self.number.value = f"{np.format_float_scientific(np.trapz(j_values, dx=1),3)}"
        return j_values, wavelengths

    # **********************************************************************

    # **********************************************************************
    def reset(self):
        """
        Reset the figure and its associated widgets to their default values.
        """
        with self.hold_trait_notifications():
            self.disabled = True
            self.figure.clean()
            self.flux_toggle.value = "HIGH"
            self.yield_slider.value = 1
            self.number.value = ""

    @traitlets.observe("disabled")
    def _observe_disabled(self, change):
        """
        Observe changes to the "disabled" trait and update the state of the flux_toggle and yield_slider widgets accordingly.

        :param change: A dictionary containing information about the change to the "disabled" trait.
        """
        disabled = change["new"]
        if disabled:
            self.flux_toggle.disabled = True
            self.yield_slider.disabled = True
        else:
            self.flux_toggle.disabled = False
            self.yield_slider.disabled = False

    # **********************************************************************

    # **********************************************************************
    def read_in_actinic(self):
        """
        Read in actinic flux data from a CSV file.

        :return: A tuple containing the wavelength and low, medium, and high actinic flux data.
        :rtype: tuple
        """
        z = np.loadtxt(
            fname="/home/jovyan/apps/aiidalab-ispg/aiidalab_ispg/static/StandardActinicFluxes2.csv",
            delimiter=",",
            skiprows=1,
            unpack=True,
            usecols=(2, 3, 4, 5),
        )
        WL = z[0]
        LOW = z[1]
        MED = z[2]
        HIGH = z[3]
        return WL, LOW, MED, HIGH

    def calculation(self, level, quantum_yield):
        """
        Calculate the J values for the given level and quantum yield.

        :param level: The level of actinic flux to use in the calculation.
        :type level: int
        :param quantum_yield: The quantum yield value to use in the calculation.
        :type quantum_yield: float
        :return: The smoothed J values.
        :rtype: np.ndarray
        """
        du = level
        interpolated_xsection = self.prepare_for_plot()
        j_vals = self.prepare_for_plot() * self.flux_data[du] * quantum_yield
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size
        j_smoothed = np.convolve(j_vals, kernel, mode="valid")
        return j_smoothed

    def prepare_for_plot(self):
        """
        Prepare the molecular intensity data for plotting by interpolating cross section onto actinic flux x values.

        :return: The interpolated cross section data.
        :rtype: np.ndarray
        """
        mol_wlength, xsection = self.spectrum_data
        mol_wlength = np.flip(mol_wlength)
        mol_intensity = xsection
        mol_intensity = np.flip(mol_intensity)
        wl_max = mol_wlength.max()
        masked_data = self.mask_data(mol_wlength, mol_intensity, 280, np.floor(wl_max))
        mol_wlength = masked_data[0]
        mol_intensity = masked_data[1]
        mol_intensity_interp = np.interp(self.flux_data[0], mol_wlength, mol_intensity)
        return mol_intensity_interp

    def process(self, file, delimiter):
        """
        Process the given CSV file and return its data as a NumPy array.

        :param file: The path to the CSV file to process.
        :type file: str
        :param delimiter: The delimiter used in the CSV file.
        :type delimiter: str
        :return: The data from the CSV file as a NumPy array.
        :rtype: np.ndarray
        """
        from csv import reader

        with open(file) as csv_file:
            csv_reader = reader(csv_file, delimiter=delimiter)
            data = list(csv_reader)
        data = np.asarray(data[1:]).transpose()
        return data.astype(float)

    def mask_data(self, array_wlength, array_intensities, minimum, maximum):
        """
        Mask the given wavelength and intensity data arrays based on the given minimum and maximum values.

        :param array_wlength: The wavelength data array to mask.
        :type array_wlength: np.ndarray
        :param array_intensities: The intensity data array to mask.
        :type array_intensities: np.ndarray
        :param minimum: The minimum wavelength value to include in the masked data.
        :type minimum: float
        :param maximum: The maximum wavelength value to include in the masked data.
        :type maximum: float
        :return: A tuple containing the masked wavelength and intensity data arrays.
        :rtype: tuple
        """
        low_cutoff = np.where(np.asarray(array_wlength) > minimum)[0][0] - 1
        array_wlength = array_wlength[low_cutoff:]
        array_intensities = array_intensities[low_cutoff:]
        high_cutoff = np.where(np.asarray(array_wlength) > maximum)[0][0]
        array_wlength = array_wlength[:high_cutoff]
        array_intensities = array_intensities[:high_cutoff]
        return array_wlength, array_intensities

    # **********************************************************************

    # **********************************************************************
    def plot_line(self, x, y, label, update=True, **args):
        """Plot a line on the figure with the given x and y data and label.

        :param x: The x data for the line.
        :type x: np.ndarray
        :param y: The y data for the line.
        :type y: np.ndarray
        :param label: The label for the line.
        :type label: str
        :param update: Whether to update the figure after plotting the line. Defaults to True.
        :type update: bool
        :param args: Additional arguments to pass to the line plot function."""
        f = self.figure.get_figure()
        line = f.select_one({"name": label})
        if line is not None:
            self.remove_line(label)

        f.line(x, y, name=label, **args, line_width=2)
        y_range_max = y.max() + y.max() * 0.2
        self.update_y_axis(0, y_range_max)

        if update:
            self.figure.update()

    def update_y_axis(self, start, end):
        """Update the y-axis range of the figure.

        :param start: The new start value for the y-axis range.
        :type start: float
        :param end: The new end value for the y-axis range.
        :type end: float"""
        f = self.figure.get_figure()
        f.y_range.start = start
        f.y_range.end = end

    def add_log_axis(self, x, level, label, update=True, **args):
        """
        Add a log axis to the figure.

        :param x: The x values for the line to be plotted.
        :param level: The level of the flux data to be plotted.
        :param label: The name of the line to be plotted.
        :param update: Whether to update the figure after adding the line. Default is True.
        :param args: Additional arguments to be passed to the line function.
        """
        f = self.figure.get_figure()
        line = f.select_one({"name": label})
        if line is not None:
            self.remove_line(label)
        y = self.flux_data[level]
        f.line(x, y, y_range_name="V", name=label, color="red")
        if update:
            self.figure.update()

    # **********************************************************************

    # **********************************************************************
    def remove_line(self, label: str, update=True):
        """
        Remove a line from the figure.

        :param label: The name of the line to be removed.
        :param update: Whether to update the figure after removing the line. Default is True.
        """
        f = self.figure.get_figure()
        line = f.select_one({"name": label})
        if line is None:
            return
        f.renderers.remove(line)
        if update:
            self.figure.update()
