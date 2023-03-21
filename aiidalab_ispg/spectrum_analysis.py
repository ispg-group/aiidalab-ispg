"""Tools for analyzing the spectra.

Authors:
    * Daniel Hollas <daniel.hollas@bristol.ac.uk>
"""

from dataclasses import dataclass
from enum import Enum, unique

import bokeh.plotting as plt
import bokeh.palettes
from bokeh.models import ColumnDataSource, Scatter

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

        self.photolysis_tab = PhotolysisPlotWidget()
        ipw.dlink(
            (self, "conformer_transitions"),
            (self.photolysis_tab, "conformer_transitions"),
        )
        ipw.dlink(
            (self, "disabled"),
            (self.photolysis_tab, "disabled"),
        )

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
        bokeh_tools = "pan,wheel_zoom,box_zoom,reset,save"
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
            energies, osc_strengths, name=self._BOKEH_LABEL, fill_color="black", size=5
        )
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


class PhotolysisPlotWidget(ipw.VBox):

    conformer_transitions = traitlets.List(
        trait=traitlets.Dict, allow_none=True, default=None
    )

    disabled = traitlets.Bool(default=True)

    _data = None

    def __init__(self):

        self.flux_toggle = ipw.ToggleButtons(
            options=[
                ("Low flux", "LOW"),
                ("Med flux", "MED"),
                ("High flux", "HIGH"),
            ],
            value="HIGH",
        )
        self.flux_toggle.observe(self._handle_flux_toggle, names="value")

        self.yield_slider = ipw.FloatSlider(
            min=0.01,
            max=1,
            step=0.01,
            value=1,
            description="Quantum yield",
            continuous_update=True,
            disabled=True,
        )
        self.yield_slider.observe(self._handle_yield_update, names="value")

        self.output = ipw.Output()

        # https://docs.bokeh.org/en/latest/docs/user_guide/tools.html?highlight=tools#specifying-tools
        bokeh_tools = "pan,wheel_zoom,box_zoom,reset,save"
        figure_size = {
            "sizing_mode": "stretch_width",
            "height": 400,
            "max_width": 400,
        }
        self.figure = self._init_figure(tools=bokeh_tools, **figure_size)
        self.figure.layout = ipw.Layout(overflow="initial")

        super().__init__(
            children=[self.yield_slider, self.flux_toggle, self.figure, self.output]
        )

    def _init_figure(self, *args, **kwargs) -> BokehFigureContext:
        """Initialize Bokeh figure. Arguments are passed to bokeh.plt.figure()"""
        figure = BokehFigureContext(plt.figure(*args, **kwargs))
        f = figure.get_figure()
        f.xaxis.axis_label = f"Excitation Energy (nm)"
        f.yaxis.axis_label = f"Oscillator strength (-)"
        return figure

    def _handle_flux_toggle(self, change):
        """Redraw spectra when user changes flux via toggle"""
        self._plot_j_values(flux=change["new"], q_yield=self.yield_slider.value)

    def _handle_yield_update(self, change):
        """Redraw spectra when user changes yield via slider"""
        self._plot_j_values(flux=self.flux_toggle.value, q_yield=change["new"])

    #     @traitlets.observe("conformer_transitions")
    def _get_data(self):
        from .spectrum import Spectrum, EnergyUnit, BroadeningKernel

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

        for conf_id, conformer in enumerate(self.conformer_transitions):
            spec = Spectrum(conformer["transitions"], conformer["nsample"])
            #             x_min = 250
            #             x_max = 500
            #             width = 0.05
            #             x = np.linspace(x_min, x_max, num=self.N_SAMPLE_POINTS)
            #             y = np.zeros(len(x))
            x, y, xs, ys = spec.get_spectrum(
                BroadeningKernel.GAUSS, 0.05, EnergyUnit.NM, x_min=x_min, x_max=x_max
            )

            y *= conformer["weight"]
            total_cross_section += y

        # Plot total spectrum
        self.plot_line(x, total_cross_section, "theory", line_width=2)

    def _get_j_values(self):
        self._get_data()
        # j calculation

    @traitlets.observe("conformer_transitions")
    def _observe_conformer_transitions(self, change):
        self.disabled = True
        if change["new"] is None or len(change["new"]) == 0:
            self.reset()
            return

        ## Here is a good place to add the comparison for flux?
        # Then call update j plot with flux_type
        self._update_j_plot(self.flux_toggle.value, self.yield_slider.value)
        self.disabled = False

    def _update_j_plot(self, flux_type: str, yield_value: float):
        if self.conformer_transitions is None:
            return
        else:
            self._plot_j_values(self.flux_toggle.value, self.yield_slider.value)

    def _plot_j_values(self, flux, q_yield):
        # Determine plot range
        self._get_j_values()
        # plot_line()

    def _plot_flux(self):
        pass

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

    def reset(self):
        with self.hold_trait_notifications():
            self.disabled = True
            self._data = None
            self.figure.clean()
            self.flux_toggle.value = "HIGH"
            self.yield_slider.value = 1

    @traitlets.observe("disabled")
    def _observe_disabled(self, change):
        disabled = change["new"]
        if disabled:
            self.flux_toggle.disabled = True
            self.yield_slider.disabled = True
        else:
            self.flux_toggle.disabled = False
            self.yield_slider.disabled = False
