"""Tools for analyzing the spectra.

Authors:
    * Daniel Hollas <daniel.hollas@bristol.ac.uk>
"""

from dataclasses import dataclass
from enum import Enum, unique

import bokeh.plotting as plt
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

    conformer_transitions = traitlets.List(
        trait=traitlets.Dict, allow_none=True, default=None
    )
    disabled = traitlets.Bool(default=True)

    def __init__(self):
        # TODO: Split the different analyses into their own widgets
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

        self.photolysis_tab = ipw.HTML("<p>Coming soon 🙂<p>")

        tab_components = [self.density_tab, self.photolysis_tab]

        tab = ipw.Tab(children=tab_components)
        tab.set_title(1, "Photolysis constant")
        tab.set_title(0, "Energy Density")
        super().__init__(children=[title, tab])

    def reset(self):
        with self.hold_trait_notifications:
            self.disabled = True
            self.density_tab.reset()


class DensityPlotWidget(ipw.VBox):

    conformer_transitions = traitlets.List(
        trait=traitlets.Dict, allow_none=True, default=None
    )
    _density: Density2D = None
    disabled = traitlets.Bool(default=True)

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

    def _update_density_plot(self, plot_type):
        if self.conformer_transitions is None:
            return
        energies, osc_strengths = self._flatten_transitions()
        # TODO: Use Enum
        if plot_type == "SCATTER":
            self.plot_scatter(energies, osc_strengths)
        elif plot_type == "DENSITY":
            self.plot_density(energies, osc_strengths)
        else:
            raise ValueError(f"Unexpected value for toggle: {plot_type}")

    def _flatten_transitions(self):
        # TODO: Merge these comprehensions
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
        self.figure.remove_renderer(self._BOKEH_LABEL, update=False)
        f = self.figure.get_figure()
        f.circle(energies, osc_strengths, fill_color="black", size=5)
        self.figure.update()

    def plot_density(self, energies, osc_strengths):
        self.figure.remove_renderer(self._BOKEH_LABEL, update=False)
        self.figure.update()
        return
        MIN_SAMPLES = 10
        colormap = mplt.cm.magma_r

        self.density_figure.clear_output()
        with self.density_figure:
            self.fig, ax = mplt.subplots(constrained_layout=True, figsize=(3, 3))
            ax.set_xlabel("Excitation energy (eV)")
            ax.set_ylabel("Oscillator strength (-)")

            if len(energies) > MIN_SAMPLES:
                ax.set_title("2D Density", loc="center", size="small")
                # Rudimentary caching, we do not want to recalculate this
                # every time the user presses the toggle button.
                if self._density is None:
                    self._density = self.get_kde(energies, osc_strengths, nbins=35)
                xi, yi, zi = self._density.xi, self._density.yi, self._density.zi
                ax.pcolormesh(
                    xi,
                    yi,
                    zi,
                    shading="gouraud",
                    cmap=colormap,
                )
                ax.contour(xi, yi, zi)
            else:
                ax.set_title("2D Histogram", loc="center", size="small")
                ax.hist2d(energies, osc_strengths, bins=50, density=True, cmap=colormap)

    @staticmethod
    def get_kde(x, y, nbins=20):
        """Evaluate a gaussian kernel density estimate (KDE)
        on a regular grid of nbins x nbins over data extents

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
        """
        xy = np.vstack((x, y))
        k = scipy.stats.gaussian_kde(xy)
        xi, yi = np.mgrid[
            x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j
        ]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        return Density2D(xi, yi, zi.reshape(xi.shape))

    def reset(self):
        self.disabled = True
        self._density = None
        self.density_toggle.value = "SCATTER"

    @traitlets.observe("disabled")
    def _observe_disabled(self, change):
        disabled = change["new"]
        if disabled:
            self.density_toggle.disabled = True
        else:
            self.density_toggle.disabled = False
