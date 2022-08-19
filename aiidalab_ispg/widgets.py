"""Widgets for the ISPG apps.

Authors:

    * Daniel Hollas <daniel.hollas@bristol.ac.uk>
"""

import base64

import ipywidgets as ipw
import traitlets
import nglview
from dataclasses import dataclass

from aiida.cmdline.utils.query.calculation import CalculationQueryBuilder
from aiida.orm import load_node, Node
from aiida.plugins import DataFactory

from aiidalab_widgets_base import register_viewer_widget
from aiidalab_widgets_base.viewers import StructureDataViewer

# trigger registration of the viewer widgets
from aiidalab_ispg.qeapp import widgets  # noqa: F401
import aiidalab_ispg.qeapp.process
from .utils import get_formula

StructureData = DataFactory("structure")
TrajectoryData = DataFactory("array.trajectory")

__all__ = [
    "TrajectoryDataViewer",
]


class WorkChainSelector(aiidalab_ispg.qeapp.process.WorkChainSelector):

    FMT_WORKCHAIN = "{wc.pk:6}{wc.ctime:>10}\t{wc.state:<16}\t{wc.formula}"

    def __init__(self, workchain_label, **kwargs):
        self.workchain_label = workchain_label
        super().__init__(**kwargs)

    @dataclass
    class WorkChainData:
        pk: int
        ctime: str
        state: str
        formula: str

    @classmethod
    def find_work_chains(cls, workchain_label):
        builder = CalculationQueryBuilder()
        filters = builder.get_filters(
            process_label=workchain_label,
        )
        query_set = builder.get_query_set(
            filters=filters,
            order_by={"ctime": "desc"},
        )
        projected = builder.get_projected(
            query_set, projections=["pk", "ctime", "state"]
        )

        for process in projected[1:]:
            pk = process[0]
            structure = load_node(pk).inputs.structure
            formula = get_formula(structure)
            yield cls.WorkChainData(formula=formula, *process)

    def refresh_work_chains(self, _=None):
        with self._refresh_lock:
            try:
                self.set_trait("busy", True)  # disables the widget

                with self.hold_trait_notifications():
                    # We need to restore the original value, because it may be reset due to this issue:
                    # https://github.com/jupyter-widgets/ipywidgets/issues/2230
                    original_value = self.work_chains_selector.value

                    self.work_chains_selector.options = [
                        ("New calculation...", self._NO_PROCESS)
                    ] + [
                        (self.FMT_WORKCHAIN.format(wc=wc), wc.pk)
                        for wc in self.find_work_chains(self.workchain_label)
                    ]

                    self.work_chains_selector.value = original_value
            finally:
                self.set_trait("busy", False)  # reenable the widget


@register_viewer_widget("data.array.trajectory.TrajectoryData.")
class TrajectoryDataViewer(StructureDataViewer):

    # TODO: Do not subclass StructureDataViewer, but have it as a component
    trajectory = traitlets.Instance(Node, allow_none=True)
    _structures = []
    _energies = None

    def __init__(self, trajectory=None, **kwargs):
        # Trajectory navigator.
        self._step_selector = ipw.IntSlider(
            min=1,
            max=1,
            disabled=True,
            description="Frame:",
        )
        self._step_selector.observe(self.update_selection, names="value")

        # Display energy if available
        self._energy = ipw.HTML(
            value="Energy ",
            placeholder="Energy",
        )

        children = [ipw.HBox(children=[self._step_selector, self._energy])]

        super().__init__(
            children=children, configuration_tabs=["Selection", "Download"], **kwargs
        )

        self.trajectory = trajectory

    def update_selection(self, change):
        """Display selected structure"""
        index = change["new"] - 1
        self.structure = self._structures[index]
        # TODO: We should pass energy units as well somehow
        if self._energies is not None:
            self._energy.value = f"Energy = {self._energies[index]:.2f} eV"

    @traitlets.observe("trajectory")
    def _update_trajectory(self, change):
        trajectory = change["new"]
        if trajectory is None:
            self._step_selector.min = 1
            self._step_selector.max = 1
            self._step_selector.disabled = True
            self._step_selector.layout.visibility = "hidden"
            self._energy.layout.visibility = "hidden"
            return

        if isinstance(trajectory, TrajectoryData):
            self._structures = [
                trajectory.get_step_structure(i) for i in self.trajectory.get_stepids()
            ]
            if "energies" in trajectory.get_arraynames():
                self._energies = trajectory.get_array("energies")
                self._energy.layout.visibility = "visible"
                self._energy.value = f"Energy = {self._energies[0]:.2f} eV"
            else:
                self._energies = None
                self._energy.layout.visibility = "hidden"
        else:
            self._structures = [trajectory]

        nframes = len(self._structures)
        self._step_selector.max = nframes
        if nframes == 1:
            self.structure = self._structures[0]
            self._step_selector.layout.visibility = "hidden"
        else:
            self._step_selector.layout.visibility = "visible"
            self._step_selector.disabled = False
            # For some reason, this does not trigger observer
            # if this value was already there, so we update manually
            if self._step_selector.value == 1:
                self.structure = self._structures[0]
            else:
                self._step_selector.value = 1

    # Slightly modified from StructureDataViewer for performance
    @traitlets.observe("displayed_structure")
    def _update_structure_viewer(self, change):
        """Update the view if displayed_structure trait was modified."""
        with self.hold_trait_notifications():
            for (
                comp_id
            ) in self._viewer._ngl_component_ids:  # pylint: disable=protected-access
                self._viewer.remove_component(comp_id)
            self.selection = list()
            if change["new"] is not None:
                self._viewer.add_component(nglview.ASEStructure(change["new"]))
                # Interestingly, this doesn't work, I am getting (True, True, True)
                # Even when supposedly it should be set to False in SmilesWidget
                # if any(change["new"].pbc):
                #    self._viewer.add_unitcell() # pylint: disable=no-member

    # Monkey patched download button to download all conformers in a single file
    # TODO: Maybe we want to have a separate button for this?
    def _prepare_payload(self, file_format=None):
        """Prepare binary information."""
        from tempfile import NamedTemporaryFile

        file_format = file_format if file_format else self.file_format.value
        tmp = NamedTemporaryFile()

        for struct in self._structures:
            struct.get_ase().write(tmp.name, format=file_format, append=True)

        with open(tmp.name, "rb") as raw:
            return base64.b64encode(raw.read()).decode()


class ResourceSelectionWidget(ipw.VBox):
    """Widget for the selection of compute resources."""

    title = ipw.HTML(
        """<div style="padding-top: 0px; padding-bottom: 0px">
        <h4>Resources</h4>
    </div>"""
    )
    prompt = ipw.HTML(
        """<div style="line-height:120%; padding-top:0px">
        <p style="padding-bottom:10px">
        Specify the number of MPI tasks for this calculation.
        (Currently ignored).
        </p></div>"""
    )

    def __init__(self, **kwargs):
        extra = {
            "style": {"description_width": "150px"},
            # "layout": {"max_width": "200px"},
            "layout": {"min_width": "310px"},
        }

        self.num_mpi_tasks = ipw.BoundedIntText(
            value=1, step=1, min=1, max=16, description="# MPI tasks", **extra
        )

        super().__init__(
            children=[
                self.title,
                ipw.HBox(children=[self.prompt, self.num_mpi_tasks]),
            ]
        )

    def reset(self):
        self.num_mpi_tasks.value = 1


class QMSelectionWidget(ipw.VBox):
    """Widget for selecting ab initio level (basis set, method, etc.)"""

    qm_title = ipw.HTML(
        """<div style="padding-top: 0px; padding-bottom: 0px">
        <h4>QM method selection</h4>
        </div>"""
    )

    spectra_title = ipw.HTML(
        """<div style="padding-top: 0px; padding-bottom: 0px">
        <h4>Spectrum settings</h4>
        </div>"""
    )

    spectra_desc = ipw.HTML(
        """<div style="line-height:120%; padding-top:0px">
        <p style="padding-bottom:10px">
        Settings for modeling UV/VIS spectrum
        </p></div>"""
    )

    def __init__(self, **kwargs):
        style = {"description_width": "initial"}

        self.method = ipw.Text(
            value="pbe",
            description="DFT functional",
            placeholder="Type DFT functional",
            style=style,
        )

        self.basis = ipw.Text(
            value="def2-svp", description="Basis set", placeholder="Type Basis Set"
        )

        self.nwigner = ipw.BoundedIntText(
            value=1,
            step=1,
            min=0,
            max=1000,
            style=style,
            description="Number of Wigner samples",
        )

        super().__init__(
            children=[
                self.qm_title,
                ipw.HBox(children=[self.method, self.basis]),
                self.spectra_title,
                self.spectra_desc,
                self.nwigner,
            ]
        )

    def reset(self):
        self.method = "pbe"
        self.basis = "def2-svp"
        self.nwigner = 1
