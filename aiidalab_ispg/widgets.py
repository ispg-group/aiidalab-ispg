"""Widgets for the ISPG apps.

Authors:

    * Daniel Hollas <daniel.hollas@bristol.ac.uk>
"""

import base64
from enum import Enum, unique
import io

import ipywidgets as ipw
import traitlets
import nglview
from dataclasses import dataclass

import ase
from ase import Atoms

from aiida.tools.query.calculation import CalculationQueryBuilder
from aiida.orm import load_node, Node, Data
from aiida.plugins import DataFactory

from aiidalab_widgets_base import register_viewer_widget
from aiidalab_widgets_base import StructureManagerWidget
from aiidalab_widgets_base.viewers import StructureDataViewer

import aiidalab_ispg.qeapp as qeapp
from .utils import get_formula

StructureData = DataFactory("core.structure")
CifData = DataFactory("core.cif")
TrajectoryData = DataFactory("core.array.trajectory")

__all__ = [
    "TrajectoryDataViewer",
]


@unique
class ExcitedStateMethod(Enum):
    TDA = "TDA/TDDFT"
    TDDFT = "TDDFT"
    CCSD = "EOM-CCSD"
    ADC2 = "ADC2"


# Taken from ORCA-5.0 manual, section 9.41
PCM_SOLVENT_LIST = (
    "None",
    "Water",
    "Acetone",
    "Acetonitrile",
    "Ammonia",
    "Benzene",
    "CCl4",
    "CH2Cl2",
    "Chloroform",
    "Cyclohexane",
    "DMF",
    "DMSO",
    "Ethanol",
    "Hexane",
    "Methanol",
    "Octanol",
    "Pyridine",
    "THF",
    "Toluene",
)


class WorkChainSelector(qeapp.WorkChainSelector):

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
        # TODO: Don't lock if dropdown open
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


@register_viewer_widget("data.core.array.trajectory.TrajectoryData.")
class TrajectoryDataViewer(StructureDataViewer):

    # TODO: Do not subclass StructureDataViewer, but have it as a component
    trajectory = traitlets.Instance(Node, allow_none=True)
    selected_structure_id = traitlets.Int(allow_none=True)

    _structures = []
    _energies = None
    _boltzmann_weights = None

    def __init__(self, trajectory=None, configuration_tabs=None, **kwargs):

        if configuration_tabs is None:
            configuration_tabs = ["Selection", "Download"]

        # Trajectory navigator.
        self._step_selector = ipw.IntSlider(
            min=1,
            max=1,
            disabled=True,
            description="Frame:",
        )
        self._step_selector.observe(self.update_selection, names="value")

        # Display energy and Boltzmann weights if available
        # TODO: Generalize this
        self._energy_label = ipw.HTML(value="", style={"description_width": "initial"})
        self._boltzmann_weight_label = ipw.HTML(
            value="", style={"description_width": "initial"}
        )
        labels = ipw.VBox(children=[self._energy_label, self._boltzmann_weight_label])

        # NOTE: Having children step selector and labels horizontally
        # does not work well in the SpectrumWidget context.
        # children = [ipw.HBox(children=[self._step_selector, labels])]
        children = [self._step_selector, labels]
        super().__init__(
            children=children, configuration_tabs=configuration_tabs, **kwargs
        )

        self.trajectory = trajectory

    def update_selection(self, change):
        """Display selected structure"""
        index = change["new"] - 1
        self.structure = self._structures[index]
        self.selected_structure_id = index
        if self._energies is not None:
            self._energy_label.value = f"{self._energies[index]:.3f}"
        if self._boltzmann_weights is not None:
            percentage = 100 * self._boltzmann_weights[index]
            self._boltzmann_weight_label.value = f"{percentage:.1f}%"

    def _reset(self):
        self.structure = None
        self.set_trait("displayed_structure", None)
        self._reset_step_selector()
        self._hide_labels()

    def _reset_step_selector(self):
        self._step_selector.layout.visibility = "hidden"
        self._step_selector.max = 1
        self._step_selector.disabled = True

    def _hide_labels(self):
        self._energies = None
        self._boltzmann_weights = None
        self._energy_label.layout.visibility = "hidden"
        self._boltzmann_weight_label.layout.visibility = "hidden"

    @traitlets.observe("trajectory")
    def _update_trajectory(self, change):
        trajectory = change["new"]
        if trajectory is None:
            self._reset()
            return

        if isinstance(trajectory, TrajectoryData):
            self._hide_labels()
            self._reset_step_selector()
            self._structures = [
                trajectory.get_step_structure(i) for i in self.trajectory.get_stepids()
            ]

            if "energies" in trajectory.get_arraynames():
                self._energies = trajectory.get_array("energies")
                energy_units = trajectory.base.extras.get("energy_units", "")
                self._energy_label.description = f"Energy ({energy_units}) ="
                self._energy_label.value = f"{self._energies[0]:.3f}"
                self._energy_label.layout.visibility = "visible"

            if "boltzmann_weights" in trajectory.get_arraynames():
                self._boltzmann_weights = trajectory.get_array("boltzmann_weights")
                temperature = trajectory.base.extras.get("temperature", "")
                percentage = 100 * self._boltzmann_weights[0]
                self._boltzmann_weight_label.description = (
                    f"Boltzmann pop. ({int(temperature)}K) ="
                )
                self._boltzmann_weight_label.value = f"{percentage:.1f}%"
                self._boltzmann_weight_label.layout.visibility = "visible"
        else:
            self._structures = [trajectory]

        nframes = len(self._structures)
        self._step_selector.max = nframes
        if nframes == 1:
            self.structure = self._structures[0]
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
            self.selection = []
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
        Number of MPI tasks for this calculation.
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


class QMSelectionWidget(ipw.HBox):
    """Widget for selecting ab initio level (basis set, method, etc.)"""

    # TODO: This should probably live elsewhere as a config
    _DEFAULT_FUNCTIONAL = "PBE"

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

        self.excited_method = ipw.Dropdown(
            options=[(method.value, method) for method in ExcitedStateMethod],
            value=ExcitedStateMethod.TDA,
            description="Excited state",
            style=style,
        )
        self.excited_method.observe(self._update_methods, names="value")

        # TODO: Have separate DFT functional selection for excited state?
        self.method = ipw.Text(
            value=self._DEFAULT_FUNCTIONAL,
            description="Ground state",
            style=style,
        )

        self.basis = ipw.Text(value="def2-SVP", description="Basis set")

        self.solvent = ipw.Dropdown(
            options=PCM_SOLVENT_LIST,
            value="None",
            description="PCM solvent",
            disabled=False,
            style=style,
        )

        # TODO: Move Wigner settings to a separate widget
        self.nwigner = ipw.BoundedIntText(
            value=1,
            step=1,
            min=0,
            max=1000,
            style=style,
            description="Number of Wigner samples",
        )

        self.wigner_low_freq_thr = ipw.BoundedFloatText(
            value=100,
            step=10,
            min=0,
            max=10000,
            style=style,
            description="Low-frequency threshold",
            title="Normal modes below this frequency will be ignored",
        )

        super().__init__(
            children=[
                ipw.VBox(
                    children=[
                        self.qm_title,
                        self.excited_method,
                        self.method,
                        self.basis,
                        self.solvent,
                    ]
                ),
                ipw.VBox(
                    children=[
                        self.spectra_title,
                        self.spectra_desc,
                        self.nwigner,
                        self.wigner_low_freq_thr,
                    ]
                ),
            ]
        )

    def _update_methods(self, change):
        """Update ground state method defaults when
        excited state method is changed"""
        es_method = change["new"]
        if es_method == change["old"]:
            return
        if es_method in (ExcitedStateMethod.ADC2, ExcitedStateMethod.CCSD):
            self.method.value = "MP2"
            self.solvent.value = "None"
            self.solvent.disabled = True
        elif change["old"] in (ExcitedStateMethod.ADC2, ExcitedStateMethod.CCSD):
            # Switching to (TDA)TDDFT from ADC2/EOM-CCSD
            # We do not want to reset the functional if switching between TDA and full TDDFT
            self.method.value = self._DEFAULT_FUNCTIONAL
            self.solvent.disabled = False

    # NOTE: It seems this method is currently not called
    def reset(self):
        self.excited_method.value = ExcitedStateMethod.TDA
        self.method.value = self._DEFAULT_FUNCTIONAL
        self.solvent.value = "None"
        self.basis.value = "def2-SVP"
        self.nwigner.value = 1
        self.wigner_low_freq_thr.value = 100


# NOTE: TrajectoryManagerWidget will hopefully note be necessary once
# the trajectory viewer is merged to AWB
class TrajectoryManagerWidget(StructureManagerWidget):
    SUPPORTED_DATA_FORMATS = {
        "CifData": "core.cif",
        "StructureData": "core.structure",
        "TrajectoryData": "core.array.trajectory",
    }

    def __init__(
        self,
        importers,
        viewer=None,
        editors=None,
        storable=True,
        node_class=None,
        **kwargs,
    ):

        # History of modifications
        self.history = []

        # Undo functionality.
        btn_undo = ipw.Button(description="Undo", button_style="success")
        btn_undo.on_click(self.undo)
        self.structure_set_by_undo = False

        # To keep track of last inserted structure object
        self._inserted_structure = None

        # Structure viewer.
        if viewer:
            self.viewer = viewer
        else:
            self.viewer = StructureDataViewer(**kwargs)

        if node_class == "TrajectoryData":
            traitlets.dlink((self, "structure_node"), (self.viewer, "trajectory"))
        else:
            traitlets.dlink((self, "structure_node"), (self.viewer, "structure"))

        # Store button.
        self.btn_store = ipw.Button(description="Store in AiiDA", disabled=True)
        self.btn_store.on_click(self.store_structure)

        # Label and description that are stored along with the new structure.
        self.structure_label = ipw.Text(description="Label")
        self.structure_description = ipw.Text(description="Description")

        # Store format selector.
        data_format = ipw.RadioButtons(
            options=self.SUPPORTED_DATA_FORMATS, description="Data type:"
        )
        traitlets.link((data_format, "label"), (self, "node_class"))

        # Store button, store class selector, description.
        store_and_description = [self.btn_store] if storable else []

        if node_class is None:
            store_and_description.append(data_format)
        elif node_class in self.SUPPORTED_DATA_FORMATS:
            self.node_class = node_class
        else:
            raise ValueError(
                "Unknown data format '{}'. Options: {}".format(
                    node_class, list(self.SUPPORTED_DATA_FORMATS.keys())
                )
            )

        self.output = ipw.HTML("")

        children = [
            self._structure_importers(importers),
            self.viewer,
            ipw.HBox(
                store_and_description
                + [self.structure_label, self.structure_description]
            ),
        ]

        super(ipw.VBox, self).__init__(children=children + [self.output], **kwargs)

    def _convert_to_structure_node(self, structure):
        """Convert structure of any type to the StructureNode object."""
        if structure is None:
            return None
        structure_node_type = DataFactory(
            self.SUPPORTED_DATA_FORMATS[self.node_class]
        )  # pylint: disable=invalid-name

        # If the input_structure trait is set to Atoms object, structure node must be created from it.
        if isinstance(structure, Atoms):
            if structure_node_type == TrajectoryData:
                structure_node = structure_node_type(
                    structurelist=(StructureData(ase=structure),)
                )
            else:
                structure_node = structure_node_type(ase=structure)

            # If the Atoms object was created by SmilesWidget,
            # attach its SMILES code as an extra.
            if "smiles" in structure.info:
                structure_node.base.extras.set("smiles", structure.info["smiles"])
            return structure_node

        # If the input_structure trait is set to AiiDA node, check what type
        elif isinstance(structure, Data):
            # Transform the structure to the structure_node_type if needed.
            if isinstance(structure, structure_node_type):
                return structure
            # TrajectoryData cannot be created from Atoms object
            if structure_node_type == TrajectoryData:
                if isinstance(structure, StructureData):
                    return structure_node_type(structurelist=(structure,))
                elif isinstance(structure, CifData):
                    return structure_node_type(
                        structurelist=(StructureData(ase=structure.get_ase()),)
                    )
                else:
                    raise ValueError(f"Unexpected node type {type(structure)}")

        # Using self.structure, as it was already converted to the ASE Atoms object.
        return structure_node_type(ase=self.structure)

    @traitlets.observe("structure_node")
    def _observe_structure_node(self, change):
        """Modify structure label and description when a new structure is provided."""
        struct = change["new"]
        if struct is None:
            self.btn_store.disabled = True
            self.structure_label.value = ""
            self.structure_label.disabled = True
            self.structure_description.value = ""
            self.structure_description.disabled = True
            return
        if struct.is_stored:
            self.btn_store.disabled = True
            self.structure_label.value = struct.label
            self.structure_label.disabled = True
            self.structure_description.value = struct.description
            self.structure_description.disabled = True
        else:
            self.btn_store.disabled = False
            self.structure_label.value = get_formula(struct)
            self.structure_label.disabled = False
            self.structure_description.value = ""
            self.structure_description.disabled = False

    @traitlets.observe("input_structure")
    def _observe_input_structure(self, change):
        """Returns ASE atoms object and sets structure_node trait."""
        # If the `input_structure` trait is set to Atoms object, then the `structure` trait should be set to it as well.
        self.history = []

        if isinstance(change["new"], Atoms):
            self.structure = change["new"]

        # If the `input_structure` trait is set to AiiDA node, then the `structure` trait should
        # be converted to an ASE Atoms object.
        elif isinstance(
            change["new"], CifData
        ):  # Special treatement of the CifData object
            str_io = io.StringIO(change["new"].get_content())
            self.structure = ase.io.read(
                str_io, format="cif", reader="ase", store_tags=True
            )
        elif isinstance(change["new"], StructureData):
            self.structure = change["new"].get_ase()

        elif isinstance(change["new"], TrajectoryData):
            # self.structure is essentially used for editing purposes.
            # We're currently not allowing editing TrajectoryData,
            # so we don't even attempt to set self.structure,
            # instead we update the structure_node directly here
            self.set_trait("structure_node", change["new"])

        else:
            self.structure = None


class Spinner(ipw.HTML):
    """Widget that shows a simple spinner if enabled."""

    enabled = traitlets.Bool()

    def __init__(self, spinner_style=None):
        self.spinner_style = f' style="{spinner_style}"' if spinner_style else ""
        super().__init__()

    @traitlets.default("enabled")
    def _default_enabled(self):  # pylint: disable=no-self-use
        return False

    @traitlets.observe("enabled")
    def _observe_enabled(self, change):
        """Show spinner if enabled, otherwise nothing."""
        if change["new"]:
            self.value = (
                f"""<i class="fa fa-spinner fa-pulse"{self.spinner_style}></i>"""
            )
        else:
            self.value = ""
