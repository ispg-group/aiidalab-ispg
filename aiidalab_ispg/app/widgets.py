"""Widgets for the ISPG apps.

Authors:

    * Daniel Hollas <daniel.hollas@bristol.ac.uk>
"""

import base64
import io
import re
from pathlib import Path
from typing import Optional

import ase
import ipywidgets as ipw
import nglview
import numpy as np
import traitlets
from ase import Atoms

from aiida.cmdline.utils.ascii_vis import calc_info
from aiida.engine import ProcessState
from aiida.orm import (
    CalcFunctionNode,
    CifData,
    Data,
    Node,
    StructureData,
    TrajectoryData,
    load_node,
)
from aiida.plugins import DataFactory
from aiidalab_widgets_base import StructureManagerWidget, register_viewer_widget
from aiidalab_widgets_base.nodes import AiidaProcessNodeTreeNode, NodesTreeWidget
from aiidalab_widgets_base.process import ProcessNodesTreeWidget
from aiidalab_widgets_base.viewers import StructureDataViewer

from .qeapp.process import WorkChainSelector
from .utils import get_formula

__all__ = [
    "TrajectoryDataViewer",
]


class ISPGWorkChainSelector(WorkChainSelector):
    extra_fields = (
        ("formula", str),
        ("method", str),
        ("label", str),
        ("description", str),
    )

    def __init__(self, process_label: str, **kwargs):
        super().__init__(process_label=process_label, **kwargs)

    def parse_extra_info(self, pk: int) -> dict:
        """Parse extra information about the work chain.

        :param pk: the pk of the workchain to parse
        :return: the parsed extra information"""
        process = load_node(pk)
        structure = process.inputs.structure
        formula = get_formula(structure)

        method = ""
        if bp := process.base.extras.get("builder_parameters", None):
            method = f"{bp['method']}/{bp['basis']}"

        description = structure.description
        if len(description) > 35:
            description = f"{description[0:34]}…"

        # By default, the label is the formula of the generated molecule.
        # In that case we do not want to show it twice so let's hide it.
        label = structure.label if structure.label != formula else ""

        return {
            "formula": formula,
            "method": method,
            "label": label,
            "description": description,
        }


class ISPGNodesTreeWidget(NodesTreeWidget):
    """A tree widget for the structured representation of a nodes graph.

    ISPG modifications:
     - do not display calcfunctions to make Tree view less confusing for non-expert user
     - do not include current function in the Process name.
    """

    @staticmethod
    def include_node(node):
        try:
            if node.process_label == "ConcatInputsToList":
                return False
            if node.process_label == "generate_wigner_structures":
                return True
        except AttributeError:
            pass

        # To make the Workflow tree less confusing, we do not display calcfunctions.
        # The only exception to this is the calcfunction for generating Wigner sampling,
        # which is handled above.
        if isinstance(node, CalcFunctionNode):
            return False
        return True

    @staticmethod
    def extract_node_name(node):
        try:
            name = calc_info(node)
        except AttributeError:
            return str(node)

        m = re.match(r".*\[[0-9*]\]", name)
        # Filter out junk at the end of the string
        if m is not None:
            name = m[0]
        return name

    @classmethod
    def _find_called(cls, root):
        process_node = load_node(root.pk)
        called = process_node.called
        called.sort(key=lambda p: p.ctime)
        for node in called:
            if not cls.include_node(node):
                continue
            if node.pk not in root.nodes_registry:
                name = cls.extract_node_name(node)
                root.nodes_registry[node.pk] = cls._to_tree_node(node, name=name)
            yield root.nodes_registry[node.pk]

    def _update_tree_node(self, tree_node):
        if isinstance(tree_node, AiidaProcessNodeTreeNode):
            process_node = load_node(tree_node.pk)
            tree_node.name = self.extract_node_name(process_node)
            # Override the process state in case that the process node has failed:
            # (This could be refactored with structural pattern matching with py>=3.10.)
            process_state = (
                ProcessState.EXCEPTED
                if process_node.is_failed
                else process_node.process_state
            )
            tree_node.icon_style = self.PROCESS_STATE_STYLE.get(
                process_state, self.PROCESS_STATE_STYLE_DEFAULT
            )


class ISPGProcessNodesTreeWidget(ProcessNodesTreeWidget):
    """A tree widget for the structured representation of a process graph."""

    def __init__(self, title="Process Tree", **kwargs):
        self.title = title  # needed for ProcessFollowerWidget

        self._tree = ISPGNodesTreeWidget()
        self._tree.observe(self._observe_tree_selected_nodes, ["selected_nodes"])
        super(ipw.VBox, self).__init__(children=[self._tree], **kwargs)
        self.update()


@register_viewer_widget("data.core.array.trajectory.TrajectoryData.")
class TrajectoryDataViewer(StructureDataViewer):
    # TODO: Do not subclass StructureDataViewer, but have it as a component
    trajectory = traitlets.Instance(Node, allow_none=True)
    selected_structure_id = traitlets.Int(allow_none=True)

    # TODO: Should probably be tuple instead
    _structures: list[StructureData] = []  # noqa: RUF012
    _energies: Optional[np.ndarray] = None
    _boltzmann_weights: Optional[np.ndarray] = None

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
            for comp_id in self._viewer._ngl_component_ids:  # pylint: disable=protected-access
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

        with Path(tmp.name).open("rb") as raw:
            return base64.b64encode(raw.read()).decode()


# NOTE: TrajectoryManagerWidget will hopefully note be necessary once
# the trajectory viewer is merged to AWB
class TrajectoryManagerWidget(StructureManagerWidget):
    SUPPORTED_DATA_FORMATS = {  # noqa: RUF012
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
            msg = f"Unknown data format '{node_class}'. Options: {list(self.SUPPORTED_DATA_FORMATS.keys())}"
            raise ValueError(msg)

        self.output = ipw.HTML("")

        children = [
            self._structure_importers(importers),
            self.viewer,
            ipw.HBox(
                [
                    *store_and_description,
                    self.structure_label,
                    self.structure_description,
                ]
            ),
        ]

        super(ipw.VBox, self).__init__(children=[*children, self.output], **kwargs)

    def _convert_to_structure_node(self, structure):
        """Convert structure of any type to the StructureNode object."""
        if structure is None:
            return None
        structure_node_type = DataFactory(self.SUPPORTED_DATA_FORMATS[self.node_class])  # pylint: disable=invalid-name

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
                    msg = f"Unexpected node type {type(structure)}"
                    raise ValueError(msg)

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


# A common spinning icon
_default_spinner_style = "color:blue;"
spinner = f"""<i class="fa fa-spinner fa-pulse" style={_default_spinner_style}></i>"""


class Spinner(ipw.HTML):
    """Widget that shows a simple spinner if enabled."""

    enabled = traitlets.Bool()

    def __init__(self, spinner_style=_default_spinner_style):
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


class HeaderWarning(ipw.HTML):
    """Class to display a warning in the header."""

    def __init__(self, dismissible=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dismissible = dismissible
        self.layout = ipw.Layout(
            display="none",
            width="auto",
            height="auto",
            margin="0px 0px 0px 0px",
            padding="0px 0px 0px 0px",
        )

    def show(self, message):
        """Show the warning."""
        dismiss = ""
        alert_classes = "alert alert-danger"
        if self.dismissible:
            alert_classes = f"{alert_classes} alert-dismissible"
            dismiss = """<a href="#" class="close" data-dismiss="alert" aria-label="close">&times;</a>"""
        self.value = (
            f"""<div class="{alert_classes}" role="alert">{dismiss}{message}</div>"""
        )
        self.layout.display = "block"

    def hide(self):
        """Hide the warning."""
        self.layout.display = "none"
