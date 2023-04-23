"""Common Steps for AiiDAlab workflows.
   Code inspired by the QeApp.

Authors:
    * Daniel Hollas <daniel.hollas@durham.ac.uk>
"""
import ipywidgets as ipw
import numpy as np
import re
import traitlets

from aiida.common import MissingEntryPointError, LinkType
from aiida.engine import ProcessState, submit
from aiida.orm import load_node
from aiida.orm import WorkChainNode, StructureData, TrajectoryData
from aiida.plugins import WorkflowFactory

from aiidalab_widgets_base import (
    AiidaNodeViewWidget,
    ProcessMonitor,
    ProcessNodesTreeWidget,
    WizardAppWidgetStep,
)

import aiidalab_ispg.qeapp as qeapp

from .parameters import DEFAULT_PARAMETERS
from .widgets import spinner
from .spectrum import EnergyUnit, Spectrum, SpectrumWidget
from .utils import get_formula, calc_boltzmann_weights, AUtoKJ

# TODO: This one will be gone soon
from aiidalab_atmospec_workchain import OrcaWignerSpectrumWorkChain

OrcaBaseWorkChain = WorkflowFactory("orca.base")


class StructureSelectionStep(qeapp.StructureSelectionStep):
    """Integrated widget for the selection of structures from different sources."""

    structure = traitlets.Union(
        [traitlets.Instance(StructureData), traitlets.Instance(TrajectoryData)],
        allow_none=True,
    )
    confirmed_structure = traitlets.Union(
        [traitlets.Instance(StructureData), traitlets.Instance(TrajectoryData)],
        allow_none=True,
    )

    @traitlets.observe("structure")
    def _observe_structure(self, change):
        structure = change["new"]
        with self.hold_trait_notifications():
            if structure is None:
                self.structure_name_text.value = ""
            else:
                self.structure_name_text.value = get_formula(self.structure)
            self._update_state()


class SubmitWorkChainStepBase(ipw.VBox, WizardAppWidgetStep):
    """Base class for workflow submission step. Must be subclassed."""

    input_structure = traitlets.Union(
        [traitlets.Instance(StructureData), traitlets.Instance(TrajectoryData)],
        allow_none=True,
    )
    process = traitlets.Instance(WorkChainNode, allow_none=True)
    disabled = traitlets.Bool()

    def __init__(self, components=None, **kwargs):
        self.submit_button = ipw.Button(
            description="Submit",
            tooltip="Submit the calculation with the selected parameters.",
            icon="play",
            button_style="success",
            layout=ipw.Layout(width="auto", flex="0 0 auto"),
            disabled=True,
        )
        self.submit_button.layout.margin = "10px 0px 10px 0px"

        self.submit_button.on_click(self._on_submit_button_clicked)

        children = [self.submit_button]
        if components is not None:
            children = components + children

        super().__init__(children=children)

    def _on_submit_button_clicked(self, _):
        self.submit_button.disabled = True
        self.submit()

    def submit(self):
        """Submit workflow, implementation must be provided by the the child class"""
        raise NotImplementedError("FATAL: submit method not implemented")

    def _get_state(self):
        # Process is already running.
        if self.process is not None:
            return self.State.SUCCESS
        # Input structure not specified.
        if self.input_structure is None:
            return self.State.INIT
        # Structure ready, but input parameters are invalid
        if not self._validate_input_parameters():
            return self.State.READY
        return self.State.CONFIGURED

    def _update_state(self, _=None):
        self.state = self._get_state()

    def _validate_input_parameters(self) -> bool:
        """Must be provided by the child class"""
        raise NotImplementedError

    @traitlets.observe("input_structure")
    def _observe_input_structure(self, change):
        self._update_state()

    @traitlets.observe("state")
    def _observe_state(self, change):
        with self.hold_trait_notifications():
            self.disabled = change["new"] not in (
                self.State.READY,
                self.State.CONFIGURED,
            )
            self.submit_button.disabled = change["new"] != self.State.CONFIGURED

    @traitlets.observe("process")
    def _observe_process(self, change):
        self._update_state()

    def can_reset(self):
        "Do not allow reset while process is running."
        return self.state is not self.State.ACTIVE

    def reset(self):
        with self.hold_trait_notifications():
            self.process = None
            self.input_structure = None


class ViewWorkChainStatusStep(ipw.VBox, WizardAppWidgetStep):
    """Widget for displaying the whole workflow as it runs"""

    process_uuid = traitlets.Unicode(allow_none=True)

    def __init__(self, progress_bar=None, children=None, **kwargs):
        self.process_tree = ProcessNodesTreeWidget()
        self.tree_toggle = ipw.ToggleButton(
            value=False,
            description="Show workflow details",
            disabled=True,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Display workflow tree with detailed results",
            icon="folder",
            layout=ipw.Layout(width="auto", height="auto"),
        )
        self.tree_toggle.observe(self._observe_tree_toggle, names="value")

        self.node_view = AiidaNodeViewWidget(layout={"width": "auto", "height": "auto"})
        ipw.dlink(
            (self.process_tree, "selected_nodes"),
            (self.node_view, "node"),
            transform=lambda nodes: nodes[0] if nodes else None,
        )

        self.process_monitor = ProcessMonitor(
            timeout=1.0,
            callbacks=[
                self.process_tree.update,
                self._update_step_state,
                self._update_workflow_state,
            ],
            on_sealed=[self._display_results],
        )
        ipw.dlink((self, "process_uuid"), (self.process_monitor, "value"))

        if progress_bar is not None:
            workflow_state = ipw.VBox([progress_bar, self.tree_toggle])
        else:
            workflow_state = ipw.VBox([self.tree_toggle])
        workflow_state.layout.width = "52%"
        components = [workflow_state, self.process_tree, self.node_view]
        if children is not None:
            components = components + children

        super().__init__(children=components, **kwargs)

    def can_reset(self):
        "Do not allow reset while process is running."
        return self.state is not self.State.ACTIVE

    def reset(self):
        self.process_uuid = None
        self.tree_toggle.value = False

    def _update_step_state(self, process_uuid):
        if process_uuid is None:
            self.state = self.State.INIT
            return

        process = load_node(process_uuid)
        process_state = process.process_state
        if process_state in (
            ProcessState.CREATED,
            ProcessState.RUNNING,
            ProcessState.WAITING,
        ):
            self.state = self.State.ACTIVE
        elif (
            process_state in (ProcessState.EXCEPTED, ProcessState.KILLED)
            or process.is_failed
        ):
            self.state = self.State.FAIL
        elif process_state is ProcessState.FINISHED and process.is_finished_ok:
            self.state = self.State.SUCCESS

    def _update_workflow_state(self, process_uuid):
        """To be implemented by child workflows
        to power the workflow-specific progress bar
        """
        pass

    def _display_results(self, process_uuid):
        """Optional function to be called when the process is finished"""
        pass

    @traitlets.observe("process_uuid")
    def _observe_process(self, change):
        process_uuid = change["new"]
        if process_uuid is None:
            self.tree_toggle.disabled = True
        else:
            self.tree_toggle.disabled = False
        self._update_step_state(process_uuid)
        self._update_workflow_state(process_uuid)

    def _observe_tree_toggle(self, change):
        if change["new"] == change["old"]:
            return
        show_tree = change["new"]
        if show_tree:
            # TODO: Spawn a new thread for this so we do not block
            # UI interaction.
            self.tree_toggle.icon = "spinner"
            self.process_tree.value = self.process_uuid
            self.tree_toggle.icon = "folder-open"
        else:
            # TODO: Should we assign None or not?
            # For large workflows, this might not be best
            self.process_tree.value = None
            self.tree_toggle.icon = "folder"


class ViewSpectrumStep(ipw.VBox, WizardAppWidgetStep):
    """Step for displaying UV/vis spectrum"""

    process_uuid = traitlets.Unicode(allow_none=True)

    def __init__(self, **kwargs):
        self.header = ipw.HTML()
        self.spectrum = SpectrumWidget()

        # NOTE: We purposefully do NOT link the process_uuid trait
        # to ProcessMonitor. We do that manually only for running processes.
        self.process_monitor = ProcessMonitor(
            timeout=1.0,
            callbacks=[self._update_state],
            on_sealed=(self._show_spectrum,),
        )
        super().__init__([self.header, self.spectrum], **kwargs)

    def reset(self):
        self.process_uuid = None
        self.spectrum.reset()

    def _orca_output_to_transitions(self, output_dict, geom_index):
        EVtoCM = Spectrum.get_energy_unit_factor(EnergyUnit.CM)
        en = output_dict["etenergies"]
        osc = output_dict["etoscs"]
        return [
            {"energy": tr[0] / EVtoCM, "osc_strength": tr[1], "geom_index": geom_index}
            for tr in zip(en, osc)
        ]

    def _wigner_output_to_transitions(self, wigner_outputs):
        transitions = []
        for i, params in enumerate(wigner_outputs):
            transitions += self._orca_output_to_transitions(params, i)
        return transitions

    def _show_spectrum(self):
        if self.process_uuid is None:
            return

        process = load_node(self.process_uuid)
        if not process.is_finished_ok:
            return

        self.spectrum.debug_output.value = f"Loading...{spinner}"

        # Number of Wigner geometries per conformer
        nsample = process.inputs.nwigner.value if process.inputs.nwigner > 0 else 1

        nconf = len(process.inputs.structure.get_stepids())
        free_energies = []
        boltzmann_weights = [1.0 for i in range(nconf)]
        if process.inputs.optimize:
            conformer_workchains = [
                link.node
                for link in process.base.links.get_outgoing(
                    link_type=LinkType.CALL_WORK, node_class=OrcaWignerSpectrumWorkChain
                )
            ]
            assert nconf == len(conformer_workchains)
            for node in conformer_workchains:
                for link in node.base.links.get_outgoing(
                    link_type=LinkType.CALL_WORK, node_class=OrcaBaseWorkChain
                ):
                    wc = link.node
                    if wc.label == "optimization":
                        temperature = wc.outputs.output_parameters["temperature"]
                        free_energy = wc.outputs.output_parameters["freeenergy"]
                        free_energies.append(free_energy)

            en0 = min(free_energies)
            free_energies = [(en - en0) * AUtoKJ for en in free_energies]
            boltzmann_weights = calc_boltzmann_weights(free_energies, T=temperature)

        # TODO: How to ensure the correct order of process.outputs.spectrum_data.get_list()?
        # with respect to boltzmann_weights that we computed above?
        conformer_transitions = [
            {
                "transitions": self._wigner_output_to_transitions(conformer),
                "nsample": nsample,
                "weight": boltzmann_weights[i],
            }
            for i, conformer in enumerate(process.outputs.spectrum_data.get_list())
        ]

        self.spectrum.conformer_transitions = conformer_transitions

        if "smiles" in process.inputs.structure.extras:
            self.spectrum.smiles = process.inputs.structure.extras["smiles"]
            # We're attaching smiles extra for the optimized structures as well
            # NOTE: You can distinguish between new / optimized geometries
            # by looking at the 'creator' attribute of the Structure node.
            if "relaxed_structures" in process.outputs:
                process.outputs.relaxed_structures.base.extras.set(
                    "smiles", self.spectrum.smiles
                )
        else:
            self.spectrum.smiles = None

        if "relaxed_structures" in process.outputs:
            assert nconf == len(process.outputs.relaxed_structures.get_stepids())
            self.spectrum.conformer_header.value = "<h4>Optimized conformers</h4>"
            conformers = process.outputs.relaxed_structures.clone()
            if nconf > 1:
                conformers.set_array("energies", np.array(free_energies))
                conformers.set_array("boltzmann_weights", np.array(boltzmann_weights))
                conformers.base.extras.set("energy_units", "kJ/mole")
                conformers.base.extras.set("temperature", temperature)
            self.spectrum.conformer_structures = conformers
        else:
            # If we did not optimize the structure, just show the input structure(s)
            self.spectrum.conformer_header.value = "<h4>Input structures</h4>"
            structures = process.inputs.structure.clone()
            # Overwrite the energy and boltzmann weights because they may come
            # from conformer sampling, i.e. xTB or MM. We do not use these
            # for spectrum weighting so displaying them would be misleading.
            if "energies" in structures.get_arraynames():
                structures.delete_array("energies")
            if "boltzmann_weights" in structures.get_arraynames():
                structures.delete_array("boltzmann_weights")
            self.spectrum.conformer_structures = structures

        # Spectrum loaded! Clear the "Loading..." text.
        self.spectrum.debug_output.value = ""

    def _update_header(self):
        if self.process_uuid is None:
            self.header.value = ""
            return
        process = load_node(self.process_uuid)
        if bp := process.base.extras.get("builder_parameters", None):
            formula = re.sub(
                r"([0-9]+)",
                r"<sub>\1</sub>",
                get_formula(process.inputs.structure),
            )
            solvent = "the gas phase"
            if bp.get("solvent") is not None:
                solvent = (
                    bp["solvent"] if bp.get("solvent") != "None" else "the gas phase"
                )
            # TODO: Compatibility hack
            es_method = bp.get("excited_method", "TDA/TDDFT")
            tddft_functional = bp.get("tddft_functional", bp.get("method", ""))
            method_string = f"{es_method}"
            if "TDDFT" in es_method:
                method_string = f"{es_method}/{tddft_functional}"
            es_basis = bp.get("es_basis", bp.get("basis", ""))
            self.header.value = (
                f"<h4>UV/vis spectrum of {formula} "
                f"at {method_string}/{es_basis} level "
                f"in {solvent}</h4>"
                f"{bp['nstates']} singlet states"
            )
            if process.inputs.optimize and process.inputs.nwigner > 0:
                self.header.value += f", {process.inputs.nwigner.value} Wigner samples"

    def _update_state(self):
        if self.process_uuid is None:
            self.state = self.State.INIT
            return

        process = load_node(self.process_uuid)
        process_state = process.process_state
        if process_state in (
            ProcessState.CREATED,
            ProcessState.RUNNING,
            ProcessState.WAITING,
        ):
            self.state = self.State.ACTIVE
        elif (
            process_state in (ProcessState.EXCEPTED, ProcessState.KILLED)
            or process.is_failed
        ):
            self.state = self.State.FAIL
        elif process_state is ProcessState.FINISHED and process.is_finished_ok:
            self.state = self.State.SUCCESS

    @traitlets.observe("process_uuid")
    def _observe_process(self, change):
        if change["new"] == change["old"]:
            return

        self.spectrum.reset()
        self._update_header()

        # Setup process monitor only for running processes,
        # This aids debugging when developing the SpectrumWidget,
        # because ProcessMonitorWidget swallows all exceptions coming from _show_spectrum().
        if self.process_uuid is None or not load_node(self.process_uuid).is_sealed:
            self.process_monitor.value = self.process_uuid
        else:
            self._show_spectrum()
        self._update_state()
