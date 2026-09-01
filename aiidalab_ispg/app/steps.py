"""Common/Base Steps for AiiDAlab workflows.
   Code inspired by the QeApp.

Authors:
    * Daniel Hollas <daniel.hollas@bristol.ac.uk>
"""

import re

import ipywidgets as ipw
import traitlets

from aiida.engine import ProcessState
from aiida.engine.processes.control import ProcessTimeoutException, kill_processes
from aiida.orm import (
    Node,
    StructureData,
    TrajectoryData,
    load_node,
)
from aiidalab_widgets_base import (
    AiidaNodeViewWidget,
    ProcessMonitor,
    WizardAppWidgetStep,
)

from .qeapp import StructureSelectionStep as QeAppStructureSelectionStep
from .spectrum import ConformerTransitions, EnergyUnit, Spectrum, Transition
from .spectrum_widget import SpectrumWidget
from .utils import get_formula
from .widgets import HeaderWarning, ISPGProcessNodesTreeWidget, spinner


class StructureSelectionStep(QeAppStructureSelectionStep):
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
    process = traitlets.Instance(Node, allow_none=True)
    disabled = traitlets.Bool()

    def __init__(self, components=None, **kwargs):
        if components is None:
            components = []

        self.header_warning = HeaderWarning(dismissible=True)
        self.header_warning.layout.width = "550px"

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

        super().__init__([self.header_warning, *components, self.submit_button])

    def _on_submit_button_clicked(self, _):
        self.submit_button.disabled = True
        self.submit()

    def submit(self):
        """Submit workflow, implementation must be provided by the the child class"""
        msg = "FATAL: submit method not implemented"
        raise NotImplementedError(msg)

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

    def reset(self):
        with self.hold_trait_notifications():
            self.header_warning.hide()
            self.process = None
            self.input_structure = None


class ViewWorkChainStatusStep(ipw.VBox, WizardAppWidgetStep):
    """Widget for displaying the whole workflow as it runs"""

    process_uuid = traitlets.Unicode(allow_none=True)

    def __init__(self, progress_bar, children=None, **kwargs):
        if children is None:
            children = []
        self.process_tree = ISPGProcessNodesTreeWidget()
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
            on_sealed=[self._display_results, self._update_kill_button],
        )
        ipw.dlink((self, "process_uuid"), (self.process_monitor, "value"))

        self.kill_button = ipw.Button(
            description="Kill workflow",
            tooltip="Stop the running workflow",
            button_style="danger",
            icon="times-circle",
            disabled=True,
        )
        self.kill_button.on_click(self._on_click_kill_button)

        workflow_btns = ipw.HBox([self.tree_toggle, self.kill_button])
        self.tree_toggle.layout.width = "70%"

        super().__init__(
            children=[
                progress_bar,
                workflow_btns,
                self.process_tree,
                self.node_view,
                *children,
            ],
            **kwargs,
        )

    def reset(self):
        self.process_uuid = None
        self.tree_toggle.value = False

    # NOTE: This is somewhat subtle: The second argument with default None is needed
    # because this function is called from ProcessMonitor, which passes process_uuid to callbacks.
    # HOWEVER, we don't want to actually use it, and instead use self.process_uuid,
    # so that the state of the widget is consistent in case ProcessMonitor is not synced yet.
    # The same comment applies to _update_workflow_state() and _display_result() as well.
    def _update_step_state(self, _=None):
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

    def _update_workflow_state(self, _=None):
        """To be implemented by child workflows
        to power the workflow-specific progress bar
        """

    def _display_results(self, _=None):
        """Optional function to be called when the process is finished"""

    @traitlets.observe("process_uuid")
    def _observe_process(self, change):
        process_uuid = change["new"]
        if process_uuid is None:
            self.tree_toggle.disabled = True
        else:
            self.tree_toggle.disabled = False
        self._update_step_state()
        self._update_workflow_state()
        self._update_kill_button()

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

    def _update_kill_button(self):
        """Update the layout of the kill button.

        Show the button only when the process is running
        """
        if self.process_uuid is not None and self.state is self.State.ACTIVE:
            self.kill_button.layout.display = "block"
            self.kill_button.disabled = False
        else:
            self.kill_button.layout.display = "none"
            self.kill_button.disabled = True

    def _on_click_kill_button(self, _=None):
        """Callback for the kill button.

        First kill the process, then update the kill button layout.
        """
        self.kill_button.disabled = True

        workchain = [load_node(self.process_uuid)]
        try:
            # TODO: Do this in a thread!
            # Wait for 5 seconds
            kill_processes(workchain, timeout=5)  # ty: ignore[invalid-argument-type]
        except ProcessTimeoutException:
            # TODO: Print a warning message
            # Should we enable the button so that user can try again?
            pass

        # update the kill button layout
        self._update_kill_button()


def _orca_output_to_transitions(output_dict: dict, geom_index: int) -> list[Transition]:
    EVtoCM = Spectrum.get_energy_unit_factor(EnergyUnit.CM)
    en = output_dict["excitation_energies_cm"]
    osc = output_dict["oscillator_strengths"]
    return [
        {"energy": tr[0] / EVtoCM, "osc_strength": tr[1], "geom_index": geom_index}
        for tr in zip(en, osc)
    ]


def _wigner_output_to_transitions(wigner_outputs) -> list[Transition]:
    transitions = []
    for i, params in enumerate(wigner_outputs):
        transitions += _orca_output_to_transitions(params, i)
    return transitions


def _get_conformer_transitions(process) -> list[ConformerTransitions]:
    """Convert process.outputs.spectrum_data into a data structure that
    is passed to the SpectrumWidget and Spectrum classes"""

    # Number of conformers
    optimized = process.inputs.optimize
    n_input_geoms = len(process.inputs.structure.get_stepids())
    # Number of Wigner geometries per conformer
    wigner_sampled = optimized and process.inputs.nwigner.value > 0
    if wigner_sampled:
        nconf = n_input_geoms
        nsample = process.inputs.nwigner.value
    elif optimized:
        nconf = n_input_geoms
        nsample = 1
    else:
        # If the input geometries were not optimized, we treat them
        # as samples, not conformers!
        nconf = 1
        nsample = n_input_geoms

    # Unfortunately, we don't have number of states as attribute in process.inputs
    nstates = None
    if bp := process.base.extras.get("builder_parameters", None):
        nstates = bp["nstates"]

    # For the case of unoptimized geometries, flatten the list
    # so that the geometries are treated as a single conformer
    spectrum_data = process.outputs.spectrum_data.get_list()
    if not optimized:
        spectrum_data = [[conf[0] for conf in spectrum_data]]

    # Use Boltzmann weighting if we optimized the molecule and have Gibbs energies
    if nconf > 1 and optimized:
        conformer_weights = process.outputs.relaxed_structures.get_array(
            "boltzmann_weights"
        )
    else:
        equal_weight = 1.0 / nconf
        conformer_weights = [equal_weight for i in range(nconf)]

    conformer_transitions: list[ConformerTransitions] = [
        ConformerTransitions(
            transitions=_wigner_output_to_transitions(conformer),
            nsample=nsample,
            weight=conformer_weights[i],
        )
        for i, conformer in enumerate(spectrum_data)
    ]

    # Make sure our data is consistent
    assert nconf == len(conformer_transitions), (
        f"{nconf=} != {len(conformer_transitions)=}"
    )
    if nstates:
        for c in conformer_transitions:
            trans: list = c["transitions"]
            nsample = c["nsample"]
            assert nsample * nstates == len(trans), (
                f"{nstates * nsample=} != {len(trans)=}: {trans=}"
            )

    return conformer_transitions


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

    def _show_spectrum(self):
        if self.process_uuid is None:
            self.spectrum.debug_output.value = ""
            return

        process = load_node(self.process_uuid)
        if not process.is_finished:
            self.spectrum.debug_output.value = "Waiting for the workflow to finish..."
            return
        elif not process.is_finished_ok:
            self.spectrum.debug_output.value = "Workflow failed :-("
            return

        self.spectrum.debug_output.value = f"Loading...{spinner}"

        self.spectrum.conformer_transitions = _get_conformer_transitions(process)

        smiles = process.inputs.structure.base.extras.get("smiles", None)
        self.spectrum.smiles = smiles
        # Attach SMILES extra for the optimized structures as well.
        # NOTE: You can distinguish between new / optimized geometries
        # by looking at the 'creator' attribute of the Structure node.
        if smiles and "relaxed_structures" in process.outputs:
            process.outputs.relaxed_structures.base.extras.set("smiles", smiles)

        if process.inputs.optimize:
            self.spectrum.conformer_header.value = "<h4>Optimized conformers</h4>"
            self.spectrum.conformer_structures = process.outputs.relaxed_structures
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
            if process.inputs.optimize and (nwigner := process.inputs.nwigner.value):
                self.header.value += f", {nwigner} Wigner samples"

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
