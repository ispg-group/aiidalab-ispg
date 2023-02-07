"""Steps specific for the optimization workflow"""

import ipywidgets as ipw
import traitlets

from aiida.engine import submit
from aiida.orm import Bool, Dict, StructureData, TrajectoryData, WorkChainNode
from aiida.orm import load_code
from aiida.plugins import DataFactory, WorkflowFactory
from aiidalab_widgets_base import WizardAppWidgetStep

from aiidalab_ispg.widgets import MoleculeDefinitionWidget, GroundStateDefinitionWidget

try:
    from aiidalab_atmospec_workchain.optimization import ConformerOptimizationWorkChain
except ImportError:
    print("ERROR: Could not find aiidalab_atmospec_workchain module!")


class SubmitWorkChainStep(ipw.VBox, WizardAppWidgetStep):
    """Base class for workflow submission stesps"""

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
            layout=ipw.Layout(width="auto", flex="1 1 auto"),
            disabled=False,
        )

        self.submit_button.on_click(self._on_submit_button_clicked)

        children = [self.submit_button]
        if components is not None:
            children = components + children

        super().__init__(children=children)

    def _on_submit_button_clicked(self, _):
        self.submit_button.disabled = True
        self.submit()

    def submit(self):
        """Must be overriden in the child class"""
        pass

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
        """Should be overriden by the child class"""
        return True

    @traitlets.observe("input_structure")
    def _observe_input_structure(self, change):
        # self.set_trait("builder_parameters", self._default_builder_parameters())
        self._update_state()
        # self._set_num_mpi_tasks_to_default()

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


class SubmitOptimizationWorkChainStep(SubmitWorkChainStep):
    """Step for submission of a optimization workchain."""

    def __init__(self, **kwargs):
        self.molecule_settings = MoleculeDefinitionWidget()
        self.ground_state_settings = GroundStateDefinitionWidget()
        components = [ipw.HBox([self.molecule_settings, self.ground_state_settings])]
        super().__init__(components=components)

    # TODO: Check the ORCA code is available, perhaps other verifications
    # (e.g. size of the molecule)
    def _validate_input_parameters(self) -> bool:
        """Validate input parameters"""
        return True

    def submit(self, _=None):

        assert self.input_structure is not None

        builder = ConformerOptimizationWorkChain.get_builder()

        # TODO: ComputationalResourceWidget
        builder.code = load_code("orca@localhost")
        builder.structure = self.input_structure
        builder.orca.parameters = Dict(self._build_orca_params())
        builder.orca.metadata = self._set_metadata()

        # Clean the remote directory by default,
        # we're copying back the main output file and gbw file anyway.
        builder.orca.clean_workdir = Bool(True)

        process = submit(builder)
        # process.base.extras.set("builder_parameters", self.builder_parameters.copy())
        # NOTE: It is important to set_extra builder_parameters before we update the traitlet
        self.process = process

    # TODO: Need to implement logic for handling more CPUs
    # and distribute them among conformers
    def _set_metadata(self):
        ncpus = 1
        metadata = {
            "options": {
                "withmpi": False,
                "resources": {
                    "tot_num_mpiprocs": ncpus,
                    "num_mpiprocs_per_machine": ncpus,
                    "num_cores_per_mpiproc": 1,
                    "num_machines": 1,
                },
            }
        }
        return metadata

    def _build_orca_params(self):
        """A bit of indirection to decouple aiida-orca plugin
        from this code"""

        basis = self.ground_state_settings.basis.value
        method = self.ground_state_settings.method.value
        charge = self.molecule_settings.charge.value
        multiplicity = self.molecule_settings.multiplicity.value

        input_keywords = [basis, method, "Opt", "AnFreq"]

        return {
            "charge": charge,
            "multiplicity": multiplicity,
            "input_blocks": {
                "scf": {"convergence": "tight", "ConvForced": "true"},
            },
            "input_keywords": input_keywords,
            # TODO: Remove this when possible
            "extra_input_keywords": [],
        }
