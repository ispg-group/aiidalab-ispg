"""Steps specific for the optimization workflow"""

import ipywidgets as ipw
import traitlets

from dataclasses import dataclass

from aiida.engine import submit
from aiida.orm import Bool, StructureData, TrajectoryData, WorkChainNode
from aiida.orm import load_code
from aiida.plugins import DataFactory, WorkflowFactory
from aiidalab_widgets_base import WizardAppWidgetStep

from .input_widgets import CodeSettings, MoleculeSettings, GroundStateSettings
from .widgets import ResourceSelectionWidget
from .steps import SubmitWorkChainStepBase
from .utils import MEMORY_PER_CPU

try:
    from aiidalab_atmospec_workchain.optimization import ConformerOptimizationWorkChain
except ImportError:
    print("ERROR: Could not find aiidalab_atmospec_workchain module!")


@dataclass(frozen=True)
class OptimizationParameters:
    charge: int
    multiplicity: int
    method: str
    basis: str
    solvent: str


DEFAULT_OPTIMIZATION_PARAMETERS = OptimizationParameters(
    charge=0,
    multiplicity=1,
    method="wB97X-D4",
    basis="def2-SVP",
    solvent="None",
)


class SubmitOptimizationWorkChainStep(SubmitWorkChainStepBase):
    """Step for submission of a optimization workchain."""

    def __init__(self, **kwargs):
        self.molecule_settings = MoleculeSettings()
        self.ground_state_settings = GroundStateSettings()
        self.code_settings = CodeSettings()
        self.resources_settings = ResourceSelectionWidget()

        # We need to observe each widget for which the validation could fail.
        self.code_settings.orca.observe(self._update_state, "value")
        components = [
            ipw.HBox(
                [
                    self.molecule_settings,
                    self.ground_state_settings,
                ]
            ),
            ipw.HBox(
                [
                    self.code_settings,
                    self.resources_settings,
                ]
            ),
        ]
        self._update_ui_from_parameters(DEFAULT_OPTIMIZATION_PARAMETERS)
        super().__init__(components=components)

    # TODO: More validations (molecule size etc)
    # TODO: display an error message when there is an issue.
    # See how "submission blockers" are handled in QeApp
    def _validate_input_parameters(self) -> bool:
        """Validate input parameters"""
        # ORCA code not selected.
        if self.code_settings.orca.value is None:
            return False
        return True

    def _update_ui_from_parameters(self, parameters: OptimizationParameters) -> None:
        """Update UI widgets according to builder parameters.

        This function is called when we load an already finished workflow,
        and we want the input widgets to be updated accordingly
        """
        self.molecule_settings.charge.value = parameters.charge
        self.molecule_settings.multiplicity.value = parameters.multiplicity
        self.molecule_settings.solvent.value = parameters.solvent
        self.ground_state_settings.method.value = parameters.method
        self.ground_state_settings.basis.value = parameters.basis

    def _get_parameters_from_ui(self) -> OptimizationParameters:
        """Prepare builder parameters from the UI input widgets"""
        return OptimizationParameters(
            charge=self.molecule_settings.charge.value,
            multiplicity=self.molecule_settings.multiplicity.value,
            solvent=self.molecule_settings.solvent.value,
            method=self.ground_state_settings.method.value,
            basis=self.ground_state_settings.basis.value,
        )

    @traitlets.observe("process")
    def _observe_process(self, change):
        with self.hold_trait_notifications():
            process = change["new"]
            if process is not None:
                self.input_structure = process.inputs.structure
                try:
                    parameters = process.base.extras.get("builder_parameters")
                    self._update_ui_from_parameters(
                        OptimizationParameters(**parameters)
                    )
                except (AttributeError, KeyError, TypeError):
                    # extras do not exist or are incompatible, ignore this problem
                    # TODO: Maybe display warning?
                    pass
            self._update_state()

    def submit(self, _=None):
        assert self.input_structure is not None

        parameters = self._get_parameters_from_ui()
        builder = ConformerOptimizationWorkChain.get_builder()

        builder.structure = self.input_structure
        builder.code = load_code(self.code_settings.orca.value)

        num_mpiprocs = self.resources_settings.num_mpi_tasks.value
        builder.orca.metadata = self._build_orca_metadata(num_mpiprocs)
        builder.orca.parameters = self._build_orca_params(parameters)
        if num_mpiprocs > 1:
            builder.orca.parameters["input_blocks"]["pal"] = {"nproc": num_mpiprocs}

        # Clean the remote directory by default,
        # We're copying back the main output file and gbw file anyway.
        builder.clean_workdir = Bool(True)

        process = submit(builder)

        # NOTE: It is important to set_extra builder_parameters before we update the traitlet
        process.base.extras.set("builder_parameters", vars(parameters))
        self.process = process

    # TODO: Need to implement logic for handling more CPUs
    # and distribute them among conformers
    def _build_orca_metadata(self, num_mpiprocs: int):
        return {
            "options": {
                "withmpi": False,
                "resources": {
                    "tot_num_mpiprocs": num_mpiprocs,
                    "num_mpiprocs_per_machine": num_mpiprocs,
                    "num_cores_per_mpiproc": 1,
                    "num_machines": 1,
                },
            }
        }

    def _build_orca_params(self, params: OptimizationParameters) -> dict:
        """Prepare dictionary of ORCA parameters, as required by aiida-orca plugin"""
        # WARNING: Here we implicitly assume, that ORCA will automatically select
        # equilibrium solvation for ground state optimization,
        # and non-equilibrium solvation for single point excited state calculations.
        # This should be the default, but it would be better to be explicit.
        input_keywords = [params.basis, params.method, "Opt", "AnFreq"]
        if params.solvent != "None":
            input_keywords.append(f"CPCM({params.solvent})")
        return {
            "charge": params.charge,
            "multiplicity": params.multiplicity,
            "input_blocks": {
                "scf": {"convergence": "tight", "ConvForced": "true"},
            },
            "input_keywords": input_keywords,
        }
