"""Steps specific for the optimization workflow"""

import ipywidgets as ipw
import traitlets

from aiida.engine import submit
from aiida.orm import Bool, Dict, StructureData, TrajectoryData, WorkChainNode
from aiida.orm import load_code
from aiida.plugins import DataFactory, WorkflowFactory
from aiidalab_widgets_base import WizardAppWidgetStep

from .widgets import MoleculeDefinitionWidget, GroundStateDefinitionWidget
from .steps import SubmitWorkChainStepBase

try:
    from aiidalab_atmospec_workchain.optimization import ConformerOptimizationWorkChain
except ImportError:
    print("ERROR: Could not find aiidalab_atmospec_workchain module!")


class SubmitOptimizationWorkChainStep(SubmitWorkChainStepBase):
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
