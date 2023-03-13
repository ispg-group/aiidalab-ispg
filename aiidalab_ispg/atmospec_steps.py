"""Steps specific for the ATMOSPEC workflow"""

from copy import deepcopy
from dataclasses import dataclass
import enum

import ipywidgets as ipw
import traitlets

from aiida.engine import submit, ProcessState
from aiida.orm import Bool, StructureData, TrajectoryData, WorkChainNode
from aiida.orm import load_code, load_node
from aiidalab_widgets_base import WizardAppWidgetStep

from .input_widgets import (
    ExcitedStateMethod,
    MolecularGeometrySettings,
    MoleculeSettings,
    GroundStateSettings,
    ExcitedStateSettings,
    WignerSamplingSettings,
    CodeSettings,
    ResourceSelectionWidget,
)
from .steps import SubmitWorkChainStepBase, ViewWorkChainStatusStep
from .optimization_steps import OptimizationParameters
from .widgets import HeaderWarning, spinner
from .utils import MEMORY_PER_CPU

try:
    from aiidalab_atmospec_workchain import AtmospecWorkChain
except ImportError:
    print("ERROR: Could not find aiidalab_atmospec_workchain module!")


@dataclass(frozen=True)
class AtmospecParameters(OptimizationParameters):
    geo_opt_type: str
    excited_method: ExcitedStateMethod
    nstates: int
    es_basis: str
    tddft_functional: str
    nwigner: int
    wigner_low_freq_thr: float


# TODO: Production parameters
DEFAULT_ATMOSPEC_PARAMETERS = AtmospecParameters(
    charge=0,
    multiplicity=1,
    method="wB97X-D4",
    basis="def2-SVP",
    solvent="None",
    geo_opt_type="OPT",
    excited_method=ExcitedStateMethod.TDA,
    nstates=3,
    es_basis="def2-SVP",
    tddft_functional="wB97X-D4",
    nwigner=1,
    wigner_low_freq_thr=100.0,
)


class SubmitAtmospecAppWorkChainStep(SubmitWorkChainStepBase):
    """Step for submission of a optimization workchain."""

    def __init__(self, **kwargs):
        self.header_warning = HeaderWarning(dismissible=True)

        self.molecule_settings = MoleculeSettings()
        self.molecule_settings.multiplicity.disabled = True

        self.geometry_settings = MolecularGeometrySettings()
        self.geometry_settings.geo_opt_type.observe(self._observe_geo_opt_type, "value")

        self.ground_state_settings = GroundStateSettings()
        self.ground_state_settings.method.observe(self._observe_gs_method, "value")
        self.ground_state_settings.basis.observe(self._observe_gs_basis, "value")
        self.ground_state_settings.method.continuous_update = False

        self.excited_state_settings = ExcitedStateSettings()
        self.excited_state_settings.ground_state_sync.observe(
            self._observe_gs_sync, "value"
        )

        self.wigner_settings = WignerSamplingSettings()

        self.codes_selector = CodeSettings()
        self.resources_settings = ResourceSelectionWidget()

        self.codes_selector.orca.observe(self._update_state, "value")

        # Set defaults
        self._update_ui_from_parameters(DEFAULT_ATMOSPEC_PARAMETERS)

        settings = [
            self.geometry_settings,
            self.ground_state_settings,
            self.wigner_settings,
            self.molecule_settings,
            self.excited_state_settings,
        ]
        grid_layout = ipw.Layout(
            width="100%",
            grid_gap="0% 3%",
            grid_template_rows="auto auto",
            grid_template_columns="31% 31% 31%",
        )

        super().__init__(
            components=[
                self.header_warning,
                ipw.GridBox(children=settings, layout=grid_layout),
                ipw.HTML("<hr>"),
                ipw.HBox([self.codes_selector, self.resources_settings]),
            ]
        )

    # TODO: More validations (molecule size etc)
    # TODO: Prevent submission if solvent is selected with EOM-CCSD or ADC2
    # TODO: display an error message when there is an issue.
    # See how "submission blockers" are handled in QeApp
    def _validate_input_parameters(self) -> bool:
        """Validate input parameters"""
        # ORCA code not selected.
        if self.codes_selector.orca.value is None:
            return False
        return True

    def _observe_geo_opt_type(self, change):
        # If we don't optimize the molecule, we cannot do Wigner sampling
        if change["new"] == "OPT":
            self.wigner_settings.disabled = False
        else:
            self.wigner_settings.disabled = True

    def _observe_gs_sync(self, change):
        if change["new"]:
            self.excited_state_settings.basis.value = (
                self.ground_state_settings.basis.value
            )
            gs_method = self.ground_state_settings.method.value
            if gs_method.lower() not in ("ri-mp2", "mp2"):
                self.excited_state_settings.tddft_functional.value = gs_method

    def _observe_gs_method(self, change):
        """Update TDDFT functional if ground state functional is changed"""
        gs_method = change["new"]
        if gs_method is not None and (
            self.excited_state_settings.ground_state_sync.value
            and gs_method.lower() not in ("ri-mp2", "mp2")
        ):
            self.excited_state_settings.tddft_functional.value = gs_method

    def _observe_gs_basis(self, change):
        """Update TDDFT functional if ground state functional is changed"""
        if self.excited_state_settings.ground_state_sync.value:
            self.excited_state_settings.basis.value = change["new"]

    def _update_ui_from_parameters(self, parameters: AtmospecParameters) -> None:
        """Update UI widgets according to builder parameters.

        This function is called when we load an already finished workflow,
        and we want the input widgets to be updated accordingly
        """
        self.geometry_settings.geo_opt_type.value = parameters.geo_opt_type
        self.molecule_settings.charge.value = parameters.charge
        self.molecule_settings.multiplicity.value = parameters.multiplicity
        self.molecule_settings.solvent.value = parameters.solvent
        self.ground_state_settings.method.value = parameters.method
        self.ground_state_settings.basis.value = parameters.basis
        self.excited_state_settings.nstates.value = parameters.nstates
        self.excited_state_settings.excited_method.value = parameters.excited_method
        self.excited_state_settings.tddft_functional.value = parameters.tddft_functional
        self.excited_state_settings.basis.value = parameters.es_basis
        self.wigner_settings.nwigner.value = parameters.nwigner
        self.wigner_settings.wigner_low_freq_thr.value = parameters.wigner_low_freq_thr

        # Infer the value of the gs_sync checkbox
        if (
            parameters.method == parameters.tddft_functional
            and parameters.basis == parameters.es_basis
        ):
            self.excited_state_settings.ground_state_sync.value = True
        else:
            self.excited_state_settings.ground_state_sync.value = False

    def _get_parameters_from_ui(self) -> AtmospecParameters:
        """Prepare builder parameters from the UI input widgets"""
        return AtmospecParameters(
            geo_opt_type=self.geometry_settings.geo_opt_type.value,
            charge=self.molecule_settings.charge.value,
            multiplicity=self.molecule_settings.multiplicity.value,
            solvent=self.molecule_settings.solvent.value,
            method=self.ground_state_settings.method.value,
            basis=self.ground_state_settings.basis.value,
            tddft_functional=self.excited_state_settings.tddft_functional.value,
            es_basis=self.excited_state_settings.basis.value,
            excited_method=self.excited_state_settings.excited_method.value,
            nstates=self.excited_state_settings.nstates.value,
            nwigner=self.wigner_settings.nwigner.value,
            wigner_low_freq_thr=self.wigner_settings.wigner_low_freq_thr.value,
        )

    @traitlets.observe("process")
    def _observe_process(self, change):
        with self.hold_trait_notifications():
            process = change["new"]
            self.header_warning.hide()
            if process is not None:
                self.input_structure = process.inputs.structure
                try:
                    parameters = process.base.extras.get("builder_parameters")
                    parameters["excited_method"] = ExcitedStateMethod(
                        parameters["excited_method"]
                    )
                    self._update_ui_from_parameters(AtmospecParameters(**parameters))
                except (AttributeError, KeyError, TypeError):
                    # extras do not exist or are incompatible, let's reset to default values
                    self.header_warning.show(
                        f"WARNING: Workflow parameters could not be loaded from process pk: {process.pk}"
                    )
                    self._update_ui_from_parameters(DEFAULT_ATMOSPEC_PARAMETERS)
            self._update_state()

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

    def build_base_orca_params(self, params: AtmospecParameters) -> dict:
        """Prepare dictionary of ORCA parameters, as required by aiida-orca plugin"""
        # WARNING: Here we implicitly assume, that ORCA will automatically select
        # equilibrium solvation for ground state optimization,
        # and non-equilibrium solvation for single point excited state calculations.
        # This should be the default, but it would be better to be explicit.
        input_keywords = []
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

    def _add_mdci_orca_params(self, orca_parameters, basis, mdci_method, nroots):
        mdci_params = deepcopy(orca_parameters)
        mdci_params["input_keywords"].append(mdci_method.value)
        mdci_params["input_keywords"].append(basis)
        if mdci_method == ExcitedStateMethod.ADC2:
            # Basis for RI approximation, this will not work for all basis sets
            mdci_params["input_keywords"].append(f"{basis}/C")

        mdci_params["input_blocks"]["mdci"] = {
            "nroots": nroots,
            "maxcore": MEMORY_PER_CPU,
        }
        # TODO: For efficiency reasons, in might not be necessary to calculated left-vectors
        # to obtain TDM, but we need to benchmark that first.
        if mdci_method == ExcitedStateMethod.CCSD:
            mdci_params["input_blocks"]["mdci"]["doTDM"] = "true"
            mdci_params["input_blocks"]["mdci"]["doLeft"] = "true"
        return mdci_params

    def _add_tddft_orca_params(
        self, base_orca_parameters, basis, es_method, functional, nroots
    ):
        tddft_params = deepcopy(base_orca_parameters)
        tddft_params["input_keywords"].append(functional)
        tddft_params["input_keywords"].append(basis)
        tddft_params["input_blocks"]["tddft"] = {
            "nroots": nroots,
            "maxcore": MEMORY_PER_CPU,
        }
        if es_method == ExcitedStateMethod.TDDFT:
            tddft_params["input_blocks"]["tddft"]["tda"] = "false"
        return tddft_params

    def _add_optimization_orca_params(self, base_orca_parameters, basis, gs_method):
        opt_params = deepcopy(base_orca_parameters)
        opt_params["input_keywords"].append(gs_method)
        opt_params["input_keywords"].append(basis)
        opt_params["input_keywords"].append("TightOpt")
        opt_params["input_keywords"].append("AnFreq")
        # For MP2, analytical frequencies are only available without Frozen Core
        # TODO: Add this to optimization workflow
        if gs_method.lower() in ("ri-mp2", "mp2"):
            opt_params["input_keywords"].append("NoFrozenCore")
            opt_params["input_keywords"].append(f"{basis}/C")
            opt_params["input_blocks"]["mp2"] = {"maxcore": MEMORY_PER_CPU}
        return opt_params

    def submit(self, _=None):
        assert self.input_structure is not None

        bp = self._get_parameters_from_ui()
        builder = AtmospecWorkChain.get_builder()

        builder.code = load_code(self.codes_selector.orca.value)
        builder.structure = self.input_structure
        base_orca_parameters = self.build_base_orca_params(bp)
        gs_opt_parameters = self._add_optimization_orca_params(
            base_orca_parameters, basis=bp.basis, gs_method=bp.method
        )
        if bp.excited_method in (
            ExcitedStateMethod.TDA,
            ExcitedStateMethod.TDDFT,
        ):
            es_parameters = self._add_tddft_orca_params(
                base_orca_parameters,
                es_method=bp.excited_method,
                basis=bp.es_basis,
                functional=bp.tddft_functional,
                nroots=bp.nstates,
            )
        elif bp.excited_method in (
            ExcitedStateMethod.ADC2,
            ExcitedStateMethod.CCSD,
        ):
            es_parameters = self._add_mdci_orca_params(
                base_orca_parameters,
                basis=bp.es_basis,
                mdci_method=bp.excited_method,
                nroots=bp.nstates,
            )
        else:
            raise NotImplementedError(
                f"Excited method {bp.excited_method} not implemented"
            )

        builder.opt.orca.parameters = gs_opt_parameters
        builder.exc.orca.parameters = es_parameters

        num_proc = self.resources_settings.num_mpi_tasks.value
        if num_proc > 1:
            # NOTE: We only paralelize the optimizations job,
            # because we suppose there will be lot's of TDDFT jobs in NEA,
            # which can be trivially launched in parallel.
            # We also paralelize EOM-CCSD as it is expensive and likely
            # used only for single point calculations.
            builder.opt.orca.parameters["input_blocks"]["pal"] = {"nproc": num_proc}
            if bp.excited_method == ExcitedStateMethod.CCSD:
                builder.exc.orca.parameters["input_blocks"]["pal"] = {"nproc": num_proc}

        metadata = self._build_orca_metadata(num_proc)
        builder.opt.orca.metadata = metadata
        builder.exc.orca.metadata = deepcopy(metadata)
        if bp.excited_method != ExcitedStateMethod.CCSD:
            builder.exc.orca.metadata.options.resources["tot_num_mpiprocs"] = 1
            builder.exc.orca.metadata.options.resources["num_mpiprocs_per_machine"] = 1

        # Clean the remote directory by default,
        # we're copying back the main output file and gbw file anyway.
        builder.exc.clean_workdir = Bool(True)
        builder.opt.clean_workdir = Bool(True)

        builder.exc.orca.metadata.description = "ORCA TDDFT calculation"
        builder.opt.orca.metadata.description = "ORCA geometry optimization"

        if bp.geo_opt_type == "NONE":
            builder.optimize = False

        # Wigner will be sampled only when optimize == True
        builder.nwigner = bp.nwigner
        builder.wigner_low_freq_thr = bp.wigner_low_freq_thr

        process = submit(builder)
        # NOTE: It is important to set_extra builder_parameters before we update the traitlet
        builder_parameters = vars(bp)
        builder_parameters["excited_method"] = builder_parameters[
            "excited_method"
        ].value
        process.base.extras.set("builder_parameters", builder_parameters)
        self.process = process

    def reset(self):
        # NOTE: We purposefully do not reset the workchain settings back to default,
        # in case one wants to submit a series of same workflows for different molecules.
        self.header_warning.hide()
        super().reset()


# TODO: Disambiguate between optimizing conformers,
# computing single point spectra and Wigner spectra
class AtmospecWorkflowStatus(enum.Enum):
    INIT = 0
    IN_PROGRESS = 1
    FINISHED = 2
    FAILED = 3


class AtmospecWorkflowProgressWidget(ipw.HBox):
    """Widget for user friendly representation of the workflow status."""

    status = traitlets.Instance(AtmospecWorkflowStatus, allow_none=True)

    def __init__(self, **kwargs):
        self._progress_bar = ipw.IntProgress(
            style={"description_width": "initial", "padding": "10px"},
            description="Workflow progress:",
            value=0,
            min=0,
            max=2,
            disabled=False,
            orientations="horizontal",
        )

        self._status_text = ipw.HTML()
        super().__init__(
            children=[self._progress_bar, self._status_text],
            # This looks slightly better to me, but the indicator would move around
            # based on the status message.
            # layout=ipw.Layout(justify_content="space-around"),
            # layout=ipw.Layout(justify_content="space-between"),
            **kwargs,
        )

    @traitlets.observe("status")
    def _observe_status(self, change):
        with self.hold_trait_notifications():
            if change["new"]:
                self._status_text.value = {
                    AtmospecWorkflowStatus.INIT: "Workflow started",
                    AtmospecWorkflowStatus.IN_PROGRESS: f"Optimizing conformers {spinner}",
                    AtmospecWorkflowStatus.FINISHED: "Worflow finished successfully! 🎉",
                    AtmospecWorkflowStatus.FAILED: "Workflow failed! 😧",
                }.get(change["new"], change["new"].name)

                self._progress_bar.value = change["new"].value
                self._progress_bar.bar_style = {
                    AtmospecWorkflowStatus.FINISHED: "success",
                    AtmospecWorkflowStatus.FAILED: "danger",
                }.get(change["new"], "info")
            else:
                self._status_text.value = ""
                self._progress_bar.value = 0
                self._progress_bar.bar_style = "info"


class ViewAtmospecAppWorkChainStatusAndResultsStep(ViewWorkChainStatusStep):

    workflow_status = traitlets.Instance(AtmospecWorkflowStatus, allow_none=True)

    def __init__(self, **kwargs):
        self.progress_bar = AtmospecWorkflowProgressWidget()
        ipw.dlink(
            (self, "workflow_status"),
            (self.progress_bar, "status"),
        )
        super().__init__(progress_bar=self.progress_bar, **kwargs)

    def _update_workflow_state(self, process_uuid):
        if process_uuid is None:
            self.workflow_status = None
            return

        process = load_node(self.process_uuid)
        process_state = process.process_state
        if process_state in (
            ProcessState.CREATED,
            ProcessState.RUNNING,
            ProcessState.WAITING,
        ):
            self.workflow_status = AtmospecWorkflowStatus.IN_PROGRESS
        elif (
            process_state in (ProcessState.EXCEPTED, ProcessState.KILLED)
            or process.is_failed
        ):
            self.workflow_status = AtmospecWorkflowStatus.FAILED
        elif process_state is ProcessState.FINISHED and process.is_finished_ok:
            self.workflow_status = AtmospecWorkflowStatus.FINISHED
