"""Steps specific for the optimization workflow"""

import ipywidgets as ipw
import traitlets

from dataclasses import dataclass

from aiida.engine import submit
from aiida.orm import Bool, StructureData, TrajectoryData, WorkChainNode
from aiida.orm import load_code
from aiida.plugins import DataFactory
from aiidalab_widgets_base import WizardAppWidgetStep

from .input_widgets import MoleculeSettings, GroundStateSettings, CodeSettings
from .steps import SubmitWorkChainStepBase, WorkChainSettings
from .widgets import ResourceSelectionWidget, QMSelectionWidget, ExcitedStateMethod

try:
    from aiidalab_atmospec_workchain.optimization import ConformerOptimizationWorkChain
except ImportError:
    print("ERROR: Could not find aiidalab_atmospec_workchain module!")


@dataclass(frozen=True)
class AtmospecParameters(OptimizationParameter):
    geo_opt: str
    excited_method: ExcitedStateMethod
    nstate: int
    nwigner: int
    wigner_low_freq_thr: float


# TODO: Production parameters
DEFAULT_ATMOSPEC_PARAMETERS = OptimizationParameters(
    charge=0,
    multiplicity=1,
    method="wB97X-D4",
    basis="def2-SVP",
    solvent="None",
    get_opt="NONE",
    excited_method=ExcitedStateMethod.ADC2,
    nstate=3,
    nwigner=1,
    wigner_low_freq_thr=150.,
)


class SubmitAtmospecAppWorkChainStep(SubmitWorkChainStepBase):
    """Step for submission of a optimization workchain."""

    def __init__(self, **kwargs):
        self.message_area = ipw.Output()
        self.workchain_settings = WorkChainSettings()
        self.codes_selector = CodeSettings()
        self.resources_config = ResourceSelectionWidget()
        self.qm_config = QMSelectionWidget()

        self.set_trait("builder_parameters", self._default_builder_parameters())
        self._setup_builder_parameters_update()

        self.codes_selector.orca.observe(self._update_state, "value")
        self.codes_selector.orca.observe(self._set_num_mpi_tasks_to_default, "value")

        self.tab = ipw.Tab(
            children=[
                self.workchain_settings,
            ],
            layout=ipw.Layout(min_height="250px"),
        )

        self.tab.set_title(0, "Workflow")
        self.tab.set_title(1, "Advanced settings")
        self.tab.set_title(2, "Codes & Resources")
        self.tab.children = [
            self.workchain_settings,
            self.qm_config,
            ipw.VBox(children=[self.codes_selector, self.resources_config]),
        ]

        self.submit_button = ipw.Button(
            description="Submit",
            tooltip="Submit the calculation with the selected parameters.",
            icon="play",
            button_style="success",
            layout=ipw.Layout(width="auto", flex="1 1 auto"),
            disabled=True,
        )

        self.submit_button.on_click(self._on_submit_button_clicked)

        self._update_ui_from_parameters(DEFAULT_ATMOSPEC_PARAMETERS)

        super().__init__(
            children=[
                self.message_area,
                self.tab,
                self.submit_button,
            ]
        )

    # TODO: More validations (molecule size etc)
    # TODO: display an error message when there is an issue.
    # See how "submission blockers" are handled in QeApp
    def _validate_input_parameters(self) -> bool:
        """Validate input parameters"""
        # ORCA code not selected.
        if self.code_settings.orca.value is None:
            return False
        return True

    def _update_ui_from_parameters(self, parameters: AtmospecParameters) -> None:
        """Update UI widgets according to builder parameters.

        This function is called when we load an already finished workflow,
        and we want the input widgets to be updated accordingly
        """
        #self.molecule_settings.charge.value = parameters.charge
        #self.molecule_settings.multiplicity.value = parameters.multiplicity
        #self.molecule_settings.solvent.value = parameters.solvent
        #self.ground_state_settings.method.value = parameters.method
        #self.ground_state_settings.basis.value = parameters.basis
        self.workchain_settings.spin_mult.value = parameters.multiplicity
        self.workchain_settings.charge.value = parameters.charge
        self.workchain_settings.nstates.value = parameters.nstates
        self.qm_config.excited_method.value = parameters.excited_method
        self.qm_config.method.value = parameters.method
        self.qm_config.basis.value = parameters.basis
        self.qm_config.solvent.value = parameters.solvent
        self.qm_config.nwigner.value = parameters.nwigner
        self.qm_config.wigner_low_freq_thr.value = parameters.wigner_low_freq_thr
        self.workchain_settings.geo_opt_type.value = parameters.geo_opt

    def _get_parameters_from_ui(self) -> AtmospecParameters:
        """Prepare builder parameters from the UI input widgets"""
        return AtomspecParameters(
            #charge=self.molecule_settings.charge.value,
            #multiplicity=self.molecule_settings.multiplicity.value,
            #solvent=self.molecule_settings.solvent.value,
            #method=self.ground_state_settings.method.value,
            #basis=self.ground_state_settings.basis.value,
            charge=self.workchain_settings.charge.value,
            multiplicity=self.workchain_settings.spin_mult.value,
            geo_opt=self.workchain_settings.geo_opt_type.value,
            solvent=self.qm_config.solvent.value,
            method=self.qm_config.method.value,
            basis=self.qm_config.basis.value,
            nstates=self.workchain_settings.nstates.value,
            excited_method=self.qm_config.excited_method.value,
            nwigner=self.qm_config.nwigner.value = parameters.nwigner,
            wigner_low_freq_thr=self.qm_config.wigner_low_freq_thr.value,
        )

    @traitlets.observe("process")
    def _observe_process(self, change):
        with self.hold_trait_notifications():
            process = change["new"]
            if process is not None:
                self.input_structure = process.inputs.structure
                parameters = process.base.extras.get("builder_parameters", None)
                if parameters is not None:
                    try:
                        self._update_ui_from_parameters(
                            AtmospecParameters(**parameters)
                        )
                    except AttributeError as e:
                        # extras do not exist or are incompatible, ignore this problem
                        pass
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

    def _update_builder_parameters(self, _=None):
        self.set_trait(
            "builder_parameters",
            self._serialize_builder_parameters(
                dict(  # noqa C408
                    orca_code=self.codes_selector.orca.value,
                    method=self.qm_config.method.value,
                    excited_method=self.qm_config.excited_method.value,
                    basis=self.qm_config.basis.value,
                    solvent=self.qm_config.solvent.value,
                    charge=self.workchain_settings.charge.value,
                    nstates=self.workchain_settings.nstates.value,
                    spin_mult=self.workchain_settings.spin_mult.value,
                )
            ),
        )

    @traitlets.observe("builder_parameters")
    def _observe_builder_parameters(self, change):
        bp = self._deserialize_builder_parameters(change["new"])

        with self.hold_trait_notifications():
            # Workchain settings
            self.workchain_settings.spin_mult.value = bp["spin_mult"]
            self.workchain_settings.charge.value = bp["charge"]
            self.workchain_settings.nstates.value = bp["nstates"]
            # Codes
            self.codes_selector.orca.value = bp.get("orca_code")
            # QM settings
            self.qm_config.excited_method.value = bp["excited_method"]
            self.qm_config.method.value = bp["method"]
            self.qm_config.basis.value = bp["basis"]
            self.qm_config.solvent.value = bp["solvent"]

    def build_base_orca_params(self, params: AtmospecParameters) -> dict:
        """Prepare dictionary of ORCA parameters, as required by aiida-orca plugin"""
        # WARNING: Here we implicitly assume, that ORCA will automatically select
        # equilibrium solvation for ground state optimization,
        # and non-equilibrium solvation for single point excited state calculations.
        # This should be the default, but it would be better to be explicit.
        input_keywords = ([params.basis],)
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
        mdci_params["input_keywords"].append(mdci_method)
        if mdci_method == ExcitedStateMethod.ADC2.value:
            # Basis for RI approximation, this will not work for all basis sets
            mdci_params["input_keywords"].append(f"{basis}/C")

        mdci_params["input_blocks"]["mdci"] = {
            "nroots": nroots,
            "maxcore": MEMORY_PER_CPU,
        }
        # TODO: For efficiency reasons, in might not be necessary to calculated left-vectors
        # to obtain TDM, but we need to benchmark that first.
        if mdci_method == ExcitedStateMethod.CCSD.value:
            mdci_params["input_blocks"]["mdci"]["doTDM"] = "true"
            mdci_params["input_blocks"]["mdci"]["doLeft"] = "true"
        return mdci_params

    def _add_tddft_orca_params(
        self, base_orca_parameters, es_method, functional, nroots
    ):
        tddft_params = deepcopy(base_orca_parameters)
        tddft_params["input_keywords"].append(functional)
        tddft_params["input_blocks"]["tddft"] = {
            "nroots": nroots,
            "maxcore": MEMORY_PER_CPU,
        }
        if es_method == ExcitedStateMethod.TDDFT.value:
            tddft_params["input_blocks"]["tddft"]["tda"] = "false"
        return tddft_params

    def _add_optimization_orca_params(self, base_orca_parameters, basis, gs_method):
        opt_params = deepcopy(base_orca_parameters)
        opt_params["input_keywords"].append(gs_method)
        opt_params["input_keywords"].append("TightOpt")
        opt_params["input_keywords"].append("AnFreq")
        # For MP2, analytical frequencies are only available without Frozen Core
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
            base_orca_parameters, basis=bp["basis"], gs_method=bp["method"]
        )
        if bp["excited_method"] in (
            ExcitedStateMethod.TDA.value,
            ExcitedStateMethod.TDDFT.value,
        ):
            es_parameters = self._add_tddft_orca_params(
                base_orca_parameters,
                es_method=bp["excited_method"],
                functional=bp["method"],
                nroots=bp["nstates"],
            )
        elif bp["excited_method"] in (
            ExcitedStateMethod.ADC2.value,
            ExcitedStateMethod.CCSD.value,
        ):
            es_parameters = self._add_mdci_orca_params(
                base_orca_parameters,
                basis=bp["basis"],
                mdci_method=bp["excited_method"],
                nroots=bp["nstates"],
            )
        else:
            raise ValueError(f"Excited method {bp['excited_method']} not implemented")

        builder.opt.orca.parameters = Dict(gs_opt_parameters)
        builder.exc.orca.parameters = Dict(es_parameters)

        num_proc = self.resources_config.num_mpi_tasks.value
        if num_proc > 1:
            # NOTE: We only paralelize the optimizations job,
            # because we suppose there will be lot's of TDDFT jobs in NEA,
            # which can be trivially launched in parallel.
            # We also paralelize EOM-CCSD as it is expensive and likely
            # used only for single point calculations.
            builder.opt.orca.parameters["input_blocks"]["pal"] = {"nproc": num_proc}
            if bp["excited_method"] == ExcitedStateMethod.CCSD.value:
                builder.exc.orca.parameters["input_blocks"]["pal"] = {"nproc": num_proc}

        metadata = self._build_orca_metadata(num_proc)
        builder.opt.orca.metadata = metadata
        builder.exc.orca.metadata = deepcopy(metadata)
        if bp["excited_method"] != ExcitedStateMethod.CCSD.value:
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
        process.base.extras.set("builder_parameters", vars(bp))
        self.process = process
