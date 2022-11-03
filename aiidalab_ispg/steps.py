"""Widgets for the submission of basic ORCA calculation.
Based on the original code from aiidalab_qe

Authors:
    * Daniel Hollas <daniel.hollas@durham.ac.uk>
    * Carl Simon Adorf <simon.adorf@epfl.ch>
"""
from pprint import pformat
import re

from copy import deepcopy

import ipywidgets as ipw
import traitlets
from traitlets import Union, Instance
from aiida.common import NotExistent
from aiida.engine import ProcessState, submit
from aiida.orm import load_node, load_code

from aiida.orm import WorkChainNode
from aiida.plugins import DataFactory
from aiidalab_widgets_base import (
    AiidaNodeViewWidget,
    ComputationalResourcesWidget,
    ProcessMonitor,
    ProcessNodesTreeWidget,
    WizardAppWidgetStep,
)

import aiidalab_ispg.qeapp as qeapp
from aiidalab_ispg.parameters import DEFAULT_PARAMETERS
from aiidalab_ispg.widgets import ResourceSelectionWidget
from aiidalab_ispg.widgets import QMSelectionWidget, ExcitedStateMethod

from .utils import get_formula

try:
    from aiidalab_atmospec_workchain import AtmospecWorkChain
except ImportError:
    print("ERROR: Could not find aiidalab_atmospec_workchain module!")

from aiidalab_ispg.spectrum import EnergyUnit, Spectrum, SpectrumWidget

StructureData = DataFactory("core.structure")
TrajectoryData = DataFactory("core.array.trajectory")
Dict = DataFactory("core.dict")
Bool = DataFactory("core.bool")


class StructureSelectionStep(qeapp.StructureSelectionStep):
    """Integrated widget for the selection of structures from different sources."""

    structure = Union(
        [Instance(StructureData), Instance(TrajectoryData)], allow_none=True
    )
    confirmed_structure = Union(
        [Instance(StructureData), Instance(TrajectoryData)], allow_none=True
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


class WorkChainSettings(ipw.VBox):

    structure_title = ipw.HTML(
        """<div style="padding-top: 0px; padding-bottom: 0px">
        <h4>Molecular geometry</h4></div>"""
    )
    structure_help = ipw.HTML(
        """<div style="line-height: 140%; padding-top: 0px; padding-bottom: 5px">
        By default, the workflow will optimize the provided geometry.<br>
        Select "Geometry as is" if this is not desired.</div>"""
    )

    electronic_structure_title = ipw.HTML(
        """<div style="padding-top: 0px; padding-bottom: 0px">
        <h4>Electronic structure</h4></div>"""
    )

    button_style_on = "info"
    button_style_off = "danger"

    def __init__(self, **kwargs):

        # Whether to optimize the molecule or not.
        self.geo_opt_type = ipw.ToggleButtons(
            options=[
                ("Geometry as is", "NONE"),
                ("Optimize geometry", "OPT"),
            ],
            value="OPT",
        )

        # TODO: Use Dropdown with Enum (Singlet, Doublet...)
        self.spin_mult = ipw.BoundedIntText(
            min=1,
            max=1,
            step=1,
            description="Multiplicity",
            disabled=True,
            value=1,
        )

        self.charge = ipw.IntText(
            description="Charge",
            disabled=False,
            value=0,
        )

        self.nstates = ipw.BoundedIntText(
            description="Nstate",
            tooltip="Number of excited states",
            disabled=False,
            value=3,
            min=1,
            max=50,
        )

        super().__init__(
            children=[
                self.structure_title,
                self.structure_help,
                self.geo_opt_type,
                self.electronic_structure_title,
                self.charge,
                self.spin_mult,
                self.nstates,
            ],
            **kwargs,
        )


class CodeSettings(ipw.VBox):

    codes_title = ipw.HTML(
        """<div style="padding-top: 0px; padding-bottom: 0px">
        <h4>Codes</h4></div>"""
    )
    codes_help = ipw.HTML(
        """<div style="line-height: 140%; padding-top: 0px; padding-bottom:
        10px"> Select the code to use for running the calculations. The codes
        on the local machine (localhost) are installed by default, but you can
        configure new ones on potentially more powerful machines by clicking on
        "Setup new code".</div>"""
    )

    def __init__(self, **kwargs):

        self.orca = ComputationalResourcesWidget(
            input_plugin="orca.orca",
            description="Main ORCA program",
        )
        super().__init__(
            children=[
                self.codes_title,
                self.codes_help,
                self.orca,
            ],
            **kwargs,
        )


class SubmitAtmospecAppWorkChainStep(ipw.VBox, WizardAppWidgetStep):
    """Step for submission of a bands workchain."""

    input_structure = Union(
        [Instance(StructureData), Instance(TrajectoryData)], allow_none=True
    )
    process = Instance(WorkChainNode, allow_none=True)
    disabled = traitlets.Bool()
    builder_parameters = traitlets.Dict()

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

        self._update_builder_parameters()

        self.builder_parameters_view = ipw.HTML(layout=ipw.Layout(width="auto"))
        ipw.dlink(
            (self, "builder_parameters"),
            (self.builder_parameters_view, "value"),
            transform=lambda p: '<pre style="line-height: 100%">'
            + pformat(p, indent=2, width=200)
            + "</pre>",
        )

        super().__init__(
            children=[
                self.message_area,
                self.tab,
                self.submit_button,
            ]
        )

    def _get_state(self):

        # Process is already running.
        if self.process is not None:
            return self.State.SUCCESS

        # Input structure not specified.
        if self.input_structure is None:
            return self.State.INIT

        # ORCA code not selected.
        if self.codes_selector.orca.value is None:
            return self.State.READY

        return self.State.CONFIGURED

    def _update_state(self, _=None):
        self.state = self._get_state()

    _ALERT_MESSAGE = """
        <div class="alert alert-{alert_class} alert-dismissible">
        <a href="#" class="close" data-dismiss="alert" aria-label="close">&times;</a>
        <span class="closebtn" onclick="this.parentElement.style.display='none';">&times;</span>
        <strong>{message}</strong>
        </div>"""

    def _show_alert_message(self, message, alert_class="info"):
        with self.message_area:
            display(  # noqa
                ipw.HTML(
                    self._ALERT_MESSAGE.format(alert_class=alert_class, message=message)
                )
            )

    def _set_num_mpi_tasks_to_default(self, _=None):
        """Set the number of MPI tasks to a reasonable value for the selected structure."""
        # DH: TODO: For now we simply set this to 1
        self.resources_config.num_mpi_tasks.value = 1

    @traitlets.observe("state")
    def _observe_state(self, change):
        with self.hold_trait_notifications():
            self.disabled = change["new"] not in (
                self.State.READY,
                self.State.CONFIGURED,
            )
            self.submit_button.disabled = change["new"] != self.State.CONFIGURED

    @traitlets.observe("input_structure")
    def _observe_input_structure(self, change):
        self.set_trait("builder_parameters", self._default_builder_parameters())
        self._update_state()
        self._set_num_mpi_tasks_to_default()

    @traitlets.observe("process")
    def _observe_process(self, change):
        with self.hold_trait_notifications():
            # process_node = change["new"]
            # DH: Not sure why this is here, but I don't think
            # it quite works for our current setup,
            # so commenting out?
            # if process_node is not None:
            # self.input_structure = process_node.inputs.structure
            # builder_parameters = process_node.get_extra("builder_parameters", None)
            # if builder_parameters is not None:
            #    self.set_trait("builder_parameters", builder_parameters)
            self._update_state()

    def _on_submit_button_clicked(self, _):
        self.submit_button.disabled = True
        self.submit()

    def _setup_builder_parameters_update(self):
        """Set up all ``observe`` calls to monitor changes in user inputs."""
        update = self._update_builder_parameters  # alias for code conciseness
        # Properties
        self.workchain_settings.geo_opt_type.observe(update, ["value"])
        self.workchain_settings.spin_mult.observe(update, ["value"])
        self.workchain_settings.charge.observe(update, ["value"])
        self.workchain_settings.nstates.observe(update, ["value"])
        # Codes
        self.codes_selector.orca.observe(update, ["value"])
        # QM settings
        self.qm_config.method.observe(update, ["value"])
        self.qm_config.excited_method.observe(update, ["value"])
        self.qm_config.basis.observe(update, ["value"])
        self.qm_config.solvent.observe(update, ["value"])

    @staticmethod
    def _serialize_builder_parameters(parameters):
        parameters = parameters.copy()  # create copy to not modify original dict
        # Serialize Enum
        parameters["excited_method"] = parameters["excited_method"].value
        return parameters

    @staticmethod
    def _deserialize_builder_parameters(parameters):
        parameters = parameters.copy()  # create copy to not modify original dict
        parameters["excited_method"] = ExcitedStateMethod(parameters["excited_method"])
        return parameters

    def _update_builder_parameters(self, _=None):
        self.set_trait(
            "builder_parameters",
            self._serialize_builder_parameters(
                dict(
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

    def build_base_orca_params(self, builder_parameters):
        """A bit of indirection to decouple aiida-orca plugin
        from this code"""
        bp = builder_parameters
        input_keywords = [bp["basis"]]

        # WARNING: Here we implicitly assume, that ORCA will automatically select
        # equilibrium solvation for ground state optimization,
        # and non-equilibrium solvation for single point excited state calculations.
        # This should be the default, but it would be better to be explicit.
        if bp["solvent"] != "None":
            input_keywords.append(f"CPCM({bp['solvent']})")

        return {
            "charge": bp["charge"],
            "multiplicity": bp["spin_mult"],
            "input_blocks": {
                "scf": {"convergence": "tight", "ConvForced": "true"},
            },
            "input_keywords": input_keywords,
            "extra_input_keywords": [],
        }

    def _add_mdci_orca_params(self, orca_parameters, basis, mdci_method, nroots):
        # TODO: Make this configurable
        # Safe default for 8 core, 32Gb machine
        # TODO: Figure out how to make this work as a global keyword
        # https://github.com/pzarabadip/aiida-orca/issues/45
        MEMORY_PER_CPU = 3000  # Mb

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
        }
        if es_method == ExcitedStateMethod.TDDFT.value:
            tddft_params["input_blocks"]["tddft"]["tda"] = "false"
        return tddft_params

    def _add_optimization_orca_params(self, base_orca_parameters, gs_method):
        opt_params = deepcopy(base_orca_parameters)
        opt_params["input_keywords"].append(gs_method)
        opt_params["input_keywords"].append("TightOpt")
        opt_params["input_keywords"].append("AnFreq")
        # For MP2, analytical frequencies are only available without Frozen Core
        if gs_method.lower() == "mp2":
            opt_params["input_keywords"].append("NoFrozenCore")
        return opt_params

    def submit(self, _=None):

        assert self.input_structure is not None

        bp = self.builder_parameters.copy()

        builder = AtmospecWorkChain.get_builder()

        orca_code = self.codes_selector.orca.value
        builder.code = orca_code
        builder.structure = self.input_structure

        base_orca_parameters = self.build_base_orca_params(bp)
        gs_opt_parameters = self._add_optimization_orca_params(
            base_orca_parameters, bp["method"]
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

        builder.opt.orca.parameters = Dict(dict=gs_opt_parameters)
        builder.exc.orca.parameters = Dict(dict=es_parameters)

        num_proc = self.resources_config.num_mpi_tasks.value
        if num_proc > 1:
            # NOTE: We only paralellize the optimizations job,
            # because we suppose there will be lot's of TDDFT jobs in NEA,
            # which can be trivially launched in parallel.
            builder.opt.orca.parameters["input_blocks"]["pal"] = {"nproc": num_proc}

        metadata = {
            "options": {
                "withmpi": False,
                "resources": {
                    "tot_num_mpiprocs": num_proc,
                    "num_mpiprocs_per_machine": num_proc,
                    "num_cores_per_mpiproc": 1,
                    "num_machines": 1,
                },
            }
        }
        builder.opt.orca.metadata = metadata
        builder.exc.orca.metadata = deepcopy(metadata)
        builder.exc.orca.metadata.options.resources["tot_num_mpiprocs"] = 1
        builder.exc.orca.metadata.options.resources["num_mpiprocs_per_machine"] = 1

        # Clean the remote directory by default,
        # we're copying back the main output file and gbw file anyway.
        builder.exc.clean_workdir = Bool(True)
        builder.opt.clean_workdir = Bool(True)

        builder.exc.orca.metadata.description = "ORCA TDDFT calculation"
        builder.opt.orca.metadata.description = "ORCA geometry optimization"

        if self.workchain_settings.geo_opt_type.value == "NONE":
            builder.optimize = False

        # Wigner will be sampled only when optimize == True
        builder.nwigner = self.qm_config.nwigner.value
        builder.wigner_low_freq_thr = self.qm_config.wigner_low_freq_thr.value

        process = submit(builder)
        process.set_extra("builder_parameters", self.builder_parameters.copy())
        # NOTE: It is important to set_extra builder_parameters before we update the traitlet
        self.process = process

    def reset(self):
        with self.hold_trait_notifications():
            self.process = None
            self.input_structure = None
            self.builder_parameters = self._default_builder_parameters()

    @traitlets.default("builder_parameters")
    def _default_builder_parameters(self):
        params = DEFAULT_PARAMETERS
        try:
            orca_code = load_code(params["orca_code"])
            params["orca_code"] = orca_code.uuid
        except NotExistent:
            params["orca_code"] = None
        return params


class ViewAtmospecAppWorkChainStatusAndResultsStep(ipw.VBox, WizardAppWidgetStep):

    process_uuid = traitlets.Unicode(allow_none=True)

    def __init__(self, **kwargs):
        self.process_tree = ProcessNodesTreeWidget()
        ipw.dlink((self, "process_uuid"), (self.process_tree, "process_uuid"))

        self.node_view = AiidaNodeViewWidget(layout={"width": "auto", "height": "auto"})
        ipw.dlink(
            (self.process_tree, "selected_nodes"),
            (self.node_view, "node"),
            transform=lambda nodes: nodes[0] if nodes else None,
        )
        self.process_status = ipw.VBox(children=[self.process_tree, self.node_view])

        # Setup process monitor
        self.process_monitor = ProcessMonitor(
            timeout=0.5,
            callbacks=[
                self.process_tree.update,
                self._update_state,
            ],
        )
        ipw.dlink((self, "process_uuid"), (self.process_monitor, "process_uuid"))

        super().__init__([self.process_status], **kwargs)

    def can_reset(self):
        "Do not allow reset while process is running."
        return self.state is not self.State.ACTIVE

    def reset(self):
        self.process_uuid = None

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
        self._update_state()


class ViewSpectrumStep(ipw.VBox, WizardAppWidgetStep):

    process_uuid = traitlets.Unicode(allow_none=True)

    def __init__(self, **kwargs):
        # Setup process monitor
        # TODO: Instead of setting another process monitor here,
        # we should just observe the process traitlet, and only set it
        # when the process is_finished_ok.
        self.process_monitor = ProcessMonitor(
            timeout=0.5,
            callbacks=[
                self._show_spectrum,
                self._update_state,
            ],
        )
        self.header = ipw.HTML()
        self.spectrum = SpectrumWidget()

        ipw.dlink((self, "process_uuid"), (self.process_monitor, "process_uuid"))

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

        # Number of Wigner geometries per conformer
        nsample = process.inputs.nwigner.value if process.inputs.nwigner > 0 else 1
        # TODO: Compute Boltzmann weight from Gibbs energy of conformers
        boltzmann_weight = 1.0 / len(process.inputs.structure.get_stepids())
        conformer_transitions = [
            {
                "transitions": self._wigner_output_to_transitions(conformer),
                "nsample": nsample,
                "weight": boltzmann_weight,
            }
            for conformer in process.outputs.spectrum_data.get_list()
        ]

        self.spectrum.conformer_transitions = conformer_transitions

        if "smiles" in process.inputs.structure.extras:
            self.spectrum.smiles = process.inputs.structure.extras["smiles"]
            # We're attaching smiles extra for the optimized structures as well
            # NOTE: You can distinguish between new / optimized geometries
            # by looking at the 'creator' attribute of the Structure node.
            if "relaxed_structures" in process.outputs:
                process.outputs.relaxed_structures.set_extra(
                    "smiles", self.spectrum.smiles
                )
        else:
            self.spectrum.smiles = None

        if "relaxed_structures" in process.outputs:
            self.spectrum.conformer_structures = process.outputs.relaxed_structures

    def _update_header(self):
        if self.process_uuid is None:
            self.header.value = ""
            return
        process = load_node(self.process_uuid)
        if bp := process.get_extra("builder_parameters", None):
            formula = re.sub(
                r"([0-9]+)",
                r"<sub>\1</sub>",
                get_formula(process.inputs.structure),
            )
            solvent = bp["solvent"] if bp["solvent"] != "None" else "the gas phase"
            # TODO: Compatibility hack
            es_method = bp.get("excited_method", "TDA-TDDFT")
            self.header.value = (
                f"<h4>UV/vis spectrum of {formula} "
                f"at {es_method}/{bp['method']}/{bp['basis']} level "
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
        self._update_state()
        self._update_header()
