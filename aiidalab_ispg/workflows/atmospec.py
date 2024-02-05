"""Base work chain to run an ORCA calculation"""

# Not sure if this is needed? Can we use self.run()?
from aiida.engine import (
    ExitCode,
    ToContext,
    WorkChain,
    append_,
    if_,
    process_handler,
    run,
)
from aiida.orm import (
    Bool,
    Dict,
    Float,
    Int,
    List,
    SinglefileData,
    StructureData,
    TrajectoryData,
    to_aiida_type,
)
from aiida.plugins import CalculationFactory, DataFactory, WorkflowFactory

from .harmonic_wigner import generate_wigner_structures
from .optimization import RobustOptimizationWorkChain
from .utils import (
    ConcatInputsToList,
    add_orca_wf_guess,
    extract_trajectory_arrays,
    pick_structure_from_trajectory,
    structures_to_trajectory,
)

Code = DataFactory("core.code.installed")
OrcaCalculation = CalculationFactory("orca.orca")
OrcaBaseWorkChain = WorkflowFactory("orca.base")


class OrcaExcitationWorkChain(OrcaBaseWorkChain):
    """A simple shim for UV/vis excitation in ORCA."""

    def _build_process_label(self) -> str:
        return "Excitation workflow"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.output(
            "excitations",
            valid_type=Dict,
            required=True,
            help="Excitation energies and oscillator strengths from a single-point excitations",
        )

    def extract_transitions_from_orca_output(self, orca_output_params):
        return {
            "oscillator_strengths": orca_output_params["etoscs"],
            # Orca returns excited state energies in cm^-1
            # Perhaps we should do the conversion here,
            # to make this less ORCA specific.
            "excitation_energies_cm": orca_output_params["etenergies"],
        }

    @process_handler(exit_codes=ExitCode(0), priority=600)
    def add_excitation_output(self, calculation):
        """Extract excitation energies and osc. strengths into a separate output node"""
        transitions = self.extract_transitions_from_orca_output(
            calculation.outputs.output_parameters
        )
        self.out("excitations", Dict(transitions).store())


class OrcaWignerSpectrumWorkChain(WorkChain):
    """Top level workchain for Nuclear Ensemble Approach UV/vis
    spectrum for a single conformer"""

    def _build_process_label(self):
        return "NEA spectrum workflow"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(
            RobustOptimizationWorkChain,
            namespace="opt",
            exclude=["orca.structure", "orca.code"],
        )
        spec.expose_inputs(
            OrcaExcitationWorkChain,
            namespace="exc",
            exclude=["orca.structure", "orca.code"],
        )
        spec.input("structure", valid_type=(StructureData, TrajectoryData))
        spec.input("code", valid_type=Code)

        # Whether to perform geometry optimization
        spec.input(
            "optimize",
            valid_type=Bool,
            default=lambda: Bool(True),
            serializer=to_aiida_type,
        )

        # Number of Wigner geometries (computed only when optimize==True)
        spec.input(
            "nwigner", valid_type=Int, default=lambda: Int(1), serializer=to_aiida_type
        )

        spec.input(
            "wigner_low_freq_thr",
            valid_type=Float,
            default=lambda: Float(10),
            serializer=to_aiida_type,
        )

        spec.output(
            "franck_condon_excitations",
            valid_type=Dict,
            required=True,
            help="Output parameters from a single-point excitations",
        )
        spec.expose_outputs(
            RobustOptimizationWorkChain,
            namespace="opt",
            include=["output_parameters", "relaxed_structure"],
            namespace_options={"required": False},
        )

        spec.output(
            "wigner_excitations",
            valid_type=List,
            required=False,
            help="Output parameters from all Wigner excited state calculation",
        )

        spec.outline(
            if_(cls.should_optimize)(
                cls.optimize,
                cls.inspect_optimization,
            ),
            cls.excite,
            cls.inspect_excitation,
            if_(cls.should_run_wigner)(
                cls.wigner_sampling,
                cls.wigner_excite,
                cls.inspect_wigner_excitation,
            ),
        )

        spec.exit_code(
            401,
            "ERROR_OPTIMIZATION_FAILED",
            "optimization encountered unspecified error",
        )
        spec.exit_code(
            402, "ERROR_EXCITATION_FAILED", "excited state calculation failed"
        )

    def excite(self):
        """Calculate excited states for a single geometry"""
        inputs = self.exposed_inputs(
            OrcaExcitationWorkChain, namespace="exc", agglomerate=False
        )
        inputs.orca.code = self.inputs.code

        if self.inputs.optimize:
            self.report("Calculating spectrum for optimized geometry")
            inputs.orca.structure = self.ctx.calc_opt.outputs.relaxed_structure

            # Pass in converged SCF wavefunction
            with self.ctx.calc_opt.outputs.retrieved.base.repository.open(
                "aiida.gbw", "rb"
            ) as handler:
                gbw_file = SinglefileData(handler)
            inputs.orca.file = {"gbw": gbw_file}
            inputs.orca.parameters = add_orca_wf_guess(inputs.orca.parameters)
        else:
            self.report("Calculating spectrum for input geometry")
            inputs.orca.structure = self.inputs.structure

        calc_exc = self.submit(OrcaExcitationWorkChain, **inputs)
        calc_exc.label = "franck-condon-excitation"
        return ToContext(calc_exc=calc_exc)

    def wigner_sampling(self):
        self.report(f"Generating {self.inputs.nwigner.value} Wigner geometries")

        n_low_freq_vibs = 0
        for freq in self.ctx.calc_opt.outputs.output_parameters["vibfreqs"]:
            if freq < self.inputs.wigner_low_freq_thr:
                n_low_freq_vibs += 1
        if n_low_freq_vibs > 0:
            self.report(
                f"Ignoring {n_low_freq_vibs} vibrations below {self.inputs.wigner_low_freq_thr.value} cm^-1"
            )

        self.ctx.wigner_structures = generate_wigner_structures(
            self.ctx.calc_opt.outputs.relaxed_structure,
            self.ctx.calc_opt.outputs.output_parameters,
            self.inputs.nwigner,
            self.inputs.wigner_low_freq_thr,
        )

    def wigner_excite(self):
        inputs = self.exposed_inputs(
            OrcaExcitationWorkChain, namespace="exc", agglomerate=False
        )
        inputs.orca.code = self.inputs.code
        # Pass in SCF wavefunction from minimum geometry
        with self.ctx.calc_opt.outputs.retrieved.base.repository.open(
            "aiida.gbw", "rb"
        ) as handler:
            gbw_file = SinglefileData(handler)
        inputs.orca.file = {"gbw": gbw_file}
        inputs.orca.parameters = add_orca_wf_guess(inputs.orca.parameters)
        for i in self.ctx.wigner_structures.get_stepids():
            inputs.orca.structure = pick_structure_from_trajectory(
                self.ctx.wigner_structures, Int(i)
            )
            calc = self.submit(OrcaExcitationWorkChain, **inputs)
            calc.label = f"wigner-excitation-{i}"
            self.to_context(wigner_calcs=append_(calc))

    def optimize(self):
        """Optimize geometry"""
        inputs = self.exposed_inputs(
            RobustOptimizationWorkChain, namespace="opt", agglomerate=False
        )
        inputs.orca.structure = self.inputs.structure
        inputs.orca.code = self.inputs.code

        calc_opt = self.submit(RobustOptimizationWorkChain, **inputs)
        calc_opt.label = "optimization"
        return ToContext(calc_opt=calc_opt)

    def inspect_optimization(self):
        """Check whether optimization succeeded"""
        if not self.ctx.calc_opt.is_finished_ok:
            self.report("Optimization failed :-(")
            return self.exit_codes.ERROR_OPTIMIZATION_FAILED

        self.out_many(
            self.exposed_outputs(
                self.ctx.calc_opt,
                RobustOptimizationWorkChain,
                namespace="opt",
                agglomerate=False,
            )
        )

    def inspect_excitation(self):
        """Check whether excitation succeeded"""
        calc = self.ctx.calc_exc
        if not calc.is_finished_ok:
            self.report("Single point excitation failed :-(")
            return self.exit_codes.ERROR_EXCITATION_FAILED
        self.out("franck_condon_excitations", calc.outputs.excitations)

    def inspect_wigner_excitation(self):
        """Check whether all wigner excitations succeeded"""
        for calc in self.ctx.wigner_calcs:
            if not calc.is_finished_ok:
                self.report("Wigner excitation failed :-(")
                return self.exit_codes.ERROR_EXCITATION_FAILED

        all_wigner_data = [
            wc.outputs.excitations.get_dict() for wc in self.ctx.wigner_calcs
        ]
        self.out("wigner_excitations", List(all_wigner_data).store())

    def should_optimize(self):
        return self.inputs.optimize.value

    def should_run_wigner(self):
        return self.should_optimize() and self.inputs.nwigner > 0


class AtmospecWorkChain(WorkChain):
    """The top-level ATMOSPEC workchain"""

    def _build_process_label(self):
        return "ATMOSPEC workflow"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(OrcaWignerSpectrumWorkChain, exclude=["structure"])
        spec.input("structure", valid_type=TrajectoryData)

        spec.output(
            "spectrum_data",
            valid_type=List,
            required=True,
            help="All data necessary to construct spectrum in SpectrumWidget",
        )

        spec.output(
            "relaxed_structures",
            valid_type=TrajectoryData,
            required=False,
            help="Minimized structures of all conformers",
        )

        spec.outline(
            cls.launch,
            cls.collect,
        )

    def launch(self):
        inputs = self.exposed_inputs(OrcaWignerSpectrumWorkChain, agglomerate=False)
        self.report(
            f"Launching ATMOSPEC for {len(self.inputs.structure.get_stepids())} conformers"
        )
        for conf_id in self.inputs.structure.get_stepids():
            inputs.structure = self.inputs.structure.get_step_structure(conf_id)
            workflow = self.submit(OrcaWignerSpectrumWorkChain, **inputs)
            workflow.label = f"atmospec-conf-{conf_id}"
            self.to_context(confs=append_(workflow))

    def collect(self):
        for wc in self.ctx.confs:
            if not wc.is_finished_ok:
                return ExitCode(wc.exit_status, wc.exit_message)

        conf_outputs = [wc.outputs for wc in self.ctx.confs]

        # Combine all spectra data
        if self.inputs.optimize and self.inputs.nwigner > 0:
            data = {
                str(i): outputs.wigner_excitations
                for i, outputs in enumerate(conf_outputs)
            }
        else:
            data = {
                str(i): [outputs.franck_condon_excitations.get_dict()]
                for i, outputs in enumerate(conf_outputs)
            }
        all_results = run(ConcatInputsToList, ns=data)
        self.out("spectrum_data", all_results["output"])

        # Combine all optimized geometries into single TrajectoryData
        if self.inputs.optimize:
            relaxed_structures = {}
            orca_output_params = {}
            for i, outputs in enumerate(conf_outputs):
                relaxed_structures[f"struct_{i}"] = outputs.opt.relaxed_structure
                orca_output_params[f"params_{i}"] = outputs.opt.output_parameters

            # For multiple conformers, we're appending relative energies and Boltzmann weights
            array_data = None
            if len(self.ctx.confs) > 1:
                array_data = extract_trajectory_arrays(**orca_output_params)

            trajectory = structures_to_trajectory(
                arrays=array_data, **relaxed_structures
            )
            self.out("relaxed_structures", trajectory)
