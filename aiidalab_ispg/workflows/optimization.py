"""AiiDA workflows for optimization of molecules."""

from aiida.engine import WorkChain
from aiida.engine import (
    append_,
    ExitCode,
    ProcessHandlerReport,
    process_handler,
)
from aiida.plugins import WorkflowFactory, DataFactory

from .utils import structures_to_trajectory, extract_trajectory_arrays

StructureData = DataFactory("core.structure")
TrajectoryData = DataFactory("core.array.trajectory")
Code = DataFactory("core.code.installed")

OrcaBaseWorkChain = WorkflowFactory("orca.base")

__all__ = [
    "RobustOptimizationWorkChain",
    "ConformerOptimizationWorkChain",
]


# TODO: For now this is just a plain optimization,
# the "robust" part needs to be implemented
class RobustOptimizationWorkChain(OrcaBaseWorkChain):
    """Molecular geometry optimization workflow.

    The workflow automatically detects imaginary frequencies
    and restarts the optimization until a true minimum is found.
    """

    def _build_process_label(self) -> str:
        return "Robust Optimization"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.exit_code(
            401,
            "ERROR_OPTIMIZATION_FAILED",
            "optimization encountered unspecified error",
        )

    # TODO: For now we simply exit if imaginary frequencies are detected
    # NOTE: aiida-quantumespresso examples
    # https://github.com/aiidateam/aiida-quantumespresso/blob/main/src/aiida_quantumespresso/workflows/pw/base.py
    @process_handler(exit_codes=ExitCode(0), priority=600)
    def handle_imaginary_frequencies(self, calculation):
        """Check successfull optimization for imaginary frequencies."""
        frequencies = calculation.outputs.output_parameters["vibfreqs"]
        # vibdisp = calculation.outputs.output_parameters["vibdisps"]
        n_imag_freq = len(list(filter(lambda x: x <= 0, frequencies)))
        # TODO: Check that nfreq is 3N-6 or 3N-5!
        if n_imag_freq > 0:
            self.report(
                f"Found {n_imag_freq} imaginary normal mode(s). Aborting the optimization."
            )
            self.report(f"All frequencies (cm^-1): {frequencies}")
            # TODO: Displace optimized geometry along the imaginary normal modes.
            # self.ctx.inputs.orca.structure =
            # self.distort_structure(self.ctx.outputs.relaxed_structure,
            # frequencies, vibdisp)
            # Note: By default there are maximum 5 restarts in the BaseRestartWorkChain, which seems reasonable
            # return ProcessHandlerReport(do_break=True)
            return ProcessHandlerReport(
                do_break=True, exit_code=self.exit_codes.ERROR_OPTIMIZATION_FAILED
            )


class ConformerOptimizationWorkChain(WorkChain):
    """Top-level workchain for optimization of molecules in Orca.

    Essentially, this is a "thin" wrapper workchain around RobustOptimizationWorkChain
    to support optimization of multiple conformers in parallel.
    """

    def _build_process_label(self) -> str:
        return "Conformer Optimization Workflow"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(RobustOptimizationWorkChain, exclude=["orca.structure"])
        spec.input("structure", valid_type=TrajectoryData)

        spec.output(
            "relaxed_structures",
            valid_type=TrajectoryData,
            required=True,
            help="Minimized structures of all conformers",
        )

        spec.outline(
            cls.launch_conformer_optimization,
            cls.inspect_conformer_optimization,
            cls.collect_optimized_conformers,
        )

    def launch_conformer_optimization(self):
        inputs = self.exposed_inputs(RobustOptimizationWorkChain, agglomerate=False)
        nconf = len(self.inputs.structure.get_stepids())
        self.report(f"Launching optimization for {nconf} conformers")
        for conf_id in self.inputs.structure.get_stepids():
            inputs.orca.structure = self.inputs.structure.get_step_structure(conf_id)
            workflow = self.submit(RobustOptimizationWorkChain, **inputs)
            workflow.label = f"optimize-conformer-{conf_id}"
            self.to_context(confs=append_(workflow))

    def inspect_conformer_optimization(self):
        """Check whether all optimizations succeeded"""
        for wc in self.ctx.confs:
            if not wc.is_finished_ok:
                return ExitCode(wc.exit_status, wc.exit_message)

    def collect_optimized_conformers(self):
        """Combine all optimized geometries into single TrajectoryData"""
        # TODO: Switch to lists in AiiDA 2.3
        relaxed_structures = {}
        orca_output_params = {}
        for wc in self.ctx.confs:
            relaxed_structures[f"struct_{wc.pk}"] = wc.outputs.relaxed_structure
            orca_output_params[f"params_{wc.pk}"] = wc.outputs.output_parameters

        array_data = None
        if len(self.ctx.confs) > 1:
            array_data = extract_trajectory_arrays(**orca_output_params)

        trajectory = structures_to_trajectory(arrays=array_data, **relaxed_structures)
        self.out("relaxed_structures", trajectory)
