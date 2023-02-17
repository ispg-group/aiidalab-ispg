# AiiDA workflows dealing with optimization of molecules.

from aiida.engine import WorkChain, calcfunction
from aiida.engine import (
    append_,
    ToContext,
    ExitCode,
    ProcessHandlerReport,
    process_handler,
)
from aiida.plugins import WorkflowFactory, DataFactory

StructureData = DataFactory("core.structure")
TrajectoryData = DataFactory("core.array.trajectory")
Array = DataFactory("core.array")
Code = DataFactory("core.code.installed")

OrcaBaseWorkChain = WorkflowFactory("orca.base")


# TODO: Switch to variadic arguments (supported since AiiDA 2.3)
@calcfunction
def structures_to_trajectory(arrays: Array = None, **structures) -> TrajectoryData:
    """Concatenate a list of StructureData to TrajectoryData
    Optionally, set additional data as Arrays.
    """
    traj = TrajectoryData([structure for structure in structures.values()])
    if arrays is not None:
        for name in arrays.get_arraynames():
            traj.set_array(name, arrays.get_array(name))
    return traj


def extract_energies(**orca_output_parameters) -> Array:
    """Extract gibbs energies and other useful stuff from the list
       of ORCA output parameters.
    Optionally, set additional data as Arrays.
    """
    gibbs_energies = [params["freeenergy"] for params in orca_output_parameters]
    en = Array()
    en.set_array("gibs_energy_au", gibs_energies)
    en.set_extra("temperature", orca_output_parameters[0]["temperature"])
    return en


# TODO: For now this is just a plain optimization,
# the "robust" part needs to be implemented
class RobustOptimizationWorkChain(OrcaBaseWorkChain):
    """Molecular geometry optimization WorkChain that automatically
    detects imaginary frequencies and restarts the optimization
    until a true minimum is found.
    """

    def _build_process_label(self):
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
        vibrational_displacements = calculation.outputs.output_parameters["vibdisps"]
        n_imag_freq = len(list(filter(lambda x: x <= 0, frequencies)))
        # TODO: Check that nfreq is 3N-6 or 3N-5!
        if n_imag_freq > 0:
            self.report(
                f"Found {n_imag_freq} imaginary normal mode(s). Aborting the optimization."
            )
            self.report(f"All frequencies (cm^-1): {frequencies}")
            # TODO: Displace optimized geometry along the imaginary normal modes.
            # self.ctx.inputs.orca.structure = self.distort_structure(self.ctx.outputs.relaxed_structure, frequencies, vibrational_displacements)
            # Note: By default there are maximum 5 restarts in the BaseRestartWorkChain, which seems reasonable
            # return ProcessHandlerReport(do_break=True)
            return ProcessHandlerReport(
                do_break=True, exit_code=self.exit_codes.ERROR_OPTIMIZATION_FAILED
            )


class ConformerOptimizationWorkChain(WorkChain):
    """Top-level workchain for optimization of molecules.

    Essentially, this is a thin wrapper workchain around RobustOptimizationWorkChain
    to support optimization of multiple conformers in parallel.
    """

    def _build_process_label(self):
        return "Conformer Optimization Workflow"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(RobustOptimizationWorkChain, exclude=["orca.structure"])
        spec.input("structure", valid_type=(StructureData, TrajectoryData))

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

        # Very generic error now
        spec.exit_code(300, "CONFORMER_ERROR", "Conformer optimization failed")

    def launch_conformer_optimization(self):
        inputs = self.exposed_inputs(RobustOptimizationWorkChain, agglomerate=False)
        # Single conformer
        # TODO: Test this!
        if isinstance(self.inputs.structure, StructureData):
            self.report("Launching Optimization for 1 conformer")
            inputs.orca.structure = self.inputs.structure
            return ToContext(confs=self.submit(RobustOptimizationWorkChain, **inputs))

        nconf = len(self.inputs.structure.get_stepids())
        self.report(f"Launching optimization for {nconf} conformers")
        for conf_id in self.inputs.structure.get_stepids():
            inputs.orca.structure = self.inputs.structure.get_step_structure(conf_id)
            workflow = self.submit(RobustOptimizationWorkChain, **inputs)
            workflow.label = f"optimize-conformer-{conf_id}"
            self.to_context(confs=append_(workflow))

    def inspect_conformer_optimization(self):
        """Check whether all optimizations succeeded"""
        # TODO: Specialize errors. Can we expose errors from child workflows?
        if isinstance(self.inputs.structure, StructureData):
            if not self.ctx.confs.is_finished_ok:
                return self.exit_codes.CONFORMER_ERROR
            return
        for wc in self.ctx.confs:
            if not wc.is_finished_ok:
                return self.exit_codes.CONFORMER_ERROR

    def collect_optimized_conformers(self):
        """Combine all optimized geometries into single TrajectoryData"""
        # TODO: Include energies in TrajectoryData for optimized structures
        # TODO: Calculate Boltzmann weights and append them to TrajectoryData
        if isinstance(self.inputs.structure, StructureData):
            relaxed_structures = {"struct_0": self.ctx.confs.outputs.relaxed_structure}
        else:
            relaxed_structures = {
                f"struct_{i}": wc.outputs.relaxed_structure
                for i, wc in enumerate(self.ctx.confs)
            }
        # TODO: We should preserve the stepids from the input TrajectoryData
        trajectory = structures_to_trajectory(**relaxed_structures)
        self.out("relaxed_structures", trajectory)
