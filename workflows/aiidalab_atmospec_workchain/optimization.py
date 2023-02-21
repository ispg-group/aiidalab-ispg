# AiiDA workflows dealing with optimization of molecules.

from aiida.engine import WorkChain, calcfunction
from aiida.engine import append_, ToContext

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


# TODO: For now this is just a plain optimization,
# the "robust" part needs to be implemented
class RobustOptimizationWorkChain(WorkChain):
    """Molecular geometry optimization WorkChain that automatically
    detects imaginary frequencies and restarts the optimization
    until a true minimum is found.
    """

    def _build_process_label(self):
        return "Robust Optimization"

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_inputs(OrcaBaseWorkChain, exclude=["orca.structure", "orca.code"])
        spec.input("structure", valid_type=StructureData)
        spec.input("code", valid_type=Code)

        spec.outline(
            cls.optimize,
            cls.inspect_optimization,
            cls.results,
        )

        spec.expose_outputs(OrcaBaseWorkChain)

        spec.exit_code(
            401,
            "ERROR_OPTIMIZATION_FAILED",
            "optimization encountered unspecified error",
        )

    def optimize(self):
        """Optimize molecular geometry"""
        inputs = self.exposed_inputs(OrcaBaseWorkChain, agglomerate=False)
        inputs.orca.structure = self.inputs.structure
        inputs.orca.code = self.inputs.code

        calc_opt = self.submit(OrcaBaseWorkChain, **inputs)
        calc_opt.label = "robust-optimization"
        return ToContext(calc_opt=calc_opt)

    def inspect_optimization(self):
        """Check whether optimization succeeded"""
        if not self.ctx.calc_opt.is_finished_ok:
            return self.exit_codes.ERROR_OPTIMIZATION_FAILED

    def results(self):
        self.out_many(self.exposed_outputs(self.ctx.calc_opt, OrcaBaseWorkChain))


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
        spec.expose_inputs(RobustOptimizationWorkChain, exclude=["structure"])
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
            inputs.structure = self.inputs.structure
            return ToContext(confs=self.submit(RobustOptimizationWorkChain, **inputs))

        nconf = len(self.inputs.structure.get_stepids())
        self.report(f"Launching optimization for {nconf} conformers")
        for conf_id in self.inputs.structure.get_stepids():
            inputs.structure = self.inputs.structure.get_step_structure(conf_id)
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
