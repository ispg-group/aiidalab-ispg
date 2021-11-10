"""Base work chain to run an ORCA calculation"""

from aiida.engine import WorkChain, calcfunction
from aiida.engine import ToContext, if_  # while_
from aiida.plugins import CalculationFactory, WorkflowFactory, DataFactory

OrcaCalculation = CalculationFactory("orca_main")
OrcaBaseWorkChain = WorkflowFactory("orca.base")
StructureData = DataFactory("structure")
Bool = DataFactory("bool")
Code = DataFactory("code")
List = DataFactory("list")
Dict = DataFactory("dict")

# TODO: aiida daemon must be able to load this class,
# as a hot fix, run export PYTHONPATH=~/apps/aiidalab-dhtest:$PYTHONPATH

# TODO: Refactor this...
# We probably don't want to use calcfunction for this.
# Instead, we should probably use the get_builder_from_protocol paradigm.


@calcfunction
def append_input_keywords(orca_parameters, input_keywords):
    new_params = orca_parameters.clone()
    for keyword in input_keywords:
        new_params["input_keywords"].append(keyword)
    return new_params


class OrcaRelaxAndTDDFTWorkChain(WorkChain):
    """Basic workchain for single point TDDFT on optimized geometry"""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(
            OrcaBaseWorkChain, namespace="opt", exclude=["orca.structure", "orca.code"]
        )
        spec.expose_inputs(
            OrcaBaseWorkChain, namespace="exc", exclude=["orca.structure", "orca.code"]
        )
        spec.input("structure", valid_type=StructureData)
        spec.input("code", valid_type=Code)

        # Whether to perform geometry optimization or not
        spec.input("optimize", valid_type=Bool, default=lambda: Bool(True))

        spec.output("relaxed_structure", valid_type=StructureData, required=False)
        spec.output(
            "output_parameters",
            valid_type=Dict,
            required=True,
            help="Output parameters from TDDFT calculation",
        )

        spec.outline(
            cls.setup,
            # WARNING: Be mega careful with while_!
            # Possibility of infinite loop is nasty here.
            # while_(cls.should_optimize) (
            if_(cls.should_optimize)(
                cls.optimize,
                cls.inspect_optimization,
            ),
            cls.excite,
            cls.inspect_excitation,
            cls.results,
        )
        # spec.expose_outputs(
        #        OrcaBaseWorkChain,
        #        include=['output_parameters']
        #        )
        # spec.expose_outputs(
        #        OrcaBaseWorkChain,
        #        include=['relaxed_structure']
        #        )
        # Cannot use namespace here due to bug in ProcessNodesTreeWidget
        #        namespace='exc')

        spec.exit_code(
            401,
            "ERROR_OPTIMIZATION_FAILED",
            "optimization encountered unspecified error",
        )
        spec.exit_code(
            402, "ERROR_EXCITATION_FAILED", "excited state calculation failed"
        )

    def setup(self):
        """Setup workchain"""
        self.report("Hello")
        self.ctx.optimize_iter = 0
        # TODO: This should be base on some input parameter
        self.ctx.nstates = 3

    def excite(self):
        """Calculate excited states for a given geometry"""
        # Either take optimized structure or input structure here
        # structure = self.ctx.structure
        self.report(f"Will calculate {self.ctx.nstates} excited states")
        inputs = self.exposed_inputs(
            OrcaBaseWorkChain, namespace="exc", agglomerate=False
        )
        if self.inputs.optimize:
            inputs.orca.structure = self.ctx.calc_opt.outputs.relaxed_structure
        else:
            inputs.orca.structure = self.inputs.structure
        inputs.orca.code = self.inputs.code
        calc_exc = self.submit(OrcaBaseWorkChain, **inputs)
        return ToContext(calc_exc=calc_exc)

    def optimize(self):
        """Optimize geometry"""
        self.ctx.optimize_iter += 1
        self.report("Hello")
        inputs = self.exposed_inputs(
            OrcaBaseWorkChain, namespace="opt", agglomerate=False
        )
        inputs.orca.structure = self.inputs.structure
        inputs.orca.code = self.inputs.code

        input_keywords = inputs.orca.parameters["input_keywords"]
        if "opt" not in map(lambda s: s.lower(), input_keywords):
            inputs.orca.parameters = append_input_keywords(
                inputs.orca.parameters, List(list=["Opt"]).store()
            )

        calc_opt = self.submit(OrcaBaseWorkChain, **inputs)
        return ToContext(calc_opt=calc_opt)

    def inspect_optimization(self):
        """Check whether optimization succeeded"""
        self.report("Hello!")
        # Not sure what to do here, maybe we should have an error handler instead?
        if not self.ctx.calc_opt.is_finished_ok:
            self.report("Optimization failed :-(")
            return self.exit_codes.ERROR_OPTIMIZATION_FAILED

    def inspect_excitation(self):
        """Check whether excitation succeeded"""
        self.report("Hello!")
        if not self.ctx.calc_exc.is_finished_ok:
            self.report("Excitation failed :-(")
            return self.exit_codes.ERROR_EXCITATION_FAILED

    def should_optimize(self):
        self.report("Hello")
        # To prevent accidental infinite loop
        if self.ctx.optimize_iter > 10:
            return False
        if self.inputs.optimize:
            return True
        return False

    def results(self):
        """Expose results from child workchains"""
        self.report("Hello")
        # TODO: Think what the output nodes should be
        # Probably "RelaxedStructure" if should_optimize == True
        # Transitions (which format?)

        if self.inputs.optimize:
            self.report(self.ctx.calc_exc)
            # Since we're not currently using namespace,
            # cannot use out_many, since we only want to take
            # relaxed_structure, and not output_parameters
            self.out("relaxed_structure", self.ctx.calc_opt.outputs.relaxed_structure)
            # self.out_many(
            #    self.exposed_outputs(
            #        self.ctx.calc_opt,
            #        OrcaBaseWorkChain,
            # namespace='exc',
            #        agglomerate=True
            #    )
            # )

        self.out("output_parameters", self.ctx.calc_exc.outputs.output_parameters)
        # self.out_many(
        #     self.exposed_outputs(
        #         self.ctx.calc_exc,
        #         OrcaBaseWorkChain,
        # namespace='exc',
        #         agglomerate=True
        #     )
        # )


# TODO:
class OrcaRobustRelaxWorkchain(WorkChain):
    """Minimization of molecular geometry in ORCA
    including the frequency calculation at the end.
    Imaginary frequencies are automatically handled by shifting
    the geometry along the imaginary mode and rerunning the optimization"""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(OrcaCalculation, namespace="orca")

    def setup(self):
        pass


# TODO:
class OrcaTddftWorkchain(WorkChain):
    """Single point TDDFT calculation"""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(OrcaBaseWorkChain, namespace="orca")
        spec.outline(
            cls.setup,
            cls.excite,
            cls.inspect_excitation,
            cls.results,
        )

    def setup(self):
        pass
