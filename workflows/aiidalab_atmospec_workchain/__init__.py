"""Base work chain to run an ORCA calculation"""

import ase
from aiida.engine import WorkChain, calcfunction
from aiida.engine import append_, ToContext, if_  # while_
from aiida.plugins import CalculationFactory, WorkflowFactory, DataFactory
from aiida.orm import to_aiida_type

from .wigner import Wigner

StructureData = DataFactory("structure")
TrajectoryData = DataFactory("array.trajectory")
Int = DataFactory("int")
Bool = DataFactory("bool")
Code = DataFactory("code")
List = DataFactory("list")
Dict = DataFactory("dict")

OrcaCalculation = CalculationFactory("orca_main")
OrcaBaseWorkChain = WorkflowFactory("orca.base")

# TODO: aiida daemon must be able to load this class,
# as a hot fix, run export PYTHONPATH=~/apps/aiidalab-dhtest:$PYTHONPATH

# TODO: Refactor this...
# We probably don't want to use calcfunction for this.
# Instead, we should probably use the get_builder_from_protocol paradigm.


@calcfunction
def pick_wigner_structure(wigner_structures, index):
    return wigner_structures.get_step_structure(index.value)


# TODO: Instead of this, we may want to do a general concatenation
# workchain, with dynamic input namespece per
# https://aiida.readthedocs.io/projects/aiida-core/en/latest/topics/processes/usage.html?highlight=dynamic%20inputs#dynamic-namespaces
@calcfunction
def concatenate_wigner_outputs(wigner_dicts):
    output_dicts = []
    for d in wigner_dicts.get_list():
        output_dicts.append(d)
    return List(list=output_dicts)


@calcfunction
def generate_wigner_structures(orca_output_dict, nsample):
    seed = orca_output_dict.extras["_aiida_hash"]

    frequencies = orca_output_dict["vibfreqs"]
    masses = orca_output_dict["atommasses"]
    normal_modes = orca_output_dict["vibdisps"]
    elements = orca_output_dict["elements"]
    min_coord = orca_output_dict["atomcoords"][-1]
    natom = orca_output_dict["natom"]
    # convert to Bohrs
    ANG2BOHRS = 1.0 / 0.529177211
    coordinates = []
    # TODO: Do the conversion in wigner.py
    # TODO: Use ASE object in wigner.py
    for iat in range(natom):
        coordinates.append(
            [
                min_coord[iat][0] * ANG2BOHRS,
                min_coord[iat][1] * ANG2BOHRS,
                min_coord[iat][2] * ANG2BOHRS,
            ]
        )

    w = Wigner(elements, masses, coordinates, frequencies, normal_modes, seed)

    wigner_list = []
    for i in range(nsample.value):
        wigner_coord = w.get_sample()
        # Convert to angstroms
        wigner_coord_ang = []
        for iat in range(natom):
            wigner_coord_ang.append(
                [
                    wigner_coord[iat][0] / ANG2BOHRS,
                    wigner_coord[iat][1] / ANG2BOHRS,
                    wigner_coord[iat][2] / ANG2BOHRS,
                ]
            )
        # TODO: We shouldn't need to specify cell
        # https://github.com/aiidateam/aiida-core/issues/5248
        ase_struct = ase.Atoms(
            positions=wigner_coord_ang,
            symbols=elements,
            cell=(1.0, 1.0, 1.0),
            pbc=False,
        )
        wigner_list.append(StructureData(ase=ase_struct))

    return TrajectoryData(structurelist=wigner_list)


class OrcaWignerSpectrumWorkChain(WorkChain):
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
        spec.input(
            "optimize",
            valid_type=Bool,
            default=lambda: Bool(True),
            serializer=to_aiida_type,
        )
        # Number of Wigner geometries (computed only when optimize==True)
        spec.input(
            "nwigner", valid_type=Int, default=lambda: Int(2), serializer=to_aiida_type
        )

        spec.output("relaxed_structure", valid_type=StructureData, required=False)
        spec.output(
            "single_point_tddft",
            valid_type=Dict,
            required=True,
            help="Output parameters from a single-point TDDFT calculation",
        )

        spec.output(
            "wigner_tddft",
            valid_type=List,
            # valid_type=Dict,
            required=False,
            help="Output parameters from all Wigner TDDFT calculation",
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
            if_(cls.should_run_wigner)(
                cls.wigner_sampling,
                cls.wigner_excite,
                cls.inspect_wigner_excitation,
            ),
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
        calc_exc.label = "single-point-tddft"
        return ToContext(calc_exc=calc_exc)

    def wigner_sampling(self):
        """Calculate excited states for a given geometry"""
        # Either take optimized structure or input structure here
        # structure = self.ctx.structure
        self.report(f"Generating {self.inputs.nwigner.value} Wigner geometries")
        self.ctx.wigner_structures = generate_wigner_structures(
            self.ctx.calc_opt.outputs.output_parameters, self.inputs.nwigner
        )

    def wigner_excite(self):
        inputs = self.exposed_inputs(
            OrcaBaseWorkChain, namespace="exc", agglomerate=False
        )
        inputs.orca.code = self.inputs.code
        # TODO: Calculate all Wigner structures
        traj = self.ctx.wigner_structures
        for i in traj.get_stepids():
            inputs.orca.structure = pick_wigner_structure(traj, Int(i))
            calc = self.submit(OrcaBaseWorkChain, **inputs)
            calc.label = "wigner-single-point-tddft"
            self.to_context(wigner_calcs=append_(calc))

    def optimize(self):
        """Optimize geometry"""
        inputs = self.exposed_inputs(
            OrcaBaseWorkChain, namespace="opt", agglomerate=False
        )
        inputs.orca.structure = self.inputs.structure
        inputs.orca.code = self.inputs.code

        calc_opt = self.submit(OrcaBaseWorkChain, **inputs)
        return ToContext(calc_opt=calc_opt)

    def inspect_optimization(self):
        """Check whether optimization succeeded"""
        # Not sure what to do here, maybe we should have an error handler instead?
        if not self.ctx.calc_opt.is_finished_ok:
            self.report("Optimization failed :-(")
            return self.exit_codes.ERROR_OPTIMIZATION_FAILED

    def inspect_excitation(self):
        """Check whether excitation succeeded"""
        if not self.ctx.calc_exc.is_finished_ok:
            self.report("Excitation failed :-(")
            return self.exit_codes.ERROR_EXCITATION_FAILED

    def inspect_wigner_excitation(self):
        """Check whether all wigner excitations succeeded"""
        for calc in self.ctx.wigner_calcs:
            if not calc.is_finished_ok:
                # TODO: Report all failed calcs at once
                self.report("Wigner excitation failed :-(")
                return self.exit_codes.ERROR_EXCITATION_FAILED

        self.report("Concatenating Wigner outputs")
        # TODO: Figure out how to do this properly
        output_dicts = []
        for calc in self.ctx.wigner_calcs:
            output_dicts.append(calc.outputs.output_parameters.get_dict())
        self.ctx.wigner_outputs = concatenate_wigner_outputs(List(list=output_dicts))

    def should_optimize(self):
        if self.inputs.optimize:
            return True
        return False

    def should_run_wigner(self):
        return self.should_optimize() and self.inputs.nwigner > 0

    def results(self):
        """Expose results from child workchains"""

        if self.inputs.optimize:
            # Since we're not currently using namespace,
            # cannot use out_many, since we only want to take
            # relaxed_structure, and not output_parameters
            self.out("relaxed_structure", self.ctx.calc_opt.outputs.relaxed_structure)

        if self.inputs.optimize and self.inputs.nwigner > 0:
            self.out("wigner_tddft", self.ctx.wigner_outputs)

        self.out("single_point_tddft", self.ctx.calc_exc.outputs.output_parameters)
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

__version__ = "1.0"
