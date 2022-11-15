"""Base work chain to run an ORCA calculation"""

from aiida.engine import WorkChain, calcfunction
from aiida.engine import append_, ToContext, if_

# not sure if this is needed? Can we use self.run()?
from aiida.engine import run
from aiida.plugins import CalculationFactory, WorkflowFactory, DataFactory
from aiida.orm import to_aiida_type

from .wigner import Wigner

StructureData = DataFactory("structure")
TrajectoryData = DataFactory("array.trajectory")
SinglefileData = DataFactory("singlefile")
Int = DataFactory("int")
Float = DataFactory("float")
Bool = DataFactory("bool")
Code = DataFactory("code")
List = DataFactory("list")
Dict = DataFactory("dict")

OrcaCalculation = CalculationFactory("orca.orca")
OrcaBaseWorkChain = WorkflowFactory("orca.base")


# Meta WorkChain for combining all inputs from a dynamic namespace into List.
# Used to combine outputs from several subworkflows into one output.
# It should be launched via run() instead of submit()
# NOTE: The code has special handling for Dict nodes,
# which otherwise fail with not being serializable,
# so we need the get the value with Dict.get_dict() first.
# We should check whether this is still needed in aiida-2.0
# Note we cannot make this more general since List and Dict
# don't have the .value attribute.
# https://github.com/aiidateam/aiida-core/issues/5313
class ConcatInputsToList(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input_namespace("ns", dynamic=True)
        spec.output("output", valid_type=List)
        spec.outline(cls.combine)

    def combine(self):
        input_list = [
            self.inputs.ns[k].get_dict()
            if isinstance(self.inputs.ns[k], Dict)
            else self.inputs.ns[k]
            for k in self.inputs.ns
        ]
        self.out("output", List(list=input_list).store())


# TODO: Allow optional inputs for array data to store energies
class ConcatStructuresToTrajectory(WorkChain):
    """WorkChain for combining a list of StructureData into TrajectoryData"""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input_namespace("structures", dynamic=True, valid_type=StructureData)
        spec.output("trajectory", valid_type=TrajectoryData)
        spec.outline(cls.combine)

    def combine(self):
        structurelist = [self.inputs.structures[k] for k in self.inputs.structures]
        self.out("trajectory", TrajectoryData(structurelist=structurelist).store())


@calcfunction
def pick_wigner_structure(wigner_structures, index):
    return wigner_structures.get_step_structure(index.value)


@calcfunction
def add_orca_wf_guess(orca_params: Dict) -> Dict:
    params = orca_params.get_dict()
    params["input_keywords"].append("MOREAD")
    params["input_blocks"]["scf"]["moinp"] = '"aiida_old.gbw"'
    return Dict(dict=params)


@calcfunction
def generate_wigner_structures(
    minimum_structure, orca_output_dict, nsample, low_freq_thr
):
    seed = orca_output_dict.extras["_aiida_hash"]
    ase_molecule = minimum_structure.get_ase()
    frequencies = orca_output_dict["vibfreqs"]
    normal_modes = orca_output_dict["vibdisps"]

    wigner = Wigner(
        ase_molecule,
        frequencies,
        normal_modes,
        seed=seed,
        low_freq_thr=low_freq_thr.value,
    )

    wigner_list = [
        StructureData(ase=wigner.get_ase_sample()) for i in range(nsample.value)
    ]
    return TrajectoryData(structurelist=wigner_list)


class OrcaWignerSpectrumWorkChain(WorkChain):
    """Top level workchain for Nuclear Ensemble Approach UV/vis
    spectrum for a single conformer"""

    def _build_process_label(self):
        return "NEA spectrum workflow"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(
            OrcaBaseWorkChain, namespace="opt", exclude=["orca.structure", "orca.code"]
        )
        spec.expose_inputs(
            OrcaBaseWorkChain, namespace="exc", exclude=["orca.structure", "orca.code"]
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

        spec.output("relaxed_structure", valid_type=StructureData, required=False)
        spec.output(
            "single_point_tddft",
            valid_type=Dict,
            required=True,
            help="Output parameters from a single-point TDDFT calculation",
        )

        # TODO: Rename this port
        spec.output(
            "wigner_tddft",
            valid_type=List,
            required=False,
            help="Output parameters from all Wigner TDDFT calculation",
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
            cls.results,
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
            OrcaBaseWorkChain, namespace="exc", agglomerate=False
        )
        inputs.orca.code = self.inputs.code

        if self.inputs.optimize:
            self.report("Calculating spectrum for optimized geometry")
            inputs.orca.structure = self.ctx.calc_opt.outputs.relaxed_structure

            # Pass in converged SCF wavefunction
            with self.ctx.calc_opt.outputs.retrieved.open("aiida.gbw", "rb") as handler:
                gbw_file = SinglefileData(handler)
            inputs.orca.file = {"gbw": gbw_file}
            inputs.orca.parameters = add_orca_wf_guess(inputs.orca.parameters)
        else:
            self.report("Calculating spectrum for input geometry")
            inputs.orca.structure = self.inputs.structure

        calc_exc = self.submit(OrcaBaseWorkChain, **inputs)
        calc_exc.label = "single-point-excitation"
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
            OrcaBaseWorkChain, namespace="exc", agglomerate=False
        )
        inputs.orca.code = self.inputs.code
        # Pass in SCF wavefunction from minimum geometry
        with self.ctx.calc_opt.outputs.retrieved.open("aiida.gbw", "rb") as handler:
            gbw_file = SinglefileData(handler)
        inputs.orca.file = {"gbw": gbw_file}
        inputs.orca.parameters = add_orca_wf_guess(inputs.orca.parameters)
        for i in self.ctx.wigner_structures.get_stepids():
            inputs.orca.structure = pick_wigner_structure(
                self.ctx.wigner_structures, Int(i)
            )
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
        calc_opt.label = "optimization"
        return ToContext(calc_opt=calc_opt)

    def inspect_optimization(self):
        """Check whether optimization succeeded"""
        if not self.ctx.calc_opt.is_finished_ok:
            self.report("Optimization failed :-(")
            return self.exit_codes.ERROR_OPTIMIZATION_FAILED

    def inspect_excitation(self):
        """Check whether excitation succeeded"""
        if not self.ctx.calc_exc.is_finished_ok:
            self.report("Single point excitation failed :-(")
            return self.exit_codes.ERROR_EXCITATION_FAILED

    def inspect_wigner_excitation(self):
        """Check whether all wigner excitations succeeded"""
        for calc in self.ctx.wigner_calcs:
            if not calc.is_finished_ok:
                # TODO: Report all failed calcs at once
                self.report("Wigner excitation failed :-(")
                return self.exit_codes.ERROR_EXCITATION_FAILED

    def should_optimize(self):
        return self.inputs.optimize.value

    def should_run_wigner(self):
        return self.should_optimize() and self.inputs.nwigner > 0

    def results(self):
        """Expose results from child workchains"""

        if self.should_optimize():
            self.out("relaxed_structure", self.ctx.calc_opt.outputs.relaxed_structure)

        if self.should_run_wigner():
            self.report("Concatenating Wigner outputs")
            # TODO: Instead of deepcopying all dicts,
            # only pick the data that we need for the spectrum to save space.
            # We should introduce a special aiida type for spectrum data
            data = {
                str(i): wc.outputs.output_parameters
                for i, wc in enumerate(self.ctx.wigner_calcs)
            }
            all_results = run(ConcatInputsToList, ns=data)
            self.out("wigner_tddft", all_results["output"])

        self.out("single_point_tddft", self.ctx.calc_exc.outputs.output_parameters)


class AtmospecWorkChain(WorkChain):
    """The top-level ATMOSPEC workchain"""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(OrcaWignerSpectrumWorkChain, exclude=["structure"])
        spec.input("structure", valid_type=(StructureData, TrajectoryData))

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

        # Very generic error now
        spec.exit_code(410, "CONFORMER_ERROR", "Conformer spectrum generation failed")

    def launch(self):
        inputs = self.exposed_inputs(OrcaWignerSpectrumWorkChain, agglomerate=False)
        # Single conformer
        # TODO: Test this!
        if isinstance(self.inputs.structure, StructureData):
            self.report("Launching ATMOSPEC for 1 conformer")
            inputs.structure = self.inputs.structure
            return ToContext(conf=self.submit(OrcaWignerSpectrumWorkChain, **inputs))

        self.report(
            f"Launching ATMOSPEC for {len(self.inputs.structure.get_stepids())} conformers"
        )
        for conf_id in self.inputs.structure.get_stepids():
            inputs.structure = self.inputs.structure.get_step_structure(conf_id)
            workflow = self.submit(OrcaWignerSpectrumWorkChain, **inputs)
            # workflow.label = 'conformer-wigner-spectrum'
            self.to_context(confs=append_(workflow))

    def collect(self):
        # For single conformer
        # TODO: This currently does not work
        if isinstance(self.inputs.structure, StructureData):
            if not self.ctx.conf.is_finished_ok:
                return self.exit_codes.CONFORMER_ERROR
            self.out_many(
                self.exposed_outputs(self.ctx.conf, OrcaWignerSpectrumWorkChain)
            )
            return

        # Check for errors
        # TODO: Specialize errors. Can we expose errors from child workflows?
        for wc in self.ctx.confs:
            if not wc.is_finished_ok:
                return self.exit_codes.CONFORMER_ERROR

        # Combine all spectra data
        # NOTE: This if duplicates the logic of OrcaWignerSpectrumWorkChain.should_run_wigner()
        if self.inputs.optimize and self.inputs.nwigner > 0:
            data = {
                str(i): wc.outputs.wigner_tddft for i, wc in enumerate(self.ctx.confs)
            }
        else:
            # TODO: We should have a separate output for single-point spectra
            data = {
                str(i): [wc.outputs.single_point_tddft.get_dict()]
                for i, wc in enumerate(self.ctx.confs)
            }
        all_results = run(ConcatInputsToList, ns=data)
        self.out("spectrum_data", all_results["output"])

        # Combine all optimized geometries into single TrajectoryData
        # TODO: Include energies in TrajectoryData for optimized structures
        if self.inputs.optimize:
            relaxed_structures = {
                str(i): wc.outputs.relaxed_structure
                for i, wc in enumerate(self.ctx.confs)
            }
            output = run(ConcatStructuresToTrajectory, structures=relaxed_structures)
            self.out("relaxed_structures", output["trajectory"])


__version__ = "0.1-alpha"
