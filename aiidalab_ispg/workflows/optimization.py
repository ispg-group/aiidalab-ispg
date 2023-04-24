# AiiDA workflows dealing with optimization of molecules.

import math
import numpy as np

from aiida.engine import WorkChain, calcfunction
from aiida.engine import (
    append_,
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

AUtoEV = 27.2114386245
AUtoKCAL = 627.04
KCALtoKJ = 4.183
AUtoKJ = AUtoKCAL * KCALtoKJ
EVtoKJ = AUtoKCAL * KCALtoKJ / AUtoEV

# TODO: Switch to variadic arguments (supported since AiiDA 2.3)
@calcfunction
def structures_to_trajectory(arrays: Array = None, **structures) -> TrajectoryData:
    """Concatenate a list of StructureData to TrajectoryData

    Optionally, set additional data as Arrays.
    """
    traj = TrajectoryData(list(structures.values()))
    if arrays is None:
        return traj

    for name in arrays.get_arraynames():
        traj.set_array(name, arrays.get_array(name))
    # Copy over extras as well, except for private ones like '_aiida_hash'
    extras = arrays.base.extras.all
    for key in list(extras.keys()):
        if key.startswith("_"):
            del extras[key]
    traj.base.extras.set_many(extras)
    return traj


def calc_boltzmann_weights(energies: list, T: float):
    """Compute Boltzmann weights for a list of energies.

    param energies: list of energies / kJ per mole
    param T: temperature / Kelvin
    returns: Boltzmann weights as numpy array
    """
    # Molar gas constant, Avogadro times Boltzmann
    R = 8.3144598
    RT = R * T
    E0 = min(energies)
    weights = [math.exp(-(1000 * (E - E0)) / RT) for E in energies]
    Q = sum(weights)
    return np.array([weight / Q for weight in weights])


@calcfunction
def extract_trajectory_arrays(**orca_output_parameters) -> Array:
    """Extract Gibbs energies and other useful stuff from the list
    of ORCA output parameter dictionaries.

    Return Array node, which will be appended to TrajectoryData node.
    """
    gibbs_energies = np.array(
        [params["freeenergy"] for params in orca_output_parameters.values()]
    )
    en0 = min(gibbs_energies)
    relative_gibbs_energies_kj = AUtoKJ * (gibbs_energies - en0)

    temperature = list(orca_output_parameters.values())[0]["temperature"]

    boltzmann_weights = calc_boltzmann_weights(relative_gibbs_energies_kj, temperature)

    en = Array()
    en.set_array("gibbs_energies_au", gibbs_energies)
    en.set_array("relative_gibbs_energies_kj", relative_gibbs_energies_kj)
    en.set_array("boltzmann_weights", boltzmann_weights)
    en.set_extra("temperature", temperature)

    # For the TrajectoryData viewer compatibility
    en.set_array("energies", relative_gibbs_energies_kj)
    en.set_extra("energy_units", "kJ/mole")
    return en


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
    """Top-level workchain for optimization of molecules in Orca.

    Essentially, this is a thin wrapper workchain around RobustOptimizationWorkChain
    to support optimization of multiple conformers in parallel.
    """

    def _build_process_label(self) -> str:
        return "Conformer Optimization Workflow"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(RobustOptimizationWorkChain, exclude=["orca.structure"])
        # TODO: Rename to trajectory?
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

        # Very generic error now
        spec.exit_code(300, "CONFORMER_ERROR", "Conformer optimization failed")

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
        # TODO: Specialize errors. Can we expose errors from child workflows?
        for wc in self.ctx.confs:
            if not wc.is_finished_ok:
                return self.exit_codes.CONFORMER_ERROR

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
