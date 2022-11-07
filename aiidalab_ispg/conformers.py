"""Widgets for the conformer generation usign RDKit.

Inspired by:
    https://rdkit.org/UGM/2012/Ebejer_20110926_RDKit_1stUGM.pdf
    https://doi.org/10.1021/ci2004658

RDKit documentation
https://www.rdkit.org/docs/RDKit_Book.html#conformer-generation
API reference
https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html?highlight=embedmultipleconfs#rdkit.Chem.rdDistGeom.EmbedMultipleConfs

https://sourceforge.net/p/rdkit/mailman/rdkit-discuss/thread/CWLP265MB0818A57240D003F146E910798C680%40CWLP265MB0818.GBRP265.PROD.OUTLOOK.COM/#msg36584689

Authors:
    * Daniel Hollas <daniel.hollas@durham.ac.uk>
"""

import numpy as np
from traitlets import Union, Instance

from ase import Atoms
from ase.optimize import GPMin  # BFGS
from rdkit import Chem
from rdkit.Chem import AllChem

from aiida.plugins import DataFactory
from aiidalab_widgets_base import SmilesWidget

from .utils import argsort

StructureData = DataFactory("structure")
TrajectoryData = DataFactory("array.trajectory")

# xTB cannot be installed automatically in official AiiDAlab Docker images,
# because the dependencies are installed via pip,
# but xtb-python package is only available on Conda.
# Hence, the xTB optimization functionality is optional for now.
DISABLE_XTB = False
try:
    from xtb.ase.calculator import XTB
except ImportError:
    DISABLE_XTB = True

# TODO: For now we always disable xTB optimization, until we extract it
# to a separate editor where it can be triggered by user optionally.
# https://github.com/ispg-group/aiidalab-ispg/issues/35
# Also, we need to make it a lot faster, see:
# https://github.com/ispg-group/aiidalab-ispg/issues/12
DISABLE_XTB = True


class ConformerSmilesWidget(SmilesWidget):

    structure = Union(
        [Instance(Atoms), Instance(StructureData), Instance(TrajectoryData)],
        allow_none=True,
    )
    _ENERGY_UNITS = "kJ/mole"
    # TODO: Select this threshold, and take care of units!
    _ENERGY_THR = 1e-8

    def _mol_from_smiles(self, smiles, steps=1000):
        """Convert SMILES to ase structure try rdkit then pybel"""
        conformers = None

        # Canonicalize the SMILES code
        # https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system#Terminology
        canonical_smiles = self.canonicalize_smiles(smiles)
        if not canonical_smiles:
            return None

        if canonical_smiles != smiles:
            self.output.value = f"Canonical SMILES: {canonical_smiles}"

        # TODO: Make a dropdown menu for algorithm selection
        rdkit_algorithm = "ETKDGv2"
        optimization_algorithm = "MMFF94"
        try:
            conformers, energies = self._rdkit_opt(
                canonical_smiles,
                steps,
                algo=rdkit_algorithm,
                opt_algo=optimization_algorithm,
            )
            if not conformers:
                return None
        except ValueError as e:
            self.output.value = str(e)
            return None

        # Fallback if XTB is not available
        if not DISABLE_XTB:
            conformers, energies = self.optimize_conformers_with_xtb(conformers)

        conformers, energies = self._filter_and_sort_conformers(conformers, energies)
        return self._create_trajectory_node(conformers, energies)

    def canonicalize_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            # Something is seriously wrong with the SMILES code,
            # just return None and don't attempt anything else.
            self.output.value = "RDkit ERROR: Invalid SMILES string"
            return None
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        if not canonical_smiles:
            self.output.value = "RDkit ERROR: Could not canonicalize SMILES"
            return None
        return canonical_smiles

    # TODO: Adjust mux number of steps and relax convergence criteria
    # fmax - maximum force per atom for convergence (0.05 default in ASE)
    # maxstep - maximum atom displacement per iteration (angstrom, 0.04 ASE default)
    def _xtb_opt(self, atoms, xtb_method="GFN2-xTB", max_steps=50, fmax=0.04):
        # https://wiki.fysik.dtu.dk/ase/gettingstarted/tut02_h2o_structure/h2o.html
        # https://xtb-python.readthedocs.io/en/latest/general-api.html
        if not xtb_method:
            return atoms

        atoms.calc = XTB(method=xtb_method)
        # opt = BFGS(atoms, maxstep=0.06, trajectory=None, logfile=None)
        opt = GPMin(atoms, trajectory=None, logfile=None)
        converged = opt.run(steps=max_steps, fmax=fmax)
        if converged:
            print(
                f"{xtb_method} minimization converged in {opt.get_number_of_steps()} iterations"
            )
        else:
            print(
                f"{xtb_method} minimization failed to converged in {opt.get_number_of_steps()} iterations"
            )
        return atoms

    def optimize_conformers_with_xtb(self, conformers):
        """Conformer optimization with XTB"""
        # method = "GFN2-xTB"
        # method = "GFNFF"
        method = "GFN2-xTB"
        max_steps = 5
        fmax = 0.15

        if len(conformers) == 1:
            self.output.value = f"Optimizing {len(conformers)} conformer with {method}"
        else:
            self.output.value = f"Optimizing {len(conformers)} conformers with {method}"

        opt_structs = []
        for ase_struct in conformers:
            opt_struct = self._xtb_opt(
                ase_struct, xtb_method=method, max_steps=max_steps, fmax=fmax
            )
            if opt_struct is not None:
                opt_structs.append(opt_struct)

        xtb_energies = [conf.get_potential_energy() for conf in opt_structs]
        return opt_structs, xtb_energies

    def _create_trajectory_node(self, conformers, energies):
        if not conformers:
            return None

        traj = TrajectoryData(
            structurelist=[StructureData(ase=conformer) for conformer in conformers]
        )
        traj.set_extra("smiles", conformers[0].info["smiles"])
        if energies is not None and len(energies) > 0:
            traj.set_extra("energy_units", self._ENERGY_UNITS)
            if not isinstance(energies, np.ndarray):
                energies = np.array(energies)
            traj.set_array("energies", energies)
        return traj

    # TODO: Automatically filter out conformers with high energy
    # Boltzmann criterion: Add conformers until reaching e.g. 95% cumulative Boltzmann population
    # To test Boltzmann population values:
    # https://www.colby.edu/chemistry/PChem/Hartree.html
    def _filter_and_sort_conformers(self, conformers, energies):
        sorted_indices = argsort(energies)
        sorted_conformers = [conformers[i] for i in sorted_indices]
        sorted_energies = sorted(energies)

        en0 = sorted_energies.pop(0)
        selected_conformers = [sorted_conformers.pop(0)]
        selected_energies = [0.0]

        # Filter out conformers that have the same energy, within threshold
        for conf, en in zip(sorted_conformers, sorted_energies):
            shifted_energy = en - en0
            if shifted_energy - selected_energies[-1] > self._ENERGY_THR:
                selected_conformers.append(conf)
                selected_energies.append(shifted_energy)
        return selected_conformers, selected_energies

    def _rdkit_opt(self, smiles, steps, algo="ETKDG", opt_algo="MMFF94"):
        """Optimize a molecule using force field and rdkit (needed for complex SMILES)."""

        # self.output.value += f"<br>Using algorithm: {algo}"
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            # Something is seriously wrong with the SMILES code,
            # just return None and don't attempt anything else.
            self.output.value = "RDkit ERROR: Invalid SMILES string"
            return None

        mol = Chem.AddHs(mol)

        # https://www.rdkit.org/docs/Cookbook.html?highlight=allchem%20embedmultipleconfs#conformer-generation-with-etkdg
        if algo == "ETKDG":
            params = AllChem.ETKDG()
        elif algo == "ETKDGv2":
            params = AllChem.ETKDGv2()
        elif algo == "ETKDGv3":
            params = AllChem.ETKDGv3()
        else:
            raise ValueError(f"Invalid RDKit algorithm '{algo}'")

        # TODO: This should probably be lower, but we need to implement filtering after optimization as well
        params.pruneRmsThresh = 0.1
        params.maxAttempts = 20
        params.randomSeed = 422
        # TODO: Determine the num_confs parameter adaptively based on the molecule size
        num_confs = 10
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)

        # Not sure what is the fail condition here
        if len(conf_ids) == 0:
            # This is supposedly a more robust setting for larger molecules, per
            # https://sourceforge.net/p/rdkit/mailman/message/21776083/
            self.output.value += "Embedding failed, retrying with random coordinates."
            params.useRandomCoords = True
            conf_ids = AllChem.EmbedMultipleConfs(
                mol, numConfs=num_confs, params=params
            )
        if len(conf_ids) == 0:
            raise ValueError("Failed to generate conformers with RDKit")

        ffenergies = None
        if opt_algo == "UFF" and AllChem.UFFHasAllMoleculeParams(mol):
            conf_opt = AllChem.UFFOptimizeMoleculeConfs(
                mol, maxIters=steps, numThreads=1
            )
            ffenergies = [energy for _, energy in conf_opt]

        elif opt_algo == "MMFF94":
            if AllChem.MMFFHasAllMoleculeParams(mol):
                conf_opt = AllChem.MMFFOptimizeMoleculeConfs(
                    mol, mmffVariant="MMFF94", maxIters=steps
                )
                ffenergies = [energy for conv, energy in conf_opt]
                # https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html?highlight=uff#rdkit.Chem.rdForceFieldHelpers.UFFOptimizeMoleculeConfs
                for converged, energy in conf_opt:
                    if converged != 0:
                        self.output.value += (
                            "<br> WARNING: MMFF94 optimization did not converge"
                        )
            else:
                self.output.value += " RDKit WARNING: Missing MMFF94 parameters"

        self.output.value += f"<br> No. conformers = {len(conf_ids)}"

        # Sort conformers based on their (optimized) energies
        if False and ffenergies is not None:
            assert len(ffenergies) == len(conf_ids)
            conf_ids = [conf_ids[i] for i in argsort(ffenergies)]
            en0 = min(ffenergies)
            ffenergies = [en - en0 for en in sorted(ffenergies)]

        # Convert conformers to an array of ASE Atoms objects
        ase_structs = []
        natoms = mol.GetNumAtoms()
        species = [mol.GetAtomWithIdx(j).GetSymbol() for j in range(natoms)]
        for conf_id in conf_ids:
            positions = mol.GetConformer(id=conf_id).GetPositions()
            ase_structs.append(self._make_ase(species, positions, smiles))
        # TODO: Find out the units of energies and convert to kJ per mole
        return ase_structs, ffenergies
