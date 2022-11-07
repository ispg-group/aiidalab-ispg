"""Widgets for the conformer generation usign RDKit.

Detailed documentation
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

    def _mol_from_smiles(self, smiles, steps=1000):
        """Convert SMILES to ase structure try rdkit then pybel"""
        conformers = None
        # TODO: Make a dropdown menu for algorithm selection
        rdkit_algorithm = "ETKDGv2"
        try:
            conformers = self._rdkit_opt(smiles, steps, algo=rdkit_algorithm)
            if conformers is None:
                # conformers = self._pybel_opt(smiles, steps)
                return None
        except ValueError as e:
            self.output.value = str(e)
            return None

        if conformers is None or len(conformers) == 0:
            return None

        # Fallback if XTB is not available
        if DISABLE_XTB:
            return self._create_trajectory_node(conformers)

        conformers = self.optimize_conformers(conformers)
        conformers = self._filter_and_sort_conformers(conformers)
        return self._create_trajectory_node(conformers)

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

    def optimize_conformers(self, conformers):
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
        return opt_structs

    def _create_trajectory_node(self, conformers):
        if conformers is None or len(conformers) == 0:
            return None

        traj = TrajectoryData(
            structurelist=[StructureData(ase=conformer) for conformer in conformers]
        )
        traj.set_extra("smiles", conformers[0].info["smiles"])
        if DISABLE_XTB:
            return traj
        # TODO: I am not sure whether get_potential_energy triggers computation,
        # or uses already computed energy from the optimization step
        en0 = conformers[0].get_potential_energy()
        energies = np.fromiter(
            (conf.get_potential_energy() - en0 for conf in conformers),
            count=len(conformers),
            dtype=float,
        )
        traj.set_array("energies", energies)
        return traj

    # TODO: Automatically filter out conformers with high energy
    # Boltzmann criterion: Add conformers until reaching e.g. 95% cumulative Boltzmann population
    # To test Boltzmann population values:
    # https://www.colby.edu/chemistry/PChem/Hartree.html
    def _filter_and_sort_conformers(self, conformers):
        energies = [conf.get_potential_energy() for conf in conformers]
        sorted_indices = argsort(energies)
        return [conformers[i] for i in sorted_indices]

    def _rdkit_opt(self, smiles, steps, algo="ETKDG", opt_algo="MMFF94", num_confs=10):
        """Optimize a molecule using force field and rdkit (needed for complex SMILES)."""

        # self.output.value += f"<br>Using algorithm: {algo}"

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Something is seriously wrong with the SMILES code,
            # just return None and don't attempt anything else.
            self.output.value = "RDkit ERROR: Invalid SMILES string"
            return None

        mol = Chem.AddHs(mol)

        if algo == "single-conformer":
            params = AllChem.ETKDG()
            params.maxAttempts = 20
            params.randomSeed = 42
            conf_id = AllChem.EmbedMolecule(mol, params=params)
            if conf_id == -1:
                # This is a more robust setting for larger molecules, per
                # https://sourceforge.net/p/rdkit/mailman/message/21776083/
                self.output.value += (
                    "Embedding failed, retrying with random coordinates."
                )
                params.useRandomCoords = True
                conf_id = AllChem.EmbedMolecule(mol, params=params)
            if conf_id == -1:
                msg = " Failed to generate conformer with RDKit. Trying OpenBabel next."
                raise ValueError(msg)

            if opt_algo == "UFF" and AllChem.UFFHasAllMoleculeParams(mol):
                AllChem.UFFOptimizeMolecule(mol, maxIters=steps)

            conf_ids = [conf_id]

        if algo == "ETKDG":
            params = AllChem.ETKDG()
            params.pruneRmsThresh = 0.1
            params.maxAttempts = 20
            params.randomSeed = 422
            conf_ids = AllChem.EmbedMultipleConfs(
                mol, numConfs=num_confs, params=params
            )
            # Not sure what is the fail condition here
            if len(conf_ids) == 0:
                # This is a more robust setting for larger molecules, per
                # https://sourceforge.net/p/rdkit/mailman/message/21776083/
                self.output.value += (
                    "Embedding failed, retrying with random coordinates."
                )
                params.useRandomCoords = True
                conf_ids = AllChem.EmbedMultipleConfs(
                    mol, numConfs=num_confs, params=params
                )
            if len(conf_ids) == -1:
                msg = " Failed to generate conformer with RDKit. Trying OpenBabel next."
                raise ValueError(msg)

        elif algo == "ETKDGv2":
            # https://www.rdkit.org/docs/Cookbook.html?highlight=allchem%20embedmultipleconfs#conformer-generation-with-etkdg
            params = AllChem.ETKDGv2()
            params.pruneRmsThresh = 0.1
            params.maxAttempts = 40
            params.randomSeed = 422
            conf_ids = AllChem.EmbedMultipleConfs(
                mol, numConfs=num_confs, params=params
            )

        else:
            raise ValueError(f"Invalid algorithm '{algo}'")

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
                # TODO: I guess we should check the return value somehow?
                for converged, energy in conf_opt:
                    if converged != 0:
                        self.output.value += (
                            "<br> WARNING: MMFF94 optimization did not converge"
                        )
            else:
                self.output.value += " RDKit WARNING: Missing MMFF94 parameters"

        # self.output.value += f"<br> No. conformers = {len(conf_ids)}"

        # TODO: Sort conformers based on FF energy if available
        ase_structs = []
        natoms = mol.GetNumAtoms()
        species = [mol.GetAtomWithIdx(j).GetSymbol() for j in range(natoms)]

        # Sort conformers based on their (optimized) energies
        if ffenergies is not None:
            assert len(ffenergies) == len(conf_ids)
            conf_ids = [conf_ids[i] for i in argsort(ffenergies)]

        for conf_id in conf_ids:
            positions = mol.GetConformer(id=conf_id).GetPositions()
            ase_structs.append(self._make_ase(species, positions, smiles))
        return ase_structs
