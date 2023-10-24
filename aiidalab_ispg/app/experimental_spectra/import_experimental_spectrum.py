# ruff: noqa: INP001
# This script needs to be run with `verdi run`
import argparse
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import yaml
from aiida.orm import QueryBuilder
from aiida.plugins import DataFactory
from rdkit import Chem

XyData = DataFactory("array.xy")


def parse_cmd():
    desc = """A script for importing experimental spectra into AiiDA database

    Run as `verdi run import_experimental_spectrum.py INPUT_FILE.yaml`
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "input_file", metavar="INPUT_FILE", help="YAML input file with metadata"
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        dest="dry_run",
        default=False,
        action="store_true",
        help="Dry run, do not store anything in DB",
    )
    return parser.parse_args()


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        msg = "RDkit ERROR: Invalid SMILES string"
        raise ValueError(msg)
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    if not canonical_smiles:
        msg = "RDKit: Could not canonicalize SMILES"
        raise ValueError(msg)
    return canonical_smiles


def main(input_file, dry_run=True):
    with Path(opts.input_file).open("r") as f:
        data = yaml.safe_load(f)

    pprint(data)

    smiles = canonicalize_smiles(data["smiles"])
    if smiles != data["smiles"]:
        print(f"Canonical SMILES: {smiles}")

    energy_nm, cross_section = np.loadtxt(data["source_file"], unpack=True)
    if scale := data.get("y_axis_scaling"):
        cross_section = cross_section * float(scale)

    xy = XyData()
    xy.set_x(energy_nm, "energy", "nm")
    xy.set_y(cross_section, "cross section", "cm^2")
    xy.set_source(source=data["metadata"]["source"])

    xy.base.extras.set("smiles", smiles)
    if desc := data.get("description"):
        xy.base.extras.set("description", desc)
    if pub := data.get("publication"):
        xy.base.extras.set("publication", pub)

    qb = QueryBuilder()
    qb.append(XyData, filters={"extras.smiles": data["smiles"]})
    if qb.count() > 0:
        print(
            f"Experimental spectrum for {data['molecule']} already exist in the database!"
        )
        spectrum = qb.all()[0][0]
        print(spectrum.get_array("x_array"))
        print(spectrum.get_array("y_array_0"))
        sys.exit(1)

    if opts.dry_run:
        print("Energies:", energy_nm)
        print("Cross sections:", cross_section)
        sys.exit(0)

    xy.store()

    qb = QueryBuilder()
    qb.append(XyData, filters={"extras.smiles": smiles})
    print(f"Stored spectrum in node {qb.all()[0]}")


if __name__ == "__main__":
    opts = parse_cmd()
    main(opts.input_file, opts.dry_run)
