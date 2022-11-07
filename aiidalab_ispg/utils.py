from aiida.plugins import DataFactory

StructureData = DataFactory("structure")
TrajectoryData = DataFactory("array.trajectory")
CifData = DataFactory("cif")


def get_formula(data_node):
    """A wrapper for getting a molecular formula out of the AiiDA Data node"""
    if isinstance(data_node, TrajectoryData):
        # TrajectoryData can only hold structures with the same chemical formula,
        # so this approach is sound.
        stepid = data_node.get_stepids()[0]
        return data_node.get_step_structure(stepid).get_formula()
    elif isinstance(data_node, StructureData):
        return data_node.get_formula()
    elif isinstance(data_node, CifData):
        return data_node.get_ase().get_chemical_formula()
    else:
        raise ValueError(f"Cannot get formula from node {type(data_node)}")


# https://stackoverflow.com/a/3382369
def argsort(seq):
    """Returns a list of indeces that sort the array"""
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)
