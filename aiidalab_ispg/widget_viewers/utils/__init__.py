"""Some utility functions used acrross the repository."""
import more_itertools as mit
from ase.io import read


def valid_arguments(arguments, valid_args):
    """Check whether provided arguments are valid."""
    result = {}
    for key, value in arguments.items():
        if key in valid_args:
            if isinstance(value, (tuple, list)):
                result[key] = "\n".join(value)
            else:
                result[key] = value
    return result


def predefine_settings(obj, **kwargs):
    """Specify some pre-defined settings."""
    for key, value in kwargs.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
        else:
            raise AttributeError("'{}' object has no attirubte '{}'".format(obj, key))


def get_ase_from_file(fname, format=None):  # pylint: disable=redefined-builtin
    """Get ASE structure object."""
    if format == "cif":
        traj = read(fname, format=format, index=":", store_tags=True)
    else:
        traj = read(fname, format=format, index=":")
    if not traj:
        print(("Could not read any information from the file {}".format(fname)))
        return False
    if len(traj) > 1:
        print(
            (
                "Warning: Uploaded file {} contained more than one structure. Selecting the first one.".format(
                    fname
                )
            )
        )
    return traj[0]


def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]


def list_to_string_range(lst, shift=1):
    """Converts a list like [0, 2, 3, 4] into a string like '1 3..5'.

    Shift used when e.g. for a user interface numbering starts from 1 not from 0"""
    return " ".join(
        [
            f"{t[0] + shift}..{t[1] + shift}"
            if isinstance(t, tuple)
            else str(t + shift)
            for t in find_ranges(sorted(lst))
        ]
    )


def string_range_to_list(strng, shift=-1):
    """Converts a string like '1 3..5' into a list like [0, 2, 3, 4].

    Shift used when e.g. for a user interface numbering starts from 1 not from 0"""
    singles = [int(s) + shift for s in strng.split() if s.isdigit()]
    ranges = [r for r in strng.split() if ".." in r]
    if len(singles) + len(ranges) != len(strng.split()):
        return list(), False
    for rng in ranges:
        try:
            start, end = rng.split("..")
            singles += [i + shift for i in range(int(start), int(end) + 1)]
        except ValueError:
            return list(), False
    return singles, True
