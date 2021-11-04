from importlib import resources

import yaml

from aiidalab_ispg import parameters

DEFAULT_PARAMETERS = yaml.safe_load(resources.read_text(parameters, "orca.yaml"))
