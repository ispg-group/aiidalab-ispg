# README

## About

The `workflows/` directory contains the `aiidalab_atmospec_workchain` package that must be installed as a dependency for the ATMOSPEC app as declared in the [setup.cfg](setup.cfg) file.

The app distributes its own AiiDA workchain implementation which constitutes an app dependency and must therefore be globally installed so that the AiiDA daemon is able to import it.

## How to update the `aiidalab_atmospec_workchain` package

Any updates to the workchain that have not been released yet must be installed manually.

```console
cd src
pip install  .
```

Consider to use the editable mode (`pip install -e .`) while actively developing the workchain.
