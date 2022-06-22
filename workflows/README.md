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

Additional notes:
- Consider to use the editable mode (`pip install -e .`) while actively developing the workchain.

## Note on alternative approaches for distributing the workchain package

The following alternatives approaches for the distribution of the workchain wheel could be considered (in rough order of preference at the time of writing):

1. Install the package directly from the app directory (something like: `aiidalab-qe-workchain@file://./src/dist/aiidalab_qe_workchain-1.0-py3-none-any.whl`).
   However this is currently not possible, because it would be difficult to reliably determine the absolute location of the package and non-local URIs are not universally supported
