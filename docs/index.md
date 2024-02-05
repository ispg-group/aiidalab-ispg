# Welcome to ATMOSPEC

**WARNING: This documentation is a work in progress, please check back later. :-)**

Welcome to the ATMOSPEC documentation!
For a quick intro to how the program looks and works, take a look at this [short screencast](https://youtu.be/1ePj1hhOFdw).

## Installation

### Install dependencies
0. [Install ORCA](https://www.orcasoftware.de/tutorials_orca/first_steps/install.html)

NOTE: If you're downloading for Linux, choose the "shared-version", which has much smaller download size.

1. [Install Docker](https://docs.docker.com/engine/install/#server)

```console
sudo apt install docker.io
```

Then add yourself to the `docker` unix group and restart the shell session

```console
sudo usermod -a -G docker $USER
```

2. Install pipx

```console
sudo apt install python-venv pipx
```

3. Install `aiidalab-launch` to manage the AiiDAlab containers

```console
pipx install aiidalab-launch
```

If pipx is giving you trouble, you should be able to install via pip as well.

TODO: Take this from README

## Quick guide

TODO: Screenshots here

## Acknowledgements

This project has been funded by European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant agreement No. 803718, [project SINDAM](https://cordis.europa.eu/project/id/803718)), and the EPSRC Grant EP/V026690/1 [UPDICE](https://updiceproject.com).
