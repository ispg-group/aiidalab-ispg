# AiiDAlab ISPG applications

ATMOSPEC - automatic ab initio workflow for UV/VIS spectroscopy
of atmospherically relevant molecules. 

## Installation

This Jupyter-based app is intended to be run within the [AiiDAlab environment](https://www.materialscloud.org/aiidalab).

If you already run AiiDAlab you can install the latest version from Github
by running the following command from within the AiiDAlab Docker container.
```
aiidalab install aiidalab-ispg@git+https://github.com/danielhollas/aiidalab-ispg.git@main
```

See below for complete installation instructions on local machine.

## Usage

Here may go a few sreenshots / animated gifs illustrating how to use the app.

## License

MIT

## Local installation

NOTE: The details of some of the commands will depend on your
OS and its version. The concrete examples are based on Ubuntu 20.04.

### Install dependencies
0. Install ab initio dependencies (ORCA)

1. Install Docker

```sh
$ sudo apt install docker.io
```

1.1 Add yourself to the docker group and restart shell session

```sh
$ sudo usermod -a -G docker $USER`
```

2. Install pipx

```sh
$ apt install python-venv pipx
```

3. Install aiidalab-launch to manage the AiiDAlab containers

```sh
$ pipx install aiidalab-launch
```

If pipx is giving you trouble, you should be able to install via pip as well.


### Build AiiDAlab ISPG Docker image

In principle, you could use the official [AiiDAlab docker image](https://hub.docker.com/r/maxcentre/aiidalab-docker-stack),
which would be downloaded automatically when launching the Docker container.
It is however preferable to build our custom image that contains additional
dependencies pre-installed and includes the SLURM queuing manager.
See [installation instructions here](https://github.com/danielhollas/aiidalab-ispg-docker-stack#readme).

### Setup AiiDAlab launch

Modify default profile to include aiidalab-ispg app

```sh
$ aiidalab-launch profiles edit default
```

Copy paste this profile configuration (substitute path to ORCA and the Docker image name if needed)
```
port = 8888
default_apps = [ "aiidalab-widgets-base", "aiidalab-ispg@git+https://github.com/danielhollas/aiidalab-ispg.git@main",]
system_user = "aiida"
image = aiidalab-ispg:latest"
home_mount = "aiidalab_ispg_home"
extra_mounts = ["/home/hollas/software/orca/5.0.3/arch/linux_x86-64_openmpi411_static:/opt/orca:ro",]
```

### Launch and manage container

 - Launch the container

```sh
$ aiidalab-launch start
```
This command will print the URL that you can then access in the browser

 - Stop the container

```sh
$ aiidalab-launch stop
```

 - Print container status

```sh
$ aiidalab-launch status
```

## Development

Re-install packages for development in the container
```sh
$ aiidalab-launch exec -- pip install -e /home/aiida/apps/aiidalab-ispg/
$ aiidalab-launch exec -- pip install -e /home/aiida/apps/aiidalab-ispg/workflows/
$ aiidalab-launch exec -- pip install -e /home/aiida/apps/aiidalab-widgets-base/
```
