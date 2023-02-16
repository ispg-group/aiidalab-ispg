# AiiDAlab ISPG applications

ATMOSPEC - ab initio workflow for UV/VIS spectroscopy of organic molecules.

## Installation

This Jupyter-based app is intended to be run within the [AiiDAlab environment](https://www.materialscloud.org/aiidalab).

If you already run AiiDAlab you can install the latest version from Github
by running the following command from within the AiiDAlab Docker container.
```
aiidalab install aiidalab-ispg@git+https://github.com/danielhollas/aiidalab-ispg.git@main
```

See below for complete installation instructions on local machine.

## Usage

TODO: A few screenshots / animated gifs illustrating how to use the app.

## License

MIT

## Local installation

NOTE: The details of some of the commands will depend on your
OS and its version. The concrete examples are based on Ubuntu 20.04.

### Install dependencies
0. [Install ORCA](https://www.orcasoftware.de/tutorials_orca/first_steps/install.html)

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


### Download the ATMOSPEC Docker image

```console
docker pull ghcr.io/ispg-group/atmospec:latest
```

This image is built and published in a separate Github repository,
which you can visit [for more information](https://github.com/ispg-group/aiidalab-ispg-docker-stack#readme).


### Setup AiiDAlab launch

While you can launch the Docker container directly using the `docker` command,
it is much more convenient to use the [aiidalab-launch application](https://github.com/aiidalab/aiidalab-launch).

Let's first modify the default profile to include to ATMOSPEC app from this repo

```sh
aiidalab-launch profiles edit default
```

This should open your default text editor.
Copy-paste the following profile configuration (substitute path to ORCA and the Docker image name if needed)
```
port = 8888
default_apps = [ "aiidalab-ispg@git+https://github.com/ispg-group/aiidalab-ispg.git@main",]
system_user = "jovyan"
image = ghcr.io/ispg-group/atmospec:latest"
home_mount = "aiidalab_atmospec_home"
extra_mounts = ["/absolute/path/to/orca/:/opt/orca:ro",]
```

With this configuration, all the data will be stored in a separate Docker volume `aiidalab_atmospec_home`.
Alternatively, you can specify the absolute path to a directory in your file system to store the data, for example

```
home_mount = "/home/username/aiidalab_atmospec_home/"
```

### Launch and manage container

 - Launch the container

```console
aiidalab-launch start
```

This command will print the URL that you can then access in the browser.
With the configuration above it should be `http://localhost:8888`

 - Stop the container

```console
aiidalab-launch stop
```

 - Print container status

```console
aiidalab-launch status
```

 - Run a command inside a running container

```console
aiidalab-launch exec -- <command>
```

 - Entering the container

```console
docker exec -it -u jovyan aiidalab_atmospec /bin/bash
```

To display all available commands

```console
aiidalab-launch --help
```

## Development

Re-install packages for development in the container
```sh
aiidalab-launch exec -- pip install --user -e /home/aiida/apps/aiidalab-ispg/
aiidalab-launch exec -- pip install --user -e /home/aiida/apps/aiidalab-ispg/workflows/
```
