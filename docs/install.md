
# Installation

## Requirements

ATMOSPEC requires a Linux machine, but should also work on older x86 Macs.
If you're interested in Mac ARM (M1/M2) support, let us know!

## Install dependencies

NOTE: The details of some of the commands will depend on your
OS and its version. The concrete examples are based on Ubuntu 20.04.


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

## Download the ATMOSPEC Docker image

```console
docker pull ghcr.io/ispg-group/atmospec:latest
```

This image is built and published in a separate Github repository,
which you can visit [for more information](https://github.com/ispg-group/aiidalab-ispg-docker-stack#readme).


### Setup AiiDAlab launch

While you can launch the Docker container directly using the `docker` command,
it is much more convenient to use the [aiidalab-launch application](https://github.com/aiidalab/aiidalab-launch).

Let's first modify the default profile to include to ATMOSPEC app from this repo

```console
aiidalab-launch profile edit default
```

This should open your default text editor.
Copy-paste the following profile configuration (substitute path to ORCA and the Docker image name if needed)
```
port = 8888
default_apps = [ "aiidalab-ispg@git+https://github.com/ispg-group/aiidalab-ispg.git@main",]
system_user = "jovyan"
image = "ghcr.io/ispg-group/atmospec:latest"
home_mount = "aiidalab_atmospec_home"
extra_mounts = ["/absolute/path/to/orca/:/opt/orca:ro",]
```

With this configuration, all the data will be stored in a separate Docker volume `aiidalab_atmospec_home`.
Alternatively, you can specify the absolute path to a directory in your file system to store the data, for example

```
home_mount = "/home/username/aiidalab_atmospec_home/"
```

## Launch and manage container

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
