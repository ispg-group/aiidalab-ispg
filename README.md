# AiiDALab Testing application

UV/VIS spectroscopy app

## Installation

This Jupyter-based app is intended to be run with [AiiDAlab](https://www.materialscloud.org/aiidalab).

If you already run AiiDaLab, you can install the latest version from Github
by running the following command from within the AiiDAlab Docker container.
```
aiidalab install aiidalab-ispg@git+https://github.com/danielhollas/aiidalab-ispg.git@main
```

## Usage

Here may go a few sreenshots / animated gifs illustrating how to use the app.

## License

MIT

## aiidalab-launch based installation

NOTE: The details of some of the commands will depend on your
OS and its version. The concrete examples are based on Ubuntu 20.04.

0. Install Docker (concrete commands will depend on your OS)
   `$ sudo apt install docker.io`
   - Add yourself to the docker group and restart shell session
   `$ sudo usermod -a -G docker $USER
1. Install pipx
   `$ apt install python-venv pipx`
2. Install aiidalab-launch to manage the aiidalab containers
   `$ pipx install aiidalab`
   If pipx is giving you trouble, you should be able to install via pip as well.
3. Modify default profile to include aiidalab-ispg app

```sh
$ aiidalab-launch profiles edit default
```

Profile configuration
```
port = 8888
default_apps = [ "aiidalab-widgets-base", "aiidalab-ispg@git+https://github.com/danielhollas/aiidalab-ispg.git@main",]
system_user = "aiida"
image = "aiidalab/aiidalab-docker-stack:latest"
home_mount = "aiidalab_ispg_home"
extra_mounts = ["/home/hollas/software/orca/5.0.3/arch/linux_x86-64_openmpi411_static:/opt/orca:ro",]
```

4. Launch the container

```sh
$ aiidalab-launch start
```

5. Install xtb-python, cannot be declared as a pip dependency since it is not published in PyPI.
   Instead, it needs to be installed by Conda.

```sh
$ aiidalab-launch exec --privileged -- conda install xtb-python
```

Note that aiidalab-launch creates a separate docker volume for Conda,
so the command above only needs to be executed once, even if you restart the container.

6. Setup ORCA code on localhost.
```sh
$ aiidalab-launch exec -- bash /home/aiida/apps/aiidalab-ispg/setup_codes_on_localhost.sh
```
This sets up the required code nodes in AiiDA DB. Since the DB is persisted in the
'home_mount' volume, this needs to only be done once for a give aiidalab profile.

If you are planning to launch codes on external computer, this step needs to be modified.

7. [Optional] Re-install packages for development
```sh
$ aiidalab-launch exec --privileged -- pip install -e /home/aiida/apps/aiidalab-ispg/
$ aiidalab-launch exec --privileged -- pip install -e /home/aiida/apps/aiidalab-widgets-base/
```
