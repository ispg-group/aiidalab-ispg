import os
from pathlib import Path
from urllib.parse import urljoin

import pytest
import requests
from requests.exceptions import ConnectionError
from selenium.webdriver.common.by import By


def is_responsive(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
    except ConnectionError:
        return False


@pytest.fixture(scope="session")
def docker_exec(docker_services, notebook_service):
    def _docker_exec(command, user="jovyan"):
        compose = "exec -T -u {user} aiidalab bash -c '{command}'"
        docker_services._docker_compose.execute(compose)

    return _docker_exec


@pytest.fixture(scope="session")
def notebook_service(docker_ip, docker_services):
    """Ensure that HTTP service is up and responsive."""

    docker_compose = docker_services._docker_compose

    # assurance for host user UID other that 1000
    # WARNING: This will render the repo directory
    # inaccessible outside of the docker container!
    # Also you'll mess up you're git if you run the tests locally!
    chown_command = "exec -T -u root aiidalab bash -c 'chown -R jovyan:users /home/jovyan/apps/aiidalab-ispg'"
    docker_compose.execute(chown_command)

    # Install dependencies via pip
    install_command = "bash -c 'pip install .'"
    command = (
        f"exec --workdir /home/jovyan/apps/aiidalab-ispg -T aiidalab {install_command}"
    )
    docker_compose.execute(command)

    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("aiidalab", 8888)
    url = f"http://{docker_ip}:{port}"
    token = os.environ["JUPYTER_TOKEN"]
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive(url)
    )
    return url, token


@pytest.fixture(scope="function")
def selenium_driver(selenium, notebook_service):
    def _selenium_driver(nb_path, wait_time=5.0):
        url, token = notebook_service
        url_with_token = urljoin(
            url, f"apps/apps/aiidalab-ispg/{nb_path}?token={token}"
        )
        selenium.get(f"{url_with_token}")
        selenium.implicitly_wait(wait_time)  # must wait until the app loaded

        selenium.find_element(By.ID, "ipython-main-app")
        selenium.find_element(By.ID, "notebook-container")
        selenium.find_element(By.CLASS_NAME, "jupyter-widgets-view")

        return selenium

    return _selenium_driver


@pytest.fixture(scope="session")
def screenshot_dir():
    sdir = Path.joinpath(Path.home(), "screenshots")
    os.mkdir(sdir)
    return sdir


@pytest.fixture
def firefox_options(firefox_options):
    firefox_options.add_argument("--headless")
    return firefox_options


@pytest.fixture
def chrome_options(chrome_options):
    chrome_options.add_argument("--headless")
    return chrome_options
