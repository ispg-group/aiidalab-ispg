import os
import time
from pathlib import Path
from urllib.parse import urljoin

import pytest
import requests
from requests.exceptions import ConnectionError
import selenium.webdriver.support.expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By


def is_responsive(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
    except ConnectionError:
        return False


@pytest.fixture(scope="session")
def docker_compose(docker_services):
    return docker_services._docker_compose


@pytest.fixture(scope="session")
def aiidalab_exec(docker_compose):
    def _execute(command, user=None, workdir=None, **kwargs):
        opts = "-T"
        if user:
            opts = f"{opts} --user={user}"
        if workdir:
            opts = f"{opts} --workdir={workdir}"
        command = f"exec {opts} aiidalab {command}"

        return docker_compose.execute(command, **kwargs)

    return _execute


@pytest.fixture(scope="session")
def nb_user(aiidalab_exec):
    return aiidalab_exec("bash -c 'echo \"${NB_USER}\"'").decode().strip()


@pytest.fixture(scope="session")
def appdir(nb_user):
    return f"/home/{nb_user}/apps/aiidalab-ispg"


@pytest.fixture(scope="session")
def notebook_service(docker_ip, docker_services, aiidalab_exec, nb_user, appdir):
    """Ensure that HTTP service is up and responsive."""

    # Directory ~/apps/aiidalab-qe/ is mounted by docker,
    # make it writeable for jovyan user, needed for `pip install`
    aiidalab_exec(f"chmod -R a+rw {appdir}", user="root")

    # Install dependencies via pip
    aiidalab_exec("pip install .", workdir=appdir, user=nb_user)

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
        WebDriverWait(selenium, 20).until(
            EC.invisibility_of_element((By.ID, "appmode-busy"))
        )
        return selenium

    return _selenium_driver


@pytest.fixture(scope="function")
def generate_mol_from_smiles(selenium):
    def _generate_mol(smiles):
        smiles_input = selenium.find_element(By.XPATH, "//input[@placeholder='C=C']")
        smiles_input.clear()
        smiles_input.send_keys(smiles)
        WebDriverWait(selenium, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[text()='Generate molecule']")
            )
        ).click()
        time.sleep(3)

    return _generate_mol


@pytest.fixture(scope="function")
def check_atoms(selenium):
    """Check that we can select atoms in a molecule given atom symbols."""

    def _select_atoms(atom_symbols: str):
        """
        atom_symbols str: For example, "CHHHH" for methane molecule
        The order of atom symbols must be the same as their indexes in the molecule
        """
        selection_box = selenium.find_element(
            By.XPATH, "//label[text()='Select atoms:']/following-sibling::input"
        )
        apply_selection = WebDriverWait(selenium, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[text()='Apply selection']"))
        )
        for i, atom in enumerate(atom_symbols):
            selection_box.clear()
            selection_box.send_keys(str(i + 1))
            apply_selection.click()
            selenium.find_element(
                By.XPATH, f"//div[starts-with(text(),'Id: {i+1}; Symbol: {atom};')]"
            )

    return _select_atoms


@pytest.fixture
def button_enabled(selenium):
    def _button_enabled(button_title):
        WebDriverWait(selenium, 15).until(
            EC.none_of(
                EC.element_attribute_to_include(
                    (By.XPATH, f"//button[text()='{button_title}']"), "disabled"
                )
            )
        )

    return _button_enabled


@pytest.fixture
def button_disabled(selenium):
    def _button_disabled(button_title):
        WebDriverWait(selenium, 15).until(
            EC.text_to_be_present_in_element_attribute(
                (By.XPATH, f"//button[text()='{button_title}']"), "disabled", "true"
            )
        )

    return _button_disabled


@pytest.fixture(scope="session")
def screenshot_dir():
    sdir = Path.joinpath(Path.cwd(), "screenshots")
    try:
        os.mkdir(sdir)
    except FileExistsError:
        pass
    return sdir


@pytest.fixture
def final_screenshot(request, screenshot_dir, selenium):
    """Take screenshot at the end of the test.
    Screenshot name is generated from the test function name
    by stripping the 'test_' prefix
    """
    screenshot_name = f"{request.function.__name__[5:]}.png"
    screenshot_path = Path.joinpath(screenshot_dir, screenshot_name)
    yield
    selenium.get_screenshot_as_file(screenshot_path)


@pytest.fixture
def firefox_options(firefox_options):
    firefox_options.add_argument("--headless")
    return firefox_options


@pytest.fixture
def chrome_options(chrome_options):
    chrome_options.add_argument("--headless")
    return chrome_options
