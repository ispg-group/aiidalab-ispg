#!/usr/bin/env python
from selenium.webdriver.common.by import By


def test_example(selenium, url):
    selenium.get(url("apps/apps/app/example.ipynb"))
    selenium.find_element(By.ID, "ipython-main-app")
    selenium.find_element(By.ID, "notebook-container")
    selenium.find_element(By.CLASS_NAME, "jupyter-widgets-view")
    selenium.get_screenshot_as_file("screenshots/example.png")
