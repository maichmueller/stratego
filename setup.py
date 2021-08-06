from __future__ import print_function

import sys
import os
import re


def find_ver(filepath: str):
    # pattern taken from:
    # https://semver.org/#is-there-a-suggested-regular-expression-regex-to-check-a-semver-string
    version_pattern = r"(?<=(##\s\[))(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?(?=\])"
    version = None
    with open(filepath, "r") as file:
        for line in file:
            if (match := re.search(version_pattern, line.strip())) is not None:
                version = match.group().strip()
                break
    if version is None:
        raise ValueError("Version string not provided.")
    return version


try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise


from setuptools import find_packages
# from setuptools import setup


setup(
    version=find_ver(os.path.join(".", "CHANGELOG.md")),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmake_install_dir="src/stratego",
)

# setup(
#     version=find_ver(os.path.join(".", "CHANGELOG.md")),
#     packages=find_packages(where="src"),
#     package_dir={"": "src"},
# )
