from setuptools import find_packages, setup


VERSION = 1.0
PACKAGE_NAME = "sparse_shift"
DESCRIPTION = "Conditional independence tools for causal learning under the sparse mechanism shift hypothesis."
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = ("Ronan Perry",)
AUTHOR_EMAIL = "rflperry@gmail.com"
with open('requirements.txt') as f:
    REQUIRED_PACKAGES = f.read().splitlines()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    install_requires=REQUIRED_PACKAGES,
    license="MIT",
    packages=find_packages(),
)