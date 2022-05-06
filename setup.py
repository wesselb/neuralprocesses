from setuptools import find_packages, setup

requirements = [
    "numpy>=1.16",
    "backends>=1.4.24",
    "backends-matrix>=1.2.10",
    "plum-dispatch>=1",
    "stheno>=1.3.10",
    "wbml>=0.3.18",
]

setup(
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)
