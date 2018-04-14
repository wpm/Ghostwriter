from setuptools import setup

from version import __version__


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="Ghostwriter",
    version=__version__,
    packages=["ghostwriter"],
    url="https://github.com/wpm/Ghostwriter",
    license="M.I.T.",
    author="W.P. McNeill",
    author_email="billmcn@gmail.com",
    description="Machine-assisted writing",
    long_description=readme(),
    entry_points={"console_scripts": ["ghostwriter=ghostwriter.command:main"]},
    install_requires=["click", "cytoolz", "keras", "numpy"]
)
