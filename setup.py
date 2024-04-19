from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="dinotrack",
    version="1.0",
    install_requires=requirements,
    packages=["dinotrack", "dinotrack/core", "dinotrack/core/image"],
)
