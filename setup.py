from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'TODO'
LONG_DESCRIPTION = 'TODO TODO TODO'

# Setting up
setup(
    name="sentle",
    version=VERSION,
    author="Clemens Mosig",
    author_email="clemens.mosig@uni-leipzig.de",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that 
    # needs to be installed along with your package. 
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ])
