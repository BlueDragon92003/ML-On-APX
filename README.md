# Cluster Analysis ML Model for CMS L1

This is a undergraduate research project developed by Quentin Schultz
(qschultz@wisc.edu; quentin@theschultzfamily.us) under advisory by Sidhara Dasu
(dasu@hep.wisc.edu) and Abhi Mallampalli (amallampalli@wisc.edu).

The project goal is to develop an Machine-Learning (ML) model to identify and
classify calorimetry events within the L1 trigger system and see if it is more
efficient or more accurate than the current algorithmic method in use.

## Project Milestones

- [ ] Develop a model for classifying identified clusters between hadronic 
    (pions, etc.) and electromagnetic (electrons, photons, etc.) clusters. 
    Data is obtained from two sources, RCT with ECAL energy clusters at the
    crystal-level (0.087x0.087) resolution and HCAL at tower-level (0.35x0.35)
    resolution. These data need to be correlated and compared to make the
    classification. The lateral cluster size in ECAL and HCAL also has some
    discrimination power.
- [ ] Evaluate if the accuracy of the model is high enough to continue with this
    project, and perhaps retrain the model to reach desired accuracies.
- [ ] Develop a model for identifying those clusters and passing them to the
    next model in the expected format
- [ ] Evaluate if the accuracy of the model is high enough to continue with this
    project, and perhaps retrain the model to reach desired accuracies.
- [ ] Translate the models with HLS4ML and test their execution time in
    comparison to the current algorithm
- [ ] Evaluate if the models' efficieny is high enough to continue with this
    project, and perhaps retrain the model(s) to improve efficiency

## User Guide

### Setup

0. Ensure Python version 3.11 or later is installed.
1. Run `git clone https://github.com/BlueDragon92003/ML-On-APX` to make a local
    copy of this repository.
2. Run `python3 -m venv .venv` to create a virtual environment. 
3. Run `source .venv` to activate the virtual environment.
4. Run `make install-dependancies` to install required packages for this
    program.

### Using

Run `python3 -m ml_on_apx` to get a set of CLI options and their behaviors.
Please ensure you are in the virtual environment when you execute any commands.

### Updating

Run `make clean-install-dependancies` to uninstall and reinstall anything.
Warning! This program was (out of laziness) not designed around even a local
database. That means that any alterations to the structure of pickled files may
cause the program to break. It is recommended to note down the structures of any
important groups or datasets and to re-create them in an updated version. ROOT
data files and `.pth` models should be transferable.

## Development

### Tools

All of these tools, excepting ruff and ty, should be installed with the other
program dependencies or do not require such installation.

- [**bubble-analysis**](https://pypi.org/project/bubble-analysis/) is a potent
    tool for analyzing raised exceptions that could potentially escape and crash
    the program. Execute it with `bubble cli audit`.
- **eliot-tree** is the CLI viewer for the
    [`eliot`](https://eliot.readthedocs.io/en/stable/) logging framework. It
    will be your best friend when digging through the logs to determine why
    something crashed or didn't work.
- [**pre_commit**](https://pre-commit.com/) adds pre-commit hooks for
    ruff and pyreqver, ensuring that committed code is nicely formatted, checked
    for code smell, and does not contain code requiring a python version higher
    than Python 3.11.
- [**pyreqver**](https://pypi.org/project/pyreqver/) is a command-line tool to
    ensure all requirements in `requirements.txt` exist at the Python version
    of the project.
- [**ruff**](https://docs.astral.sh/ruff/) and
    [**ty**](https://docs.astral.sh/ty/) are `astral.sh`'s linting and type
    checking libraries, respectively. They exist as VSCode extensions or can be
    run locally. Ruff is already configured in `ruff.toml`, so simply install
    the tool.
- [**vermin**](https://pypi.org/project/vermin/) is the tool pre-commit uses to
    ensure committed code uses Python features introduced no later than version
    3.11.

To ensure all tools are properly installed:
0. Follow the steps to set up the project.
1. Run `pre-commit install`

### Reference Material

- [Eliot docs](https://eliot.readthedocs.io/en/stable/)
- [PyFakeFS docs](https://pytest-pyfakefs.readthedocs.io/en/stable/intro.html)
- [Textual docs](https://textual.textualize.io/)
- [Torch 2.9.1 docs](https://docs.pytorch.org/docs/2.9/index.html)
- [Uproot docs](https://uproot.readthedocs.io/en/latest/)
