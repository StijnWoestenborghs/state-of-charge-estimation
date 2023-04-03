<h1 align='center'><b>State of Charge Estimation</b></h1>

<p align='center'>
    __
</p>
<p align='center'>
    The rising efficiency and energy density of Li-ion batteries is driving an exciting wave of innovation, which not only accelerates adoption, but also enables new technologies with an even greater impact. Needless to say that an accurate State of Charge estimation is vital.
</p>
<p align='center'>
    __
</p>

## Project setup

### Prerequisit
1. Python versions and virtual environments are controlled with (the very convenient) [pyenv](https://github.com/pyenv/pyenv).  
Pyenv installation depends on the OS used:
> - MacOS: `brew install pyenv`
> - Windows: `choco install pyenv-win` (or follow [pyenv-win](https://github.com/pyenv-win/pyenv-win))
> - Switching the global python version: `pyenv install 3.9.13`, `pyenv global 3.9.13`

2. All the necessary data for this project has been added in git via [Git Large File Storage](https://git-lfs.github.com/). After cloning the project, make sure to first unpack that data: 
> - `git lfs install`  
> - `git lfs pull`

### Configure project setup
A virtual environment with all required packages is created by the simple command:  
(Note: MacOS users replace `. .venv/Scripts/activate` with `. .venv/bin/activate` in the makefile)
> - `make setup` (make sure to use a bash-like shell or follow similar commands)


## State of Charge Estimation

Initial discovery of the data is done in `notebooks/`.

Training is initiated by a configuration file `src/config.json`. This way of working allows for easy tracking of experiments. All logs of an experiment can be found in `logs/` and can be analysed with tensorboard using `tensorboard --logdir logs/<experiment_name>`.

1. **Training**

> Make sure you have configured the right settings in the configuration file. Note that different DNN architectures can be found in `src/models.py`.  
> - `make train`

2. **Test**

> Test the model performance of a given experiment for some unique Drive Cycles.  
> - `make test`