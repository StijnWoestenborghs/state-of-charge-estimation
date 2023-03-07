# state of charge estimation

## Prerequisit 1: Menaging python versions with pyenv
No need to customise env path variables and easily switch between python versions
- Windows: "choco install pyenv-win"
- Switching the global python version: "pyenv install 3.9.13", "pyenv global 3.9.13"

Standardize python version across projects
- When there is a .python-version file inside the current directory "python --version" will switch regardless of the global python

## Prerequisit 2: Creating virtual environment
The whole setup can is controlled and virtualenvs are controlled in a makefile
- Windows (Run in git bash terminal): "make setup"
After this point the interpreter of the IDE of your choise to the new python venv

TIP: Activating virtual environment using alias
- Windows: (install nano: "choco install nano")
    Git bash:
    - "touch ~/.bashrc"
    - "nano ~/.bashrc" to include 'alias pyact=". .venv/Scripts/activate" '

    Alternative Powershell 
    - "nano $profile" to include: "set-alias -Name pyact -Value .venv/Scripts/activate"

## Prerequisit 3: Seting up Git LFS
- download from: https://git-lfs.com/
- run "git lfs install"