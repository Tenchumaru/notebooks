@ECHO OFF

CALL %SystemDrive%\local\bin\CreateVenv.cmd
IF ERRORLEVEL 1 GOTO error
CALL venv\Scripts\activate.bat
IF ERRORLEVEL 1 GOTO error
python -m pip install --upgrade pip
IF ERRORLEVEL 1 GOTO error
pip install -U setuptools
IF ERRORLEVEL 1 GOTO error
pip install -r requirements.txt
IF ERRORLEVEL 1 GOTO error
GOTO :EOF

:error
ECHO An error occurred.
EXIT /B
