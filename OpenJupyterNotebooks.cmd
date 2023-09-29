@ECHO OFF
SETLOCAL

CD /D "%~dp0"
IF EXIST venv\Scripts\activate.bat (
	CALL venv\Scripts\activate.bat
) ELSE (
	CALL %SystemDrive%\local\bin\CreateVenv.cmd
	IF ERRORLEVEL 1 EXIT /B 1
	CALL venv\Scripts\activate.bat
	python -m pip install -r requirements.txt
	IF ERRORLEVEL 1 EXIT /B 1
)
jupyter notebook
