@ECHO OFF
SETLOCAL

CD /D "%~dp0.."
CALL venv\Scripts\activate.bat
CD /D "%~dp0"
jupyter notebook
