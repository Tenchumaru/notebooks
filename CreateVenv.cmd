@ECHO OFF

"C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\python.exe" -m venv --system-site-packages venv
IF ERRORLEVEL 1 GOTO error
CALL venv\Scripts\activate.bat
IF ERRORLEVEL 1 GOTO error
python -m pip install --upgrade pip
IF ERRORLEVEL 1 GOTO error
pip install -U setuptools
IF ERRORLEVEL 1 GOTO error
pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
IF ERRORLEVEL 1 GOTO error
pip install -r requirements.txt
IF ERRORLEVEL 1 GOTO error
GOTO :EOF

:error
ECHO An error occurred.
EXIT /B
