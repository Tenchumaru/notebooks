@ECHO OFF

"C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\python.exe" -m venv --system-site-packages venv
IF ERRORLEVEL 1 EXIT /B
CALL venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -U setuptools
pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
IF ERRORLEVEL 1 EXIT /B
pip install -r requirements.txt
