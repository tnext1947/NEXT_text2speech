@echo off
REM Prompt user for GPU or CPU usage
set /p use_gpu="Do you want to use GPU? (y/n): "

REM Create a virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install required dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

if /I "%use_gpu%"=="y" (
    echo Installing GPU-specific dependencies...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

REM Run the application
echo Running app.py...
python app.py

REM Deactivate the virtual environment
echo Deactivating virtual environment...
deactivate

echo Setup complete. The virtual environment is ready.
pause