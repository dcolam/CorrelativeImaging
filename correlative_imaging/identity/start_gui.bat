@echo off
rem Start the particle-identity tool (napari + docks) in conda env corrimg. Double-click, or run from a terminal.
cd /d "%~dp0..\.."
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" corrimg
python -m correlative_imaging.identity gui %*
if errorlevel 1 pause
