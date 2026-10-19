@echo off
rem Start the downstream analyzer GUI (conda env corrimg). Double-click, or run from a terminal.
cd /d "%~dp0..\.."
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" corrimg
python -m correlative_imaging.downstream gui %*
if errorlevel 1 pause
