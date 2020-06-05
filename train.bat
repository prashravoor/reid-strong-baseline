@echo off
if "%1"=="" goto usage
python tools/train.py --config_file=%1 MODEL.DEVICE_ID "('0')" 
goto :eof
:usage
@echo A config file is required