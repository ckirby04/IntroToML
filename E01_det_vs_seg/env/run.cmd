@echo off
REM Run an E01 module with the E01 environment:  run.cmd -m e01.setup_dataset [args]
call "%~dp0e01_env.cmd"
cd /d "%E01%"
"%E01_PY%" %*
