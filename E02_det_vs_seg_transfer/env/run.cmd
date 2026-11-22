@echo off
REM Run an E02 module with the E02 environment:  run.cmd -m e02.setup_dataset [args]
call "%~dp0e02_env.cmd"
cd /d "%E02%"
"%E02_PY%" %*
