@echo off
REM E02: go-ahead for the one test-set evaluation (runs after any unfinished training). See code/e02/control.py
call "%~dp0run.cmd" -m e02.control eval
