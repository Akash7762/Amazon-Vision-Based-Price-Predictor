@echo off
rem Double-click to start the app: the price model, the web app, and the
rem browser. Press Q in the window, or close it, to stop. Details and options
rem are in scripts\start_app.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_app.ps1" %*
if errorlevel 1 pause
