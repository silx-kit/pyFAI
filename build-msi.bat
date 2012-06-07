@ECHO OFF
REM  Script that builds a msi package from this library 

c:\python26\python.exe setup.py build --force --compiler mingw32 bdist_msi


