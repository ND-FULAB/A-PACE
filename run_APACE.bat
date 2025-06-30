@echo off
CALL "C:\Users\fulab\anaconda3\Scripts\activate.bat" APACE

cd /d "C:\Users\fulab\OneDrive\Desktop\APACE"

start "" /B python app.py

timeout /t 6 >nul
start "" "http://127.0.0.1:5000/"

exit /b
