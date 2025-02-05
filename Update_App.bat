@echo off
taskkill /f /im DocAnalyzer.exe 2>nul
curl -L -o DocAnalyzer.zip https://github.com/mhdsedighi/DOC-Analyzer/releases/download/latest/DocAnalyzer.zip
powershell -Command "Expand-Archive -Path DocAnalyzer.zip -DestinationPath . -Force"
del DocAnalyzer.zip
echo Update complete!
pause