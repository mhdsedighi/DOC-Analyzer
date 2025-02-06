@echo off
setlocal
set "repo_url=https://github.com/mhdsedighi/DOC-Analyzer/archive/refs/heads/master.zip"
curl -L -o repo.zip %repo_url%
where powershell >nul 2>&1
if %errorlevel% equ 0 (
    powershell -Command "Expand-Archive -Force -Path repo.zip -DestinationPath temp_folder"
    powershell -Command "Get-ChildItem -Path temp_folder\DOC-Analyzer-master -Recurse | Move-Item -Destination . -Force"
    powershell -Command "Remove-Item -Recurse -Force temp_folder"
) else (
    tar -xf repo.zip --strip-components=1 -C .
)

del repo.zip
echo Project downloaded and files replaced successfully.