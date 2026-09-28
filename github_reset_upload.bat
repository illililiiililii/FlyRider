@echo off
chcp 65001 >nul
setlocal

title GitHub 초기화 및 업로드

echo ==========================================
echo   GitHub 초기화 + 코드 업로드
echo ==========================================
echo.

echo 현재 폴더:
echo %CD%
echo.

set /p GITHUB_ID=GitHub 아이디: 
set /p GITHUB_EMAIL=GitHub 이메일: 
set /p REPO_NAME=GitHub 저장소 이름: 
set /p COMMIT_MSG=커밋 메시지 [Initial commit]: 

if "%COMMIT_MSG%"=="" set "COMMIT_MSG=Initial commit"

set "REPO_URL=https://github.com/%GITHUB_ID%/%REPO_NAME%.git"

echo.
echo ==========================================
echo GitHub : %GITHUB_ID%
echo 저장소 : %REPO_URL%
echo ==========================================
echo.

set /p CONFIRM=기존 Git 기록을 삭제하고 다시 올릴까요? (Y/N): 

if /i not "%CONFIRM%"=="Y" (
    echo 취소되었습니다.
    pause
    exit /b
)

echo.
echo [1/5] 기존 Git 기록 삭제...

if exist ".git" (
    rmdir /s /q .git
)

echo 완료.

echo.
echo [2/5] Git 초기화...

git init

if errorlevel 1 (
    echo Git 초기화 실패
    pause
    exit /b
)

git config user.name "%GITHUB_ID%"
git config user.email "%GITHUB_EMAIL%"

echo.
echo [3/5] 파일 추가...

git add .

if errorlevel 1 (
    echo 파일 추가 실패
    pause
    exit /b
)

echo.
echo 현재 커밋될 파일:
echo ------------------------------------------

git status --short

echo ------------------------------------------
echo.

echo [4/5] 커밋...

git commit -m "%COMMIT_MSG%"

if errorlevel 1 (
    echo.
    echo 커밋 실패
    pause
    exit /b
)

echo.
echo [5/5] GitHub 연결 및 업로드...

git branch -M main
git remote add origin "%REPO_URL%"

git push -u origin main

if errorlevel 1 (
    echo.
    echo ==========================================
    echo             업로드 실패
    echo ==========================================
    echo.
    echo 저장소 주소:
    echo %REPO_URL%
    echo.
    pause
    exit /b
)

echo.
echo ==========================================
echo          업로드 완료!
echo ==========================================
echo.
echo %REPO_URL%
echo.

pause