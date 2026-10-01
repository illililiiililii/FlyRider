@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

title FlyRider - Auto Setup

:: Always work from the folder that contains this launcher. This prevents
:: downloads, datasets, and captures from being created in an unrelated cwd.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%.") do set "SCRIPT_DIR=%%~fI"
set "PROJECT_DIR=%SCRIPT_DIR%"
cd /d "%PROJECT_DIR%"

:: Support a previous installation that was cloned one level below the launcher.
if not exist "%PROJECT_DIR%\FlyRider_connectome.py" if exist "%PROJECT_DIR%\FlyRider\FlyRider_connectome.py" set "PROJECT_DIR=%PROJECT_DIR%\FlyRider"
cd /d "%PROJECT_DIR%"

echo ==========================================
echo       FLYRIDER AUTO LAUNCHER
echo ==========================================
echo.

:: ==========================================
:: GitHub Release 설정
:: ==========================================

set "GITHUB_USER=illililiiililii"
set "GITHUB_REPO=FlyRider"
set "RELEASE_TAG=v1.0"

set "BASE_URL=https://github.com/%GITHUB_USER%/%GITHUB_REPO%/releases/download/%RELEASE_TAG%"

:: 다운로드할 데이터
set "DATA1=neurons.csv"
set "DATA2=connections_princeton.csv"
set "DATA3=chopari_connectome_learning.npz"

:: ==========================================
:: git 확인
:: ==========================================

if exist "%PROJECT_DIR%\FlyRider_connectome.py" goto CHECK_PYTHON

echo [1/9] Git 확인...

git --version > nul 2>&1

if errorlevel 1 (
    echo.
    echo [ERROR] Git이 설치되어 있지 않습니다.
    echo.
    choice /c YN /n /m "Git을 설치하시겠습니까? (Y/N): "
    if errorlevel 2 (
        echo Git 설치를 취소했습니다.
    ) else (
        echo Git 설치를 시도합니다...
        winget install --id Git.Git -e --source winget
        if errorlevel 1 (
            echo [ERROR] Git 설치에 실패했습니다.
            pause
            exit /b 1
        )
        echo Git 설치가 완료되었습니다. 새 터미널에서 다시 실행해 주세요.
        echo 필요시 로그아웃이 필요할 수 있습니다.
    )
    pause
    exit /b 1
)

:: ==========================================
:: Python 확인
:: ==========================================

:CHECK_PYTHON

echo [2/9] Python 확인...

set "PYTHON_CMD=python"
%PYTHON_CMD% --version > nul 2>&1

if errorlevel 1 (
    py -3 --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)

%PYTHON_CMD% --version > nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python이 설치되어 있지 않습니다.
    echo.
    choice /c YN /n /m "Python을 설치하시겠습니까? (Y/N): "
    if errorlevel 2 (
        echo Python 설치를 취소했습니다.
    ) else (
        echo Python 설치를 시도합니다...
        winget install --id Python.Python.3.10 -e --source winget
        if errorlevel 1 (
            echo [ERROR] Python 설치에 실패했습니다.
            pause
            exit /b 1
        )
        echo Python 설치가 완료되었습니다. 새 터미널에서 다시 실행해 주세요.
        echo 필요시 로그아웃이 필요할 수 있습니다.
    )
    pause
    exit /b 1
)

echo ==========================================
echo Python 모듈 확인
echo ==========================================

set "MISSING="

call :CHECK_MODULE numpy numpy
call :CHECK_MODULE pandas pandas
call :CHECK_MODULE cv2 opencv-python
call :CHECK_MODULE mss mss
call :CHECK_MODULE scipy scipy
call :CHECK_MODULE pyvjoy pyvjoy
call :CHECK_MODULE PIL pillow
call :CHECK_MODULE ultralytics ultralytics

:: ==================================================
:: 누락 패키지가 없으면 종료
:: ==================================================

if not defined MISSING (
    echo ==========================================
    echo 모든 Python 의존성이 설치되어 있습니다.
    echo ==========================================
    goto CHECK_VJOY
)

:: ==================================================
:: 누락 패키지 설치
:: ==================================================

echo ==========================================
echo 누락된 패키지를 설치합니다.
echo ==========================================
echo.

%PYTHON_CMD% -m pip install --upgrade pip

if errorlevel 1 (
    echo.
    echo [WARN] pip 업데이트에 실패했습니다.
    echo 기존 pip로 설치를 계속 진행합니다.
    echo.
)

for %%P in (%MISSING%) do (
    echo.
    echo [INSTALL] %%P
    %PYTHON_CMD% -m pip install %%P

    if errorlevel 1 (
        echo.
        echo [ERROR] %%P 설치에 실패했습니다.
        echo 인터넷 연결 또는 Python 환경을 확인하세요.
        echo.
        pause
        exit /b 1
    )
)

echo.
echo [OK] 누락된 패키지 설치 완료!
echo.

:: ==================================================
:: vJoy 안내
:: ==================================================

:CHECK_VJOY

echo ==========================================
echo Python용 vJoy 확인
echo ==========================================
echo.

%PYTHON_CMD% -c "import pyvjoy" >nul 2>&1

if errorlevel 1 (
    echo [ERROR] pyvjoy 모듈이 없습니다.
    echo.
    choice /c YN /n /m "Pyvjoy을 설치하시겠습니까? (Y/N): "
    if errorlevel 2 (
        echo pyvjoy 설치를 취소했습니다.
        pause
        exit /b 1
    ) else (
        echo pyvjoy 설치를 시도합니다...
        %PYTHON_CMD% -m pip install pyvjoy
        if errorlevel 1 (
            echo [ERROR] pyvjoy 설치에 실패했습니다.
            pause
            exit /b 1
        )
        echo pyvjoy 설치가 완료되었습니다
    )
    
)
%PYTHON_CMD% -c "import pyvjoy" >nul 2>&1
if errorlevel 1 (
    echo.
    echo pyvjoy 설치에 실패했습니다.
    echo.
    pause
    exit /b 1

)

echo.
echo [OK] pyvjoy 모듈이 설치되어 있습니다.
echo 주의:
echo pyvjoy는 Python 모듈이고,
echo 실제 vJoy 드라이버는 Windows에 별도로 설치되어 있어야 합니다.
echo.
echo 현재 FlyRider에서 vJoy가 정상 작동한다면 추가 작업은 필요 없습니다.


::========================================
::         FlyRider vJoy 검사
::========================================


echo [3/9] vJoy 드라이버 확인...
sc query vJoy >nul 2>&1

if %errorlevel%==0 (
    echo [OK] vJoy 드라이버가 설치되어 있습니다.
) else (
    echo [X] vJoy 드라이버를 찾을 수 없습니다.
    echo     vJoy를 설치해 주세요.
    echo     vJoy 다운로드: https://downloads.sourceforge.net/project/vjoystick/Beta%202.x/2.1.9.1-160719/vJoySetup.exe?ts=gAAAAABquySvB1aIeQiMnFk-VyCQof470fNn34H3aFii3UEHABIdkK6gs6Z6YSA--riHOK4x-AixG93KSS-ZKqKll_ac48W-Zw%3D%3D&r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Fvjoystick%2Ffiles%2Flatest%2Fdownload
    pause
    exit /b 1
)

::==================================================
:: 최종 확인
::==================================================

echo ==========================================
echo       FlyRider 의존성 확인 완료
echo ==========================================
echo.
echo Python packages:
echo   numpy
echo   pandas
echo   opencv-python
echo   mss
echo   scipy
echo   pyvjoy
echo   pillow
echo.
echo vJoy driver: 설치됨
echo.



:: ==========================================
:: 메인 프로그램 설치(업데이트)/확인
:: ==========================================

echo [4/9] 메인 프로그램 확인...

if exist "%PROJECT_DIR%\FlyRider_connectome.py" (
    echo [OK] 현재 폴더에서 프로젝트를 실행합니다.
) else (
    rem A new destination folder may contain only this launcher. Clone to a
    rem temporary folder, then copy files here to avoid an extra FlyRider level.
    dir /b /a "%PROJECT_DIR%" | findstr /v /i /x "run.bat" >nul
    if not errorlevel 1 (
        echo.
        echo [ERROR] 이 폴더에 메인 프로그램이 없습니다.
        echo 빈 프로젝트 폴더에 전체 프로젝트를 복사하거나 run.bat만 두고 실행해 주세요.
        pause
        exit /b 1
    )
    echo [DOWNLOAD] 임시 폴더에 프로젝트를 받습니다...
    if not defined TEMP set "TEMP=%PROJECT_DIR%\.tmp"
    if not exist "!TEMP!" mkdir "!TEMP!"
    set "CLONE_DIR=!TEMP!\FlyRider_setup_%RANDOM%_%RANDOM%"
    git clone "https://github.com/%GITHUB_USER%/%GITHUB_REPO%.git" "!CLONE_DIR!"
    if errorlevel 1 (
        echo.
        echo [ERROR] 메인 프로그램 다운로드에 실패했습니다.
        pause
        exit /b 1
    )
    robocopy "!CLONE_DIR!" "%PROJECT_DIR%" /E /COPY:DAT /R:2 /W:1 /XF run.bat >nul
    if errorlevel 8 (
        echo.
        echo [ERROR] 프로젝트 파일을 대상 폴더로 복사하지 못했습니다.
        pause
        exit /b 1
    )
    rmdir /s /q "!CLONE_DIR!"
    if not exist "%PROJECT_DIR%\FlyRider_connectome.py" (
        echo.
        echo [ERROR] 복사한 프로젝트에서 메인 프로그램을 찾을 수 없습니다.
        pause
        exit /b 1
    )
    echo [OK] 프로젝트 파일을 현재 폴더에 설치했습니다.
)

cd /d "%PROJECT_DIR%"
echo [OK] 프로젝트 경로: %PROJECT_DIR%
echo.

:: ==========================================
:: 데이터 폴더
:: ==========================================

if not exist "data" mkdir "data"

:: ==========================================
:: neurons.csv
:: ==========================================

echo [5/9] Connectome 데이터 확인...

if exist "data\%DATA1%" (
    echo [OK] %DATA1% 이미 존재합니다.
) else (
    echo [DOWNLOAD] %DATA1%
    
    curl.exe -L --fail --progress-bar ^
        "%BASE_URL%/%DATA1%" ^
        -o "data\%DATA1%"

    if errorlevel 1 (
        echo.
        echo [ERROR] %DATA1% 다운로드 실패
        echo.
        echo Release 주소:
        echo %BASE_URL%/%DATA1%
        echo.
        pause
        exit /b 1
    )

    echo [OK] %DATA1% 다운로드 완료
)

echo.

:: ==========================================
:: connections_princeton.csv
:: ==========================================

if exist "data\%DATA2%" (
    echo [OK] %DATA2% 이미 존재합니다.
) else (
    echo [DOWNLOAD] %DATA2%
    echo.
    echo 약 213MB 정도이므로 시간이 걸릴 수 있습니다.
    echo.

    curl.exe -L --fail --progress-bar ^
        "%BASE_URL%/%DATA2%" ^
        -o "data\%DATA2%"

    if errorlevel 1 (
        echo.
        echo [ERROR] %DATA2% 다운로드 실패
        echo.
        echo Release 주소:
        echo %BASE_URL%/%DATA2%
        echo.
        pause
        exit /b 1
    )

    echo [OK] %DATA2% 다운로드 완료
)

echo.
:: ==========================================
:: %DATA3%
:: ==========================================

if exist "data\%DATA3%" (
    echo [OK] %DATA3% 이미 존재합니다.
) else (
    echo [DOWNLOAD] %DATA3%
    echo.

    curl.exe -L --fail --progress-bar ^
        "%BASE_URL%/%DATA3%" ^
        -o "data\%DATA3%"

    if errorlevel 1 (
        echo.
        echo [ERROR] %DATA3% 다운로드 실패
        echo.
        echo Release 주소:
        echo %BASE_URL%/%DATA3%
        echo.
        pause
        exit /b 1
    )

    echo [OK] %DATA3% 다운로드 완료
)

echo.

:: ==========================================
:: Python 패키지 확인
:: ==========================================

echo [7/9] Python 패키지 확인...

if exist "requirements.txt" (
    echo requirements.txt 발견
    echo 패키지를 확인합니다.
    echo.
    
    %PYTHON_CMD% -m pip install -r requirements.txt

    if errorlevel 1 (
        echo.
        echo [WARNING] 일부 패키지 설치에 실패했습니다.
        echo.
    )
) else (
    echo requirements.txt가 없습니다.
    echo 패키지 설치를 건너뜁니다.
)

echo.

:: ==========================================
:: 데이터 경로 환경변수
:: ==========================================

echo [8/9] 데이터 경로 설정...

set "CHOPARI_DATA_DIR=%PROJECT_DIR%\data"

echo.
echo Connectome 데이터:
echo %CHOPARI_DATA_DIR%
echo.

:: ==========================================
:: 프로그램 실행
:: ==========================================


echo.
echo ==========================================
echo.
echo [9/9] FlyRider 실행...
echo.
echo.
if not exist "%PROJECT_DIR%\FlyRider_connectome.py" (
    echo [ERROR] 메인 파일이 없습니다: %PROJECT_DIR%\FlyRider_connectome.py
    pause
    exit /b 1
)

if not exist "%PROJECT_DIR%\main.py" (
    echo [ERROR] GUI 실행 파일이 없습니다: %PROJECT_DIR%\main.py
    pause
    exit /b 1
)

echo [9/9] FlyRider Control Hub 실행...
for /f "delims=" %%P in ('%PYTHON_CMD% -c "import os,sys; print(os.path.join(os.path.dirname(sys.executable),'pythonw.exe'))"') do set "PYTHONW_CMD=%%P"
if exist "%PYTHONW_CMD%" (
    start "FlyRider Control Hub" /D "%PROJECT_DIR%" "%PYTHONW_CMD%" "%PROJECT_DIR%\main.py"
) else (
    echo [INFO] pythonw.exe를 찾지 못해 Python GUI 실행을 사용합니다.
    %PYTHON_CMD% "%PROJECT_DIR%\main.py"
)
exit /b 0
:: ==================================================
:: 모듈 확인 함수
:: ==================================================

:CHECK_MODULE
set "IMPORT_NAME=%~1"
set "PIP_NAME=%~2"
echo Checking Python module: %IMPORT_NAME%...

%PYTHON_CMD% -c "import %IMPORT_NAME%" >nul 2>&1

if errorlevel 1 (
    echo [MISSING] %PIP_NAME%

    if not defined MISSING (
        set "MISSING=%PIP_NAME%"
    ) else (
        set "MISSING=%MISSING% %PIP_NAME%"
    )
) else (
    echo [OK] %PIP_NAME%
)

