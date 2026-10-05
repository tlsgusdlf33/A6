@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   Drawing2FEA 설치 (Windows)
echo ============================================
set "PY=python"
where py >nul 2>&1 && set "PY=py -3"
%PY% --version >nul 2>&1
if errorlevel 1 (
  echo [오류] Python 3.10 이상이 필요합니다.
  echo https://www.python.org/downloads/ 에서 설치하세요. 설치 화면에서 "Add python.exe to PATH" 를 체크하세요.
  pause
  exit /b 1
)
echo [1/3] 필요한 라이브러리를 설치합니다... (처음 한 번, 수 분 소요)
%PY% -m pip install --upgrade pip >nul
%PY% -m pip install -e .
if errorlevel 1 (
  echo [오류] 라이브러리 설치에 실패했습니다. 위 메시지를 확인하세요.
  pause
  exit /b 1
)
echo [2/3] 고속 솔버(선택)를 설치합니다...
%PY% -m pip install pypardiso >nul 2>&1 || echo     (건너뜀: 기본 솔버를 사용합니다)
echo [3/3] 바탕화면과 시작 메뉴에 실행 아이콘을 만듭니다...
%PY% -m drawing2fea shortcut
echo.
echo 완료! 바탕화면의 Drawing2FEA 아이콘을 더블 클릭하세요.
pause
