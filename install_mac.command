#!/bin/sh
# Double-click in Finder: installs Drawing2FEA and puts Drawing2FEA.app on the Desktop.
cd "$(dirname "$0")" || exit 1
PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python 3.10+ 이 필요합니다: https://www.python.org/downloads/"
  read -r _
  exit 1
fi
echo "[1/2] 필요한 라이브러리를 설치합니다..."
"$PY" -m pip install --user -e . || { echo "설치 실패"; read -r _; exit 1; }
echo "[2/2] 바탕화면에 Drawing2FEA.app 을 만듭니다..."
"$PY" -m drawing2fea shortcut
echo "완료! 바탕화면의 Drawing2FEA 아이콘을 더블 클릭하세요."
read -r _
