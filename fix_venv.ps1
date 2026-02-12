# venv가 "No Python at C:\Python310\python.exe" 로 깨진 경우 실행
# 사용법: PowerShell에서 .\fix_venv.ps1

$venvPath = Join-Path $PSScriptRoot "venv"
$requirementsPath = Join-Path $PSScriptRoot "requirements.txt"

# 1) 기존 venv 백업 후 제거
if (Test-Path $venvPath) {
    $backup = "${venvPath}_backup_$(Get-Date -Format 'yyyyMMdd_HHmm')"
    Write-Host "기존 venv를 $backup 으로 이동합니다..."
    Rename-Item $venvPath $backup -Force
}

# 2) 새 venv 생성 (D:\python310 우선, 없으면 py -3)
if (Test-Path "D:\python310\python.exe") {
    Write-Host "새 가상환경 생성 중 (D:\python310\python.exe -m venv venv)..."
    & "D:\python310\python.exe" -m venv $venvPath
} else {
    Write-Host "새 가상환경 생성 중 (py -3 -m venv venv)..."
    py -3 -m venv $venvPath
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "오류: venv 생성 실패. D:\python310\python.exe 가 있는지 확인하세요."
    exit 1
}

# 3) 활성화 후 패키지 설치
Write-Host "패키지 설치 중..."
& "$venvPath\Scripts\pip.exe" install -r $requirementsPath --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip install 실패. 수동으로: venv\Scripts\activate 후 pip install -r requirements.txt"
    exit 1
}

Write-Host "완료. 다음처럼 실행하세요:"
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host "  uvicorn main:app --reload"
