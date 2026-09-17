# ============================
# Script: ativa_venv.ps1
# Objetivo: criar/ativar venv, instalar dependências e contornar bloqueio de execução
# Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
# .\ativa_venv.ps1

# ============================
# ============================
# Script: ativa_venv.ps1
# Objetivo: criar/ativar venv, instalar dependências e contornar bloqueio de execução
# ============================
# powershell -ExecutionPolicy Bypass -File .\ativa_venv.ps1
# pip install --upgrade "openpyxl>=3.1.5"
# Exibir cabeçalho
Write-Host "==== Ativando ambiente virtual Python ====" -ForegroundColor Cyan

# Guardar política original de execução
$originalPolicy = Get-ExecutionPolicy

# Permitir execução local temporariamente
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# Caminho base do projeto
$projectPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectPath

function Get-AvailablePythonCmd {
    # Retorna um array com o comando (e opcionalmente um argumento), por exemplo @('py','-3') ou @('python')
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            & py -3 --version > $null 2>&1
            if ($LASTEXITCODE -eq 0) { return @('py','-3') }
        } catch { }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) { return @('python') }
    return $null
}

# Verifica se venv existe e se referencia um interpretador inexistente
if (Test-Path "$projectPath\venv") {
    $cfgPath = Join-Path $projectPath 'venv\pyvenv.cfg'
    $needRecreate = $false
    if (Test-Path $cfgPath) {
        $cfg = Get-Content $cfgPath | ForEach-Object { $_.Trim() }
        $homeLine = $cfg | Where-Object { $_ -like 'home*' }
        if ($homeLine) {
            $homePath = $homeLine -replace '^home\s*=\s*',''
            $homePath = $homePath.Trim()
            if (-not (Test-Path (Join-Path $homePath 'python.exe'))) {
                Write-Host "venv aponta para intérprete ausente: $homePath" -ForegroundColor Yellow
                $needRecreate = $true
            }
        }
    }
    if ($needRecreate) {
        Write-Host "Removendo venv antigo para recriar com intérprete disponível..." -ForegroundColor Yellow
        try {
            Remove-Item -Recurse -Force "$projectPath\venv"
        } catch {
            Write-Host "Falha ao remover venv antigo: $_" -ForegroundColor Red
        }
    }
}

# Criar venv se não existir (ou foi removido)
if (-Not (Test-Path "$projectPath\venv")) {
    Write-Host "Criando ambiente virtual em: $projectPath\venv" -ForegroundColor Yellow
    $pythonCmd = Get-AvailablePythonCmd
    if ($pythonCmd -ne $null) {
        if ($pythonCmd.Length -gt 1) {
            & $pythonCmd[0] $pythonCmd[1] -m venv venv
        } else {
            & $pythonCmd[0] -m venv venv
        }
    } else {
        Write-Host "Nenhum interpretador Python encontrado no PATH. Instale o Python ou use o launcher 'py'." -ForegroundColor Red
        # Restaurar política antes de sair
        Set-ExecutionPolicy -Scope Process -ExecutionPolicy $originalPolicy -Force
        exit 1
    }
}

# Ativar a venv
Write-Host "Ativando ambiente virtual..." -ForegroundColor Yellow
& "$projectPath\venv\Scripts\Activate.ps1"

# Atualizar pip (usar o python do venv quando possível)
Write-Host "Atualizando pip..." -ForegroundColor Yellow
$venvPython = Join-Path $projectPath 'venv\Scripts\python.exe'
if (Test-Path $venvPython) {
    & $venvPython -m pip install --upgrade pip
} else {
    $pythonCmd = Get-AvailablePythonCmd
    if ($pythonCmd -ne $null) {
        if ($pythonCmd.Length -gt 1) { & $pythonCmd[0] $pythonCmd[1] -m pip install --upgrade pip } else { & $pythonCmd[0] -m pip install --upgrade pip }
    } else { Write-Host "Não foi possível atualizar pip: nenhum python disponível" -ForegroundColor Red }
}

# Instalar dependências
if (Test-Path "$projectPath\requirements.txt") {
    Write-Host "Instalando dependências de requirements.txt..." -ForegroundColor Yellow
    if (Test-Path $venvPython) {
        & $venvPython -m pip install -r "$projectPath\requirements.txt"
    } else {
        $pythonCmd = Get-AvailablePythonCmd
        if ($pythonCmd -ne $null) {
            if ($pythonCmd.Length -gt 1) { & $pythonCmd[0] $pythonCmd[1] -m pip install -r "$projectPath\requirements.txt" } else { & $pythonCmd[0] -m pip install -r "$projectPath\requirements.txt" }
        } else { Write-Host "Não foi possível instalar dependências: nenhum python disponível" -ForegroundColor Red }
    }
} else {
    Write-Host "⚠️  Nenhum arquivo requirements.txt encontrado." -ForegroundColor Red
}

# Mensagem final
Write-Host "Ambiente virtual ativado e dependências instaladas!" -ForegroundColor Green

# Restaurar política original
Set-ExecutionPolicy -Scope Process -ExecutionPolicy $originalPolicy -Force