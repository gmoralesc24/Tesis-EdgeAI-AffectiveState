@echo off
echo ==============================================
echo  Configuracion de Entorno Edge AI - Tesis MIA
echo ==============================================

REM 1. Crear entorno virtual si no existe
if not exist ".venv" (
    echo Creando entorno virtual .venv...
    python -m venv .venv
) else (
    echo Entorno .venv ya existe.
)

REM 2. Activar
echo Activando entorno...
call .venv\Scripts\activate

REM 3. Instalar dependencias
echo Instalando dependencias desde requirements.txt...
pip install -r requirements.txt
pip install ipykernel notebook jupyterlab kaggle

REM 4. Registrar kernel para Jupyter
echo Registrando kernel para Jupyter...
python -m ipykernel install --user --name=edge_ai_env --display-name "Python (Edge AI Tesis)"

echo.
echo ==============================================
echo  Instalacion Completada.
echo  Para activar: .venv\Scripts\activate
echo  Para Jupyter: jupyter lab
echo ==============================================
pause
