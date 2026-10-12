@echo off
REM Abre o Cifrador de Brácsa como app no computador (precisa do Python instalado).
REM Depois que abrir no Chrome ou Edge, clique no ícone "Instalar" na barra de endereço.
cd /d "%~dp0app"
start "" http://localhost:8765/
python -m http.server 8765
