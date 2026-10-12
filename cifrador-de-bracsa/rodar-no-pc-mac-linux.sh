#!/bin/sh
# Abre o Cifrador de Brácsa como app no computador (precisa do Python 3).
# Depois que abrir no Chrome ou Edge, clique no ícone "Instalar" na barra de endereço.
cd "$(dirname "$0")/app"
( sleep 1; (open http://localhost:8765/ 2>/dev/null || xdg-open http://localhost:8765/ 2>/dev/null) ) &
python3 -m http.server 8765
