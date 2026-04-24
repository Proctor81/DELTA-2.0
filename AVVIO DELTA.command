#!/bin/bash
# ═══════════════════════════════════════════════════════
#  AVVIO DELTA — Launcher macOS
#  Doppio clic per avviare il sistema DELTA AI Agent.
#  Non richiede password. Accesso hardware completo.
# ═══════════════════════════════════════════════════════

# Posizionati nella directory del file (compatibile con doppio clic)
cd "$(dirname "$0")"

# Se esiste un virtualenv locale, usa quello.
# In alternativa ripiega su python3 di sistema.
if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo "✘ Python 3 non trovato. Installare da https://www.python.org"
    read -n 1 -s -r -p "Premi un tasto per chiudere..."
    exit 1
fi

# Avvia DELTA
"$PYTHON_BIN" AVVIO_DELTA.py

echo ""
echo "─────────────────────────────────────────"
echo "  DELTA si è chiuso. Premi un tasto..."
echo "─────────────────────────────────────────"
read -n 1 -s -r
