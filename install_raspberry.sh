#!/usr/bin/env bash
# =============================================================================
# DELTA AI Agent — Installazione automatica Raspberry Pi 5
# Compatibile con: Raspberry Pi OS (64-bit, Bookworm o superiore)
# Richiede Python 3.12+
# Hardware target: Raspberry Pi 5 (4/8/16 GB) + AI HAT 2+
# =============================================================================
# Utilizzo:
#   chmod +x install_raspberry.sh
#   sudo ./install_raspberry.sh
# =============================================================================

set -euo pipefail

# ─── Colori terminale ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; }
ok()      { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}   $*"; }
error()   { echo -e "${RED}[ERROR]${RESET}  $*" >&2; }
header()  { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════${RESET}"; \
            echo -e "${BOLD}${BLUE}  $*${RESET}"; \
            echo -e "${BOLD}${BLUE}══════════════════════════════════════════${RESET}"; }

# ─── Controllo root ───────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    error "Questo script deve essere eseguito come root (sudo)."
    exit 1
fi

# ─── Variabili configurabili ──────────────────────────────────────────────────
DELTA_USER="${SUDO_USER:-pi}"                         # utente non-root che esegue DELTA
DELTA_HOME=$(getent passwd "$DELTA_USER" | cut -d: -f6)
DELTA_DIR="${DELTA_HOME}/DELTA"                       # directory installazione
PYTHON_MIN="3.12"
SERVICE_NAME="delta"
VENV_DIR="${DELTA_DIR}/.venv"

header "DELTA AI Agent — Setup Raspberry Pi 5"
echo ""
info "Utente target:     ${DELTA_USER}"
info "Directory target:  ${DELTA_DIR}"
info "Python minimo:     ${PYTHON_MIN}"
echo ""

# ─── 1. Aggiornamento sistema ─────────────────────────────────────────────────
header "1/9  Aggiornamento sistema operativo"
apt-get update -qq
apt-get upgrade -y -qq
ok "Sistema aggiornato."

# ─── 2. Dipendenze di sistema ─────────────────────────────────────────────────
header "2/9  Installazione dipendenze di sistema"
PKGS=(
    python3 python3-pip python3-venv python3-dev
    python3-numpy python3-opencv
    libatlas-base-dev libhdf5-dev libhdf5-serial-dev
    libopenblas-dev libblas-dev liblapack-dev
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
    git curl wget unzip
    i2c-tools python3-smbus
    libcamera-dev libcamera-tools python3-libcamera
    # Picamera2 dipende da:
    python3-picamera2
    # Font per la generazione del manuale
    fonts-liberation
)
apt-get install -y -qq "${PKGS[@]}" || warn "Alcuni pacchetti potrebbero non essere disponibili."
ok "Dipendenze di sistema installate."

# ─── 3. Abilitazione interfacce hardware ─────────────────────────────────────
header "3/9  Configurazione hardware (I2C, Camera, SPI)"
# I2C
if ! grep -q "^dtparam=i2c_arm=on" /boot/firmware/config.txt 2>/dev/null; then
    echo "dtparam=i2c_arm=on" >> /boot/firmware/config.txt
    info "I2C abilitato in /boot/firmware/config.txt"
fi
# SPI
if ! grep -q "^dtparam=spi=on" /boot/firmware/config.txt 2>/dev/null; then
    echo "dtparam=spi=on" >> /boot/firmware/config.txt
    info "SPI abilitato."
fi
# Camera (libcamera stack, Pi 5)
if ! grep -q "^camera_auto_detect=1" /boot/firmware/config.txt 2>/dev/null; then
    echo "camera_auto_detect=1" >> /boot/firmware/config.txt
    info "Rilevamento automatico camera abilitato."
fi
# AI HAT 2+ overlay
if ! grep -q "^dtoverlay=ai-hat\+" /boot/firmware/config.txt 2>/dev/null; then
    echo "dtoverlay=ai-hat+" >> /boot/firmware/config.txt
    info "Overlay AI HAT 2+ aggiunto."
fi
# Aggiunge utente ai gruppi hardware
usermod -aG i2c,spi,gpio,video,dialout "$DELTA_USER" || true
ok "Interfacce hardware configurate."

# ─── 4. Copia sorgenti DELTA ─────────────────────────────────────────────────
header "4/9  Installazione sorgenti DELTA"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$SCRIPT_DIR" != "$DELTA_DIR" ]]; then
    info "Copia da $SCRIPT_DIR a $DELTA_DIR ..."
    rsync -a --exclude='.venv' --exclude='__pycache__' \
              --exclude='*.pyc' --exclude='delta.db-shm' --exclude='delta.db-wal' \
              "${SCRIPT_DIR}/" "${DELTA_DIR}/"
    chown -R "${DELTA_USER}:${DELTA_USER}" "${DELTA_DIR}"
else
    info "Sorgenti già nella directory di destinazione."
fi

# Crea directory necessarie
DIRS=("${DELTA_DIR}/input_images" "${DELTA_DIR}/exports" "${DELTA_DIR}/logs"
      "${DELTA_DIR}/datasets/captures" "${DELTA_DIR}/datasets/training" "${DELTA_DIR}/models")
for d in "${DIRS[@]}"; do
    mkdir -p "$d"
    chown "${DELTA_USER}:${DELTA_USER}" "$d"
done
ok "Sorgenti DELTA installati in ${DELTA_DIR}."

# ─── 5. Ambiente virtuale Python ─────────────────────────────────────────────
header "5/9  Creazione ambiente virtuale Python"
if [[ ! -d "$VENV_DIR" ]]; then
    sudo -u "$DELTA_USER" python3 -m venv --system-site-packages "$VENV_DIR"
    ok "Ambiente virtuale creato: ${VENV_DIR}"
else
    ok "Ambiente virtuale già esistente: ${VENV_DIR}"
fi

VENV_PIP="${VENV_DIR}/bin/pip"
VENV_PYTHON="${VENV_DIR}/bin/python"

# ─── 6. Installazione dipendenze Python ──────────────────────────────────────
header "6/9  Installazione dipendenze Python"
sudo -u "$DELTA_USER" "$VENV_PIP" install --upgrade pip wheel setuptools -q

# Core requirements
sudo -u "$DELTA_USER" "$VENV_PIP" install -r "${DELTA_DIR}/requirements.txt" -q || \
    warn "Alcuni pacchetti potrebbero richiedere installazione manuale."

# TFLite runtime (Raspberry Pi)
info "Tentativo installazione tflite-runtime ..."
sudo -u "$DELTA_USER" "$VENV_PIP" install tflite-runtime -q 2>/dev/null || {
    warn "tflite-runtime non disponibile via pip."
    warn "Installeremo TensorFlow 2.21.0 come fallback per il runtime TFLite."
    sudo -u "$DELTA_USER" "$VENV_PIP" install tensorflow==2.21.0 flatbuffers==25.12.19 -q || \
        warn "installazione fallback TensorFlow fallita — verificare connessione e repository pip."
}

# Adafruit sensor libraries
info "Installazione librerie sensori Adafruit ..."
ADAFRUIT_PKGS=(
    "RPi.GPIO"
    "smbus2"
    "adafruit-circuitpython-bme680"
    "adafruit-circuitpython-veml7700"
    "adafruit-circuitpython-scd4x"
    "adafruit-circuitpython-ads1x15"
    "adafruit-blinka"
)
sudo -u "$DELTA_USER" "$VENV_PIP" install "${ADAFRUIT_PKGS[@]}" -q 2>/dev/null || \
    warn "Alcune librerie sensori non installate. Verificare connettività."

# Generazione manuale PDF
sudo -u "$DELTA_USER" "$VENV_PIP" install fpdf2 -q
ok "Dipendenze Python installate."

# ─── 7. Generazione manuale PDF ──────────────────────────────────────────────
header "7/9  Generazione Manuale Utente PDF"
cd "${DELTA_DIR}"
sudo -u "$DELTA_USER" "$VENV_PYTHON" Manuale/genera_manuale.py && \
    ok "Manuale generato: ${DELTA_DIR}/Manuale/DELTA_Manuale_Utente.pdf" || \
    warn "Impossibile generare il manuale (verranno generati al primo avvio)."

# ─── 8. Servizio systemd ─────────────────────────────────────────────────────
header "8/9  Configurazione servizio systemd"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=DELTA AI Agent - Diagnosi Piante
Documentation=file://${DELTA_DIR}/Manuale/DELTA_Manuale_Utente.pdf
After=network.target

[Service]
Type=simple
User=${DELTA_USER}
WorkingDirectory=${DELTA_DIR}
ExecStart=${VENV_DIR}/bin/python ${DELTA_DIR}/main.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1
Environment=DELTA_HOME=${DELTA_DIR}

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}.service"
ok "Servizio systemd '${SERVICE_NAME}' installato e abilitato all'avvio."

# ─── 9. Script di avvio rapido ────────────────────────────────────────────────
header "9/9  Creazione script di avvio rapido"
LAUNCHER="/usr/local/bin/delta"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
# Avvio rapido DELTA AI Agent
cd "${DELTA_DIR}"
exec "${VENV_DIR}/bin/python" "${DELTA_DIR}/main.py" "\$@"
EOF
chmod +x "$LAUNCHER"
ok "Comando rapido disponibile: 'delta' (da qualsiasi directory)"

# ─── Riepilogo finale ─────────────────────────────────────────────────────────
header "Installazione completata"
echo ""
echo -e "${GREEN}${BOLD}DELTA AI Agent è stato installato correttamente!${RESET}"
echo ""
echo "  Directory:   ${DELTA_DIR}"
echo "  Python venv: ${VENV_DIR}"
echo "  Servizio:    systemctl start ${SERVICE_NAME}"
echo ""
echo -e "${BOLD}Comandi utili:${RESET}"
echo "  delta                         # Avvio manuale interattivo"
echo "  sudo systemctl start delta    # Avvio come servizio"
echo "  sudo systemctl status delta   # Stato servizio"
echo "  journalctl -u delta -f        # Log in tempo reale"
echo ""
echo -e "${BOLD}Cartella immagini input (modalità no-camera):${RESET}"
echo "  ${DELTA_DIR}/input_images/"
echo "  Copiare qui le immagini JPG/PNG da analizzare"
echo ""
echo -e "${YELLOW}${BOLD}NOTA: Riavviare il sistema per attivare I2C, SPI e Camera.${RESET}"
echo -e "  sudo reboot"
echo ""
