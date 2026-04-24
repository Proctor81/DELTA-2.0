# DELTA AI Agent

## Requisiti principali
- Raspberry Pi 5 (64-bit)
- Raspberry Pi OS Bookworm o superiore
- Python 3.12.13
- AI HAT 2+ (consigliato per l'accelerazione inferenza)

## Installazione su Raspberry Pi
1. Copia il repository in `/home/<utente>/DELTA` oppure lascia i file nella directory corrente.
2. Rendi eseguibile lo script di installazione:
   ```bash
   chmod +x install_raspberry.sh
   ```
3. Esegui l'installazione come root:
   ```bash
   sudo ./install_raspberry.sh
   ```
4. Al termine, riavvia il sistema per applicare le modifiche a I2C, SPI e camera:
   ```bash
   sudo reboot
   ```

## Ambiente Python 3.12
Lo script `install_raspberry.sh` ora richiede Python 3.12.
Se `tflite-runtime` non è disponibile, viene applicato un fallback installando:
- `tensorflow==2.21.0`
- `flatbuffers==25.12.19`

### Creazione manuale del venv
Se desideri eseguire manualmente l'ambiente virtuale:
```bash
cd /home/<utente>/DELTA
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install opencv-python-headless
python -m pip install tensorflow==2.21.0 flatbuffers==25.12.19
```

## Verifica installazione
Esegui la validazione preflight:
```bash
cd /home/<utente>/DELTA
source .venv/bin/activate
python main.py --preflight-only
```

## Avvio DELTA
- Avvio manuale: `delta`
- Avvio come servizio:
  - `sudo systemctl start delta`
  - `sudo systemctl status delta`
  - `journalctl -u delta -f`

## Generazione manuale PDF
Per rigenerare il manuale utente:
```bash
cd /home/<utente>/DELTA
source .venv/bin/activate
python Manuale/genera_manuale.py
```

## Note aggiuntive
- La directory del progetto contiene:
  - `models/` per i modelli AI
  - `input_images/` per immagini da analizzare in modalità no-camera
  - `Manuale/` per il PDF dell'utente
- Se il runtime TFLite non viene trovato, installa TensorFlow come fallback secondo le istruzioni sopra.
