# Release v2.0.0 — 25 April 2026

> Generato automaticamente da DELTA il 25/04/2026 00:43

## Changelog

- feat: aggiungi comando globale 'delta' e aggiorna manuale sezione 5.1
- fix: setup_raspberry.py — ai-edge-litert check e gestione NameError opzionali
- fix: install_raspberry.sh — usa python3.12 per venv, fix rsync exclude, fix TFLite
- docs: aggiorna manuale con protezione .gitignore e file LICENSE
- chore: aggiungi file LICENSE (DELTA 2.0 Software License)
- chore: proteggi dati operativi locali con .gitignore
- docs: allinea manuale §20 alle modifiche privacy GitHub Publisher
- privacy: rimuovi dati operativi locali dalla pubblicazione GitHub
- docs: aggiorna manuale con sezione GitHub Publisher (§20)
- feat: aggiungi GitHub Publisher nel Pannello Amministratore [7]
- docs: allinea manuale al fix ai-edge-litert==1.2.0 su RPi5
- fix: pin ai-edge-litert==1.2.0 per evitare segfault su RPi5 aarch64
- DELTA 2.0 — Pubblicazione ufficiale progetto completo

## Informazioni tecniche

| | |
|---|---|
| Classi modello AI | 7 |
| Dimensione modello | 2675 KB |
| Branch | `main` |
| Tag precedente | `N/A` |

## Note di installazione

- Raspberry Pi 5 (aarch64): `pip install ai-edge-litert==1.2.0`
- Versioni `ai-edge-litert >= 1.3.0` causano segfault su BCM2712 — **non aggiornare**
- Python 3.12 richiesto — Python ≥ 3.13 non supportato da TensorFlow/TFLite

---
*Pubblicato con DELTA GitHub Publisher — `interface/github_publisher.py`*
