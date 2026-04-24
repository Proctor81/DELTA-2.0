"""
DELTA - core/auth.py
Modulo di autenticazione per il pannello amministratore.
Gestisce hashing password (PBKDF2-SHA256 + salt random), verifica e
aggiornamento credenziali. Le credenziali sono salvate in data/auth.json.
"""

import hashlib
import hmac
import json
import secrets
from pathlib import Path

_AUTH_FILE = Path(__file__).resolve().parent.parent / "data" / "auth.json"
_DEFAULT_PASSWORD = "Mol036k6*Mol036k6*"
_ITERATIONS = 260_000          # PBKDF2 cost factor


# ── Hashing ────────────────────────────────────────────────────────────────

def _hash_password(password: str, salt: str) -> str:
    """Restituisce l'hash PBKDF2-SHA256 della password con il salt dato."""
    raw = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=_ITERATIONS,
    )
    return raw.hex()


def _write_auth(salt: str, hashed: str) -> None:
    _AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    _AUTH_FILE.write_text(
        json.dumps({"salt": salt, "hash": hashed}, indent=2),
        encoding="utf-8",
    )


def _read_auth() -> dict:
    if not _AUTH_FILE.exists():
        initialize_password()
    return json.loads(_AUTH_FILE.read_text(encoding="utf-8"))


# ── API pubblica ───────────────────────────────────────────────────────────

def initialize_password() -> None:
    """Crea auth.json con la password di default se non esiste ancora."""
    if not _AUTH_FILE.exists():
        salt = secrets.token_hex(32)
        _write_auth(salt, _hash_password(_DEFAULT_PASSWORD, salt))


def verify_password(password: str) -> bool:
    """Verifica la password in ingresso contro quella salvata (constant-time)."""
    data = _read_auth()
    expected = _hash_password(password, data["salt"])
    return hmac.compare_digest(expected, data["hash"])


def change_password(old_password: str, new_password: str) -> tuple[bool, str]:
    """
    Cambia la password amministratore.

    Returns:
        (True, "")         — in caso di successo
        (False, messaggio) — in caso di errore
    """
    if not verify_password(old_password):
        return False, "Password corrente non corretta."
    if len(new_password) < 8:
        return False, "La nuova password deve avere almeno 8 caratteri."
    salt = secrets.token_hex(32)
    _write_auth(salt, _hash_password(new_password, salt))
    return True, "Password aggiornata con successo."
