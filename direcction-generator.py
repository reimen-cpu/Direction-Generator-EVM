#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVM Address Generator — Derivación de direcciones Ethereum desde seed phrases.

Implementa la cadena criptográfica completa BIP-39 → BIP-32 → BIP-44 → EVM:

  1. BIP-39: Seed phrase → Seed binario (512 bits)
     ▸ PBKDF2-HMAC-SHA512, 2048 iteraciones, salt="mnemonic"+passphrase

  2. BIP-32: Seed binario → Master private key + Master chain code
     ▸ HMAC-SHA512 con clave "Bitcoin seed"

  3. BIP-44: Derivación jerárquica para Ethereum
     ▸ Ruta: m/44'/60'/0'/0/i  (i = 0..9)
     ▸ Derivación hardened (') y normal según BIP-32

  4. Dirección EVM:
     ▸ Clave pública: multiplicación escalar en secp256k1
     ▸ Clave pública sin comprimir (quitar prefijo 0x04)
     ▸ Hash: Keccak-256 (NO SHA3-256, difieren en padding)
     ▸ Dirección: últimos 20 bytes del hash, prefijo 0x
"""

import os
import sys
import hashlib
import hmac
import struct
import tkinter as tk
import tempfile
from tkinter import filedialog, messagebox, scrolledtext, ttk
from pathlib import Path

# Verificación de dependencias
try:
    from Crypto.Hash import keccak as _keccak
except ImportError:
    print("❌ Error: Falta la librería 'pycryptodome'.")
    print("Instálala ejecutando: pip install pycryptodome")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
#  secp256k1 — Aritmética de curva elíptica (Python puro)
# ═══════════════════════════════════════════════════════════════════════

_P  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
_N  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
_Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
_Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


def _modinv(a: int, m: int) -> int:
    """Inverso modular usando el pequeño teorema de Fermat (m primo).
    Maneja correctamente entradas negativas normalizando con módulo."""
    return pow(a % m, m - 2, m)


def _point_add(P, Q):
    """Suma de dos puntos en secp256k1. Maneja la duplicación correctamente."""
    if P is None:
        return Q
    if Q is None:
        return P
    
    x1, y1 = P
    x2, y2 = Q
    
    if x1 == x2:
        if y1 == y2:
            # Duplicación de punto
            lam = (3 * x1 * x1 * _modinv(2 * y1, _P)) % _P
        else:
            # P + (-P) = Punto en el infinito
            return None  
    else:
        # Suma de puntos distintos
        lam = ((y2 - y1) * _modinv(x2 - x1, _P)) % _P
        
    x3 = (lam * lam - x1 - x2) % _P
    y3 = (lam * (x1 - x3) - y1) % _P
    return (x3, y3)


def _point_mul(k: int, P) -> tuple:
    """Multiplicación escalar por el método double-and-add."""
    R = None
    Q = P
    while k:
        if k & 1:
            R = _point_add(R, Q)
        Q = _point_add(Q, Q)
        k >>= 1
    return R


def _privkey_to_pubkey_uncompressed(privkey: bytes) -> bytes:
    """Devuelve la clave pública sin comprimir (65 bytes, prefijo 0x04).
    Asegura padding de 32 bytes para retener los ceros iniciales."""
    k = int.from_bytes(privkey, "big")
    x, y = _point_mul(k, (_Gx, _Gy))
    return b'\x04' + x.to_bytes(32, "big") + y.to_bytes(32, "big")


def _privkey_to_pubkey_compressed(privkey: bytes) -> bytes:
    """Devuelve la clave pública comprimida (33 bytes)."""
    k = int.from_bytes(privkey, "big")
    x, y = _point_mul(k, (_Gx, _Gy))
    prefix = b'\x02' if y % 2 == 0 else b'\x03'
    return prefix + x.to_bytes(32, "big")


# ═══════════════════════════════════════════════════════════════════════
#  Keccak-256
# ═══════════════════════════════════════════════════════════════════════

def _keccak256(data: bytes) -> bytes:
    """Keccak-256 vía pycryptodome."""
    k = _keccak.new(digest_bits=256)
    k.update(data)
    return k.digest()


# ═══════════════════════════════════════════════════════════════════════
#  BIP-39: mnemonic → seed
# ═══════════════════════════════════════════════════════════════════════

def _mnemonic_to_seed(mnemonic: str, passphrase: str = "") -> bytes:
    """PBKDF2-HMAC-SHA512, 2048 iteraciones, salt='mnemonic'+passphrase."""
    password = mnemonic.strip().encode("utf-8")
    salt = ("mnemonic" + passphrase).encode("utf-8")
    return hashlib.pbkdf2_hmac("sha512", password, salt, 2048)


# ═══════════════════════════════════════════════════════════════════════
#  BIP-32: derivación jerárquica de claves
# ═══════════════════════════════════════════════════════════════════════

def _derive_master(seed: bytes):
    """Devuelve (master_privkey, master_chaincode) desde la semilla."""
    I = hmac.new(b"Bitcoin seed", seed, hashlib.sha512).digest()
    return I[:32], I[32:]


def _derive_child(privkey: bytes, chaincode: bytes, index: int):
    """Derivación de clave hija BIP-32."""
    if index >= 0x80000000:
        # Hardened: data = 0x00 || privkey || index (BE 4 bytes)
        data = b'\x00' + privkey + struct.pack(">I", index)
    else:
        # Normal: data = pubkey_comprimida || index
        data = _privkey_to_pubkey_compressed(privkey) + struct.pack(">I", index)

    I = hmac.new(chaincode, data, hashlib.sha512).digest()
    IL, IR = I[:32], I[32:]
    child_key = (int.from_bytes(IL, "big") + int.from_bytes(privkey, "big")) % _N
    # Siempre to_bytes(32) para evitar perder el padding
    return child_key.to_bytes(32, "big"), IR


def _derive_path(seed: bytes, path: str):
    """Deriva la clave privada a lo largo de un path BIP-44."""
    privkey, chaincode = _derive_master(seed)
    parts = path.split("/")
    for part in parts[1:]:  # saltar "m"
        hardened = part.endswith("'")
        idx = int(part.rstrip("'"))
        if hardened:
            idx += 0x80000000
        privkey, chaincode = _derive_child(privkey, chaincode, idx)
    return privkey, chaincode


# ═══════════════════════════════════════════════════════════════════════
#  EVM address desde clave privada
# ═══════════════════════════════════════════════════════════════════════

def _privkey_to_address(privkey: bytes) -> tuple[str, str]:
    """Devuelve (address_checksum, privkey_hex)."""
    pub_uncompressed = _privkey_to_pubkey_uncompressed(privkey)
    pub_body = pub_uncompressed[1:]          # 64 bytes, sin prefijo 0x04
    h = _keccak256(pub_body)
    addr_raw = h[-20:].hex()                 # últimos 20 bytes

    # EIP-55 checksum
    addr_hash = _keccak256(addr_raw.encode("ascii")).hex()
    checksum = "".join(
        c.upper() if int(addr_hash[i], 16) >= 8 else c
        for i, c in enumerate(addr_raw)
    )
    return "0x" + checksum, "0x" + privkey.hex()


# ═══════════════════════════════════════════════════════════════════════
#  Procesamiento completo: archivo de frases → direcciones (CPU puro)
# ═══════════════════════════════════════════════════════════════════════

def process_phrases_file(
    file_path: str,
    num_addresses: int,
    output_path: str,
    passphrase: str = "",
    log_fn=None,
) -> int:
    """Lee seed phrases, deriva EVM accounts y escribe en el archivo de salida."""
    def log(msg: str):
        if log_fn:
            log_fn(msg)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            total_phrases = len(lines)
    except Exception as e:
        log(f"❌ Error al leer archivo: {e}")
        return 0

    if total_phrases == 0:
        log("⚠ El archivo está vacío o no contiene frases.")
        return 0

    log(f"📂 Archivo: {os.path.basename(file_path)}")
    log(f"   Frases encontradas: {total_phrases}")
    log(f"   Direcciones por frase: {num_addresses} (i=0..{num_addresses - 1})")
    log(f"   Ruta BIP-44: m/44'/60'/0'/0/i")
    log("")

    try:
        with open(output_path, "w", encoding="utf-8") as out:
            for phrase_idx, mnemonic in enumerate(lines, 1):
                log(f"🔑 [{phrase_idx}/{total_phrases}] Procesando frase...")
                seed = _mnemonic_to_seed(mnemonic, passphrase)
                for i in range(num_addresses):
                    path = f"m/44'/60'/0'/0/{i}"
                    privkey, _ = _derive_path(seed, path)
                    address, privkey_hex = _privkey_to_address(privkey)
                    out.write(f"{mnemonic} | {path} | {privkey_hex} | {address}\n")
                log(f"   ✅ {num_addresses} dirección(es) derivada(s)")
    except Exception as e:
        log(f"❌ Error durante la generación: {e}")
        return 0

    log("")
    log(f"✅ Proceso finalizado. {total_phrases} frase(s) procesada(s).")
    log(f"   Total de direcciones: {total_phrases * num_addresses}")
    log(f"   Archivo de salida: {output_path}")
    return total_phrases


# ═══════════════════════════════════════════════════════════════════════
#  Interfaz gráfica con Tkinter
# ═══════════════════════════════════════════════════════════════════════

class EVMAddressGeneratorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EVM Address Generator — BIP-39/32/44")
        self.root.geometry("850x720")
        self.root.resizable(True, True)

        # Centrar ventana
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"+{x}+{y}")

        home = os.path.expanduser("~")
        desktop_dir = os.path.join(home, "Escritorio")
        if not os.path.exists(desktop_dir):
            desktop_dir = os.path.join(home, "Desktop")
        self.shared_dir = os.path.join(desktop_dir, "seed-tools-txt")
        os.makedirs(self.shared_dir, exist_ok=True)

        self.selected_file: str = ""
        self._build_ui()

    def _build_ui(self):
        bg = "#1e1e2e"
        fg = "#cdd6f4"
        accent = "#f9e2af"
        btn_bg = "#313244"
        btn_active = "#45475a"
        entry_bg = "#313244"
        font_main = ("Segoe UI", 10)
        font_title = ("Segoe UI", 14, "bold")
        font_mono = ("Consolas", 9)

        self.root.configure(bg=bg)

        # Título
        tk.Label(self.root, text="🔗  EVM Address Generator", font=font_title, bg=bg, fg=accent).pack(pady=(15, 3))
        tk.Label(self.root, text="BIP-39 → BIP-32 → BIP-44 → secp256k1 → Keccak-256", font=("Segoe UI", 9), bg=bg, fg="#a6adc8").pack(pady=(0, 12))

        # Notebook (Pestañas)
        style = ttk.Style()
        style.theme_use('default')
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", background=btn_bg, foreground=fg, padding=[10, 5], font=font_main)
        style.map("TNotebook.Tab", background=[("selected", accent)], foreground=[("selected", "#1e1e2e")])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=False, padx=20, pady=5)

        tab_lote = tk.Frame(self.notebook, bg=bg)
        self.notebook.add(tab_lote, text="Desde Archivo (Lote)")

        tab_single = tk.Frame(self.notebook, bg=bg)
        self.notebook.add(tab_single, text="Frase Individual")

        # TAB 1: DESDE ARCHIVO
        frame_file = tk.Frame(tab_lote, bg=bg)
        frame_file.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(frame_file, text="Archivo de seed phrases (.txt):", font=font_main, bg=bg, fg=fg).pack(side=tk.LEFT)
        tk.Button(frame_file, text="Seleccionar archivo…", font=font_main, bg=btn_bg, fg=fg, activebackground=btn_active, cursor="hand2", command=self._select_file).pack(side=tk.RIGHT)

        self.file_var = tk.StringVar(value="Ningún archivo seleccionado.")
        tk.Label(tab_lote, textvariable=self.file_var, font=("Segoe UI", 9), bg=bg, fg="#a6adc8").pack(fill=tk.X, padx=25, pady=(0, 10))

        frame_pass_lote = tk.Frame(tab_lote, bg=bg)
        frame_pass_lote.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(frame_pass_lote, text="Secret Key / Passphrase (opcional):", font=font_main, bg=bg, fg=fg).pack(side=tk.LEFT)
        self.pass_lote_entry = tk.Entry(frame_pass_lote, font=font_main, bg=entry_bg, fg=fg, insertbackground=fg, relief=tk.FLAT, width=20, show="*")
        self.pass_lote_entry.pack(side=tk.RIGHT, padx=(10, 0))

        frame_qty = tk.Frame(tab_lote, bg=bg)
        frame_qty.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(frame_qty, text="Direcciones por frase (i=0..n):", font=font_main, bg=bg, fg=fg).pack(side=tk.LEFT)
        self.qty_entry = tk.Entry(frame_qty, font=font_main, bg=entry_bg, fg=fg, insertbackground=fg, relief=tk.FLAT, width=10, justify=tk.CENTER)
        self.qty_entry.insert(0, "1")  # <-- 1 POR DEFECTO
        self.qty_entry.pack(side=tk.RIGHT, padx=(10, 0))

        frame_btn_lote = tk.Frame(tab_lote, bg=bg)
        frame_btn_lote.pack(pady=15)
        self.generate_btn = tk.Button(frame_btn_lote, text="⚡  Derivar Lote → direcctions.txt", font=("Segoe UI", 11, "bold"), bg=accent, fg="#1e1e2e", activebackground="#fcecc8", cursor="hand2", command=self._generate, padx=20, pady=8)
        self.generate_btn.pack()

        # TAB 2: FRASE INDIVIDUAL
        frame_single_phrase = tk.Frame(tab_single, bg=bg)
        frame_single_phrase.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(frame_single_phrase, text="Seed phrase (12 o 24 palabras):", font=font_main, bg=bg, fg=fg).pack(anchor=tk.W)
        self.single_phrase_text = tk.Text(frame_single_phrase, font=font_main, bg=entry_bg, fg=fg, insertbackground=fg, relief=tk.FLAT, height=3, wrap=tk.WORD)
        self.single_phrase_text.pack(fill=tk.X, pady=(5, 10))

        frame_pass_single = tk.Frame(tab_single, bg=bg)
        frame_pass_single.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(frame_pass_single, text="Secret Key / Passphrase (opcional):", font=font_main, bg=bg, fg=fg).pack(side=tk.LEFT)
        self.pass_single_entry = tk.Entry(frame_pass_single, font=font_main, bg=entry_bg, fg=fg, insertbackground=fg, relief=tk.FLAT, width=25, show="*")
        self.pass_single_entry.pack(side=tk.RIGHT, padx=(10, 0))

        frame_btn_single = tk.Frame(tab_single, bg=bg)
        frame_btn_single.pack(pady=15)
        self.generate_single_btn = tk.Button(frame_btn_single, text="⚡  Generar 1 Dirección", font=("Segoe UI", 11, "bold"), bg=accent, fg="#1e1e2e", activebackground="#fcecc8", cursor="hand2", command=self._generate_single, padx=20, pady=8)
        self.generate_single_btn.pack()

        # Consola de LOG
        tk.Label(self.root, text="Registro de operaciones:", font=font_main, bg=bg, fg=fg).pack(anchor=tk.W, padx=20, pady=(10, 0))
        self.log_area = scrolledtext.ScrolledText(self.root, font=font_mono, bg="#11111b", fg="#a6e3a1", insertbackground=fg, relief=tk.FLAT, height=18, state=tk.DISABLED)
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=20, pady=(5, 15))

    def _select_file(self):
        f = filedialog.askopenfilename(
            title="Seleccionar archivo de seed phrases",
            initialdir=self.shared_dir,
            filetypes=[("Archivos de texto", "*.txt"), ("Todos", "*.*")],
        )
        if f:
            self.selected_file = f
            self.file_var.set(f"📄 {os.path.basename(f)}")

    def _log(self, msg: str):
        self.log_area.configure(state=tk.NORMAL)
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)
        self.log_area.configure(state=tk.DISABLED)
        self.root.update_idletasks()

    def _generate(self):
        self.log_area.configure(state=tk.NORMAL)
        self.log_area.delete("1.0", tk.END)
        self.log_area.configure(state=tk.DISABLED)

        if not self.selected_file:
            messagebox.showwarning("Sin archivo", "Selecciona un archivo .txt con seed phrases.")
            return

        qty_str = self.qty_entry.get().strip()
        if not qty_str.isdigit() or int(qty_str) < 1:
            messagebox.showwarning("Cantidad inválida", "Ingresa un número entero positivo para la cantidad de direcciones.")
            return

        num_addresses = int(qty_str)
        output_path = os.path.join(self.shared_dir, "direcctions.txt")

        self._log("🔗 EVM Address Generator")
        self._log("   Aceleración: CPU (Python puro + PyCryptodome)")
        self._log("")

        self.generate_btn.configure(state=tk.DISABLED)
        self.root.update_idletasks()

        try:
            passphrase = self.pass_lote_entry.get()
            total = process_phrases_file(self.selected_file, num_addresses, output_path, passphrase, self._log)

            if total > 0:
                messagebox.showinfo("Completado", f"Derivación finalizada.\nTotal procesado: {total}\nArchivo generado:\n{output_path}")
            else:
                messagebox.showwarning("Sin resultados", "No se procesó ninguna frase. Revisa el registro.")
        except Exception as e:
            self._log(f"\n⚠ Error inesperado: {e}")
        finally:
            self.generate_btn.configure(state=tk.NORMAL)

    def _generate_single(self):
        phrase = self.single_phrase_text.get("1.0", tk.END).strip()
        passphrase = self.pass_single_entry.get()

        if not phrase:
            messagebox.showwarning("Falta frase", "Por favor ingresa una seed phrase.")
            return

        self.log_area.configure(state=tk.NORMAL)
        self.log_area.delete("1.0", tk.END)
        self.log_area.configure(state=tk.DISABLED)

        self._log("⚡ Generando dirección para frase individual...")
        self.generate_single_btn.configure(state=tk.DISABLED)
        self.root.update_idletasks()

        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f_in:
                f_in.write(phrase.replace('\n', ' ') + "\n")
                temp_in = f_in.name

            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f_out:
                temp_out = f_out.name

            total = process_phrases_file(temp_in, 1, temp_out, passphrase, self._log)

            if total > 0:
                with open(temp_out, "r", encoding="utf-8") as f:
                    for line in f:
                        if " | " in line:
                            parts = line.strip().split(" | ")
                            if len(parts) >= 4:
                                privkey = parts[2]
                                address = parts[3]
                                self._log("\n🎉 ¡Dirección generada exitosamente!")
                                self._log(f"Dirección: {address}")
                                self._log(f"Clave Privada: {privkey}")
                                messagebox.showinfo("Dirección Generada", f"Dirección:\n{address}\n\nRevisa el registro para ver la clave privada.")
                            break

            os.remove(temp_in)
            os.remove(temp_out)

        except Exception as e:
            self._log(f"\n⚠ Error inesperado: {e}")
        finally:
            self.generate_single_btn.configure(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = EVMAddressGeneratorApp(root)
    root.mainloop()
