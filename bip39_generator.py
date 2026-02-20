#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIP-39 Seedphrase Generator — Generador de frases mnemónicas válidas.

Genera frases BIP-39 de 12 o 24 palabras con checksum SHA-256 correcto,
para poder probar el obfuscador (bip39_obfuscator.py).

Cada frase generada cumple el estándar BIP-39 completo:
  1. Se genera entropía aleatoria criptográfica (128 o 256 bits).
  2. Se calcula el checksum: primeros CS bits de SHA-256(entropía).
  3. Se concatenan entropía + checksum en un flujo de bits.
  4. Se divide en grupos de 11 bits y se mapea cada grupo a una palabra.

Sin dependencias externas — solo stdlib de Python 3.
"""

import hashlib
import os
import secrets
import sys
import tkinter as tk
from tkinter import messagebox, scrolledtext
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════
#  Parámetros BIP-39
# ═══════════════════════════════════════════════════════════════════════
# word_count → (entropy_bits, checksum_bits)
BIP39_PARAMS = {
    12: (128, 4),
    24: (256, 8),
}


# ═══════════════════════════════════════════════════════════════════════
#  Carga de la lista de palabras BIP-39
# ═══════════════════════════════════════════════════════════════════════

def load_wordlist(path: str) -> list[str]:
    """
    Lee el archivo BIP-39 (una palabra por línea) y devuelve una lista
    ordenada de 2048 palabras.  Lanza excepción si el archivo no tiene
    exactamente 2048 entradas.
    """
    with open(path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    if len(words) != 2048:
        raise ValueError(
            f"El archivo de palabras debe contener exactamente 2048 "
            f"entradas, pero tiene {len(words)}."
        )
    return words


# ═══════════════════════════════════════════════════════════════════════
#  Generación de una frase BIP-39 válida
# ═══════════════════════════════════════════════════════════════════════

def generate_mnemonic(word_count: int, wordlist: list[str]) -> str:
    """
    Genera una frase mnemónica BIP-39 válida con checksum correcto.

    1. Genera entropía aleatoria criptográfica (secrets.token_bytes).
    2. Calcula SHA-256 de la entropía.
    3. Toma los primeros CS bits del hash como checksum.
    4. Concatena entropía + checksum en binario.
    5. Divide en grupos de 11 bits → índice → palabra.
    """
    if word_count not in BIP39_PARAMS:
        raise ValueError(f"Cantidad de palabras inválida: {word_count}. Debe ser 12 o 24.")

    ent_bits, cs_bits = BIP39_PARAMS[word_count]
    ent_bytes_len = ent_bits // 8

    # ── 1. Generar entropía aleatoria criptográfica ──
    entropy = secrets.token_bytes(ent_bytes_len)

    # ── 2. Calcular checksum SHA-256 ──
    sha256_hash = hashlib.sha256(entropy).digest()
    hash_bits = bin(int.from_bytes(sha256_hash, "big"))[2:].zfill(256)
    checksum = hash_bits[:cs_bits]

    # ── 3. Concatenar entropía + checksum ──
    entropy_int = int.from_bytes(entropy, "big")
    entropy_bits = bin(entropy_int)[2:].zfill(ent_bits)
    full_bits = entropy_bits + checksum

    # ── 4. Dividir en grupos de 11 bits → palabras ──
    words = []
    for i in range(word_count):
        chunk = full_bits[i * 11 : (i + 1) * 11]
        index = int(chunk, 2)
        words.append(wordlist[index])

    return " ".join(words)


# ═══════════════════════════════════════════════════════════════════════
#  Interfaz gráfica con Tkinter
# ═══════════════════════════════════════════════════════════════════════

class BIP39GeneratorApp:
    """Interfaz gráfica para el generador de frases BIP-39."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("BIP-39 Seedphrase Generator")
        self.root.geometry("780x650")
        self.root.resizable(True, True)

        # Intentar centrar la ventana
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"+{x}+{y}")

        # ── Cargar lista de palabras ──
        self.wordlist_path = self._find_wordlist()
        try:
            self.wordlist = load_wordlist(self.wordlist_path)
        except Exception as e:
            messagebox.showerror(
                "Error fatal",
                f"No se pudo cargar la lista BIP-39:\n{e}"
            )
            sys.exit(1)

        self._build_ui()

    # ── Buscar archivo de lista de palabras ──────────────────────────
    def _find_wordlist(self) -> str:
        """
        Busca el archivo de palabras BIP-39 en varias ubicaciones:
        1. Junto al script (mismo directorio).
        2. En el directorio de trabajo actual.
        Soporta los nombres: bip39.txt, english.txt, wordlist.txt
        """
        script_dir = Path(__file__).resolve().parent
        candidates = ["bip39.txt", "english.txt", "wordlist.txt"]

        for name in candidates:
            p = script_dir / name
            if p.is_file():
                return str(p)

        for name in candidates:
            p = Path.cwd() / name
            if p.is_file():
                return str(p)

        messagebox.showerror(
            "Error fatal",
            "No se encontró el archivo de palabras BIP-39.\n"
            "Asegúrate de que 'bip39.txt' (o 'english.txt') esté "
            "en el mismo directorio que este script."
        )
        sys.exit(1)

    # ── Construir la interfaz ────────────────────────────────────────
    def _build_ui(self):
        # Configuración de estilos base (mismo tema que el obfuscador)
        bg = "#1e1e2e"
        fg = "#cdd6f4"
        accent = "#a6e3a1"       # verde para diferenciar del obfuscador
        btn_bg = "#313244"
        btn_active = "#45475a"
        entry_bg = "#313244"
        font_main = ("Segoe UI", 10)
        font_title = ("Segoe UI", 14, "bold")
        font_mono = ("Consolas", 9)

        self.root.configure(bg=bg)

        # ── Título ──
        tk.Label(
            self.root,
            text="🌱  BIP-39 Seedphrase Generator",
            font=font_title,
            bg=bg,
            fg=accent,
        ).pack(pady=(15, 5))

        tk.Label(
            self.root,
            text=f"Lista cargada: {os.path.basename(self.wordlist_path)} "
                 f"({len(self.wordlist)} palabras)",
            font=("Segoe UI", 9),
            bg=bg,
            fg="#a6adc8",
        ).pack(pady=(0, 15))

        # ── Tipo de frase (12 o 24 palabras) ──
        frame_type = tk.Frame(self.root, bg=bg)
        frame_type.pack(fill=tk.X, padx=20, pady=5)

        tk.Label(
            frame_type,
            text="Tipo de frase:",
            font=font_main,
            bg=bg,
            fg=fg,
        ).pack(side=tk.LEFT)

        self.phrase_type = tk.StringVar(value="12")

        radio_frame = tk.Frame(frame_type, bg=bg)
        radio_frame.pack(side=tk.RIGHT)

        tk.Radiobutton(
            radio_frame,
            text="12 palabras (128 bits)",
            variable=self.phrase_type,
            value="12",
            font=font_main,
            bg=bg,
            fg=fg,
            selectcolor=btn_bg,
            activebackground=bg,
            activeforeground=fg,
        ).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(
            radio_frame,
            text="24 palabras (256 bits)",
            variable=self.phrase_type,
            value="24",
            font=font_main,
            bg=bg,
            fg=fg,
            selectcolor=btn_bg,
            activebackground=bg,
            activeforeground=fg,
        ).pack(side=tk.LEFT)

        # ── Cantidad de frases ──
        frame_qty = tk.Frame(self.root, bg=bg)
        frame_qty.pack(fill=tk.X, padx=20, pady=5)

        tk.Label(
            frame_qty,
            text="Cantidad de frases a generar:",
            font=font_main,
            bg=bg,
            fg=fg,
        ).pack(side=tk.LEFT)

        self.qty_entry = tk.Entry(
            frame_qty,
            font=font_main,
            bg=entry_bg,
            fg=fg,
            insertbackground=fg,
            relief=tk.FLAT,
            width=10,
            justify=tk.CENTER,
        )
        self.qty_entry.insert(0, "5")
        self.qty_entry.pack(side=tk.RIGHT, padx=(10, 0))

        # ── Botón generar ──
        frame_buttons = tk.Frame(self.root, bg=bg)
        frame_buttons.pack(pady=15)

        self.generate_btn = tk.Button(
            frame_buttons,
            text="⚡  Generar → frases.txt",
            font=("Segoe UI", 11, "bold"),
            bg=accent,
            fg="#1e1e2e",
            activebackground="#c6f3c1",
            activeforeground="#1e1e2e",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._generate,
            padx=20,
            pady=8,
        )
        self.generate_btn.pack()

        # ── Consola de log ──
        tk.Label(
            self.root,
            text="Registro de operaciones:",
            font=font_main,
            bg=bg,
            fg=fg,
        ).pack(anchor=tk.W, padx=20)

        self.log_area = scrolledtext.ScrolledText(
            self.root,
            font=font_mono,
            bg="#11111b",
            fg="#a6e3a1",
            insertbackground=fg,
            relief=tk.FLAT,
            height=16,
            state=tk.DISABLED,
        )
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=20, pady=(5, 15))

    # ── Log al área de texto ─────────────────────────────────────────
    def _log(self, msg: str):
        self.log_area.configure(state=tk.NORMAL)
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)
        self.log_area.configure(state=tk.DISABLED)
        self.root.update_idletasks()

    # ── Generar frases ───────────────────────────────────────────────
    def _generate(self):
        # Limpiar log
        self.log_area.configure(state=tk.NORMAL)
        self.log_area.delete("1.0", tk.END)
        self.log_area.configure(state=tk.DISABLED)

        # Validar cantidad
        qty_str = self.qty_entry.get().strip()
        if not qty_str.isdigit() or int(qty_str) < 1:
            messagebox.showwarning(
                "Cantidad inválida",
                "Ingresa un número entero positivo para la cantidad de frases."
            )
            return

        qty = int(qty_str)
        word_count = int(self.phrase_type.get())
        ent_bits, cs_bits = BIP39_PARAMS[word_count]

        self._log(f"🌱 BIP-39 Seedphrase Generator")
        self._log(f"   Tipo: {word_count} palabras ({ent_bits} bits de entropía)")
        self._log(f"   Cantidad: {qty} frases")
        self._log(f"   Fuente de aleatoriedad: secrets (CSPRNG del SO)")
        self._log("")

        # Deshabilitar botón durante generación
        self.generate_btn.configure(state=tk.DISABLED)

        try:
            phrases = []
            for i in range(1, qty + 1):
                phrase = generate_mnemonic(word_count, self.wordlist)
                phrases.append(phrase)
                self._log(f"  ✅ Frase #{i}: {phrase}")

            # Escribir archivo de salida
            output_dir = str(Path(__file__).resolve().parent)
            output_path = os.path.join(output_dir, "frases.txt")

            with open(output_path, "w", encoding="utf-8") as f:
                for phrase in phrases:
                    f.write(phrase + "\n")

            self._log("")
            self._log(f"✅ Archivo generado: {output_path}")
            self._log(f"   Total de frases: {len(phrases)}")

            messagebox.showinfo(
                "Completado",
                f"Generación finalizada.\n\n"
                f"Frases generadas: {len(phrases)}\n"
                f"Tipo: {word_count} palabras\n\n"
                f"Archivo generado:\n{output_path}"
            )
        except Exception as e:
            self._log(f"\n⚠ Error inesperado: {e}")
            messagebox.showerror("Error", f"Error inesperado:\n{e}")
        finally:
            self.generate_btn.configure(state=tk.NORMAL)


# ═══════════════════════════════════════════════════════════════════════
#  Punto de entrada principal
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app = BIP39GeneratorApp(root)
    root.mainloop()
