import sys
print("STARTING SCRIPT", file=sys.stderr)
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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Estándares implementados:
  • BIP-39  — Mnemonic code for generating deterministic keys
  • BIP-32  — Hierarchical Deterministic Wallets
  • BIP-44  — Multi-Account Hierarchy for Deterministic Wallets
  • secp256k1 — Parámetros de curva elíptica (SEC 2, sección 2.7.1)
  • EIP-55  — Mixed-case checksum address encoding
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencias: pycryptodome (para Keccak-256).  Curva secp256k1 implementada
en Python puro — sin librerías externas de curvas elípticas.
"""

import os
import sys
import tkinter as tk
import subprocess
import tempfile
from tkinter import filedialog, messagebox, scrolledtext, ttk
from tkinter.ttk import Progressbar
from tkinter.font import Font
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════
#  Procesamiento completo: archivo de frases → direcciones (Vía CUDA)
# ═══════════════════════════════════════════════════════════════════════

def process_phrases_file(
    file_path: str,
    num_addresses: int,
    output_path: str,
    passphrase: str = "",
    log_fn=None,
) -> int:
    """
    Lee un archivo con seed phrases (una por línea), y para cada una
    deriva las primeras `num_addresses` direcciones EVM usando CUDA.
    Escribe el resultado en output_path.
    """

    def log(msg: str):
        if log_fn:
            log_fn(msg)

    # Leer frases del archivo para contar y validar
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

    # Ruta al binario compilado CUDA
    if getattr(sys, 'frozen', False):
        script_dir = sys._MEIPASS
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
    exe_path = os.path.join(script_dir, "direcction-generator-bin")
    
    if not os.path.exists(exe_path):
        log(f"❌ Error: No se encontró el binario compilado '{exe_path}'.")
        log("Compílalo primero ejecutando 'make' en la carpeta del script.")
        return 0

    log(f"🚀 Ejecutando aceleración CUDA para {total_phrases} frases...")
    
    cmd = [exe_path, file_path, output_path, str(num_addresses)]
    if passphrase:
        cmd.append(passphrase)
    try:
        # Ejecutar y capturar stderr en tiempo real para el log
        # El binario C++ escribe el progreso en stderr y el resultado en el output_file
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        for line in process.stderr:
            log(line.rstrip())
            
        process.wait()
        
        if process.returncode != 0:
            log(f"❌ Error durante la generación (código {process.returncode}).")
            return 0
            
    except Exception as e:
        log(f"❌ Error al ejecutar el binario CUDA: {e}")
        return 0

    return total_phrases


# ═══════════════════════════════════════════════════════════════════════
#  Interfaz gráfica con Tkinter
# ═══════════════════════════════════════════════════════════════════════

class EVMAddressGeneratorApp:
    """Interfaz gráfica para el generador de direcciones EVM."""

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

        self.selected_file: str = ""
        self._build_ui()

    # ── Construir la interfaz ────────────────────────────────────────
    def _build_ui(self):
        # Configuración de estilos (mismo tema Catppuccin que los otros scripts)
        bg = "#1e1e2e"
        fg = "#cdd6f4"
        accent = "#f9e2af"       # amarillo dorado para diferenciar
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
            text="🔗  EVM Address Generator",
            font=font_title,
            bg=bg,
            fg=accent,
        ).pack(pady=(15, 3))

        tk.Label(
            self.root,
            text="BIP-39 → BIP-32 → BIP-44 → secp256k1 → Keccak-256 → Dirección",
            font=("Segoe UI", 9),
            bg=bg,
            fg="#a6adc8",
        ).pack(pady=(0, 12))

        # ── Estilos Notebook ──
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

        # ═════════ TAB 1: DESDE ARCHIVO ═════════
        frame_file = tk.Frame(tab_lote, bg=bg)
        frame_file.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(
            frame_file,
            text="Archivo de seed phrases (.txt):",
            font=font_main,
            bg=bg,
            fg=fg,
        ).pack(side=tk.LEFT)

        tk.Button(
            frame_file,
            text="Seleccionar archivo…",
            font=font_main,
            bg=btn_bg,
            fg=fg,
            activebackground=btn_active,
            activeforeground=fg,
            relief=tk.FLAT,
            cursor="hand2",
            command=self._select_file,
        ).pack(side=tk.RIGHT)

        self.file_var = tk.StringVar(value="Ningún archivo seleccionado.")
        tk.Label(
            tab_lote,
            textvariable=self.file_var,
            font=("Segoe UI", 9),
            bg=bg,
            fg="#a6adc8",
            wraplength=780,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=25, pady=(0, 10))

        frame_pass_lote = tk.Frame(tab_lote, bg=bg)
        frame_pass_lote.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(
            frame_pass_lote,
            text="Secret Key / Passphrase (opcional):",
            font=font_main,
            bg=bg,
            fg=fg,
        ).pack(side=tk.LEFT)
        
        self.pass_lote_entry = tk.Entry(
            frame_pass_lote,
            font=font_main,
            bg=entry_bg,
            fg=fg,
            insertbackground=fg,
            relief=tk.FLAT,
            width=20,
            show="*",
        )
        self.pass_lote_entry.pack(side=tk.RIGHT, padx=(10, 0))

        frame_qty = tk.Frame(tab_lote, bg=bg)
        frame_qty.pack(fill=tk.X, padx=20, pady=5)

        tk.Label(
            frame_qty,
            text="Direcciones por frase (i=0..n):",
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
        self.qty_entry.insert(0, "10")
        self.qty_entry.pack(side=tk.RIGHT, padx=(10, 0))

        frame_btn_lote = tk.Frame(tab_lote, bg=bg)
        frame_btn_lote.pack(pady=15)

        self.generate_btn = tk.Button(
            frame_btn_lote,
            text="⚡  Derivar Lote → direcctions.txt",
            font=("Segoe UI", 11, "bold"),
            bg=accent,
            fg="#1e1e2e",
            activebackground="#fcecc8",
            activeforeground="#1e1e2e",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._generate,
            padx=20,
            pady=8,
        )
        self.generate_btn.pack()

        # ═════════ TAB 2: FRASE INDIVIDUAL ═════════
        frame_single_phrase = tk.Frame(tab_single, bg=bg)
        frame_single_phrase.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(
            frame_single_phrase,
            text="Seed phrase (12 o 24 palabras):",
            font=font_main,
            bg=bg,
            fg=fg,
        ).pack(anchor=tk.W)

        self.single_phrase_text = tk.Text(
            frame_single_phrase,
            font=font_main,
            bg=entry_bg,
            fg=fg,
            insertbackground=fg,
            relief=tk.FLAT,
            height=3,
            wrap=tk.WORD,
        )
        self.single_phrase_text.pack(fill=tk.X, pady=(5, 10))

        frame_pass_single = tk.Frame(tab_single, bg=bg)
        frame_pass_single.pack(fill=tk.X, padx=20, pady=5)

        tk.Label(
            frame_pass_single,
            text="Secret Key / Passphrase (opcional):",
            font=font_main,
            bg=bg,
            fg=fg,
        ).pack(side=tk.LEFT)

        self.pass_single_entry = tk.Entry(
            frame_pass_single,
            font=font_main,
            bg=entry_bg,
            fg=fg,
            insertbackground=fg,
            relief=tk.FLAT,
            width=25,
            show="*",
        )
        self.pass_single_entry.pack(side=tk.RIGHT, padx=(10, 0))

        frame_btn_single = tk.Frame(tab_single, bg=bg)
        frame_btn_single.pack(pady=15)

        self.generate_single_btn = tk.Button(
            frame_btn_single,
            text="⚡  Generar 1 Dirección",
            font=("Segoe UI", 11, "bold"),
            bg=accent,
            fg="#1e1e2e",
            activebackground="#fcecc8",
            activeforeground="#1e1e2e",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._generate_single,
            padx=20,
            pady=8,
        )
        self.generate_single_btn.pack()

        # ── Consola de log (GLOBAL) ──
        tk.Label(
            self.root,
            text="Registro de operaciones:",
            font=font_main,
            bg=bg,
            fg=fg,
        ).pack(anchor=tk.W, padx=20, pady=(10, 0))

        self.log_area = scrolledtext.ScrolledText(
            self.root,
            font=font_mono,
            bg="#11111b",
            fg="#a6e3a1",
            insertbackground=fg,
            relief=tk.FLAT,
            height=18,
            state=tk.DISABLED,
        )
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=20, pady=(5, 15))

    # ── Seleccionar archivo ──────────────────────────────────────────
    def _select_file(self):
        f = filedialog.askopenfilename(
            title="Seleccionar archivo de seed phrases",
            filetypes=[("Archivos de texto", "*.txt"), ("Todos", "*.*")],
        )
        if f:
            self.selected_file = f
            self.file_var.set(f"📄 {os.path.basename(f)}")
        else:
            self.selected_file = ""
            self.file_var.set("Ningún archivo seleccionado.")

    # ── Log al área de texto ─────────────────────────────────────────
    def _log(self, msg: str):
        self.log_area.configure(state=tk.NORMAL)
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)
        self.log_area.configure(state=tk.DISABLED)
        self.root.update_idletasks()

    # ── Generar direcciones ──────────────────────────────────────────
    def _generate(self):
        # Limpiar log
        self.log_area.configure(state=tk.NORMAL)
        self.log_area.delete("1.0", tk.END)
        self.log_area.configure(state=tk.DISABLED)

        # Validar archivo
        if not self.selected_file:
            messagebox.showwarning(
                "Sin archivo",
                "Selecciona un archivo .txt con seed phrases."
            )
            return

        # Validar cantidad
        qty_str = self.qty_entry.get().strip()
        if not qty_str.isdigit() or int(qty_str) < 1:
            messagebox.showwarning(
                "Cantidad inválida",
                "Ingresa un número entero positivo para la cantidad "
                "de direcciones."
            )
            return

        num_addresses = int(qty_str)

        # Ruta de salida
        output_dir = str(Path(__file__).resolve().parent)
        output_path = os.path.join(output_dir, "direcctions.txt")

        self._log("🔗 EVM Address Generator")
        self._log("   Cadena: BIP-39 → BIP-32 → BIP-44 → secp256k1 → "
                  "Keccak-256")
        self._log(f"   Ruta: m/44'/60'/0'/0/i  (i=0..{num_addresses - 1})")
        self._log(f"   Aceleración: CUDA (NVIDIA GPU)")
        self._log("")

        # Deshabilitar botón
        self.generate_btn.configure(state=tk.DISABLED)
        self.root.update_idletasks()

        try:
            passphrase = getattr(self, "pass_lote_entry", tk.Entry(self.root)).get()

            total = process_phrases_file(
                file_path=self.selected_file,
                num_addresses=num_addresses,
                output_path=output_path,
                passphrase=passphrase,
                log_fn=self._log,
            )

            if total > 0:
                messagebox.showinfo(
                    "Completado",
                    f"Derivación finalizada.\n\n"
                    f"Frases procesadas: {total}\n"
                    f"Direcciones por frase: {num_addresses}\n"
                    f"Total direcciones: {total * num_addresses}\n\n"
                    f"Archivo generado:\n{output_path}"
                )
            else:
                messagebox.showwarning(
                    "Sin resultados",
                    "No se procesó ninguna frase.\n"
                    "Revisa el registro para más detalles."
                )
        except Exception as e:
            self._log(f"\n⚠ Error inesperado: {e}")
            messagebox.showerror("Error", f"Error inesperado:\n{e}")
        finally:
            self.generate_btn.configure(state=tk.NORMAL)

    # ── Generar dirección individual ─────────────────────────────────
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

            total = process_phrases_file(
                file_path=temp_in,
                num_addresses=1,
                output_path=temp_out,
                passphrase=passphrase,
                log_fn=self._log,
            )

            if total > 0:
                with open(temp_out, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    address_line = ""
                    for line in reversed(lines):
                        if "0x" in line:
                            address_line = line.strip()
                            break
                    if address_line:
                        parts = address_line.split("0x")
                        if len(parts) >= 3:
                            address = "0x" + parts[-1].strip()
                            privkey = "0x" + parts[-2].split(" ")[0].strip()
                        elif len(parts) >= 2:
                            address = address_line
                            privkey = ""
                        else:
                            address = address_line
                            privkey = ""
                        
                        self._log("\n🎉 ¡Dirección generada exitosamente!")
                        self._log(f"Dirección: {address}")
                        if privkey:
                            self._log(f"Clave Privada: {privkey}")
                        
                        messagebox.showinfo("Dirección Generada", f"Dirección:\n{address}\n\nRevisa el registro para ver la clave privada.")

            os.remove(temp_in)
            os.remove(temp_out)

        except Exception as e:
            self._log(f"\n⚠ Error inesperado: {e}")
            messagebox.showerror("Error", f"Error inesperado:\n{e}")
        finally:
            self.generate_single_btn.configure(state=tk.NORMAL)



# ═══════════════════════════════════════════════════════════════════════
#  Punto de entrada principal
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app = EVMAddressGeneratorApp(root)
    root.mainloop()
