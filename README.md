# EVM Address Generator — BIP-39/32/44 & CUDA

High-performance EVM (Ethereum Virtual Machine) address generator that implements the full cryptographic chain from seed phrases to public addresses. This tool features a Python-based GUI and a high-speed CUDA-accelerated C++ backend for massive batch processing.

## 🚀 Features

- **Full Standards Support**: Implements BIP-39 (Mnemonic), BIP-32 (HD Wallets), BIP-44 (Multi-Account Hierarchy), and EIP-55 (Checksum addresses).
- **GPU Acceleration**: Uses CUDA for parallelized derivation of addresses (secp256k1 scalar multiplication and Keccak-256 hashing).
- **Dual Interface**:
  - **Single Mode**: Generate a single address and its private key from a phrase.
  - **Batch Mode**: Process thousands of seed phrases from a text file and generate multiple addresses per phrase.
- **Pure Python Fallback**: Non-critical paths (like GUI and verification) use pure Python for portability.

## 🛠 Prerequisites

- **Python 3.x**
- **NVIDIA GPU** with CUDA support.
- **CUDA Toolkit** installed (to compile the C++ backend).
- **Dependencies**:
  ```bash
  pip install pycryptodome
  ```

## 📦 Installation & Compilation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Direction-Generator-EVM
   ```

2. **Compile the CUDA Backend**:
   Ensure you have `nvcc` in your path and run:
   ```bash
   make
   ```
   This will generate the `direcction-generator` binary.

## 🖥 Usage

### Graphical Interface
Run the main Python script:
```bash
python3 direcction-generator.py
```

### Batch Processing
1. Select a text file containing one seed phrase per line.
2. Specify the number of addresses to derive per phrase (e.g., `m/44'/60'/0'/0/0` to `m/44'/60'/0'/0/9`).
3. The results will be saved to `direcctions.txt`.

## 🛡 Security Note

This tool is designed for offline use. For maximum security, run it on an air-gapped machine. Never share your seed phrases or private keys.

## 📜 Standards Implemented

- **BIP-39**: Mnemonic code for generating deterministic keys.
- **BIP-32**: Hierarchical Deterministic Wallets.
- **BIP-44**: Multi-Account Hierarchy for Deterministic Wallets.
- **secp256k1**: Elliptic curve parameters.
- **Keccak-256**: Ethereum's hashing algorithm.
- **EIP-55**: Mixed-case checksum address encoding.
