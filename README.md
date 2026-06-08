# 🧭 EVM Address Generator

[![Download Latest Executable](https://img.shields.io/badge/Download-Latest_Release-blue?style=for-the-badge&logo=linux)](https://github.com/reimen-cpu/Direction-Generator-EVM/releases/latest/download/direction-generator)

> **Implementation of BIP-39 / BIP-32 / BIP-44 / EIP-55**

A high-performance EVM (Ethereum Virtual Machine) address generator that implements the full cryptographic chain from seed phrases to public addresses. This tool features a Python-based GUI and a high-speed CUDA-accelerated C++ backend designed for massive batch processing.

## 🚀 Features

- **Full Standards Support**: Implements BIP-39 (Mnemonic), BIP-32 (HD Wallets), BIP-44 (Multi-Account Hierarchy), and EIP-55 (Checksum addresses).
- **GPU Acceleration**: Utilizes CUDA for highly parallelized derivation of addresses (secp256k1 scalar multiplication and Keccak-256 hashing).
- **Dual Interface**:
  - **Single Mode**: Generate a single address and its private key directly from a phrase.
  - **Batch Mode**: Process thousands of seed phrases from a text file and generate multiple addresses per phrase at lightning speed.
- **Pure Python Fallback**: Non-critical paths (like GUI and basic verification) use pure Python for portability and ease of use.

## 🛠 Prerequisites

If you intend to run from source or compile the backend, you will need:
- **Python 3.x**
- **NVIDIA GPU** with CUDA support.
- **CUDA Toolkit** installed (to compile the C++ backend).
- **Python Dependencies**:
  ```bash
  pip install pycryptodome

📦 Installation

Option 1: Download Pre-compiled Binary (Linux)

You can download the latest compiled executable directly using the button at the
top of this page, or via terminal:

wget https://github.com/reimen-cpu/Direction-Generator-EVM/releases/latest/download/direction-generator
chmod +x direction-generator

Option 2: Compile from Source

1.  Clone the repository:

    git clone https://github.com/reimen-cpu/Direction-Generator-EVM.git
    cd Direction-Generator-EVM

2.  Compile the CUDA Backend: Ensure you have nvcc in your path and run:

    make

    This will generate the direction-generator binary.

🖥 Usage

Graphical Interface (GUI)

To launch the user-friendly graphical interface, run the main Python script:

python3 direction-generator.py

Batch Processing Workflow

1.  Select a text file containing one seed phrase per line.
2.  Specify the number of addresses to derive per phrase. (e.g., deriving 10
    addresses will calculate from m/44'/60'/0'/0/0 to m/44'/60'/0'/0/9).
3.  The generated addresses and keys will be saved automatically to
    directions.txt.

🛡 Security Note

⚠️ USE AT YOUR OWN RISK. This tool is designed primarily for offline use. For
maximum security when handling real funds, always run this software on an
air-gapped machine (disconnected from the internet). Never share your seed
phrases or private keys with anyone.

📜 Standards Implemented

  - BIP-39: Mnemonic code for generating deterministic keys.
  - BIP-32: Hierarchical Deterministic Wallets.
  - BIP-44: Multi-Account Hierarchy for Deterministic Wallets.
  - secp256k1: Elliptic curve parameters used by Ethereum/Bitcoin.
  - Keccak-256: Ethereum's native hashing algorithm.
  - EIP-55: Mixed-case checksum address encoding for EVM networks.

