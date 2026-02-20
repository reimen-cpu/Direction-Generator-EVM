// ═══════════════════════════════════════════════════════════════════════
//  EVM Address Generator — C++/CUDA
//
//  Derivación BIP-39 → BIP-32 → BIP-44 → secp256k1 → Keccak-256
//  con aceleración GPU NVIDIA para las operaciones de curva elíptica.
//
//  CPU (OpenSSL): PBKDF2-HMAC-SHA512, HMAC-SHA512, derivaciones hardened
//  GPU (CUDA):    secp256k1 scalar mult, Keccak-256, derivación normal
//
//  Uso: ./direcction-generator <input.txt> <output.txt> <num_addresses>
// ═══════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// OpenSSL for PBKDF2 and HMAC
#include <openssl/evp.h>
#include <openssl/hmac.h>

// CUDA
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────
//  Macros
// ─────────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ═══════════════════════════════════════════════════════════════════════
//  256-bit unsigned integer arithmetic (CUDA device)
//
//  Representación: 8 × uint32_t en little-endian (limb[0] = LSW).
//  Todas las funciones son __device__ para ejecución en GPU.
// ═══════════════════════════════════════════════════════════════════════

typedef uint32_t u256[8];

// secp256k1 field prime: p = 2^256 - 2^32 - 977
__constant__ u256 FIELD_P = {0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu,
                             0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
                             0xFFFFFFFFu, 0xFFFFFFFFu};

// secp256k1 group order: n
__constant__ u256 GROUP_N = {0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu,
                             0xBAAEDCE6u, 0xFFFFFFFEu, 0xFFFFFFFFu,
                             0xFFFFFFFFu, 0xFFFFFFFFu};

// Generator point G
// Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
// Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A6855419 9C47D08FFB10D4B8
__constant__ u256 GEN_X = {0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
                           0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu};
__constant__ u256 GEN_Y = {0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
                           0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u};

// ── 256-bit basic ops ────────────────────────────────────────────────

__device__ __host__ void u256_zero(u256 r) {
  for (int i = 0; i < 8; i++)
    r[i] = 0;
}

__device__ __host__ void u256_copy(u256 r, const u256 a) {
  for (int i = 0; i < 8; i++)
    r[i] = a[i];
}

__device__ __host__ int u256_is_zero(const u256 a) {
  for (int i = 0; i < 8; i++)
    if (a[i])
      return 0;
  return 1;
}

// Compare: returns -1, 0, 1
__device__ __host__ int u256_cmp(const u256 a, const u256 b) {
  for (int i = 7; i >= 0; i--) {
    if (a[i] < b[i])
      return -1;
    if (a[i] > b[i])
      return 1;
  }
  return 0;
}

// r = a + b, returns carry
__device__ __host__ uint32_t u256_add(u256 r, const u256 a, const u256 b) {
  uint64_t carry = 0;
  for (int i = 0; i < 8; i++) {
    carry += (uint64_t)a[i] + b[i];
    r[i] = (uint32_t)carry;
    carry >>= 32;
  }
  return (uint32_t)carry;
}

// r = a - b, returns borrow
__device__ __host__ uint32_t u256_sub(u256 r, const u256 a, const u256 b) {
  uint64_t borrow = 0;
  for (int i = 0; i < 8; i++) {
    uint64_t diff = (uint64_t)a[i] - b[i] - borrow;
    r[i] = (uint32_t)diff;
    borrow = (diff >> 63); // borrow if negative
  }
  return (uint32_t)borrow;
}

// ── Modular arithmetic (mod p) ──────────────────────────────────────

__device__ void mod_add(u256 r, const u256 a, const u256 b, const u256 mod) {
  uint32_t carry = u256_add(r, a, b);
  if (carry || u256_cmp(r, mod) >= 0) {
    u256_sub(r, r, mod);
  }
}

__device__ void mod_sub(u256 r, const u256 a, const u256 b, const u256 mod) {
  uint32_t borrow = u256_sub(r, a, b);
  if (borrow) {
    u256_add(r, r, mod);
  }
}

// Multiply 256×256 → 512 bits (schoolbook), then Barrett/simple reduction
// We use a simpler approach: multiply then reduce via repeated subtraction
// Actually, for modular multiplication we use Montgomery or schoolbook + mod

// Full 256×256 → 512 bit multiplication
__device__ void u256_mul_full(uint32_t result[16], const u256 a, const u256 b) {
  uint64_t carry;
  for (int i = 0; i < 16; i++)
    result[i] = 0;

  for (int i = 0; i < 8; i++) {
    carry = 0;
    for (int j = 0; j < 8; j++) {
      uint64_t prod = (uint64_t)a[i] * b[j] + result[i + j] + carry;
      result[i + j] = (uint32_t)prod;
      carry = prod >> 32;
    }
    result[i + 8] = (uint32_t)carry;
  }
}

// Reduce 512-bit number mod p (secp256k1 specific fast reduction)
// p = 2^256 - 2^32 - 977 = 2^256 - 4294968273
// So: x mod p = x_lo + x_hi * (2^32 + 977) mod p
// where x = x_hi * 2^256 + x_lo
__device__ void mod_reduce_512(u256 r, const uint32_t x[16], const u256 mod) {
  // For secp256k1: p = 2^256 - c where c = 4294968273 (0x1000003D1)
  // x mod p = x[0..7] + x[8..15] * c  (mod p)
  // c fits in 33 bits: c = 0x1_000003D1

  uint32_t hi[8];
  for (int i = 0; i < 8; i++)
    hi[i] = x[i + 8];

  // Multiply hi by c = 0x1000003D1
  // c = (1 << 32) + 0x3D1 = 4294968273
  // hi * c = hi * 2^32 + hi * 0x3D1
  // hi * 2^32 = shift hi left by one limb
  // We compute: result = x_lo + hi * 0x3D1 + (hi shifted left by 1 limb)

  // Step 1: t = hi * 0x3D1 (32-bit scalar multiply)
  uint32_t t[9];
  uint64_t carry = 0;
  for (int i = 0; i < 8; i++) {
    uint64_t prod = (uint64_t)hi[i] * 0x3D1u + carry;
    t[i] = (uint32_t)prod;
    carry = prod >> 32;
  }
  t[8] = (uint32_t)carry;

  // Step 2: result = x_lo + t
  uint32_t tmp[10];
  for (int i = 0; i < 10; i++)
    tmp[i] = 0;
  carry = 0;
  for (int i = 0; i < 8; i++) {
    uint64_t sum = (uint64_t)x[i] + t[i] + carry;
    tmp[i] = (uint32_t)sum;
    carry = sum >> 32;
  }
  tmp[8] = (uint32_t)carry + t[8];

  // Step 3: add hi shifted left by 1 limb (= hi * 2^32)
  carry = 0;
  for (int i = 0; i < 8; i++) {
    uint64_t sum = (uint64_t)tmp[i + 1] + hi[i] + carry;
    tmp[i + 1] = (uint32_t)sum;
    carry = sum >> 32;
  }
  tmp[9] = (uint32_t)carry;

  // Now tmp is at most ~258 bits. We may need another round of reduction.
  // If tmp >= 2^256, we have overflow in tmp[8..9].
  // Reduce again: overflow * c
  uint32_t overflow[2] = {tmp[8], tmp[9]};
  // overflow * c (at most ~66 bits * 33 bits = ~99 bits, but overflow is small)
  uint32_t red[4] = {0, 0, 0, 0};
  uint64_t ov_carry = 0;
  for (int i = 0; i < 2; i++) {
    uint64_t prod = (uint64_t)overflow[i] * 0x3D1u + ov_carry;
    red[i] = (uint32_t)prod;
    ov_carry = prod >> 32;
  }
  red[2] = (uint32_t)ov_carry;

  // Add overflow (shifted by 1 limb) for the * 2^32 part
  uint32_t red2[4] = {0, overflow[0], overflow[1], 0};

  // Final: r = tmp[0..7] + red + red2
  carry = 0;
  for (int i = 0; i < 8; i++) {
    uint64_t sum = (uint64_t)tmp[i] + carry;
    if (i < 4)
      sum += red[i] + red2[i];
    r[i] = (uint32_t)sum;
    carry = sum >> 32;
  }

  // Final reduction: if r >= p, subtract p (at most twice)
  while (u256_cmp(r, mod) >= 0) {
    u256_sub(r, r, mod);
  }
}

__device__ void mod_mul(u256 r, const u256 a, const u256 b, const u256 mod) {
  uint32_t full[16];
  u256_mul_full(full, a, b);
  mod_reduce_512(r, full, mod);
}

__device__ void mod_sqr(u256 r, const u256 a, const u256 mod) {
  mod_mul(r, a, a, mod);
}

// Modular inverse via Fermat's little theorem: a^(p-2) mod p
// Using square-and-multiply with the specific structure of p-2
__device__ void mod_inv(u256 r, const u256 a, const u256 mod) {
  // p-2 for secp256k1
  u256 exp;
  u256_copy(exp, mod);
  // exp = p - 2
  uint64_t borrow = 2;
  for (int i = 0; i < 8; i++) {
    uint64_t diff = (uint64_t)exp[i] - borrow;
    exp[i] = (uint32_t)diff;
    borrow = (diff >> 63);
  }

  u256 base, result;
  u256_copy(base, a);
  u256_zero(result);
  result[0] = 1; // result = 1

  // Square-and-multiply (LSB to MSB)
  for (int i = 0; i < 8; i++) {
    uint32_t word = exp[i];
    for (int bit = 0; bit < 32; bit++) {
      if (word & 1) {
        mod_mul(result, result, base, mod);
      }
      mod_sqr(base, base, mod);
      word >>= 1;
    }
  }
  u256_copy(r, result);
}

// ═══════════════════════════════════════════════════════════════════════
//  secp256k1 — Elliptic curve operations (Jacobian coordinates)
//
//  Jacobian: (X, Y, Z) represents affine point (X/Z², Y/Z³)
//  Point at infinity: Z = 0
//
//  a = 0 for secp256k1, which simplifies the doubling formula.
// ═══════════════════════════════════════════════════════════════════════

struct JacobianPoint {
  u256 x, y, z;
};

__device__ void jac_set_infinity(JacobianPoint &p) {
  u256_zero(p.x);
  u256_zero(p.y);
  u256_zero(p.z);
  p.x[0] = 0;
  p.y[0] = 1; // Convention: (0, 1, 0)
}

__device__ int jac_is_infinity(const JacobianPoint &p) {
  return u256_is_zero(p.z);
}

// Point doubling: R = 2*P (Jacobian, a=0)
// Formulas (from
// https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html):
//   A = Y1^2
//   B = X1*A
//   C = A^2
//   D = (X1+A)^2 - B - C    -- This is incorrect, use standard formulas
//
// Standard doubling for y²=x³+7 (a=0):
//   S = 4*X*Y²
//   M = 3*X²     (since a=0)
//   X' = M² - 2*S
//   Y' = M*(S - X') - 8*Y⁴
//   Z' = 2*Y*Z
__device__ void jac_double(JacobianPoint &r, const JacobianPoint &p) {
  if (jac_is_infinity(p)) {
    jac_set_infinity(r);
    return;
  }

  u256 ysq, s, m, t1, t2;

  // ysq = Y^2
  mod_sqr(ysq, p.y, FIELD_P);

  // s = 4*X*Y^2
  mod_mul(s, p.x, ysq, FIELD_P);
  mod_add(s, s, s, FIELD_P); // 2*X*Y²
  mod_add(s, s, s, FIELD_P); // 4*X*Y²

  // m = 3*X^2 (a=0 for secp256k1)
  mod_sqr(m, p.x, FIELD_P);
  u256_copy(t1, m);
  mod_add(m, m, m, FIELD_P);  // 2*X²
  mod_add(m, m, t1, FIELD_P); // 3*X²

  // x' = M^2 - 2*S
  mod_sqr(r.x, m, FIELD_P);
  mod_sub(r.x, r.x, s, FIELD_P);
  mod_sub(r.x, r.x, s, FIELD_P);

  // y' = M*(S - X') - 8*Y^4
  mod_sub(t1, s, r.x, FIELD_P);
  mod_mul(r.y, m, t1, FIELD_P);
  mod_sqr(t2, ysq, FIELD_P);    // Y^4
  mod_add(t2, t2, t2, FIELD_P); // 2*Y^4
  mod_add(t2, t2, t2, FIELD_P); // 4*Y^4
  mod_add(t2, t2, t2, FIELD_P); // 8*Y^4
  mod_sub(r.y, r.y, t2, FIELD_P);

  // z' = 2*Y*Z
  mod_mul(r.z, p.y, p.z, FIELD_P);
  mod_add(r.z, r.z, r.z, FIELD_P);
}

// Point addition: R = P + Q (Jacobian, mixed: Q can have Z=1)
// Standard addition formulas for Jacobian coordinates:
//   U1 = X1*Z2²,  U2 = X2*Z1²
//   S1 = Y1*Z2³,  S2 = Y2*Z1³
//   H = U2 - U1
//   R = S2 - S1
//   X3 = R² - H³ - 2*U1*H²
//   Y3 = R*(U1*H² - X3) - S1*H³
//   Z3 = H*Z1*Z2
__device__ void jac_add(JacobianPoint &r, const JacobianPoint &p,
                        const JacobianPoint &q) {
  if (jac_is_infinity(p)) {
    u256_copy(r.x, q.x);
    u256_copy(r.y, q.y);
    u256_copy(r.z, q.z);
    return;
  }
  if (jac_is_infinity(q)) {
    u256_copy(r.x, p.x);
    u256_copy(r.y, p.y);
    u256_copy(r.z, p.z);
    return;
  }

  u256 z1sq, z2sq, u1, u2, z1cu, z2cu, s1, s2, h, rr, hsq, hcu, t;

  mod_sqr(z1sq, p.z, FIELD_P);
  mod_sqr(z2sq, q.z, FIELD_P);
  mod_mul(u1, p.x, z2sq, FIELD_P);
  mod_mul(u2, q.x, z1sq, FIELD_P);

  mod_mul(z1cu, z1sq, p.z, FIELD_P);
  mod_mul(z2cu, z2sq, q.z, FIELD_P);
  mod_mul(s1, p.y, z2cu, FIELD_P);
  mod_mul(s2, q.y, z1cu, FIELD_P);

  // Check if points are the same
  if (u256_cmp(u1, u2) == 0) {
    if (u256_cmp(s1, s2) == 0) {
      jac_double(r, p);
      return;
    } else {
      jac_set_infinity(r);
      return;
    }
  }

  mod_sub(h, u2, u1, FIELD_P);
  mod_sub(rr, s2, s1, FIELD_P);

  mod_sqr(hsq, h, FIELD_P);
  mod_mul(hcu, hsq, h, FIELD_P);

  mod_mul(t, u1, hsq, FIELD_P); // U1*H²

  // X3 = R² - H³ - 2*U1*H²
  mod_sqr(r.x, rr, FIELD_P);
  mod_sub(r.x, r.x, hcu, FIELD_P);
  mod_sub(r.x, r.x, t, FIELD_P);
  mod_sub(r.x, r.x, t, FIELD_P);

  // Y3 = R*(U1*H² - X3) - S1*H³
  u256 tmp;
  mod_sub(tmp, t, r.x, FIELD_P);
  mod_mul(r.y, rr, tmp, FIELD_P);
  mod_mul(tmp, s1, hcu, FIELD_P);
  mod_sub(r.y, r.y, tmp, FIELD_P);

  // Z3 = H*Z1*Z2
  mod_mul(r.z, p.z, q.z, FIELD_P);
  mod_mul(r.z, r.z, h, FIELD_P);
}

// Scalar multiplication: k * P (double-and-add, MSB to LSB)
__device__ void jac_scalar_mul(JacobianPoint &r, const u256 k,
                               const JacobianPoint &p) {
  jac_set_infinity(r);

  // Find highest set bit
  int highest = -1;
  for (int i = 7; i >= 0; i--) {
    if (k[i]) {
      for (int b = 31; b >= 0; b--) {
        if (k[i] & (1u << b)) {
          highest = i * 32 + b;
          goto found;
        }
      }
    }
  }
found:
  if (highest < 0)
    return; // k = 0

  for (int bit = highest; bit >= 0; bit--) {
    JacobianPoint tmp;
    jac_double(tmp, r);
    int word = bit / 32;
    int pos = bit % 32;
    if (k[word] & (1u << pos)) {
      jac_add(r, tmp, p);
    } else {
      u256_copy(r.x, tmp.x);
      u256_copy(r.y, tmp.y);
      u256_copy(r.z, tmp.z);
    }
  }
}

// Convert Jacobian → Affine (x, y)
__device__ void jac_to_affine(u256 ax, u256 ay, const JacobianPoint &p) {
  u256 zinv, zinv2, zinv3;
  mod_inv(zinv, p.z, FIELD_P);
  mod_sqr(zinv2, zinv, FIELD_P);
  mod_mul(zinv3, zinv2, zinv, FIELD_P);
  mod_mul(ax, p.x, zinv2, FIELD_P);
  mod_mul(ay, p.y, zinv3, FIELD_P);
}

// Load 32 bytes big-endian into u256 (little-endian limbs)
__device__ __host__ void bytes_to_u256(u256 r, const uint8_t bytes[32]) {
  for (int i = 0; i < 8; i++) {
    int off = (7 - i) * 4;
    r[i] = ((uint32_t)bytes[off] << 24) | ((uint32_t)bytes[off + 1] << 16) |
           ((uint32_t)bytes[off + 2] << 8) | ((uint32_t)bytes[off + 3]);
  }
}

// Store u256 to 32 bytes big-endian
__device__ __host__ void u256_to_bytes(uint8_t bytes[32], const u256 a) {
  for (int i = 0; i < 8; i++) {
    int off = (7 - i) * 4;
    bytes[off] = (a[i] >> 24) & 0xFF;
    bytes[off + 1] = (a[i] >> 16) & 0xFF;
    bytes[off + 2] = (a[i] >> 8) & 0xFF;
    bytes[off + 3] = a[i] & 0xFF;
  }
}

// ═══════════════════════════════════════════════════════════════════════
//  Keccak-256 (CUDA device implementation)
//
//  Ethereum uses original Keccak (NOT FIPS 202 SHA3-256).
//  Padding byte: 0x01 (Keccak) vs 0x06 (SHA3).
//  Rate for Keccak-256: 1088 bits = 136 bytes.
// ═══════════════════════════════════════════════════════════════════════

__device__ const uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};

__device__ const int KECCAK_ROT[24] = {1,  3,  6,  10, 15, 21, 28, 36,
                                       45, 55, 2,  14, 27, 41, 56, 8,
                                       25, 43, 62, 18, 39, 61, 20, 44};

__device__ const int KECCAK_PI[24] = {10, 7,  11, 17, 18, 3,  5,  16,
                                      8,  21, 24, 4,  15, 23, 19, 13,
                                      12, 2,  20, 14, 22, 9,  6,  1};

__device__ uint64_t rotl64(uint64_t x, int n) {
  return (x << n) | (x >> (64 - n));
}

__device__ void keccak_f1600(uint64_t state[25]) {
  for (int round = 0; round < 24; round++) {
    // θ step
    uint64_t C[5];
    for (int x = 0; x < 5; x++)
      C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^
             state[x + 20];

    uint64_t D[5];
    for (int x = 0; x < 5; x++)
      D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);

    for (int x = 0; x < 5; x++)
      for (int y = 0; y < 25; y += 5)
        state[y + x] ^= D[x];

    // ρ and π steps
    uint64_t last = state[1];
    for (int i = 0; i < 24; i++) {
      int j = KECCAK_PI[i];
      uint64_t tmp = state[j];
      state[j] = rotl64(last, KECCAK_ROT[i]);
      last = tmp;
    }

    // χ step
    for (int y = 0; y < 25; y += 5) {
      uint64_t t0 = state[y], t1 = state[y + 1], t2 = state[y + 2],
               t3 = state[y + 3], t4 = state[y + 4];
      state[y] = t0 ^ ((~t1) & t2);
      state[y + 1] = t1 ^ ((~t2) & t3);
      state[y + 2] = t2 ^ ((~t3) & t4);
      state[y + 3] = t3 ^ ((~t4) & t0);
      state[y + 4] = t4 ^ ((~t0) & t1);
    }

    // ι step
    state[0] ^= KECCAK_RC[round];
  }
}

// Keccak-256 of exactly 64 bytes input (public key without 0x04 prefix)
// Rate = 136 bytes, so 64 bytes < 136, single block absorption
__device__ void keccak256_64(uint8_t hash[32], const uint8_t data[64]) {
  uint64_t state[25];
  for (int i = 0; i < 25; i++)
    state[i] = 0;

  // Absorb: XOR data into state (little-endian u64 words)
  for (int i = 0; i < 8; i++) { // 64 bytes = 8 words
    uint64_t word = 0;
    for (int b = 0; b < 8; b++)
      word |= (uint64_t)data[i * 8 + b] << (b * 8);
    state[i] ^= word;
  }

  // Keccak padding: 0x01 at byte 64, 0x80 at byte 135 (rate-1)
  // Byte 64 is in word 8, byte offset 0
  state[8] ^= 0x01ULL;
  // Byte 135 is in word 16 (135/8=16, 135%8=7)
  state[16] ^= 0x80ULL << 56;

  keccak_f1600(state);

  // Squeeze: first 32 bytes = first 4 words
  for (int i = 0; i < 4; i++) {
    for (int b = 0; b < 8; b++)
      hash[i * 8 + b] = (state[i] >> (b * 8)) & 0xFF;
  }
}

// ═══════════════════════════════════════════════════════════════════════
//  CUDA Kernel — Derive addresses in parallel
//
//  Each thread:
//    1. Takes the parent key + chain code at level m/44'/60'/0'/0
//    2. Computes the normal BIP-32 child derivation for index i
//       (requires computing parent public key via secp256k1)
//    3. Computes the child public key
//    4. Hashes with Keccak-256 to get the EVM address
// ═══════════════════════════════════════════════════════════════════════

// HMAC-SHA512 is done on CPU. The kernel receives pre-computed
// child private keys and just needs to compute pubkey + Keccak-256.

// Simplified kernel: receives child private keys, computes addresses.
__global__ void
compute_addresses_kernel(const uint8_t *d_privkeys, // num_keys × 32 bytes
                         uint8_t *d_addresses,      // num_keys × 20 bytes
                         int num_keys) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_keys)
    return;

  const uint8_t *privkey = d_privkeys + idx * 32;
  uint8_t *address = d_addresses + idx * 20;

  // 1. Load private key
  u256 k;
  bytes_to_u256(k, privkey);

  // 2. Compute public key: Q = k * G
  JacobianPoint G_jac;
  u256_copy(G_jac.x, GEN_X);
  u256_copy(G_jac.y, GEN_Y);
  u256_zero(G_jac.z);
  G_jac.z[0] = 1;

  JacobianPoint Q;
  jac_scalar_mul(Q, k, G_jac);

  // 3. Convert to affine
  u256 pub_x, pub_y;
  jac_to_affine(pub_x, pub_y, Q);

  // 4. Serialize uncompressed pubkey (without 0x04 prefix): x || y
  uint8_t pubkey_body[64];
  u256_to_bytes(pubkey_body, pub_x);
  u256_to_bytes(pubkey_body + 32, pub_y);

  // 5. Keccak-256
  uint8_t hash[32];
  keccak256_64(hash, pubkey_body);

  // 6. Last 20 bytes = address
  for (int i = 0; i < 20; i++)
    address[i] = hash[12 + i];
}

// Also need a kernel to compute compressed public key for BIP-32 normal
// derivation
__global__ void
compute_compressed_pubkey_kernel(const uint8_t *d_privkey, // 32 bytes
                                 uint8_t *d_compressed_pub // 33 bytes
) {
  u256 k;
  bytes_to_u256(k, d_privkey);

  JacobianPoint G_jac;
  u256_copy(G_jac.x, GEN_X);
  u256_copy(G_jac.y, GEN_Y);
  u256_zero(G_jac.z);
  G_jac.z[0] = 1;

  JacobianPoint Q;
  jac_scalar_mul(Q, k, G_jac);

  u256 pub_x, pub_y;
  jac_to_affine(pub_x, pub_y, Q);

  // Compressed: prefix (02 if y even, 03 if y odd) || x
  uint8_t prefix = (pub_y[0] & 1) ? 0x03 : 0x02;
  d_compressed_pub[0] = prefix;
  u256_to_bytes(d_compressed_pub + 1, pub_x);
}

// ═══════════════════════════════════════════════════════════════════════
//  CPU-side: BIP-39, BIP-32, BIP-44 using OpenSSL
// ═══════════════════════════════════════════════════════════════════════

// PBKDF2-HMAC-SHA512
void bip39_mnemonic_to_seed(const std::string &mnemonic,
                            const std::string &passphrase, uint8_t seed[64]) {
  std::string salt = "mnemonic" + passphrase;
  PKCS5_PBKDF2_HMAC(mnemonic.c_str(), mnemonic.length(),
                    (const unsigned char *)salt.c_str(), salt.length(), 2048,
                    EVP_sha512(), 64, seed);
}

// HMAC-SHA512
void hmac_sha512(const uint8_t *key, int key_len, const uint8_t *data,
                 int data_len, uint8_t output[64]) {
  unsigned int len = 64;
  HMAC(EVP_sha512(), key, key_len, data, data_len, output, &len);
}

// BIP-32: master key from seed
void bip32_master_key(const uint8_t seed[64], uint8_t privkey[32],
                      uint8_t chaincode[32]) {
  uint8_t I[64];
  hmac_sha512((const uint8_t *)"Bitcoin seed", 12, seed, 64, I);
  memcpy(privkey, I, 32);
  memcpy(chaincode, I + 32, 32);
}

// secp256k1 order n (big-endian bytes for CPU-side arithmetic)
static const uint8_t SECP256K1_N_BYTES[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48,
    0xA0, 0x3B, 0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x41};

// Add two 256-bit integers mod n (big-endian byte arrays)
void add_mod_n(uint8_t r[32], const uint8_t a[32], const uint8_t b[32]) {
  // Convert to little-endian u256 for arithmetic
  // Simple: use 64-bit math
  uint16_t carry = 0;
  uint8_t sum[32];
  for (int i = 31; i >= 0; i--) {
    carry += (uint16_t)a[i] + b[i];
    sum[i] = carry & 0xFF;
    carry >>= 8;
  }

  // Check if sum >= n, subtract n if so
  bool ge = (carry > 0);
  if (!ge) {
    for (int i = 0; i < 32; i++) {
      if (sum[i] < SECP256K1_N_BYTES[i]) {
        ge = false;
        break;
      }
      if (sum[i] > SECP256K1_N_BYTES[i]) {
        ge = true;
        break;
      }
    }
  }

  if (ge) {
    uint16_t borrow = 0;
    for (int i = 31; i >= 0; i--) {
      int16_t diff = (int16_t)sum[i] - SECP256K1_N_BYTES[i] - borrow;
      if (diff < 0) {
        diff += 256;
        borrow = 1;
      } else {
        borrow = 0;
      }
      r[i] = diff & 0xFF;
    }
  } else {
    memcpy(r, sum, 32);
  }
}

// BIP-32: hardened child key derivation (no EC needed)
void bip32_derive_hardened(const uint8_t parent_key[32],
                           const uint8_t parent_chain[32], uint32_t index,
                           uint8_t child_key[32], uint8_t child_chain[32]) {
  // Data = 0x00 || parent_key || ser32(index)
  uint8_t data[37];
  data[0] = 0x00;
  memcpy(data + 1, parent_key, 32);
  data[33] = (index >> 24) & 0xFF;
  data[34] = (index >> 16) & 0xFF;
  data[35] = (index >> 8) & 0xFF;
  data[36] = index & 0xFF;

  uint8_t I[64];
  hmac_sha512(parent_chain, 32, data, 37, I);

  add_mod_n(child_key, I, parent_key);
  memcpy(child_chain, I + 32, 32);
}

// BIP-32: normal child key derivation (needs compressed pubkey)
// compressed_pubkey is 33 bytes computed via GPU
void bip32_derive_normal(const uint8_t parent_key[32],
                         const uint8_t parent_chain[32],
                         const uint8_t compressed_pubkey[33], uint32_t index,
                         uint8_t child_key[32], uint8_t child_chain[32]) {
  // Data = serP(parent_pub) || ser32(index)
  uint8_t data[37];
  memcpy(data, compressed_pubkey, 33);
  data[33] = (index >> 24) & 0xFF;
  data[34] = (index >> 16) & 0xFF;
  data[35] = (index >> 8) & 0xFF;
  data[36] = index & 0xFF;

  uint8_t I[64];
  hmac_sha512(parent_chain, 32, data, 37, I);

  add_mod_n(child_key, I, parent_key);
  memcpy(child_chain, I + 32, 32);
}

// Compute compressed public key using GPU (single key)
void gpu_compressed_pubkey(const uint8_t privkey[32], uint8_t compressed[33]) {
  uint8_t *d_priv, *d_pub;
  CUDA_CHECK(cudaMalloc(&d_priv, 32));
  CUDA_CHECK(cudaMalloc(&d_pub, 33));
  CUDA_CHECK(cudaMemcpy(d_priv, privkey, 32, cudaMemcpyHostToDevice));

  compute_compressed_pubkey_kernel<<<1, 1>>>(d_priv, d_pub);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(compressed, d_pub, 33, cudaMemcpyDeviceToHost));
  cudaFree(d_priv);
  cudaFree(d_pub);
}

// Full BIP-44 derivation for Ethereum: m/44'/60'/0'/0
void derive_bip44_eth_parent(const uint8_t master_key[32],
                             const uint8_t master_chain[32],
                             uint8_t parent_key[32], uint8_t parent_chain[32]) {
  uint8_t k1[32], c1[32], k2[32], c2[32], k3[32], c3[32];

  // m/44' (hardened)
  bip32_derive_hardened(master_key, master_chain, 0x80000000 | 44, k1, c1);
  // m/44'/60' (hardened)
  bip32_derive_hardened(k1, c1, 0x80000000 | 60, k2, c2);
  // m/44'/60'/0' (hardened)
  bip32_derive_hardened(k2, c2, 0x80000000 | 0, k3, c3);

  // m/44'/60'/0'/0 (normal — needs compressed pubkey)
  uint8_t compressed_pub[33];
  gpu_compressed_pubkey(k3, compressed_pub);
  bip32_derive_normal(k3, c3, compressed_pub, 0, parent_key, parent_chain);
}

// Derive child keys at m/44'/60'/0'/0/i for i=0..num_keys-1 (CPU loop + GPU for
// pubkeys)
void derive_child_keys(const uint8_t parent_key[32],
                       const uint8_t parent_chain[32], int num_keys,
                       std::vector<uint8_t> &child_privkeys) {
  child_privkeys.resize(num_keys * 32);

  // Need compressed pubkey of parent once
  uint8_t compressed_pub[33];
  gpu_compressed_pubkey(parent_key, compressed_pub);

  for (int i = 0; i < num_keys; i++) {
    uint8_t ck[32], cc[32];
    bip32_derive_normal(parent_key, parent_chain, compressed_pub, i, ck, cc);
    memcpy(child_privkeys.data() + i * 32, ck, 32);
  }
}

// ═══════════════════════════════════════════════════════════════════════
//  EIP-55 checksum (CPU)
// ═══════════════════════════════════════════════════════════════════════

// CPU-side Keccak-256 (reused from GPU logic but on CPU)
// Simpler: use OpenSSL / reimplement
// We'll implement a simple CPU keccak for the EIP-55 checksum

static const uint64_t KECCAK_RC_CPU[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};

static uint64_t rotl64_cpu(uint64_t x, int n) {
  return (x << n) | (x >> (64 - n));
}

void keccak_f1600_cpu(uint64_t state[25]) {
  static const int rot[24] = {1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
                              27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44};
  static const int pi[24] = {10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
                             15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1};

  for (int round = 0; round < 24; round++) {
    uint64_t C[5], D[5];
    for (int x = 0; x < 5; x++)
      C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^
             state[x + 20];
    for (int x = 0; x < 5; x++)
      D[x] = C[(x + 4) % 5] ^ rotl64_cpu(C[(x + 1) % 5], 1);
    for (int i = 0; i < 25; i++)
      state[i] ^= D[i % 5];

    uint64_t last = state[1];
    for (int i = 0; i < 24; i++) {
      uint64_t tmp = state[pi[i]];
      state[pi[i]] = rotl64_cpu(last, rot[i]);
      last = tmp;
    }

    for (int y = 0; y < 25; y += 5) {
      uint64_t t[5];
      for (int x = 0; x < 5; x++)
        t[x] = state[y + x];
      for (int x = 0; x < 5; x++)
        state[y + x] = t[x] ^ ((~t[(x + 1) % 5]) & t[(x + 2) % 5]);
    }
    state[0] ^= KECCAK_RC_CPU[round];
  }
}

void keccak256_cpu(const uint8_t *data, size_t len, uint8_t hash[32]) {
  uint64_t state[25] = {0};
  const int rate = 136; // bytes

  // Absorb
  size_t offset = 0;
  while (offset + rate <= len) {
    for (int i = 0; i < rate / 8; i++) {
      uint64_t word = 0;
      for (int b = 0; b < 8; b++)
        word |= (uint64_t)data[offset + i * 8 + b] << (b * 8);
      state[i] ^= word;
    }
    keccak_f1600_cpu(state);
    offset += rate;
  }

  // Final block with padding
  uint8_t block[136] = {0};
  size_t remaining = len - offset;
  memcpy(block, data + offset, remaining);
  block[remaining] = 0x01; // Keccak padding (NOT SHA3's 0x06)
  block[rate - 1] |= 0x80;

  for (int i = 0; i < rate / 8; i++) {
    uint64_t word = 0;
    for (int b = 0; b < 8; b++)
      word |= (uint64_t)block[i * 8 + b] << (b * 8);
    state[i] ^= word;
  }
  keccak_f1600_cpu(state);

  // Squeeze
  for (int i = 0; i < 4; i++)
    for (int b = 0; b < 8; b++)
      hash[i * 8 + b] = (state[i] >> (b * 8)) & 0xFF;
}

std::string eip55_checksum(const uint8_t address[20]) {
  // Convert to hex lowercase
  char hex[41];
  for (int i = 0; i < 20; i++)
    sprintf(hex + i * 2, "%02x", address[i]);
  hex[40] = 0;

  // Keccak-256 of the hex string
  uint8_t hash[32];
  keccak256_cpu((const uint8_t *)hex, 40, hash);

  // Apply checksum
  std::string result = "0x";
  for (int i = 0; i < 40; i++) {
    int nibble = (hash[i / 2] >> (((i % 2) == 0) ? 4 : 0)) & 0xF;
    if (hex[i] >= 'a' && hex[i] <= 'f') {
      result += (nibble >= 8) ? (char)(hex[i] - 32) : hex[i];
    } else {
      result += hex[i];
    }
  }
  return result;
}

// ═══════════════════════════════════════════════════════════════════════
//  Main — Read phrases, derive, write output
// ═══════════════════════════════════════════════════════════════════════

std::string bytes_to_hex(const uint8_t *data, int len) {
  std::ostringstream oss;
  for (int i = 0; i < len; i++)
    oss << std::hex << std::setfill('0') << std::setw(2) << (int)data[i];
  return oss.str();
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    fprintf(stderr,
            "Uso: %s <input.txt> <output.txt> <num_addresses> [passphrase]\n",
            argv[0]);
    return 1;
  }

  std::string input_file = argv[1];
  std::string output_file = argv[2];
  int num_addresses = atoi(argv[3]);
  std::string passphrase = (argc >= 5) ? argv[4] : "";

  if (num_addresses < 1 || num_addresses > 10000) {
    fprintf(stderr, "Error: num_addresses debe ser entre 1 y 10000\n");
    return 1;
  }

  // Read phrases
  std::ifstream fin(input_file);
  if (!fin.is_open()) {
    fprintf(stderr, "Error: no se pudo abrir %s\n", input_file.c_str());
    return 1;
  }

  std::vector<std::string> phrases;
  std::string line;
  while (std::getline(fin, line)) {
    // Trim
    size_t s = line.find_first_not_of(" \t\r\n");
    if (s == std::string::npos)
      continue;
    size_t e = line.find_last_not_of(" \t\r\n");
    line = line.substr(s, e - s + 1);
    if (!line.empty())
      phrases.push_back(line);
  }
  fin.close();

  if (phrases.empty()) {
    fprintf(stderr, "Error: archivo vacio\n");
    return 1;
  }

  fprintf(stderr, "Frases: %zu, Direcciones por frase: %d\n", phrases.size(),
          num_addresses);

  // Allocate GPU memory for batch address computation
  uint8_t *d_privkeys, *d_addresses;
  CUDA_CHECK(cudaMalloc(&d_privkeys, num_addresses * 32));
  CUDA_CHECK(cudaMalloc(&d_addresses, num_addresses * 20));

  std::ofstream fout(output_file);

  for (size_t pi = 0; pi < phrases.size(); pi++) {
    const std::string &mnemonic = phrases[pi];
    int word_count = 1;
    for (char c : mnemonic)
      if (c == ' ')
        word_count++;

    fprintf(stderr, "Procesando frase #%zu (%d palabras)...\n", pi + 1,
            word_count);

    // BIP-39: mnemonic → seed (with passphrase)
    uint8_t seed[64];
    bip39_mnemonic_to_seed(mnemonic, passphrase, seed);

    // BIP-32: seed → master key
    uint8_t master_key[32], master_chain[32];
    bip32_master_key(seed, master_key, master_chain);

    // BIP-44: derive to m/44'/60'/0'/0 (3 hardened + 1 normal)
    uint8_t parent_key[32], parent_chain[32];
    derive_bip44_eth_parent(master_key, master_chain, parent_key, parent_chain);

    // Derive child keys on CPU (HMAC-SHA512)
    std::vector<uint8_t> child_privkeys;
    derive_child_keys(parent_key, parent_chain, num_addresses, child_privkeys);

    // Copy child private keys to GPU
    CUDA_CHECK(cudaMemcpy(d_privkeys, child_privkeys.data(), num_addresses * 32,
                          cudaMemcpyHostToDevice));

    // Launch kernel: compute public keys + Keccak-256 addresses
    int threads = 256;
    int blocks = (num_addresses + threads - 1) / threads;
    compute_addresses_kernel<<<blocks, threads>>>(d_privkeys, d_addresses,
                                                  num_addresses);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy addresses back
    std::vector<uint8_t> h_addresses(num_addresses * 20);
    CUDA_CHECK(cudaMemcpy(h_addresses.data(), d_addresses, num_addresses * 20,
                          cudaMemcpyDeviceToHost));

    // Write output
    fout << "═══ Frase #" << (pi + 1) << " (" << word_count << "w) ═══\n";
    fout << mnemonic << "\n";
    fout << "Ruta: m/44'/60'/0'/0/i\n\n";
    fout << std::left << std::setw(5) << "Idx" << std::setw(68)
         << "Clave Privada"
         << "Dirección" << "\n";
    for (int i = 0; i < 120; i++)
      fout << "─";
    fout << "\n";

    for (int i = 0; i < num_addresses; i++) {
      std::string privkey_hex =
          "0x" + bytes_to_hex(child_privkeys.data() + i * 32, 32);
      std::string address = eip55_checksum(h_addresses.data() + i * 20);

      fout << std::left << std::setw(5) << i << std::setw(68) << privkey_hex
           << address << "\n";

      fprintf(stderr, "  [%d] %s\n", i, address.c_str());
    }
    fout << "\n";

    fprintf(stderr, "  Frase #%zu completada.\n\n", pi + 1);
  }

  fout.close();
  cudaFree(d_privkeys);
  cudaFree(d_addresses);

  fprintf(stderr, "Archivo generado: %s\n", output_file.c_str());
  return 0;
}
