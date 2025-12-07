#include <openssl/bn.h>
#include <stdio.h>
#define NBITS 256
void printBN(char *msg, BIGNUM *a) {
  /* Use BN_bn2hex(a) for hex string
  Use BN_bn2dec(a) for decimal string*/
  char *number_str = BN_bn2hex(a);
  printf("%s %s\n", msg, number_str);
  OPENSSL_free(number_str);
}
int main() {
  BN_CTX *ctx = BN_CTX_new();
  BIGNUM *m = BN_new();
  BIGNUM *n = BN_new();
  BIGNUM *d = BN_new();
  BIGNUM *sign = BN_new();
  // Initialize
  BN_hex2bn(&m, "49206f7765202433303030");
  BN_hex2bn(&n,
            "DCBFFE3E51F62E09CE7032E2677A78946A849DC4CDDE3A4D0CB81629242FB1A5");
  BN_hex2bn(&d,
            "74D806F9F3A62BAE331FFE3F0A68AFE35B3D2E4794148AACBC26AA381CD7D30D");
  // Signing
  BN_mod_exp(sign, m, d, n, ctx);
  printBN("Sign =", sign);
  return 0;
}
