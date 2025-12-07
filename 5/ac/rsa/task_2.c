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
  BIGNUM *p = BN_new();
  BIGNUM *q = BN_new();
  BIGNUM *e = BN_new();
  BIGNUM *d = BN_new();
  BIGNUM *res1 = BN_new();
  BIGNUM *res2 = BN_new();
  BIGNUM *res3 = BN_new();
  BIGNUM *one = BN_new();
  // Initialize
  BN_hex2bn(&p, "F7E75FDC469067FFDC4E847C51F452DF");
  BN_hex2bn(&q, "E85CED54AF57E53E092113E62F436F4F");
  BN_hex2bn(&e, "0D88C3");
  BN_hex2bn(&one, "1");
  BN_sub(res1, p, one);
  BN_sub(res2, q, one);
  BN_mul(res3, res1, res2, ctx);
  BN_mod_inverse(d, e, res3, ctx);
  printBN("d=", d);
  return 0;
}
