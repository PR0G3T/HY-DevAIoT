#include "infer.h"
#include "model.h"

// Bit-exact int8 inference: int8 weights/activations, int32 accumulate,
// int32 folded bias, integer-only requantization (out = acc * m0 * 2^(e-31)).
// Mirrors ml/train.py:forward_i8 exactly.

static int8_t a1[W1 * WIN], p1[W1 * WIN / POOL];
static int8_t a2[W2 * WIN / POOL], p2[W2 * WIN / POOL / POOL];

static int8_t requant(int32_t acc, int32_t m0, int8_t e) {
  int64_t r = (int64_t)acc * m0;
  int s = 31 - e;
  int64_t y;
  if (s > 0) {
    int64_t h = 1LL << (s - 1);
    y = r >= 0 ? (r + h) >> s : -((-r + h) >> s);
  } else {
    y = r << (-s);
  }
  return y > 127 ? 127 : y < -128 ? -128 : (int8_t)y;
}

static void conv(const int8_t *x, int ci, int tin, const int8_t *w,
                 const int32_t *b, const int32_t *m, const int8_t *e, int co,
                 int8_t *out) {
  for (int o = 0; o < co; o++)
    for (int t = 0; t < tin; t++) {
      int32_t acc = b[o];
      for (int i = 0; i < ci; i++)
        for (int j = 0; j < KS; j++) {
          int xt = t + j - KS / 2;
          if (xt >= 0 && xt < tin)
            acc += (int32_t)w[(o * ci + i) * KS + j] * x[i * tin + xt];
        }
      out[o * tin + t] = requant(acc, m[o], e[o]);
    }
}

static void relu(int8_t *x, int n) {
  for (int i = 0; i < n; i++)
    if (x[i] < 0)
      x[i] = 0;
}

static void pool(const int8_t *x, int c, int tin, int8_t *out) {
  int tout = tin / POOL;
  for (int o = 0; o < c; o++)
    for (int t = 0; t < tout; t++) {
      int8_t m = x[o * tin + t * POOL];
      for (int j = 1; j < POOL; j++)
        if (x[o * tin + t * POOL + j] > m)
          m = x[o * tin + t * POOL + j];
      out[o * tout + t] = m;
    }
}

int infer(const int8_t *x, int8_t *out) {
  conv(x, IN_C, WIN, w1, b1, m1, e1, W1, a1);
  relu(a1, W1 * WIN);
  pool(a1, W1, WIN, p1);
  conv(p1, W1, WIN / POOL, w2, b2, m2, e2, W2, a2);
  relu(a2, W2 * WIN / POOL);
  pool(a2, W2, WIN / POOL, p2);
  int nin = W2 * WIN / POOL / POOL;
  for (int o = 0; o < NCLASS; o++) {
    int32_t acc = b3[o];
    for (int i = 0; i < nin; i++)
      acc += (int32_t)w3[o * nin + i] * p2[i];
    out[o] = requant(acc, m3[o], e3[o]);
  }
  int am = 0;
  for (int o = 1; o < NCLASS; o++)
    if (out[o] > out[am])
      am = o;
  return am;
}
