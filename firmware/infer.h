#pragma once
#include <stdint.h>

// x: IN_C*WIN int8 input, layout [channel][t].
// Returns argmax class index; logits (int8, scale S3) written to out.
int infer(const int8_t *x, int8_t *out);
