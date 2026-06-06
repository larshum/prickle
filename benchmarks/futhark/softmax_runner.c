#include "softmax.h"
#include <string.h>

int main(int argc, char **argv) {
  struct futhark_context_config *cfg = futhark_context_config_new();
  struct futhark_context *ctx = futhark_context_new(cfg);

  int dim0 = atoi(argv[1]);
  int dim1 = atoi(argv[2]);

  float *data = (float*)malloc(dim0 * dim1 * sizeof(float));
  float x;
  scanf("[");
  for (int i = 0; i < dim0; i++) {
    scanf("[%f", &data[i * dim1]);
    for (int j = 0; j < dim1; j++) {
      scanf(", %f", &data[i * dim1 + j]);
    }
    scanf("]");
  }
  scanf("]");

  struct futhark_f32_2d *input = futhark_new_f32_2d(ctx, data, dim0, dim1);
  struct futhark_f32_2d *output;
  for (int i = 0; i < 100; i++) {
    futhark_entry_softmax(ctx, &output, input);
  }
  futhark_context_sync(ctx);

  futhark_values_f32_2d(ctx, output, data);
  fprintf(stderr, "[");
  for (int i = 0; i < dim0; i++) {
    fprintf(stderr, "[%f", data[i * dim1]);
    for (int j = 1; j < dim1; j++) {
      fprintf(stderr, " %f", data[i * dim1 + j]);
    }
    fprintf(stderr, "]");
  }
  fprintf(stderr, "]");

  return 0;
}
