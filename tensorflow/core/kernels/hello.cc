// Author: stakky
#include "tensorflow/core/kernels/hello.h"

namespace tensorflow {
REGISTER_KERNEL_BUILDER(Name("Hello").Device(DEVICE_CPU), HelloOp);
}  // namespace tensorflow
