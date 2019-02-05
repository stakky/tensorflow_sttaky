// Author: stakky
#include "tensorflow/core/kernels/string_add.h"

namespace tensorflow {
REGISTER_KERNEL_BUILDER(Name("StringAdd").Device(DEVICE_CPU), StringAddOp);
}  // namespace tensorflow
