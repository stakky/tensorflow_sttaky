// Author: stakky
#ifndef TENSORFLOW_CORE_KERNELS_HELLO_H_
#define TENSORFLOW_CORE_KERNELS_HELLO_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class HelloOp : public OpKernel {
 public:
  explicit HelloOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    Tensor* hello = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({}), &hello));
    hello->scalar<string>()() = "Hello";
    LOG(INFO) << "tfinternal!!!";
  }
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_HELLO_H_
