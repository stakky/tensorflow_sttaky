// Author: stakky
#ifndef TENSORFLOW_CORE_KERNELS_STRING_ADD_H_
#define TENSORFLOW_CORE_KERNELS_STRING_ADD_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {

static constexpr char kErrorMessage[] =
    "StringAddOp could not correctly convert string: ";

class StringAddOp : public OpKernel {
 public:
  explicit StringAddOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    // string to number
    const Tensor& input0_tensor = context->input(0);
    const Tensor& input1_tensor = context->input(1);
    const auto& input0_flat = input0_tensor.flat<string>();
    const auto& input1_flat = input1_tensor.flat<string>();

    Tensor temp0_tensor;
    Tensor temp1_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(context->expected_output_dtype(0),
					  input0_tensor.shape(),
                                          &temp0_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(context->expected_output_dtype(0),
					  input1_tensor.shape(),
                                          &temp1_tensor));
    auto temp0_flat = temp0_tensor.flat<int32>();
    auto temp1_flat = temp1_tensor.flat<int32>();

    for (int i = 0; i < input0_flat.size(); ++i) {
      OP_REQUIRES(
          context,
          strings::SafeStringToNumeric<int32>(input0_flat(i).c_str(),
					      &temp0_flat(i)),
          errors::InvalidArgument(kErrorMessage, input0_flat(i).c_str()));

      OP_REQUIRES(
          context,
          strings::SafeStringToNumeric<int32>(input1_flat(i).c_str(),
					      &temp1_flat(i)),
          errors::InvalidArgument(kErrorMessage, input1_flat(i).c_str()));
    }

    // math add
    const int* temp0_data = static_cast<const int*>(DMAHelper::base(&temp0_tensor));
    const int* temp1_data = static_cast<const int*>(DMAHelper::base(&temp1_tensor));
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, temp0_tensor.shape(),
						     &output_tensor));
    int* output_data = static_cast<int*>(DMAHelper::base(output_tensor));
    const int64 N = temp0_tensor.NumElements();
    for (int64 i = 0; i < N; i++) {
      output_data[i] = temp0_data[i] + temp1_data[i];
    }
    LOG(INFO) << "tfinternal!!!";
  }
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_STRING_ADD_H_
