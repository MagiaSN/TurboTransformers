#include "prepare_bert_masks.h"
#include <stdint.h>
#include "fast_transformers/core/common.h"
#ifdef FT_WITH_CUDA
#include "fast_transformers/layers/kernels/gpu_utils.h"
#endif

namespace fast_transformers {
namespace layers {

void PrepareBertMasks::operator()(const core::Tensor& inputs,
                                  core::Tensor* att_mask,
                                  core::Tensor* seq_type,
                                  core::Tensor* position_ids,
                                  core::Tensor* extended_attention_mask) const {
  if (position_ids->is_null()) {
    auto pos_ids_ptr = position_ids->Reshape<int64_t>(
        {inputs.shape(0), inputs.shape(1)}, inputs.device_type(),
        inputs.device_id());

    // fill range
    for (int64_t row_id = 0; row_id < inputs.shape(0); ++row_id) {
      core::common::ft_seqence(pos_ids_ptr, inputs.shape(1),
                               inputs.device_type());
      pos_ids_ptr += inputs.shape(1);
    }
  }

  if (seq_type->is_null()) {
    // seq_type.zeros_like(inputs)
    seq_type->Reshape<int64_t>({inputs.shape(0), inputs.shape(1)},
                               inputs.device_type(), inputs.device_id());
    core::common::ft_fill(seq_type->mutableData<int64_t>(), seq_type->numel(),
                          static_cast<int64_t>(0), inputs.device_type());
  }

  if (att_mask->is_null()) {
    att_mask->Reshape<int64_t>({inputs.shape(0), inputs.shape(1)},
                               inputs.device_type(), inputs.device_id());
    core::common::ft_fill(att_mask->mutableData<int64_t>(), att_mask->numel(),
                          static_cast<int64_t>(1), inputs.device_type());
  }

  // cast att_mask to float
  extended_attention_mask->Reshape<float>(
      {att_mask->shape(0), 1, 1, att_mask->shape(1)}, inputs.device_type(),
      inputs.device_id());
  core::common::ft_transform(att_mask->mutableData<int64_t>(),
                             extended_attention_mask->mutableData<float>(),
                             att_mask->numel(), inputs.device_type());
}

}  // namespace layers
}  // namespace fast_transformers