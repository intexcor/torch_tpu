/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "csrc/ops/copy_from/cpu_to_tpu.h"

#include <memory>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/layout_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/host_allocator.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/copy_common.h"
#include "csrc/ops/copy_from/tpu_to_tpu.h"
#include "csrc/pjrt/pjrt_transfer.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

absl::Status CopyCpuToTpu(const at::Tensor& src, const at::Tensor& dest,
                          TransferSyncMode transfer_sync_mode) {
  ABSL_VLOG(1) << "[AtenCopyFrom] CPU -> TPU copy path for "
               << ToString(src, "src");
  if (src.numel() == 0) {
    // Copying nothing is a no-op.
    return absl::OkStatus();
  }

  // In the most common case of `.to("tpu")`, src and dest will be the same size
  // and dtype. However, `copy_()` allows broadcasting and dtype conversion.
  // In these cases, we do a CPU->TPU copy with no broadcasting or dtype
  // conversion, and then do a TPU->TPU copy to the target shape and dtype.
  // The TPU->TPU copy will handle broadcasting and dtype conversion in a
  // non-blocking fashion.
  if (src.sizes() != dest.sizes() || src.scalar_type() != dest.scalar_type()) {
    TT_ASSIGN_OR_RETURN(
        at::Tensor src_tpu,
        MakeEmptyTensor(src.sizes(), src.scalar_type(), dest.device()));
    TT_RETURN_IF_ERROR(CopyCpuToTpu(src, src_tpu, transfer_sync_mode));
    return CopyTpuToTpu(src_tpu, dest);
  }

  // Ensure the source tensor is contiguous.
  at::Tensor effective_src = src.contiguous();

  // Ensure that we're copying from pinned memory if the copy is non-blocking.
  // This is reference-counted to prevent use-after-free if the user drops the
  // source tensor before the copy completes.
  // If the copy is blocking, then the copy will complete before we drop
  // effective_src, so we don't need to pin.
  if (transfer_sync_mode == TransferSyncMode::kEagerNonBlocking &&
      !IsTpuPinnedPtr(effective_src.storage().data_ptr().get())) {
    effective_src = effective_src.pin_memory(dest.device());
  }

  TT_ASSIGN_OR_RETURN(TpuLayout tpu_layout, ResolveTpuLayout(effective_src));

  // NOTE: we are intentionally copying from Tensor::data_ptr() here,
  // and **not** Tensor::storage().data_ptr().get().
  // On CPU, Tensor::data_ptr() will account for the storage offset, while
  // Tensor::storage().data_ptr().get() will not.
  // However, we also need to pass in the storage pointer to indicate that the
  // host buffer should be kept alive during the transfer if it's pinned.
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef tpu_buf,
      TpuMallocAndMemcpyHtoD(effective_src.data_ptr(), tpu_layout.element_type,
                             tpu_layout.sizes, /*backing_data=*/{},
                             effective_src.storage().data_ptr().get()));
  if (transfer_sync_mode == TransferSyncMode::kIsolated) {
    // Block until the transfer completes, assign the buffer, and return without
    // touching any streams.
    TT_RETURN_IF_ERROR(WrapXlaError(
        tpu_buf.Synchronize(),
        /* context= */ "failed to copy buffer from host to device"));
    TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(tpu_buf), dest));
    return absl::OkStatus();
  }

  // Record the transfer on the stream in either kEagerBlocking or
  // kEagerNonBlocking mode, but only block if kEagerBlocking.
  RecordAsyncHostToDevice(tpu_buf);
  TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(tpu_buf), dest));

  if (transfer_sync_mode == TransferSyncMode::kEagerBlocking) {
    TT_ASSIGN_OR_RETURN(
        std::shared_ptr<EventSnapshot> snapshot,
        MaterializeCurrentStream(MaterializationReason::kCpuTransfer).Await());
    TT_RETURN_IF_ERROR(snapshot->Synchronize());
    TT_RETURN_IF_ERROR(SyncAndReturnStickyError());
  }
  return absl::OkStatus();
}

}  // namespace torch_tpu
