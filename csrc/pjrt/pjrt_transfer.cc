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

#include "csrc/pjrt/pjrt_transfer.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/absl_vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/context_states.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/host_allocator.h"
#include "csrc/pjrt/pjrt_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {
// Packs unpacked float4_e2m1fn_x2 data (1 element per byte) from unpacked_data
// into packed_data where each byte contains 2 elements (lower 4 bits of first
// element, upper 4 bits of second element).
// Contract:
// - packed_data must not be null.
// - packed_data must point to a buffer containing at least
//   unpacked_data.size() / 2 bytes.
void PackFp4Vector(const std::vector<uint8_t>& unpacked_data,
                   uint8_t* absl_nonnull packed_data) {
  ABSL_CHECK(packed_data != nullptr);          // CRASH_OK
  ABSL_CHECK_EQ(unpacked_data.size() % 2, 0);  // CRASH_OK
  size_t num_packed_bytes = unpacked_data.size() / 2;

  for (int64_t i = 0; i < num_packed_bytes; ++i) {
    uint8_t val0 = unpacked_data[2 * i] & 0x0F;
    uint8_t val1 = unpacked_data[2 * i + 1] & 0x0F;
    packed_data[i] = (val1 << 4) | val0;
  }
}

// Unpacks packed float4_e2m1fn_x2 data from host_data into a byte vector
// where each element occupies one byte (lower 4 bits). Contract:
// - host_data must not be null.
// - num_elements must be non-negative and even.
// - host_data must point to a buffer containing at least num_elements / 2
// bytes.
std::vector<uint8_t> UnpackFp4(const uint8_t* absl_nonnull host_data,
                               int64_t num_elements) {
  ABSL_CHECK(host_data != nullptr);    // CRASH_OK
  ABSL_CHECK_GE(num_elements, 0);      // CRASH_OK
  ABSL_CHECK_EQ(num_elements % 2, 0);  // CRASH_OK

  std::vector<uint8_t> temp_unpacked_data(num_elements);
  for (int64_t i = 0; i < num_elements / 2; ++i) {
    uint8_t byte = host_data[i];
    temp_unpacked_data[2 * i] = byte & 0x0F;
    temp_unpacked_data[2 * i + 1] = (byte >> 4) & 0x0F;
  }
  return temp_unpacked_data;
}

void ValidateFp4Dimensions(absl::Span<const int64_t> dimensions) {
  ABSL_CHECK(!dimensions.empty())  // CRASH_OK
      << "expected float4_e2m1fn_x2 tensors to be at least 1-dimensional, got "
         "0-dimensional";
  ABSL_CHECK_EQ(dimensions.back() % 2, 0)  // CRASH_OK
      << "expected even size in the last dimension for float4_e2m1fn_x2 "
         "tensors, got "
      << dimensions.back();
}

}  // namespace

// TODO(bawilson): refactor this to simplify
absl::StatusOr<DeviceBufferRef> TpuMallocAndMemcpyHtoD(
    const void* host_data, mlir::ElementType element_type,
    absl::Span<const int64_t> dimensions, std::vector<uint8_t> backing_data,
    const void* pinned_ptr) {
  tsl::profiler::TraceMe trace("TpuMallocAndMemcpyHtoD");
  if (!backing_data.empty()) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=not visible to Python
        backing_data.data() == host_data, error::kInvalidArgument)
        << "expected backing data pointer to be " << host_data << ", got "
        << backing_data.data();
  }

  if (element_type == mlir::ElementType::F4E2M1FN) {
    ValidateFp4Dimensions(dimensions);
  }

  const xla::PrimitiveType type = ConvertTo<xla::PrimitiveType>(element_type);
  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL ENTRY] host_data_is_null: "
               << (host_data == nullptr) << ", type: "
               << xla::primitive_util::LowercasePrimitiveTypeName(type)
               << ", dimensions: [" << absl::StrJoin(dimensions, ",") << "]";

  TT_ASSIGN_OR_RETURN(xla::PjRtClient* const client, GetPjRtClient());
  TT_ASSIGN_OR_RETURN(xla::PjRtDevice* const device, GetPjRtDevice());

  int64_t num_elements = 1;
  if (dimensions.empty() && type != xla::TUPLE) {
    // Do nothing.
  } else {
    TT_ASSIGN_OR_RETURN(num_elements, NumElements(dimensions));
  }

  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] Getting default memory "
                  "space for device: "
               << device->DebugString();
  TT_ASSIGN_OR_RETURN(xla::PjRtMemorySpace* const memory_space,
                      GetDefaultMemorySpace(device));
  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] Got memory space: "
               << memory_space->DebugString();

  xla::PjRtClient::HostBufferSemantics semantics =
      xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes;
  const void* effective_host_data = host_data;

  if (host_data != nullptr && element_type == mlir::ElementType::F4E2M1FN) {
    // Unpack FP4 data from host_data into backing_data and use that as the
    // effective_host_data.
    backing_data =
        UnpackFp4(static_cast<const uint8_t*>(host_data), num_elements);
    effective_host_data = backing_data.data();
  } else if (host_data == nullptr && num_elements > 0) {
    size_t buffer_size_bytes =
        xla::ShapeUtil::ByteSizeOf(xla::ShapeUtil::MakeShape(type, dimensions));
    if (buffer_size_bytes > 0) {
      ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] host_data is nullptr, "
                      "num_elements > 0. Creating temporary zeroed host buffer "
                      "of size "
                   << buffer_size_bytes << " bytes.";
      backing_data.clear();
      backing_data.resize(buffer_size_bytes, 0);
      effective_host_data = backing_data.data();
    } else {
      ABSL_VLOG(1)
          << "[TpuMallocAndMemcpyHtoD INTERNAL] host_data is nullptr, "
             "num_elements > 0, but calculated byte size is 0. Passing "
             "nullptr as effective_host_data.";
    }
  }

  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] Calling "
                  "client->BufferFromHostBuffer with"
               << " effective_host_data_is_null: "
               << (effective_host_data == nullptr) << ", type: "
               << xla::primitive_util::LowercasePrimitiveTypeName(type)
               << ", dimensions: [" << absl::StrJoin(dimensions, ",") << "]"
               << ", semantics: " << static_cast<int>(semantics);

  const xla::Layout* const layout_ptr =
      GetContextState<LayoutContextState>().value_or(nullptr).get();

  bool should_hold =
      (host_data != nullptr && pinned_ptr != nullptr && backing_data.empty());

  TT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      WrapXlaError(
          client->BufferFromHostBuffer(
              effective_host_data, type, dimensions, std::nullopt, semantics,
              [backing_data = std::move(backing_data)]() mutable {
                // This lambda ensures that the backing bytes vector is kept
                // alive until the transfer completes.
                backing_data = std::vector<uint8_t>();
              },
              memory_space, layout_ptr),
          /* context= */ "failed to start copy of buffer from host to device"));
  if (should_hold) {
    bool is_host_pinned =
        HoldPinnedPointerUntil(pinned_ptr, buffer->GetReadyFuture());
    if (ABSL_VLOG_IS_ON(1)) {
      if (is_host_pinned) {
        ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] Host buffer is "
                        "pinned memory; keeping it alive until the "
                        "host-to-device transfer completes.";
      } else {
        ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] Host buffer is not "
                        "pinned, continuing";
      }
    }
  }

  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] "
                  "client->BufferFromHostBuffer SUCCEEDED.";
  return DeviceBufferList::CreateMaterialized(std::move(buffer));
}

namespace {

// TODO(bawilson): refactor this to simplify
void StartD2H(absl::Status materialization_status,
              const DeviceBufferRef& buffer_ref,
              std::vector<uint8_t> fp4_receiver, void* dst_ptr,
              xla::Promise<void> promise) {
  // Verify that the materialization succeeded by retrieving the PjRtBuffer.
  // If we can't, forward the error to the promise.
  if (!materialization_status.ok()) {
    promise.Set(materialization_status);
    return;
  }
  absl::StatusOr<xla::PjRtBuffer*> buffer_or = buffer_ref.AwaitBuffer();
  if (!buffer_or.ok()) {
    promise.Set(buffer_or.status());
    return;
  }
  if ((*buffer_or)->IsDeleted()) {
    promise.Set(TT_ERROR(error::kUnavailable)
                << "cannot start a device to host copy from a donated or "
                   "deleted buffer");
    return;
  }
  // Materialization succeeded.
  // If we are copying FP4 data, we will copy to the temporary fp4_receiver
  // buffer; otherwise, we will copy directly to the destination pointer.
  void* effective_dst_ptr =
      (fp4_receiver.empty() ? dst_ptr : fp4_receiver.data());
  xla::Shape xla_shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(
      ConvertTo<xla::PrimitiveType>(buffer_ref.element_type()),
      buffer_ref.dimensions());
  auto literal = std::make_unique<xla::MutableBorrowingLiteral>(
      static_cast<char*>(effective_dst_ptr), xla_shape);

  // Start the transfer.
  xla::Future<> to_literal_future = (*buffer_or)->ToLiteral(literal.get());

  // When the transfer completes we will:
  //   1. Pack FP4 data if necessary.
  //   2. Release the underlying DeviceBufferList.
  //   3. Format the transfer error using WrapXlaError.
  //   4. Set the chained promise.
  // `literal` and `device_buffer_list` are kept alive until the transfer
  // completes.
  to_literal_future.OnReady(
      [literal = std::move(literal),
       device_buffer_list = buffer_ref.device_buffer_list(),

       promise = std::move(promise), fp4_receiver = std::move(fp4_receiver),
       dst_ptr](absl::Status s) mutable {
        if (s.ok() && !fp4_receiver.empty()) {
          PackFp4Vector(fp4_receiver, static_cast<uint8_t*>(dst_ptr));
        }
        device_buffer_list.reset();
        promise.Set(WrapXlaError(
            s,
            /*context=*/"failed to copy buffer from device to host"));
      });
}

}  // namespace

// TODO(bawilson): refactor this to simplify
xla::Future<void> TpuMemcpyDtoHDirect(const DeviceBufferRef& buffer_ref,
                                      void* dst_ptr,
                                      const void* dst_pinned_ptr) {
  if (buffer_ref.size_bytes() == 0) {
    return absl::OkStatus();
  }
  std::vector<uint8_t> fp4_receiver;
  if (buffer_ref.element_type() == mlir::ElementType::F4E2M1FN) {
    // buffer_ref.size_bytes() reflects the unpacked size of the data;
    // it uses xla::ShapeUtil::ByteSizeOfPrimitiveType which cannot report less
    // than 1 byte per element.
    fp4_receiver.resize(buffer_ref.size_bytes());
  }

  auto [chained_promise, chained_future] = xla::MakePromise<void>();
  if (dst_pinned_ptr != nullptr) {
    bool dst_held = HoldPinnedPointerUntil(dst_pinned_ptr, chained_future);
    if (ABSL_VLOG_IS_ON(1)) {
      if (dst_held) {
        ABSL_VLOG(1) << "[TpuMemcpyDtoHDirect] Destination pointer is pinned "
                        "memory; keeping it alive until the "
                        "device-to-host transfer completes.";
      } else {
        ABSL_VLOG(1)
            << "[TpuMemcpyDtoHDirect] Destination pointer is not pinned "
               "memory, continuing";
      }
    }
  }

  if (buffer_ref.is_materialized()) {
    // The buffer is already materialized, so we can start the transfer now.
    StartD2H(absl::OkStatus(), buffer_ref, std::move(fp4_receiver), dst_ptr,
             std::move(chained_promise));
  } else {
    // The buffer is not materialized, so we will start the transfer when it
    // becomes materialized.
    buffer_ref.GetMaterializationFuture().OnReady(
        [buffer_ref, fp4_receiver = std::move(fp4_receiver), dst_ptr,
         chained_promise = std::move(chained_promise)](absl::Status s) mutable {
          StartD2H(s, buffer_ref, std::move(fp4_receiver), dst_ptr,
                   std::move(chained_promise));
        });
  }
  return chained_future;
}

// RAII management holder for a PjRt RawBuffer alias and its underlying
// DeviceBufferRef.
//
// During asynchronous TPU DMA transfers (Device-to-Host or Host-to-Device),
// the DMA engine accesses physical device memory asynchronously. This RAII
// struct guarantees that:
// 1. The underlying DeviceBufferRef and its PjRtBuffer remain alive and
//    referenced until the DMA completes.
// 2. When the last reference to this holder is dropped, the PJRT_RawBuffer
//    alias is cleanly destroyed via the PjRt C API extension BEFORE
//    device_buffer_ref is destroyed.
struct PreparedRawDmaBuffer::Holder {
  // Reference-counted handle holding the underlying PyTorch TPU DeviceBuffer
  // alive. Declared first so it is destroyed last after ~Holder() releases the
  // PJRT_RawBuffer alias.
  DeviceBufferRef device_buffer_ref;
  // Global PjRt C API function table pointer.
  const PJRT_Api* c_api{nullptr};
  // PjRt RawBuffer extension function table pointer.
  const PJRT_RawBuffer_Extension* extension{nullptr};
  // Handle to the active raw buffer alias on the TPU device.
  PJRT_RawBuffer* buffer{nullptr};

  Holder(DeviceBufferRef ref, const PJRT_Api* api,
         const PJRT_RawBuffer_Extension* ext, PJRT_RawBuffer* buf)
      : device_buffer_ref(std::move(ref)),
        c_api(api),
        extension(ext),
        buffer(buf) {}

  Holder(const Holder&) = delete;
  Holder& operator=(const Holder&) = delete;

  ~Holder() {
    // Release the PjRt raw buffer alias before device_buffer_ref is destroyed.
    if (buffer != nullptr && c_api != nullptr && extension != nullptr &&
        extension->PJRT_RawBuffer_Destroy != nullptr) {
      PJRT_RawBuffer_Destroy_Args args;
      args.struct_size = PJRT_RawBuffer_Destroy_Args_STRUCT_SIZE;
      args.extension_start = nullptr;
      args.buffer = buffer;
      pjrt::LogFatalIfPjrtError(extension->PJRT_RawBuffer_Destroy(&args),
                                c_api);
    }
  }
};

absl::StatusOr<PreparedRawDmaBuffer> PreparedRawDmaBuffer::Prepare(
    DeviceBufferRef buffer_ref) {
  // Await the PjRtBuffer from the reference and verify computation readiness.
  TT_ASSIGN_OR_RETURN(xla::PjRtBuffer * buffer, buffer_ref.AwaitBuffer());
  TT_RETURN_IF_ERROR(
      WrapXlaError(buffer->GetReadyFuture().Await(),
                   "TPU buffer computation failed or not ready"));

  // Ensure the buffer is backed by the PjRt C API client.
  auto* capi_buffer = dynamic_cast<xla::PjRtCApiBuffer*>(buffer);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=initialization condition
      capi_buffer != nullptr, error::kInvalidArgument)
      << "Expected PjRtCApiBuffer.";

  const PJRT_Api* c_api = capi_buffer->pjrt_c_api();
  auto* capi_client = dynamic_cast<xla::PjRtCApiClient*>(capi_buffer->client());
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=initialization condition
      capi_client != nullptr, error::kInvalidArgument)
      << "Expected PjRtCApiClient.";

  // Locate the RawBuffer extension required for direct DMA copy.
  const auto* extension = capi_client->FindExtension<PJRT_RawBuffer_Extension>(
      PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=initialization condition
      extension != nullptr, error::kUnavailable)
      << "RawBuffer extension not available.";

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=initialization condition
      extension->PJRT_RawBuffer_CreateRawAliasOfBuffer != nullptr,
      error::kUnavailable)
      << "PJRT_RawBuffer_CreateRawAliasOfBuffer not implemented.";

  // Create a raw alias of the PjRt buffer for direct memory access.
  PJRT_RawBuffer_CreateRawAliasOfBuffer_Args create_args;
  create_args.struct_size =
      PJRT_RawBuffer_CreateRawAliasOfBuffer_Args_STRUCT_SIZE;
  create_args.extension_start = nullptr;
  create_args.buffer = capi_buffer->c_buffer();
  create_args.raw_buffer = nullptr;

  PJRT_Error* create_error =
      extension->PJRT_RawBuffer_CreateRawAliasOfBuffer(&create_args);
  if (create_error != nullptr) {
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error_cleanup(
        create_error, pjrt::MakeErrorDeleter(c_api));
    return WrapXlaError(pjrt::PjrtErrorToStatus(create_error, c_api),
                        "Failed to create raw alias of buffer");
  }

  return PreparedRawDmaBuffer(std::make_shared<Holder>(
      std::move(buffer_ref), c_api, extension, create_args.raw_buffer));
}

absl::StatusOr<xla::Future<>> PreparedRawDmaBuffer::CopyDtoH(
    void* dst, int64_t offset, int64_t transfer_size) const {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=initialization condition
      holder_ != nullptr &&
          holder_->extension->PJRT_RawBuffer_CopyRawDeviceToHost != nullptr,
      error::kUnavailable)
      << "PJRT_RawBuffer_CopyRawDeviceToHost not implemented.";

  PJRT_RawBuffer_CopyRawDeviceToHost_Args args;
  args.struct_size = PJRT_RawBuffer_CopyRawDeviceToHost_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = holder_->buffer;
  args.dst = dst;
  args.offset = offset;
  args.transfer_size = transfer_size;
  args.event = nullptr;

  PJRT_Error* error =
      holder_->extension->PJRT_RawBuffer_CopyRawDeviceToHost(&args);
  if (error != nullptr) {
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error_cleanup(
        error, pjrt::MakeErrorDeleter(holder_->c_api));
    return WrapXlaError(pjrt::PjrtErrorToStatus(error, holder_->c_api),
                        "PJRT CopyRawDeviceToHost invocation failed");
  }

  auto [promise, future] = xla::MakePromise<void>();
  pjrt::ConvertCEventToCppFuture(args.event, holder_->c_api)
      .OnReady([holder = holder_,
                promise = std::move(promise)](absl::Status status) mutable {
        holder.reset();
        promise.Set(std::move(status));
      });
  return future;
}

absl::StatusOr<xla::Future<>> PreparedRawDmaBuffer::CopyHtoD(
    const void* src, int64_t offset, int64_t transfer_size) const {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=initialization condition
      holder_ != nullptr &&
          holder_->extension->PJRT_RawBuffer_CopyRawHostToDevice != nullptr,
      error::kUnavailable)
      << "PJRT_RawBuffer_CopyRawHostToDevice not implemented.";

  PJRT_RawBuffer_CopyRawHostToDevice_Args args;
  args.struct_size = PJRT_RawBuffer_CopyRawHostToDevice_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = holder_->buffer;
  args.src = src;
  args.offset = offset;
  args.transfer_size = transfer_size;
  args.event = nullptr;

  PJRT_Error* error =
      holder_->extension->PJRT_RawBuffer_CopyRawHostToDevice(&args);
  if (error != nullptr) {
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error_cleanup(
        error, pjrt::MakeErrorDeleter(holder_->c_api));
    return WrapXlaError(pjrt::PjrtErrorToStatus(error, holder_->c_api),
                        "PJRT CopyRawHostToDevice invocation failed");
  }

  auto [promise, future] = xla::MakePromise<void>();
  pjrt::ConvertCEventToCppFuture(args.event, holder_->c_api)
      .OnReady([holder = holder_,
                promise = std::move(promise)](absl::Status status) mutable {
        holder.reset();
        promise.Set(std::move(status));
      });
  return future;
}

absl::StatusOr<xla::Future<>> TpuAsyncDmaCopyDtoH(
    const DeviceBufferRef& src_buffer_ref, void* dst_host_ptr,
    int64_t copy_bytes) {
  TT_ASSIGN_OR_RETURN(PreparedRawDmaBuffer raw_buffer,
                      PreparedRawDmaBuffer::Prepare(src_buffer_ref));
  return raw_buffer.CopyDtoH(dst_host_ptr, /*offset=*/0, copy_bytes);
}

absl::StatusOr<xla::Future<>> TpuAsyncDmaCopyHtoD(
    const void* src_host_ptr, const DeviceBufferRef& dst_buffer_ref,
    int64_t dst_byte_offset, int64_t copy_bytes) {
  TT_ASSIGN_OR_RETURN(PreparedRawDmaBuffer raw_buffer,
                      PreparedRawDmaBuffer::Prepare(dst_buffer_ref));
  return raw_buffer.CopyHtoD(src_host_ptr, dst_byte_offset, copy_bytes);
}

}  // namespace torch_tpu
