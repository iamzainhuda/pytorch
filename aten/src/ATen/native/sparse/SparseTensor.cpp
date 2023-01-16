// Basic functions on sparse tensors
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/Layout.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/NamedTensorUtils.h>

#include <ATen/native/Copy.h>
#include <ATen/native/CPUBlas.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_coalesce.h>
#include <ATen/ops/_coalesce_native.h>
#include <ATen/ops/_coalesced_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/_dimI_native.h>
#include <ATen/ops/_dimV_native.h>
#include <ATen/ops/_indices_native.h>
#include <ATen/ops/_nnz_native.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_native.h>
#include <ATen/ops/_sparse_mask_helper_native.h>
#include <ATen/ops/_validate_sparse_coo_tensor_args_native.h>
#include <ATen/ops/_values_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/coalesce_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/copy_sparse_to_sparse.h>
#include <ATen/ops/copy_sparse_to_sparse_native.h>
#include <ATen/ops/dense_dim_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/indices_native.h>
#include <ATen/ops/is_coalesced_native.h>
#include <ATen/ops/resize_as_sparse.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/sparse_coo_tensor_native.h>
#include <ATen/ops/sparse_dim_native.h>
#include <ATen/ops/sparse_mask_native.h>
#include <ATen/ops/sparse_resize_and_clear_native.h>
#include <ATen/ops/sparse_resize_native.h>
#include <ATen/ops/to_dense_native.h>
#include <ATen/ops/to_sparse_native.h>
#include <ATen/ops/unique_dim.h>
#include <ATen/ops/values_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/ones.h>
#endif

namespace at {
namespace native {

using namespace at::sparse;

/******************************************************************************
 * access methods
 ******************************************************************************/

int64_t sparse_dim_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->sparse_dim();
}

int64_t dense_dim_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->dense_dim();
}

bool is_coalesced_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->coalesced();
}

int64_t _nnz_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->nnz();
}

// Why are there so many methods to get indices and value?
// See Note [ Sparse: different methods to get indices and values ] in
// native_functions.yaml

Tensor _indices_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->indices();
}

Tensor _values_sparse(const SparseTensor& self) {
  return get_sparse_impl(self)->values();
}

Tensor& _coalesced_sparse_(SparseTensor& self, bool coalesced) {
  get_sparse_impl(self)->set_coalesced(coalesced);
  return self;
}

Tensor indices_sparse(const Tensor& self) {
  TORCH_CHECK(
      self.is_coalesced(),
      "Cannot get indices on an uncoalesced tensor, please call .coalesce() first");
  return get_sparse_impl(self)->indices().alias();
}

Tensor values_sparse(const Tensor& self) {
  TORCH_CHECK(
      self.is_coalesced(),
      "Cannot get values on an uncoalesced tensor, please call .coalesce() first");
  return get_sparse_impl(self)->values().alias();
}

/******************************************************************************
 * creation methods
 * See NOTE [ Sparse: autograd and API ] for details
 ******************************************************************************/

/*** Helper methods ***/

SparseTensor new_sparse(
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  AT_ASSERT(layout.has_value() && *layout == kSparse);
  DispatchKey dispatch_key;
  switch (device_or_default(device).type()) {
#define DO_CASE(device, _) \
    case DeviceType::device: \
      dispatch_key = DispatchKey::Sparse##device; \
      break;
    C10_FORALL_BACKEND_DEVICE_TYPES(DO_CASE, unused)
#undef DO_CASE
    default:
      TORCH_CHECK(false, "device type not supported for sparse ", device_or_default(device))
  }
  return detail::make_tensor<SparseTensorImpl>(
      DispatchKeySet(dispatch_key),
      scalarTypeToTypeMeta(dtype_or_default(dtype)));
}

/** Actual dispatched creation methods ***/

SparseTensor new_with_dims_sparse(
    int64_t sparse_dim,
    int64_t dense_dim,
    ArrayRef<int64_t> size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  SparseTensor self = new_sparse(dtype, layout, device, pin_memory);
  get_sparse_impl(self)->resize_and_clear_(sparse_dim, dense_dim, size);
  return self;
}

SparseTensor new_with_dims_and_tensor_sparse_symint(
    int64_t sparse_dim,
    int64_t dense_dim,
    c10::SymIntArrayRef size,
    const Tensor& indices,
    const Tensor& values,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  SparseTensor self = new_sparse(dtype, layout, device, pin_memory);
  get_sparse_impl(self)->resize_(sparse_dim, dense_dim, size);
  // NOTE: There is no guarantee that `indices` and `values` don't contain
  // AutogradMeta. However, we want to maintain the invariant that `indices_`
  // and `values_` of a sparse tensor don't contain AutogradMeta, and to achieve
  // that we shallow-copy `indices` and `values` here.
  auto indices_shallow_copy =
      Tensor(indices.unsafeGetTensorImpl()->shallow_copy_and_detach(
          /*version_counter=*/indices.unsafeGetTensorImpl()->version_counter(),
          /*allow_tensor_metadata_change=*/true));
  auto values_shallow_copy =
      Tensor(values.unsafeGetTensorImpl()->shallow_copy_and_detach(
          /*version_counter=*/values.unsafeGetTensorImpl()->version_counter(),
          /*allow_tensor_metadata_change=*/true));
  alias_into_sparse(self, indices_shallow_copy, values_shallow_copy);
  return self;
}

/** Public creation API that dispatch to methods above **/

/** Empty init **/
Tensor empty_sparse(
    IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !pin_memory.has_value() || !*pin_memory,
      "Only dense CPU tensors can be pinned");
  return new_with_dims_sparse(
      size.size(), 0, size, dtype, layout, device, pin_memory);
}

/* Shape init */
Tensor sparse_coo_tensor(IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  return at::_sparse_coo_tensor_with_dims(size.size(), 0, size, options.layout(at::kSparse));
}

/* Pointer-copy init */

// helper
namespace {
static inline Tensor expand_values_if_needed(const Tensor& values) {
  // expand
  if (values.dim() == 0) {
    // Mimic Numpy behavior here and treat it as a 1D tensor
    return values.expand({1});
  } else {
    return values;
  }
}
} // namespace

Tensor sparse_coo_tensor(const Tensor& indices, const Tensor& values_,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  Tensor values = expand_values_if_needed(values_);

  // arg checking
  TORCH_CHECK(
      !options.has_layout() || options.layout() == kSparse,
      "expected sparse layout, but got layout ",
      options.layout());
  // the following checks are redundant because they are also checked in
  // SparseTensorImpl::set_indices_and_values_unsafe but we need to ensure them
  // in order to infer the shape.
  TORCH_CHECK(
      indices.dim() == 2,
      "indices must be sparse_dim x nnz, but got: ",
      indices.sizes())
  TORCH_CHECK(
      !indices.is_sparse(),
      "expected indices to be a dense tensor, but got indices of layout ",
      indices.layout());

  // If sizes are not given, it is inferred as max index of each dim.
  int64_t sparse_dim = indices.size(0);
  int64_t dense_dim = values.dim() - 1;

  std::vector<int64_t> computed_sizes(sparse_dim + dense_dim);
  if (indices.numel() > 0) {
    // If the indices has elements in it, we infer the minimum sparse dimension
    // sizes as the max value of each dim in indices. NB: It used to keepdim. I
    // think that was wrong.
    Tensor min_indices =
        std::get</* values */ 0>(indices.min(/* dim */ 1, /* keepdim */ false));
    Tensor computed_indices_sizes =
        std::get</* values */ 0>(indices.max(/* dim */ 1, /* keepdim */ false));
    computed_indices_sizes.add_(1); // len = max_index + 1
    Tensor cpu_min_indices = min_indices.to(at::DeviceType::CPU);
    Tensor cpu_computed_indices_sizes =
        computed_indices_sizes.to(at::DeviceType::CPU);
    auto cpu_min_indices_accessor = cpu_min_indices.accessor<int64_t, 1>();
    auto cpu_computed_indices_sizes_accessor =
        cpu_computed_indices_sizes.accessor<int64_t, 1>();
    for (const auto d : c10::irange(sparse_dim)) {
      int64_t min_index_in_dim = cpu_min_indices_accessor[d];
      TORCH_CHECK(
          min_index_in_dim >= 0,
          "found negative index ",
          min_index_in_dim,
          " for dim ",
          d);
      computed_sizes[static_cast<size_t>(d)] =
          cpu_computed_indices_sizes_accessor[d];
    }
  } else {
    // If the indices doesn't have elements in it, there is not enough
    // information to know what the minimum sparse dimension sizes should be,
    // and in this case we set them to 0
    for (const auto d : c10::irange(sparse_dim)) {
      computed_sizes[static_cast<size_t>(d)] = 0;
    }
  }
  for (const auto d : c10::irange(dense_dim)) {
    computed_sizes[static_cast<size_t>(sparse_dim + d)] = values.size(d + 1);
  }

  return at::_sparse_coo_tensor_with_dims_and_tensors(
      sparse_dim,
      dense_dim,
      computed_sizes,
      indices,
      values,
      values.options().layout(kSparse));
}

void _validate_sparse_coo_tensor_args(
    const Tensor& indices,
    const Tensor& values_,
    ArrayRef<int64_t> size) {
  Tensor values = expand_values_if_needed(values_);

  // the following checks are redundant because they are also checked in
  // SparseTensorImpl::set_indices_and_values_unsafe but we need to ensure them
  // in order to infer the shape.
  TORCH_CHECK(
      indices.dim() == 2,
      "indices must be sparse_dim x nnz, but got: ",
      indices.sizes())
  TORCH_CHECK(
      !indices.is_sparse(),
      "expected indices to be a dense tensor, but got indices of layout ",
      indices.layout());
  int64_t sparse_dim = indices.size(0);
  int64_t dense_dim = values.dim() - 1;
  TORCH_CHECK(
      static_cast<int64_t>(size.size()) == sparse_dim + dense_dim,
      "number of dimensions must be sparse_dim (",
      sparse_dim,
      ") + dense_dim (",
      dense_dim,
      "), but got ",
      size.size());

  // Check to make sure all indices are within the boundaries of `size`
  if (indices.numel() > 0) {
    Tensor min_indices =
        std::get</* values */ 0>(indices.min(/* dim */ 1, /* keepdim */ false));
    Tensor max_indices =
        std::get</* values */ 0>(indices.max(/* dim */ 1, /* keepdim */ false));
    Tensor cpu_min_indices, cpu_max_indices;
    if (!indices.is_cpu()) {
      cpu_min_indices = min_indices.to(at::DeviceType::CPU);
      cpu_max_indices = max_indices.to(at::DeviceType::CPU);
    } else {
      cpu_min_indices = min_indices;
      cpu_max_indices = max_indices;
    }
    auto cpu_min_indices_accessor = cpu_min_indices.accessor<int64_t, 1>();
    auto cpu_max_indices_accessor = cpu_max_indices.accessor<int64_t, 1>();
    for (const auto d : c10::irange(sparse_dim)) {
      // NB: This used to sync ndim times to access each entry; now we copy
      // everything to CPU first and then access it.
      int64_t min_index_in_dim = cpu_min_indices_accessor[d];
      TORCH_CHECK(
          min_index_in_dim >= 0,
          "found negative index ",
          min_index_in_dim,
          " for dim ",
          d);
      int64_t max_index_in_dim = cpu_max_indices_accessor[d];
      int64_t dim_size = size[static_cast<size_t>(d)];
      TORCH_CHECK(
          max_index_in_dim < dim_size,
          "size is inconsistent with indices: for dim ",
          d,
          ", size is ",
          dim_size,
          " but found index ",
          max_index_in_dim);
    }
  }
}

// NB: Got rid of the sizes == NULL case

Tensor sparse_coo_tensor(const Tensor& indices, const Tensor& values, IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
  // arg checking
  TORCH_CHECK(
      !options.has_layout() || options.layout() == kSparse,
      "expected sparse layout, but got layout ",
      options.layout());
  return at::native::_sparse_coo_tensor_unsafe(
      indices,
      values,
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

Tensor _sparse_coo_tensor_unsafe(const Tensor& indices, const Tensor& values_, at::IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  if (at::globalContext().checkSparseTensorInvariants()) {
    at::native::_validate_sparse_coo_tensor_args(indices, values_, size);
  }
  return at::native::_sparse_coo_tensor_unsafe_symint(indices, values_, c10::fromIntArrayRefSlow(size), dtype, layout, device, pin_memory);
}

// NOTE: _sparse_coo_tensor_unsafe() differs from sparse_coo_tensor()
// in that we don't check whether any indices are out of boundaries of `size`, thus avoiding a
// copy from CUDA to CPU. However, this function should ONLY be used where we know that the indices
// are guaranteed to be within bounds or if the caller is going to call
// _validate_sparse_coo_tensor_args before using the tensor.
// NB: Got rid of the size == NULL case
Tensor _sparse_coo_tensor_unsafe_symint(const Tensor& indices, const Tensor& values_, c10::SymIntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]

  Tensor values = expand_values_if_needed(values_);

  // This guard is intentional: we don't support dynamic shapes along the
  // indices dimension because that implies variable dimensionality
  auto sparse_dim = indices.sym_size(0).guard_int(__FILE__, __LINE__);
  auto dense_dim = values.dim() - 1;

  return at::_sparse_coo_tensor_with_dims_and_tensors_symint(
      sparse_dim,
      dense_dim,
      size,
      indices,
      values,
      values.options().layout(kSparse));
}

// NB: Deleted newWithSizeNd variants

SparseTensor clone_sparse(
    const SparseTensor& self,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "unsupported memory format option ",
      optional_memory_format.value());
  SparseTensor other = new_with_dims_sparse(
      self.sparse_dim(),
      self.dense_dim(),
      self.sizes(),
      optTypeMetaToScalarType(self.options().dtype_opt()),
      self.options().layout_opt(),
      self.options().device_opt(),
      self.options().pinned_memory_opt());
  copy_into_sparse(other, self._indices(), self._values(), true);
  return other._coalesced_(self.is_coalesced());
}

/******************************************************************************
 * reshaping methods
 ******************************************************************************/

const SparseTensor& sparse_resize_(
    const SparseTensor& self,
    ArrayRef<int64_t> size,
    int64_t sparse_dim,
    int64_t dense_dim) {
  get_sparse_impl(self)->resize_(sparse_dim, dense_dim, size);
  return self;
}

const SparseTensor& sparse_resize_and_clear_(
    const SparseTensor& self,
    ArrayRef<int64_t> size,
    int64_t sparse_dim,
    int64_t dense_dim) {
  get_sparse_impl(self)->resize_and_clear_(sparse_dim, dense_dim, size);
  return self;
}

namespace {
bool _is_same_size_as_sparse(
    const SparseTensor& self,
    const SparseTensor& src) {
  return self.sparse_dim() == src.sparse_dim() &&
      self.dense_dim() == src.dense_dim() && self.sizes().equals(src.sizes());
}
} // namespace

// Invoked from native/Resize.cpp (no dynamic dispatch necessary)
const SparseTensor& resize_as_sparse_(const SparseTensor& self, const SparseTensor& src) {
  if (!_is_same_size_as_sparse(self, src)) {
    sparse_resize_(self, src.sizes(), src.sparse_dim(), src.dense_dim());
  }
  return self;
}

SparseTensor dense_to_sparse(const Tensor& self, c10::optional<c10::Layout> layout, OptionalIntArrayRef blocksize, c10::optional<int64_t> dense_dim_opt) {
  if (layout.has_value()) {
    if (blocksize.has_value() && !(*layout == kSparseBsr || *layout == kSparseBsc)) {
      AT_ERROR("to_sparse for ", self.layout(), " to ", *layout, " conversion does not use specified blocksize");
    }
    if (self.layout() == *layout) {
      return self;
    }
    switch (*layout) {
    case kStrided:
      return self;
    case kSparse:
      return dense_to_sparse(self, self.dim() - dense_dim_opt.value_or(0));
    case kSparseCsr:
      return self.to_sparse_csr(dense_dim_opt);
    case kSparseCsc:
      return self.to_sparse_csc(dense_dim_opt);
    case kSparseBsr:
      if (blocksize.has_value()) {
        return self.to_sparse_bsr(*blocksize, dense_dim_opt);
      }
      AT_ERROR("to_sparse for ", self.layout(), " to ", *layout, " conversion requires blocksize");
      break;
    case kSparseBsc:
      if (blocksize.has_value()) {
        return self.to_sparse_bsc(*blocksize, dense_dim_opt);
      }
      break;
      AT_ERROR("to_sparse for ", self.layout(), " to ", *layout, " conversion requires blocksize");
    default:
      break;
    }
    AT_ERROR("to_sparse not implemented for ", self.layout(), " to ", *layout, " conversion");
  }
  return dense_to_sparse(self, self.dim() - dense_dim_opt.value_or(0));
}

SparseTensor dense_to_sparse(const Tensor& self, int64_t sparse_dim) {
  int64_t dims = self.dim();
  // TODO: it seems like sparse_dim == 0 could be supported even if self.dim() >
  // 0, but this would take some work and doesn't seem particularly useful.
  TORCH_CHECK(
      sparse_dim > 0 || self.dim() == 0,
      "sparse_dim must be >0 if dimensionality > 0");
  TORCH_CHECK(
      sparse_dim <= dims,
      "sparse_dim must be less than or equal to self.dim()");
  at::TensorOptions sparse_options = self.options().layout(kSparse);
  std::vector<int64_t> sizes = self.sizes().vec();

  Tensor nz = self.nonzero().transpose(0, 1);
  if (nz.size(1) == 0) {
    auto sparse = new_with_dims_sparse(
        sparse_dim,
        dims - sparse_dim,
        sizes,
        optTypeMetaToScalarType(sparse_options.dtype_opt()),
        sparse_options.layout_opt(),
        sparse_options.device_opt(),
        sparse_options.pinned_memory_opt());
    return sparse._coalesced_(true);
  }
  Tensor indices;
  if (sparse_dim == dims) {
    indices = nz.clone();
  } else {
    Tensor i = nz.narrow(0, 0, sparse_dim);
    std::tie(indices, std::ignore, std::ignore) = unique_dim(i, 1);
    indices = indices.contiguous(); // many sparse CUDA kernels require
                                    // contiguity, see issue #12633
  }

  Tensor values;
  if (self.dim() > 0) {
    auto ix = toListOfOptionalTensors(indices.chunk(indices.size(0), 0));
    values = self.index(ix).squeeze(0).clone(at::MemoryFormat::Preserve);
  } else {
    AT_ASSERT(nz.sizes().equals({0, 1}));
    // In this cases, indices is a clone of nz, which is a tensor of shape (0,
    // 1). Given sparse tensor invariants, values should be shape (1,)
    values = self.unsqueeze(0).clone(at::MemoryFormat::Preserve);
  }

  Tensor sparse = at::sparse_coo_tensor(indices, values, sizes, sparse_options);
  return sparse._coalesced_(true);
}

// NB: Dropped the resizeNd variants

SparseTensor& copy_sparse_wrapper_(
    Tensor& self,
    const Tensor& src,
    bool non_blocking) {
  // TODO: Once copy_ is fully migrated to use dispatcher, handle named
  // inference using dispatcher instead of doing it everywhere
  auto maybe_outnames = namedinference::compute_broadcast_outnames(self, src);
  {
    NoNamesGuard guard;
    if (!self.is_sparse() || !src.is_sparse()) {
      AT_ERROR(
          "copy_() between dense and sparse Tensors is not implemented! Found self type = ",
          self.toString(),
          " and src type = ",
          src.toString());
    }
    at::copy_sparse_to_sparse_(self, src, non_blocking);
  }
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

SparseTensor& copy_sparse_(
    SparseTensor& self,
    const SparseTensor& src,
    bool non_blocking) {
  if (is_same_tensor(self, src))
    return self;
  get_sparse_impl(self)->resize_(
      src.sparse_dim(), src.dense_dim(), src.sizes());
  copy_into_sparse(self, src._indices(), src._values(), non_blocking);
  return self._coalesced_(src.is_coalesced());
}

SparseTensor coalesce(const SparseTensor& self) {
  // See NOTE: [ coalesce autograd ]
  if (self.is_coalesced()) {
    return self;
  }
  return at::_coalesce(self);
}

SparseTensor _coalesce_sparse_cpu(const SparseTensor& self) {
  AT_ASSERT(self.defined());
  TORCH_INTERNAL_ASSERT(at::impl::variable_excluded_from_dispatch());
  AT_ASSERT(self.is_sparse());
  TORCH_INTERNAL_ASSERT(!self.is_coalesced());

  // NOTE: Since `coalesce` is not an in-place operation when `is_coalesced` is false,
  // we should keep the original tensor intact and do coalesce on a copy of the tensor
  if (self._nnz() < 2) {
    SparseTensor dst = self.clone();
    dst._coalesced_(true);
    return dst;
  }

  Tensor indices = self._indices();
  Tensor values = self._values().contiguous();
  int64_t sparse_dim = self.sparse_dim();
  int64_t dense_dim = self.dense_dim();
  int64_t nnz = self._nnz();

  Tensor indices_scalar = flatten_indices(indices, self.sizes());

  SparseTensor dst = new_sparse(
      optTypeMetaToScalarType(self.options().dtype_opt()),
      self.options().layout_opt(),
      self.options().device_opt(),
      self.options().pinned_memory_opt());
  get_sparse_impl(dst)->resize_(sparse_dim, dense_dim, self.sizes());
  // TODO: is there a more idiomatic way to do this?
  Tensor newIndices = at::empty(indices.sizes(), indices.options());
  Tensor newValues = at::empty(values.sizes(), values.options());
  alias_into_sparse(dst, newIndices, newValues);

  Tensor indicesBuffer;
  Tensor indicesPermutation;
  std::tie(indicesBuffer, indicesPermutation) = indices_scalar.sort(0);
  // NB: The accessor accesses here rely on self._nnz() > 0 (tested earlier in
  // this function)
  auto newIndicesAccessor = newIndices.accessor<int64_t, 2>();
  auto indicesAccessor = indices.accessor<int64_t, 2>();
  auto indicesPermutationAccessor = indicesPermutation.accessor<int64_t, 1>();
  auto indicesBufferAccessor = indicesBuffer.accessor<int64_t, 1>();

  int64_t i = -1;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::ComplexHalf, at::ScalarType::BFloat16, at::ScalarType::Half, at::ScalarType::Bool,
      values.scalar_type(), "coalesce", [&] {
    int64_t prev = -1;
    int64_t blockSize = values.stride(0);
    scalar_t* values_ptr = values.data_ptr<scalar_t>();
    scalar_t* newValues_ptr = newValues.data_ptr<scalar_t>();
    for (const auto j : c10::irange(nnz)) {
      int64_t pos = indicesPermutationAccessor[j];
      int64_t curr = indicesBufferAccessor[j];
      if (curr == prev) {
        if (values.numel() >
            0) { // if values is an empty tensor, there are no elements to copy
          at::native::cpublas::axpy<scalar_t>(
              blockSize,
              static_cast<scalar_t>(1),
              values_ptr + pos * blockSize,
              1,
              newValues_ptr + i * blockSize,
              1);
        }
      } else {
        ++i;
        for (const auto d : c10::irange(sparse_dim)) {
          newIndicesAccessor[d][i] = indicesAccessor[d][pos];
        }
        if (values.numel() >
            0) { // if values is an empty tensor, there are no elements to copy
          at::native::cpublas::copy<scalar_t>(
              blockSize,
              values_ptr + pos * blockSize,
              1,
              newValues_ptr + i * blockSize,
              1);
        }
      }
      prev = curr;
    }
  });

  dst._coalesced_(true);
  get_sparse_impl(dst)->set_nnz_and_narrow(i + 1);

  return dst;
}

DEFINE_DISPATCH(sparse_mask_intersection_out_stub);

SparseTensor sparse_mask(const Tensor& t, const SparseTensor& mask) {
  TORCH_CHECK(
      mask.sizes().equals(t.sizes()),
      "sparse_mask(): operands have incompatible sizes; self has size ",
      t.sizes(),
      " but mask has size ",
      mask.sizes());

  if (!mask.numel()) {
    return mask.clone().to(t.device(), t.scalar_type());
  }

  if (t.layout() == at::kSparse) {
    auto intersection = at::empty({0}, t.options());
    sparse_mask_intersection_out_stub(intersection.device().type(), intersection, t, mask);

    if (intersection._nnz() == 0) {
      return mask.clone().to(t.device(), t.scalar_type());
    }

    // TODO: once union kernels are fast, reimplement with something along the lines of
    // return intersection + zeros_like(mask)
    const auto union_indices = at::cat(
        {intersection._indices(), mask._indices()},
        /*dim=*/-1);
    const auto union_values = at::cat(
        {intersection._values(), at::zeros({1}, t._values().options()).expand_as(mask._values())},
        /*dim=*/0);
    const auto union_sparse_t = at::sparse_coo_tensor(
        union_indices,
        union_values,
        t.sizes());
    return union_sparse_t;
  }

  const auto mask_values = mask._values();
  auto mask_template = at::sparse_coo_tensor(
      mask._indices(),
      at::ones({1}, mask_values.options()).expand_as(mask_values),
      mask.sizes())._coalesced_(mask.is_coalesced());
  return t.mul(mask_template).to(t.scalar_type());
}

Tensor sparse_mask_helper_cpu(
    const SparseTensor& t,
    const Tensor& mask_indices) {
  /*
    This is a helper function which filter values from `t._values()` using the
    `mask_indices`. This CPU implementation uses a simple hash_map to filter
    values by matching the `mask_indices` with the indices at tensor input `t`.

    Inputs:
      `t`             - coalesced sparse tensor input
      `mask_indices`  - mask indices tensor

    Note: The nnz in the output tensor will be same as the `mask_indices`. So it
    will works independently if the mask is coalesced or not.
  */
  TORCH_CHECK(t.is_sparse(), "t: input is not a sparse tensor");
  TORCH_CHECK(t.is_coalesced(), "t:  input is uncoalesced");
  TORCH_CHECK(
      mask_indices.dim() == t._indices().dim(),
      "mask_indices: operands have incompatible indices dim; self has dim ",
      t._indices().dim(),
      " but mask has dim ",
      mask_indices.dim());
  TORCH_CHECK(
      mask_indices.is_contiguous(), "mask_indices: mask is not contiguous");

  int64_t r_nnz = mask_indices.size(1);
  auto t_v = t._values();
  auto vsize = t_v.sizes().vec();
  vsize[0] = r_nnz;

  Tensor r_values = at::zeros(vsize, t_v.options());
  auto t_i = t._indices();
  auto t_nnz = t._nnz();

  std::unordered_map<int64_t, int64_t> t_flatten_indices =
      std::unordered_map<int64_t, int64_t>{};
  auto full_size = t.sizes();
  auto ti_flattened_indices = at::sparse::flatten_indices(t_i, full_size);

  // Step 1: flatten the sparse indices `t._indices()` tensor and then  map this
  // flatten value `index` to the original position `i`
  for (const auto i : c10::irange(t_nnz)) {
    int64_t index = ti_flattened_indices.data_ptr<int64_t>()[i];
    t_flatten_indices[index] = i;
  }

  // Step 2: Filter `t._values()` values by matching the flatten `mask_indices`
  // with the flatten `t._indices()` using the hash_map `t_flatten_indices`

  auto flattened_mask_indices =
      at::sparse::flatten_indices(mask_indices, full_size);

  const auto copy_iter = TensorIteratorConfig()
    .add_output(r_values)
    .add_input(t_v)
    .resize_outputs(false)
    .declare_static_shape(r_values.sizes(), /*squash_dims=*/0)
    .build();

  at::parallel_for(0, r_nnz, 0, [&](int64_t start, int64_t end) {
    TensorIterator copy_iter_local(copy_iter);
    const auto r_values_data = reinterpret_cast<char*>(r_values.data_ptr());
    const auto t_values_data = reinterpret_cast<char*>(t_v.data_ptr());
    const auto r_values_stride = r_values.strides()[0] * r_values.element_size();
    const auto t_values_stride = t_v.strides()[0] * t_v.element_size();

    for (const auto i : c10::irange(start, end)) {
      int64_t index = flattened_mask_indices.data_ptr<int64_t>()[i];
      auto iter = t_flatten_indices.find(index);
      if (iter != t_flatten_indices.end()) {
        // r_values[i].copy_(t_v[iter->second])
        copy_iter_local.unsafe_replace_operand(
            0, r_values_data + i * r_values_stride);
        copy_iter_local.unsafe_replace_operand(
            1, t_values_data + iter->second * t_values_stride);
        copy_stub(kCPU, copy_iter_local, /*non_blocking=*/false);
      }
    }
  });
  return r_values;
}

Tensor empty_like_sparse_coo(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  TORCH_CHECK(
    !(options_.has_memory_format() && optional_memory_format.has_value()),
    "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
    "the redundant setter.");

  TensorOptions options =
      self.options()
          .merge_in(options_)
          .merge_memory_format(optional_memory_format);

  TORCH_CHECK(
      !(options.layout() != kStrided &&
          optional_memory_format.has_value()),
      "memory format option is only supported by strided tensors");

  if (options.layout() == kSparse) {
    auto result = at::empty({0}, options);
    result.sparse_resize_and_clear_(
        self.sizes(), self.sparse_dim(), self.dense_dim());
    return result;
  } else {
    return at::native::empty_like(self, dtype, layout, device, pin_memory, optional_memory_format);
  }
}

} // namespace native
} // namespace at
