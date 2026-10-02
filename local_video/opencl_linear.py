"""Bounded OpenCL 1.2 BF16-weight/FP32-activation linear inference.

No CUDA/XPU framework or full GPU model copy is required. A single mapped
matrix is uploaded at a time. Attention, VAE and safety remain in the isolated
CPU runtime. This is partial neural GPU execution, not video encoding.
"""
from __future__ import annotations
import ctypes as C
import os
import time

P, U, I, S, B = C.c_void_p, C.c_uint32, C.c_int32, C.c_size_t, C.c_uint64
KERNEL = r"""
__kernel void linear_bf16(__global const float *a, __global const ushort *w,
                         __global const float *bias, __global float *out,
                         int M, int N, int K, int has_bias) {
    const int col = get_global_id(0), row = get_global_id(1);
    const int x = get_local_id(0), y = get_local_id(1);
    __local float aa[16][16], ww[16][16];
    float sum = 0.0f;
    for (int base = 0; base < K; base += 16) {
        aa[y][x] = (row < M && base+x < K) ? a[row*K+base+x] : 0.0f;
        ww[y][x] = (col < N && base+y < K) ? as_float(((uint)w[col*K+base+y]) << 16) : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < 16; ++k) sum += aa[y][k] * ww[k][x];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) out[row*N+col] = sum + (has_bias ? bias[col] : 0.0f);
}
"""


def api():
    if os.name != "nt":
        raise RuntimeError("이 포터블 OpenCL 실행은 Windows용입니다.")
    dll = C.WinDLL("OpenCL.dll", use_last_error=True)
    signatures = {
        "clGetPlatformIDs": (I, [U, C.POINTER(P), C.POINTER(U)]),
        "clGetDeviceIDs": (I, [P, B, U, C.POINTER(P), C.POINTER(U)]),
        "clGetDeviceInfo": (I, [P, U, S, P, C.POINTER(S)]),
        "clCreateContext": (P, [P, U, C.POINTER(P), P, P, C.POINTER(I)]),
        "clCreateCommandQueue": (P, [P, P, B, C.POINTER(I)]),
        "clCreateProgramWithSource": (P, [P, U, C.POINTER(C.c_char_p), C.POINTER(S), C.POINTER(I)]),
        "clBuildProgram": (I, [P, U, C.POINTER(P), C.c_char_p, P, P]),
        "clGetProgramBuildInfo": (I, [P, P, U, S, P, C.POINTER(S)]),
        "clCreateKernel": (P, [P, C.c_char_p, C.POINTER(I)]),
        "clCreateBuffer": (P, [P, B, S, P, C.POINTER(I)]),
        "clSetKernelArg": (I, [P, U, S, P]),
        "clEnqueueWriteBuffer": (I, [P, P, U, S, S, P, U, P, P]),
        "clEnqueueReadBuffer": (I, [P, P, U, S, S, P, U, P, P]),
        "clEnqueueNDRangeKernel": (I, [P, P, U, P, C.POINTER(S), C.POINTER(S), U, P, P]),
        "clFinish": (I, [P]),
    }
    for name in ("clReleaseMemObject", "clReleaseKernel", "clReleaseProgram", "clReleaseCommandQueue", "clReleaseContext"):
        signatures[name] = (I, [P])
    for name, (result, arguments) in signatures.items():
        function = getattr(dll, name)
        function.restype, function.argtypes = result, arguments
    return dll


def check(code, operation):
    if code:
        raise RuntimeError(f"OpenCL {operation} 실패 ({code})")


def inventory(with_handles=False):
    dll = api()
    count = U()
    code = dll.clGetPlatformIDs(0, None, C.byref(count))
    if code == -1001:
        return []
    check(code, "platform")
    platforms = (P * count.value)()
    check(dll.clGetPlatformIDs(count, platforms, None), "platform list")
    result = []
    def info(device, key, numeric=None):
        if numeric:
            value = numeric()
            check(dll.clGetDeviceInfo(device, key, C.sizeof(value), C.byref(value), None), "device property")
            return value.value
        size = S()
        check(dll.clGetDeviceInfo(device, key, 0, None, C.byref(size)), "device info size")
        text = C.create_string_buffer(size.value)
        check(dll.clGetDeviceInfo(device, key, size, text, None), "device info")
        return text.value.decode("utf-8", "replace")
    for platform in platforms:
        count = U()
        code = dll.clGetDeviceIDs(platform, 4, 0, None, C.byref(count))
        if code == -1:
            continue
        check(code, "GPU list size")
        devices = (P * count.value)()
        check(dll.clGetDeviceIDs(platform, 4, count, devices, None), "GPU list")
        for handle in devices:
            name, vendor = info(handle, 0x102B), info(handle, 0x102C).lower()
            shared = bool(info(handle, 0x1035, U))
            item = {"id": "OpenCL" + str(len(result)), "name": name,
                    "vendor": "intel" if "intel" in vendor else "nvidia" if "nvidia" in vendor else "amd" if any(x in vendor for x in ["amd", "advanced micro"]) else "other",
                    "shared_memory": shared, "kind": "integrated" if shared else "discrete",
                    "maximum_allocation_bytes": info(handle, 0x1010, B),
                    "maximum_work_group_size": info(handle, 0x1004, S),
                    "driver": info(handle, 0x102D)}
            if with_handles:
                item["handle"] = handle
            result.append(item)
    return result


def plan(backend="auto", requested_id=None):
    from .devices import choose_device
    if backend == "intel-vulkan":
        raise ValueError("Neodragon GPU는 OpenCL 경로입니다. intel-gpu 또는 auto를 선택하세요.")
    if backend == "cpu":
        devices = []
    else:
        try:
            devices = inventory()
        except (OSError, RuntimeError) as exc:
            if backend != "auto":
                raise RuntimeError("GPU 드라이버를 확인할 수 없습니다: " + str(exc)) from exc
            return {"gpu_inference": False, "backend_assignment": "cpu", "selected_device": None,
                    "fallback_reason": str(exc), "minimum_free_ram_gib": 3.0}
    selected = choose_device(devices, "intel-vulkan" if backend == "intel-gpu" else backend, requested_id)
    if selected and selected["maximum_work_group_size"] < 256:
        raise RuntimeError("GPU의 연산 블록 크기가 이 저메모리 경로를 지원하지 않습니다.")
    return {"gpu_inference": selected is not None, "backend_assignment": "linear=" + selected["id"] if selected else "cpu",
            "selected_device": selected, "minimum_free_ram_gib": 3.0,
            "maximum_gpu_budget_gib": 0.0625 if selected else None,
            "policy": "one_mapped_bf16_matrix_at_a_time", "gpu_scope": "video_transformer_linear_layers_only",
            "target_iris_generation_verified": False}


class LinearEngine:
    """One sequential queue and at most 64 MiB of explicit GPU buffers."""
    def __init__(self, requested_id, budget_bytes=64 * 2**20):
        devices = inventory(with_handles=True)
        matches = [d for d in devices if d["id"] == requested_id]
        if not matches:
            raise RuntimeError("선택한 OpenCL GPU가 더 이상 없습니다.")
        self.device = dict(matches[0])
        device = P(self.device.pop("handle"))
        self.dll, self.buffers, self.resources = api(), {}, []
        self.budget = min(int(budget_bytes), 64 * 2**20)
        if self.budget < 1:
            raise ValueError("GPU 버퍼 예산은 양수여야 합니다.")
        self.calls, self.uploaded, self.peak, self.seconds = 0, 0, 0, 0.0
        self.allocations, self.reuses, self.releases = 0, 0, 0
        self.linear_calls = 0
        self.working_set_trims = 0
        error = I()
        try:
            self.context = self.dll.clCreateContext(None, 1, C.byref(device), None, None, C.byref(error))
            check(error.value, "context")
            self.resources.append(("clReleaseContext", self.context))
            self.queue = self.dll.clCreateCommandQueue(self.context, device, 0, C.byref(error))
            check(error.value, "queue")
            self.resources.append(("clReleaseCommandQueue", self.queue))
            source = C.c_char_p(KERNEL.encode())
            self.program = self.dll.clCreateProgramWithSource(self.context, 1, C.byref(source), None, C.byref(error))
            check(error.value, "program")
            self.resources.append(("clReleaseProgram", self.program))
            code = self.dll.clBuildProgram(self.program, 1, C.byref(device), b"-cl-std=CL1.2", None, None)
            if code:
                log = C.create_string_buffer(8192)
                self.dll.clGetProgramBuildInfo(self.program, device, 0x1183, len(log), log, None)
                raise RuntimeError("GPU 커널 컴파일 실패: " + log.value.decode("utf-8", "replace"))
            self.kernel = self.dll.clCreateKernel(self.program, b"linear_bf16", C.byref(error))
            check(error.value, "kernel")
            self.resources.append(("clReleaseKernel", self.kernel))
        except Exception:
            self.close()
            raise

    def buffer(self, key, size, flags):
        old = self.buffers.get(key)
        if old and old[1] >= size:
            self.reuses += 1
            return old[0]
        if old:
            check(self.dll.clReleaseMemObject(old[0]), "release buffer")
            self.releases += 1
            del self.buffers[key]
        if size > self.device["maximum_allocation_bytes"] or sum(x[1] for x in self.buffers.values()) + size > self.budget:
            raise MemoryError("GPU 임시 버퍼가 저메모리 상한을 초과했습니다.")
        error = I()
        handle = self.dll.clCreateBuffer(self.context, flags, size, None, C.byref(error))
        check(error.value, "buffer")
        self.buffers[key] = (handle, size)
        self.allocations += 1
        self.peak = max(self.peak, sum(x[1] for x in self.buffers.values()))
        return handle

    def prepare_buffers(self, sizes):
        """Reuse capacity, reclaiming oversized buffers before any growth.

        Every tensor is overwritten before use. No weight/tensor identity cache
        is retained, so mutable host arrays are safe. Validate the entire next
        layout before changing existing allocations.
        """
        flags = {"w": 4, "b": 4, "a": 4, "c": 2}
        if set(sizes) != set(flags) or any(size < 1 or size > self.device["maximum_allocation_bytes"] for size in sizes.values()) or sum(sizes.values()) > self.budget:
            raise MemoryError("GPU 임시 버퍼가 저메모리 상한을 초과했습니다.")
        capacities = {key: max(size, self.buffers.get(key, (None, 0))[1]) for key, size in sizes.items()}
        total = sum(capacities.values())
        for key in sorted(sizes, key=lambda key: capacities[key] - sizes[key], reverse=True):
            if total <= self.budget:
                break
            total -= capacities[key] - sizes[key]
            capacities[key] = sizes[key]
        # Reclaim before growth; even temporary double allocation must fit on UMA.
        for key, (handle, capacity) in list(self.buffers.items()):
            if capacity != capacities[key]:
                check(self.dll.clReleaseMemObject(handle), "release buffer")
                self.releases += 1
                del self.buffers[key]
        return {key: self.buffer(key, capacities[key], flags[key]) for key in sizes}

    def linear(self, activation, weight_bits, bias=None):
        import numpy as np
        if activation.dtype != np.float32 or weight_bits.dtype != np.uint16 or activation.ndim != 2 or weight_bits.ndim != 2 or activation.shape[1] != weight_bits.shape[1]:
            raise ValueError("GPU Linear는 연속 FP32 입력과 BF16 원본 행렬이 필요합니다.")
        if not activation.flags.c_contiguous or not weight_bits.flags.c_contiguous:
            raise ValueError("비연속 텐서는 GPU에 전달하지 않습니다.")
        m, k = activation.shape
        n = weight_bits.shape[0]
        if not m or not k or not n:
            raise ValueError("GPU 행렬의 각 크기는 양수여야 합니다.")
        start = time.monotonic()
        bias_values = np.zeros(1, dtype=np.float32) if bias is None else np.ascontiguousarray(bias, dtype=np.float32)
        if bias is not None and bias_values.shape != (n,):
            raise ValueError("GPU bias 크기가 일치하지 않습니다.")
        remaining = self.budget - weight_bits.nbytes - bias_values.nbytes
        rows = min(m, remaining // (4 * (k + n)), self.device["maximum_allocation_bytes"] // (4 * max(k, n)))
        if rows < 1:
            raise MemoryError("현재 행렬은 GPU 버퍼 상한으로 처리할 수 없습니다.")
        layout = self.prepare_buffers({"w": weight_bits.nbytes, "b": bias_values.nbytes,
                                       "a": rows * k * 4, "c": rows * n * 4})
        w, b, a, c = (layout[key] for key in ("w", "b", "a", "c"))
        for handle, values in [(w, weight_bits), (b, bias_values)]:
            check(self.dll.clEnqueueWriteBuffer(self.queue, handle, 1, 0, values.nbytes, P(values.ctypes.data), 0, None, None), "upload")
            self.uploaded += values.nbytes
        result = np.empty((m, n), dtype=np.float32)
        for position in range(0, m, rows):
            count = min(rows, m-position)
            values = activation[position:position+count]
            check(self.dll.clEnqueueWriteBuffer(self.queue, a, 1, 0, values.nbytes, P(values.ctypes.data), 0, None, None), "activation upload")
            self.uploaded += values.nbytes
            arguments = [P(a), P(w), P(b), P(c), I(count), I(n), I(k), I(bias is not None)]
            for index, value in enumerate(arguments):
                check(self.dll.clSetKernelArg(self.kernel, index, C.sizeof(value), C.byref(value)), "argument")
            global_size, local_size = (S * 2)((n+15)//16*16, (count+15)//16*16), (S * 2)(16, 16)
            check(self.dll.clEnqueueNDRangeKernel(self.queue, self.kernel, 2, None, global_size, local_size, 0, None, None), "neural matrix multiplication")
            out = result[position:position+count]
            check(self.dll.clEnqueueReadBuffer(self.queue, c, 1, 0, out.nbytes, P(out.ctypes.data), 0, None, None), "read result")
            self.calls += 1
        self.seconds += time.monotonic() - start
        self.linear_calls += 1
        # Mapped BF16 pages accumulate in the working set as layers are read.
        # Under RAM pressure, let Windows reclaim this owned process's pages;
        # the next matrix can be read again from the selected SSD mapping.
        from .backend import _memory
        working, available = _memory(os.getpid())
        if available < 2.5 * 2**30 and working > 1.0 * 2**30:
            kernel, psapi = C.WinDLL("kernel32"), C.WinDLL("psapi")
            kernel.GetCurrentProcess.restype = P
            psapi.EmptyWorkingSet.argtypes, psapi.EmptyWorkingSet.restype = [P], I
            if psapi.EmptyWorkingSet(kernel.GetCurrentProcess()):
                self.working_set_trims += 1
        return result

    def metrics(self):
        return {"device": self.device, "kernel_calls": self.calls, "uploaded_bytes": self.uploaded,
                "linear_calls": self.linear_calls, "buffer_allocations": self.allocations,
                "buffer_reuses": self.reuses, "buffer_releases": self.releases,
                "buffer_policy": "reuse_capacity_overwrite_every_tensor_within_fixed_cap",
                "peak_explicit_gpu_buffer_bytes": self.peak, "buffer_limit_bytes": self.budget,
                "linear_elapsed_seconds": round(self.seconds, 3), "gpu_scope": "video_transformer_linear_layers_only",
                "owned_process_working_set_trims": self.working_set_trims,
                "total_driver_memory_measured": False, "target_iris_generation_verified": False}

    def close(self):
        for handle, _ in self.buffers.values():
            self.dll.clReleaseMemObject(handle)
        self.buffers.clear()
        for release, handle in reversed(self.resources):
            getattr(self.dll, release)(handle)
        self.resources.clear()
