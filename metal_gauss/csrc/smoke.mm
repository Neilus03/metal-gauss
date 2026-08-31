// Smallest possible custom Metal kernel driven from PyTorch MPS tensors.
// Exists to prove the toolchain works before any real kernel is written.
//
// NOT part of the build: nothing references this file, and metal_backend.py
// compiles rasterize.mm only. It is kept as a standalone reproducer for
// diagnosing a broken Metal/PyTorch toolchain, where the question is "does
// ANY custom kernel run here" and the real rasterizer is too large to answer
// it. Compile it by hand against the same flags metal_backend.py uses.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/mps.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static const char* kSource = R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void scale_kernel(device const float* in  [[buffer(0)]],
                         device float*       out [[buffer(1)]],
                         constant float&     k   [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    out[gid] = in[gid] * k;
}
)METAL";

static id<MTLComputePipelineState> pipeline(id<MTLDevice> dev) {
    static id<MTLComputePipelineState> pso = nil;
    if (pso) return pso;
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                                           options:nil error:&err];
    TORCH_CHECK(lib, "Metal library compile failed: ",
                err ? err.localizedDescription.UTF8String : "unknown");
    id<MTLFunction> fn = [lib newFunctionWithName:@"scale_kernel"];
    TORCH_CHECK(fn, "kernel scale_kernel not found");
    pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    TORCH_CHECK(pso, "pipeline creation failed");
    return pso;
}

torch::Tensor scale(torch::Tensor input, double k) {
    TORCH_CHECK(input.device().is_mps(), "input must be an MPS tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "input must be float32");
    input = input.contiguous();
    torch::Tensor out = torch::empty_like(input);

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLComputePipelineState> pso = pipeline(dev);
        id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
        TORCH_CHECK(cb, "no MPS command buffer");
        dispatch_queue_t q = torch::mps::get_dispatch_queue();

        dispatch_sync(q, ^{
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(input) offset:input.storage_offset()*sizeof(float) atIndex:0];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(out)   offset:out.storage_offset()*sizeof(float)   atIndex:1];
            float kf = (float)k;
            [enc setBytes:&kf length:sizeof(float) atIndex:2];
            NSUInteger n = input.numel();
            NSUInteger tg = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, n);
            [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
            [enc endEncoding];
            torch::mps::commit();
        });
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("scale", &scale); }
