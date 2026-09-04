#include <metal_stdlib>
using namespace metal;

// Forward rasterisation: one thread per pixel. Each thread walks its own tile's
// depth-sorted Gaussian list front to back, accumulating colour and
// transmittance, and stops early once the pixel is effectively opaque.
//
// Sorting and tile binning stay in PyTorch (torch.sort on MPS), matching the
// design of the abandoned gsplat MPS PR. A Metal radix sort would remove that
// round trip but is not the bottleneck: the per-pixel loop below is.
#define STAGE 256

// Metal versions used by some Apple Silicon CI images do not expose
// `atomic_float`. The gradient buffers are still float tensors, so use an
// atomic uint as a bitwise CAS lock-free accumulator over the same 32-bit
// storage. Floating-point addition is not associative, and the native float
// atomic path is also order-dependent; this preserves that behavior while
// keeping the kernel source compatible with older Metal compilers.
inline void atomic_add_float(device atomic_uint* address, float value)
{
    if (value == 0.0f) return;
    uint expected = atomic_load_explicit(address, memory_order_relaxed);
    while (true) {
        const float current = as_type<float>(expected);
        const uint desired = as_type<uint>(current + value);
        if (atomic_compare_exchange_weak_explicit(
                address, &expected, desired,
                memory_order_relaxed, memory_order_relaxed)) return;
    }
}

kernel void rasterize_forward(
    device const float2*  uv            [[buffer(0)]],
    device const packed_float3* conic   [[buffer(1)]],
    device const float*   opacity       [[buffer(2)]],
    device const packed_float3* color   [[buffer(3)]],
    device const int*     gauss_ids     [[buffer(4)]],
    device const int*     tile_offsets  [[buffer(5)]],
    device packed_float3* out_rgb       [[buffer(6)]],
    device float*         out_alpha     [[buffer(7)]],
    device float*         out_T         [[buffer(8)]],
    device int*           out_ncontrib  [[buffer(9)]],
    constant uint4&       dims          [[buffer(10)]],  // W, H, tile, tiles_x
    uint2 gid  [[thread_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 tgs  [[threads_per_threadgroup]])
{
    const uint W = dims.x, H = dims.y, tile = dims.z, tiles_x = dims.w;
    // No early return: every thread must reach the threadgroup barriers, so
    // out-of-image threads become passive loaders.
    const bool inimg = (gid.x < W) && (gid.y < H);

    const uint tile_id = (min(gid.y, H - 1) / tile) * tiles_x + (min(gid.x, W - 1) / tile);
    const int  start   = tile_offsets[tile_id];
    const int  end     = tile_offsets[tile_id + 1];

    const float2 px = float2(float(gid.x) + 0.5f, float(gid.y) + 0.5f);
    const uint lane = lid.y * tgs.x + lid.x;

    // Cooperative staging (classic Inria pattern): each of the 256 threads
    // loads one gaussian's attributes per batch into threadgroup memory; all
    // pixels then walk the staged batch. Turns 5 scattered device loads per
    // (pixel, gaussian) into one per (threadgroup, gaussian).
    threadgroup float2 s_uv[STAGE];
    threadgroup float3 s_conic[STAGE];
    threadgroup float  s_op[STAGE];
    threadgroup float3 s_col[STAGE];
    threadgroup atomic_int s_done;

    float  T   = 1.0f;
    float3 acc = float3(0.0f);
    float  a_acc = 0.0f;
    int    stop = start;
    bool   done = !inimg;

    for (int base = start; base < end; base += STAGE) {
        if (lane == 0) atomic_store_explicit(&s_done, 0, memory_order_relaxed);
        const int n = min(STAGE, end - base);
        // Grid-stride load: STAGE need not equal the threadgroup size, so the
        // tile size is a free parameter. The earlier version assumed
        // threadgroup == STAGE == 256 and silently read uninitialised
        // threadgroup memory for any other tile size.
        for (int t = int(lane); t < n; t += int(tgs.x * tgs.y)) {
            const int g = gauss_ids[base + t];
            s_uv[t] = uv[g];
            s_conic[t] = float3(conic[g]);
            s_op[t] = opacity[g];
            s_col[t] = float3(color[g]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (!done) {
            for (int j = 0; j < n; ++j) {
                const float2 d = px - s_uv[j];
                const float3 c = s_conic[j];
                const float power = -0.5f * (c.x * d.x * d.x + 2.0f * c.y * d.x * d.y
                                             + c.z * d.y * d.y);
                if (power > 0.0f) continue;
                float alpha = min(0.999f, s_op[j] * exp(power));
                if (alpha < 1.0f / 255.0f) continue;
                const float w = alpha * T;
                acc   += w * s_col[j];
                a_acc += w;
                T     *= (1.0f - alpha);
                stop   = base + j + 1;
                if (T < 1e-4f) { done = true; break; }
            }
        }
        if (done) atomic_fetch_add_explicit(&s_done, 1, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atomic_load_explicit(&s_done, memory_order_relaxed) >= int(tgs.x * tgs.y))
            break;                                   // whole tile is opaque
    }
    if (!inimg) return;

    const uint o = gid.y * W + gid.x;
    out_rgb[o]      = packed_float3(acc);
    out_alpha[o]    = a_acc;
    out_T[o]        = T;
    // Taming-3DGS-style: store the stop INDEX, not a contributor count, so the
    // backward can start its reverse walk directly instead of re-walking the
    // whole list forward to find where the forward pass ended.
    out_ncontrib[o] = stop;
}

// Backward pass. Walks the same list BACK to front, reconstructing the
// transmittance at each Gaussian by dividing it out, which avoids storing a
// per-(pixel, gaussian) buffer. Gradients accumulate atomically because many
// pixels touch the same Gaussian.
kernel void rasterize_backward(
    device const float2*  uv            [[buffer(0)]],
    device const packed_float3* conic   [[buffer(1)]],
    device const float*   opacity       [[buffer(2)]],
    device const packed_float3* color   [[buffer(3)]],
    device const int*     gauss_ids     [[buffer(4)]],
    device const int*     tile_offsets  [[buffer(5)]],
    device const float*   final_T       [[buffer(6)]],
    device const int*     n_contrib    [[buffer(7)]],
    device const packed_float3* grad_rgb [[buffer(8)]],
    device const float*   grad_alpha    [[buffer(9)]],
    device atomic_uint*    d_uv          [[buffer(10)]],  // 2 per gaussian
    device atomic_uint*    d_conic       [[buffer(11)]],  // 3 per gaussian
    device atomic_uint*    d_opacity     [[buffer(12)]],  // 1 per gaussian
    device atomic_uint*    d_color       [[buffer(13)]],  // 3 per gaussian
    constant uint4&       dims          [[buffer(14)]],
    // absgrad: sum over PIXELS of |d_uv|, not the magnitude of the summed
    // gradient. A gaussian straddling an edge gets opposing per-pixel pushes
    // that cancel in d_uv while the gaussian is plainly under-fitting; this
    // statistic does not cancel, which is what makes it a better densification
    // signal (gsplat's absgrad). Written to a separate buffer because it is a
    // statistic, not an adjoint -- nothing differentiates through it.
    device atomic_uint*    d_absuv       [[buffer(15)]],  // 1 per gaussian
    // Uniform across the whole dispatch, so the branches below are perfectly
    // predicted and cost nothing when absgrad is not requested. Without this
    // the reduction, the sqrt and a tenth atomic ran on EVERY backward,
    // including every run that discards the statistic -- which is the default.
    constant uint&        want_absgrad  [[buffer(16)]],
    uint2 gid  [[thread_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 tgs  [[threads_per_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]])
{
    const uint W = dims.x, H = dims.y, tile = dims.z, tiles_x = dims.w;
    const bool inimg = (gid.x < W) && (gid.y < H);

    const uint tile_id = (min(gid.y, H - 1) / tile) * tiles_x + (min(gid.x, W - 1) / tile);
    const int  start   = tile_offsets[tile_id];
    const int  end     = tile_offsets[tile_id + 1];

    const uint o  = min(gid.y, H - 1) * W + min(gid.x, W - 1);
    const float2 px = float2(float(gid.x) + 0.5f, float(gid.y) + 0.5f);
    const uint lane = lid.y * tgs.x + lid.x;

    const float3 dL_drgb = inimg ? float3(grad_rgb[o]) : float3(0);
    const float  dL_da   = inimg ? grad_alpha[o] : 0.0f;
    // Forward stored its stop as a list index (Taming-3DGS style).
    const int my_stop = inimg ? n_contrib[o] : start;

    // All lanes walk the same full [start, end) range back to front, staged
    // batch by staged batch, and PREDICATE their contribution instead of
    // branching. That keeps every lane of a simdgroup on the same gaussian at
    // the same time, which is what makes simd_sum aggregation legal: 32 lanes
    // reduce their gradient contributions in-register and lane 0 issues ONE
    // atomic per component instead of 32 (LichtFeld PR #1673 pattern; their
    // ncu profile showed the naive version at 12% compute, 94% atomic-stall).
    threadgroup float2 s_uv[STAGE];
    threadgroup float3 s_conic[STAGE];
    threadgroup float  s_op[STAGE];
    threadgroup float3 s_col[STAGE];

    float  T = inimg ? final_T[o] : 1.0f;
    float3 suffix = float3(0.0f);
    float  suffix_a = 0.0f;

    const int nbatch = (end - start + STAGE - 1) / STAGE;
    for (int bi = nbatch - 1; bi >= 0; --bi) {
        const int base = start + bi * STAGE;
        const int n = min(STAGE, end - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = int(lane); t < n; t += int(tgs.x * tgs.y)) {
            const int g = gauss_ids[base + t];
            s_uv[t] = uv[g];
            s_conic[t] = float3(conic[g]);
            s_op[t] = opacity[g];
            s_col[t] = float3(color[g]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int j = n - 1; j >= 0; --j) {
            const int i = base + j;
            const float2 d = px - s_uv[j];
            const float3 c = s_conic[j];
            const float power = -0.5f * (c.x * d.x * d.x + 2.0f * c.y * d.x * d.y
                                         + c.z * d.y * d.y);
            const float gauss = exp(min(power, 0.0f));
            const float op = s_op[j];
            float alpha_raw = op * gauss;
            const bool clamped = alpha_raw > 0.999f;
            const float alpha = min(0.999f, alpha_raw);

            // this lane actually blended gaussian i in the forward pass?
            const bool active = inimg && (i < my_stop) && (power <= 0.0f)
                                && (alpha >= 1.0f / 255.0f);

            float3 dc = float3(0);
            float  dux = 0, duy = 0, dcx = 0, dcy = 0, dcz = 0, dop = 0;
            if (active) {
                T = T / max(1.0f - alpha, 1e-8f);
                const float w = alpha * T;
                dc = w * dL_drgb;
                float dL_dalpha = dot(dL_drgb, T * s_col[j] - suffix * (1.0f / max(1.0f - alpha, 1e-8f)))
                                + dL_da * (T - suffix_a * (1.0f / max(1.0f - alpha, 1e-8f)));
                suffix   += alpha * T * s_col[j];
                suffix_a += alpha * T;
                if (!clamped) {
                    dop = dL_dalpha * gauss;
                    const float dL_dpower = dL_dalpha * op * gauss;
                    const float dL_ddx = dL_dpower * -(c.x * d.x + c.y * d.y);
                    const float dL_ddy = dL_dpower * -(c.y * d.x + c.z * d.y);
                    dux = -dL_ddx;
                    duy = -dL_ddy;
                    dcx = dL_dpower * -0.5f * d.x * d.x;
                    dcy = dL_dpower * -1.0f * d.x * d.y;
                    dcz = dL_dpower * -0.5f * d.y * d.y;
                }
            }

            // simdgroup reduction: 10 sums, one atomic each from lane 0
            const float r_cx = simd_sum(dc.x), r_cy = simd_sum(dc.y), r_cz = simd_sum(dc.z);
            const float r_ux = simd_sum(dux),  r_uy = simd_sum(duy);
            const float r_k0 = simd_sum(dcx),  r_k1 = simd_sum(dcy), r_k2 = simd_sum(dcz);
            const float r_op = simd_sum(dop);
            // magnitude BEFORE the reduction, so opposing pixels add instead of
            // cancelling. simd_sum(|.|) != |simd_sum(.)| is the entire point.
            // simd_sum is convergent, so it must stay outside any divergent
            // branch -- want_absgrad is dispatch-uniform, which is why it is
            // safe here and why a per-lane condition would not be.
            float r_abs = 0.0f;
            if (want_absgrad) r_abs = simd_sum(sqrt(dux * dux + duy * duy));
            if (simd_lane == 0 && (r_cx != 0 || r_cy != 0 || r_cz != 0 || r_ux != 0
                                   || r_uy != 0 || r_k0 != 0 || r_k1 != 0 || r_k2 != 0
                                   || r_op != 0)) {
                const int g = gauss_ids[i];
                atomic_add_float(&d_color[3*g+0], r_cx);
                atomic_add_float(&d_color[3*g+1], r_cy);
                atomic_add_float(&d_color[3*g+2], r_cz);
                atomic_add_float(&d_uv[2*g+0], r_ux);
                atomic_add_float(&d_uv[2*g+1], r_uy);
                atomic_add_float(&d_conic[3*g+0], r_k0);
                atomic_add_float(&d_conic[3*g+1], r_k1);
                atomic_add_float(&d_conic[3*g+2], r_k2);
                atomic_add_float(&d_opacity[g], r_op);
                if (want_absgrad && r_abs != 0)
                    atomic_add_float(&d_absuv[g], r_abs);
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Pack per-intersection attributes into sorted, contiguous arrays.
//
// The rasteriser stages each tile's gaussians into threadgroup memory with
//     g = gauss_ids[base + t];  s_uv[t] = uv[g];  ...
// which is four SCATTERED device loads per (threadgroup, gaussian): 256 lanes
// each fetching 8-12 bytes from unrelated addresses, the worst case for the
// memory system. Packing pays that gather ONCE here and lets both the forward
// and the backward read `packed[base + t]` -- 256 lanes over contiguous
// memory, fully coalesced.
//
// This is msplat's prefix_sort_pack / packed_xy_opac / packed_conic /
// packed_rgb, read out of their metallib's argument names. Storage is 9 floats
// per intersection: 34 MB at 100k splats / 800x800, where the count is 956k.
// ---------------------------------------------------------------------------
kernel void pack_intersections(
    device const float2*        uv          [[buffer(0)]],
    device const packed_float3* conic       [[buffer(1)]],
    device const float*         opacity     [[buffer(2)]],
    device const packed_float3* color       [[buffer(3)]],
    device const int*           gauss_ids   [[buffer(4)]],
    device packed_float3*       p_xy_opac   [[buffer(5)]],   // (u, v, opacity)
    device packed_float3*       p_conic     [[buffer(6)]],
    device packed_float3*       p_rgb       [[buffer(7)]],
    constant uint&              n_isect     [[buffer(8)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n_isect) return;
    const int g = gauss_ids[i];
    const float2 t = uv[g];
    p_xy_opac[i] = packed_float3(t.x, t.y, opacity[g]);
    p_conic[i]   = conic[g];
    p_rgb[i]     = color[g];
}
