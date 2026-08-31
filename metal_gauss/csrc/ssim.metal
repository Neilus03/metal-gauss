#include <metal_stdlib>
using namespace metal;

// Fused SSIM tail: the elementwise stage AFTER the separable gaussian blur.
//
// The two grouped convolutions stay in torch -- measured 6.2 ms combined at
// 900x1600, and torch's conv backward is well optimised. What was expensive
// was the ~14 elementwise ops after them (9.6 ms forward) and their autograd
// (most of a 69 ms backward), because every one materialises a full-resolution
// temporary. Here they stay in registers.
//
// Per pixel, per colour channel, with all quantities already blurred:
//   sxx = exx - mx^2,  syy = eyy - my^2,  sxy = exy - mx*my
//   A = 2 mx my + C1,  B = 2 sxy + C2,  C = mx^2 + my^2 + C1,  D = sxx + syy + C2
//   S = (A B) / (C D)
//
// Input layout is the 15-channel blurred stack, contiguous (1,15,H,W):
//   channels 0-2 mx, 3-5 my, 6-8 exx, 9-11 eyy, 12-14 exy.

constant float SSIM_C1 = 0.01f * 0.01f;
constant float SSIM_C2 = 0.03f * 0.03f;

kernel void ssim_tail_forward(device const float* b   [[buffer(0)]],
                              device float*       out [[buffer(1)]],
                              constant uint2&     dim [[buffer(2)]],  // (npix, stride=H*W)
                              uint gid [[thread_position_in_grid]])
{
    const uint npix = dim.x, N = dim.y;
    if (gid >= 3u * npix) return;
    const uint c = gid / npix;          // colour channel 0..2
    const uint i = gid - c * npix;      // pixel index
    const uint o = c * N + i;

    const float MX  = b[(0u + c) * N + i];
    const float MY  = b[(3u + c) * N + i];
    const float EXX = b[(6u + c) * N + i];
    const float EYY = b[(9u + c) * N + i];
    const float EXY = b[(12u + c) * N + i];

    const float sxx = EXX - MX * MX;
    const float syy = EYY - MY * MY;
    const float sxy = EXY - MX * MY;

    const float A = 2.0f * MX * MY + SSIM_C1;
    const float B = 2.0f * sxy + SSIM_C2;
    const float C = MX * MX + MY * MY + SSIM_C1;
    const float D = sxx + syy + SSIM_C2;

    out[o] = (A * B) / (C * D);
}

kernel void ssim_tail_backward(device const float* b    [[buffer(0)]],
                               device const float* gout [[buffer(1)]],
                               device float*       gb   [[buffer(2)]],
                               constant uint2&     dim  [[buffer(3)]],
                               uint gid [[thread_position_in_grid]])
{
    const uint npix = dim.x, N = dim.y;
    if (gid >= 3u * npix) return;
    const uint c = gid / npix;
    const uint i = gid - c * npix;

    const float MX  = b[(0u + c) * N + i];
    const float MY  = b[(3u + c) * N + i];
    const float EXX = b[(6u + c) * N + i];
    const float EYY = b[(9u + c) * N + i];
    const float EXY = b[(12u + c) * N + i];

    const float sxx = EXX - MX * MX;
    const float syy = EYY - MY * MY;
    const float sxy = EXY - MX * MY;

    const float A = 2.0f * MX * MY + SSIM_C1;
    const float B = 2.0f * sxy + SSIM_C2;
    const float C = MX * MX + MY * MY + SSIM_C1;
    const float D = sxx + syy + SSIM_C2;

    const float invCD = 1.0f / (C * D);
    const float S = A * B * invCD;
    const float g = gout[c * N + i];

    // dS/dA = B/(CD), dS/dB = A/(CD), dS/dC = -S/C, dS/dD = -S/D
    const float dA = B * invCD;
    const float dB = A * invCD;
    const float dC = -S / C;
    const float dD = -S / D;

    // exx, eyy enter only through D (via sxx, syy); exy only through B (via sxy)
    gb[(6u  + c) * N + i] = g * dD;
    gb[(9u  + c) * N + i] = g * dD;
    gb[(12u + c) * N + i] = g * (2.0f * dB);

    // mx: A(+2my), B(-2my via sxy), C(+2mx), D(-2mx via sxx)
    gb[(0u + c) * N + i] = g * (2.0f * MY * (dA - dB) + 2.0f * MX * (dC - dD));
    // my: A(+2mx), B(-2mx via sxy), C(+2my), D(-2my via syy)
    gb[(3u + c) * N + i] = g * (2.0f * MX * (dA - dB) + 2.0f * MY * (dC - dD));
}

// ---------------------------------------------------------------------------
// Fused SSIM blur. Removes the 15-channel stack materialisation (86 MB at
// 900x1600) and torch's four convolution dispatches.
//
// Boundary handling matches F.conv2d with zero padding EXACTLY -- taps that
// fall outside the image contribute nothing. Getting that wrong shows up only
// in an 5px border, which is easy to miss in a mean-reduced loss.
//
// A symmetric gaussian is self-adjoint, so `ssim_blur15` serves three roles:
// the forward vertical pass, and both adjoint passes in the backward. Only the
// stack construction needs its own forward and chain kernels.
//
// Channel layout throughout is the tail's: 0-2 x, 3-5 y, 6-8 xx, 9-11 yy,
// 12-14 xy (pre-blur), becoming mx/my/exx/eyy/exy after both blurs.

constant int SSIM_R = 5;   // 11-tap radius

// x,y -> the 5 stacked quantities, horizontally blurred. Forward only.
kernel void ssim_stack_blur_h(device const float* x    [[buffer(0)]],
                              device const float* y    [[buffer(1)]],
                              device const float* wgt  [[buffer(2)]],
                              device float*       out  [[buffer(3)]],
                              constant uint2&     hw   [[buffer(4)]],
                              uint3 gid [[thread_position_in_grid]])
{
    const uint W = hw.x, H = hw.y;
    if (gid.x >= W || gid.y >= H || gid.z >= 3u) return;
    const uint c = gid.z, N = W * H;
    const uint row = gid.y * W;

    // x and y are read in (H,W,3) -- the layout render() and the ground truth
    // already have. The torch path had to permute to (1,3,H,W) and make it
    // contiguous first, which is two 17 MB copies per step at 900x1600.
    float sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (int k = -SSIM_R; k <= SSIM_R; ++k) {
        const int u = int(gid.x) + k;
        if (u < 0 || u >= int(W)) continue;      // zero padding, as conv2d
        const float w = wgt[k + SSIM_R];
        const float xv = x[(row + uint(u)) * 3u + c];
        const float yv = y[(row + uint(u)) * 3u + c];
        sx += w * xv;  sy += w * yv;
        sxx += w * xv * xv;  syy += w * yv * yv;  sxy += w * xv * yv;
    }
    const uint o = row + gid.x;
    out[(0u  + c) * N + o] = sx;
    out[(3u  + c) * N + o] = sy;
    out[(6u  + c) * N + o] = sxx;
    out[(9u  + c) * N + o] = syy;
    out[(12u + c) * N + o] = sxy;
}

// Generic 15-channel separable blur. dir=0 horizontal, dir=1 vertical.
// Self-adjoint, so the backward reuses it unchanged.
kernel void ssim_blur15(device const float* src  [[buffer(0)]],
                        device const float* wgt  [[buffer(1)]],
                        device float*       dst  [[buffer(2)]],
                        constant uint4&     hwd  [[buffer(3)]],  // W,H,dir,unused
                        uint3 gid [[thread_position_in_grid]])
{
    const uint W = hwd.x, H = hwd.y, dir = hwd.z;
    if (gid.x >= W || gid.y >= H || gid.z >= 15u) return;
    const uint N = W * H, base = gid.z * N;

    float s = 0;
    for (int k = -SSIM_R; k <= SSIM_R; ++k) {
        int u = int(gid.x), v = int(gid.y);
        if (dir == 0u) u += k; else v += k;
        if (u < 0 || u >= int(W) || v < 0 || v >= int(H)) continue;
        s += wgt[k + SSIM_R] * src[base + uint(v) * W + uint(u)];
    }
    dst[base + gid.y * W + gid.x] = s;
}

// Adjoint of the stack construction: d_stack (15ch) -> dL/dx (3ch).
//   x contributes through channel 0 (as x), 6 (as x*x), 12 (as x*y)
kernel void ssim_chain_backward(device const float* x       [[buffer(0)]],
                                device const float* y       [[buffer(1)]],
                                device const float* d_stack [[buffer(2)]],
                                device float*       d_x     [[buffer(3)]],
                                constant uint2&     hw      [[buffer(4)]],
                                uint3 gid [[thread_position_in_grid]])
{
    const uint W = hw.x, H = hw.y;
    if (gid.x >= W || gid.y >= H || gid.z >= 3u) return;
    const uint c = gid.z, N = W * H, o = gid.y * W + gid.x;
    const float xv = x[o * 3u + c], yv = y[o * 3u + c];   // (H,W,3), as above
    d_x[o * 3u + c] = d_stack[(0u + c) * N + o]
                    + 2.0f * xv * d_stack[(6u + c) * N + o]
                    + yv * d_stack[(12u + c) * N + o];
}
