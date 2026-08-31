#include <metal_stdlib>
using namespace metal;

// Tile binning in Metal: bounds, the exact ellipse-vs-tile test, and the
// packed sort key, in two passes over the gaussians.
//
// The torch version costs ~15 ms at 600k/900x1600 before any filtering, spread
// over nonzero / four floors / repeat_interleave / arange / modulo / gathers.
// Adding the exact ellipse test there cost another 14.6 ms and only saved 7.6
// ms of rasterisation, so the test -- which removes 37.5% of pairs and is
// bit-exact -- was a net loss. Here it is a few extra ALU ops on data already
// in registers.
//
// Two passes because the surviving pair count per gaussian is not known until
// the test runs: pass 1 counts, the host scans, pass 2 writes at the scanned
// offsets. Both passes run the identical loop so their counts cannot disagree.

struct BinParams {
    uint N, W, H, tile, tiles_x, tiles_y, pad0, pad1;
};

// Minimum of the conic's quadratic form over an axis-aligned tile rectangle.
// Zero if the centre is inside; otherwise attained on an edge, where it is a
// 1-D quadratic with a closed-form minimiser.
inline float min_q_over_rect(float2 mu, float3 c, float x0, float x1,
                             float y0, float y1)
{
    if (mu.x >= x0 && mu.x <= x1 && mu.y >= y0 && mu.y <= y1) return 0.0f;
    const float a = max(c.x, 1e-12f), b = c.y, cc = max(c.z, 1e-12f);
    float best = INFINITY;
    for (int e = 0; e < 2; ++e) {                  // horizontal edges
        const float yc = (e == 0) ? y0 : y1;
        const float xs = clamp(mu.x - b * (yc - mu.y) / a, x0, x1);
        const float ex = xs - mu.x, ey = yc - mu.y;
        best = min(best, a * ex * ex + 2.0f * b * ex * ey + cc * ey * ey);
    }
    for (int e = 0; e < 2; ++e) {                  // vertical edges
        const float xc = (e == 0) ? x0 : x1;
        const float ys = clamp(mu.y - b * (xc - mu.x) / cc, y0, y1);
        const float ex = xc - mu.x, ey = ys - mu.y;
        best = min(best, a * ex * ex + 2.0f * b * ex * ey + cc * ey * ey);
    }
    return best;
}

// Shared tile walk. When `keys` is null this only counts; otherwise it writes.
inline uint bin_walk(uint g,
                     device const float2* uv, device const float2* rxy,
                     device const packed_float3* conic, device const float* opacity,
                     device const float* depth,
                     constant BinParams& P,
                     device ulong* keys, device int* ids, uint write_at)
{
    const float2 mu = uv[g];
    const float2 r  = rxy[g];
    const float  tf = float(P.tile);

    int x0 = int(floor((mu.x - r.x) / tf)), x1 = int(floor((mu.x + r.x) / tf));
    int y0 = int(floor((mu.y - r.y) / tf)), y1 = int(floor((mu.y + r.y) / tf));
    x0 = clamp(x0, 0, int(P.tiles_x) - 1);  x1 = clamp(x1, 0, int(P.tiles_x) - 1);
    y0 = clamp(y0, 0, int(P.tiles_y) - 1);  y1 = clamp(y1, 0, int(P.tiles_y) - 1);

    const float3 c = float3(conic[g]);
    // alpha >= 1/255  <=>  Q <= 2 ln(255 * opacity). Below 1/255 opacity the
    // gaussian cannot reach the cutoff anywhere, so it claims no tiles.
    const float oa = 255.0f * opacity[g];
    if (oa <= 1.0f) return 0u;
    const float thr = 2.0f * log(oa);

    uint dbits = 0u;
    if (keys) dbits = as_type<uint>(depth[g]);

    uint n = 0u;
    for (int ty = y0; ty <= y1; ++ty) {
        const float ry0 = float(ty * int(P.tile));
        const float ry1 = min(float((ty + 1) * int(P.tile)), float(P.H));
        for (int tx = x0; tx <= x1; ++tx) {
            const float rx0 = float(tx * int(P.tile));
            const float rx1 = min(float((tx + 1) * int(P.tile)), float(P.W));
            if (min_q_over_rect(mu, c, rx0, rx1, ry0, ry1) >= thr) continue;
            if (keys) {
                const ulong tid = ulong(uint(ty) * P.tiles_x + uint(tx));
                keys[write_at + n] = (tid << 32) | ulong(dbits);
                ids[write_at + n]  = int(g);
            }
            ++n;
        }
    }
    return n;
}

kernel void bin_count(device const float2*        uv      [[buffer(0)]],
                      device const float2*        rxy     [[buffer(1)]],
                      device const packed_float3* conic   [[buffer(2)]],
                      device const float*         opacity [[buffer(3)]],
                      device const int*           valid   [[buffer(4)]],
                      device uint*                counts  [[buffer(5)]],
                      constant BinParams&         P       [[buffer(6)]],
                      uint g [[thread_position_in_grid]])
{
    if (g >= P.N) return;
    counts[g] = (valid[g] == 0) ? 0u
              : bin_walk(g, uv, rxy, conic, opacity, nullptr, P, nullptr, nullptr, 0u);
}

kernel void bin_write(device const float2*        uv      [[buffer(0)]],
                      device const float2*        rxy     [[buffer(1)]],
                      device const packed_float3* conic   [[buffer(2)]],
                      device const float*         opacity [[buffer(3)]],
                      device const int*           valid   [[buffer(4)]],
                      device const float*         depth   [[buffer(5)]],
                      device const uint*          offsets [[buffer(6)]],
                      device ulong*               keys    [[buffer(7)]],
                      device int*                 ids     [[buffer(8)]],
                      constant BinParams&         P       [[buffer(9)]],
                      uint g [[thread_position_in_grid]])
{
    if (g >= P.N || valid[g] == 0) return;
    bin_walk(g, uv, rxy, conic, opacity, depth, P, keys, ids, offsets[g]);
}
