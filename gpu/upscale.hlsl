// Original SI-GPU reference compute pass. GPU-resident float32 RGB(A).
cbuffer Settings : register(b0) { uint width; uint height; uint scale; uint learned; };
StructuredBuffer<float4> source : register(t0);
StructuredBuffer<float> weights : register(t1);
RWStructuredBuffer<float4> target : register(u0);

float3 pixel(int2 p) {
    p = clamp(p, int2(0,0), int2(width-1,height-1));
    return source[p.y*width+p.x].rgb;
}

[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint ow=width*scale, oh=height*scale;
    if (id.x>=ow || id.y>=oh) return;
    float2 pos=(float2(id.xy)+0.5)/scale-0.5;
    int2 lo=int2(floor(pos));
    float2 f=frac(pos);
    float3 value=lerp(lerp(pixel(lo),pixel(lo+int2(1,0)),f.x),
                      lerp(pixel(lo+int2(0,1)),pixel(lo+int2(1,1)),f.x),f.y);
    if (learned) {
        uint phase=(id.y%scale)*scale+id.x%scale;
        uint phases=scale*scale;
        int2 base=int2(id.xy/scale);
        float3 residual=weights[9*phases+phase];
        [unroll] for (int y=0;y<3;y++) {
            [unroll] for (int x=0;x<3;x++) {
                residual+=pixel(base+int2(x-1,y-1))*weights[(y*3+x)*phases+phase];
            }
        }
        value+=residual;
    }
    target[id.y*ow+id.x]=float4(saturate(value),1);
}
