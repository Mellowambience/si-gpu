// CPU-reference parity path: unjittered display RGB and comparable linear depth.
cbuffer Settings : register(b0) {
    uint width; uint height; uint scale; uint useHistory;
    float historyWeight; float depthThreshold;
};
StructuredBuffer<float4> current : register(t0);
StructuredBuffer<float4> metadata : register(t1); // depth, motion x/y, reactive
StructuredBuffer<float4> history : register(t2);
StructuredBuffer<float4> previousMetadata : register(t3);
RWStructuredBuffer<float4> target : register(u0);

float3 currentPixel(int2 p) {
    uint ow=width*scale, oh=height*scale;
    p=clamp(p,int2(0,0),int2(ow-1,oh-1));
    return current[p.y*ow+p.x].rgb;
}
float4 metaPixel(int2 p) {
    p=clamp(p,int2(0,0),int2(width-1,height-1));
    return metadata[p.y*width+p.x];
}
float3 historyPixel(int2 p) {
    uint ow=width*scale, oh=height*scale;
    p=clamp(p,int2(0,0),int2(ow-1,oh-1));
    return history[p.y*ow+p.x].rgb;
}

[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint ow=width*scale, oh=height*scale;
    if(id.x>=ow || id.y>=oh) return;
    float3 value=currentPixel(int2(id.xy));
    if(!useHistory) { target[id.y*ow+id.x]=float4(value,1); return; }
    float4 meta=metaPixel(int2(id.xy/scale));
    float2 displacement=clamp(meta.yz,-float(max(width,height)*2),float(max(width,height)*2))*scale;
    float2 pos=float2(id.xy)+displacement;
    bool valid=all(pos>=0) && pos.x<=ow-1 && pos.y<=oh-1;
    pos=clamp(pos,float2(0,0),float2(ow-1,oh-1));
    int2 nearest=int2(floor(pos+0.5));
    nearest=clamp(nearest,int2(0,0),int2(ow-1,oh-1));
    int2 lowNearest=nearest/int(scale);
    float oldDepth=previousMetadata[lowNearest.y*width+lowNearest.x].x;
    valid=valid && abs(oldDepth-meta.x)<=depthThreshold*max(meta.x,1e-6);
    int2 lo=int2(floor(pos));
    float2 f=frac(pos);
    float3 old=lerp(lerp(historyPixel(lo),historyPixel(lo+int2(1,0)),f.x),
                    lerp(historyPixel(lo+int2(0,1)),historyPixel(lo+int2(1,1)),f.x),f.y);
    float3 lower=value,upper=value;
    [unroll] for(int y=-1;y<=1;y++) {
        [unroll] for(int x=-1;x<=1;x++) {
            float3 neighbor=currentPixel(int2(id.xy)+int2(x,y));
            lower=min(lower,neighbor); upper=max(upper,neighbor);
        }
    }
    old=clamp(old,lower,upper);
    // Match CPU bilinear reactive-mask enlargement; depth and motion stay nearest.
    float2 lowPos=(float2(id.xy)+0.5)/scale-0.5;
    int2 base=int2(floor(lowPos)); float2 phase=frac(lowPos);
    float reactive=lerp(lerp(metaPixel(base).w,metaPixel(base+int2(1,0)).w,phase.x),
                        lerp(metaPixel(base+int2(0,1)).w,metaPixel(base+int2(1,1)).w,phase.x),phase.y);
    float alpha=historyWeight*(valid?1:0)*(1-reactive);
    target[id.y*ow+id.x]=float4(value*(1-alpha)+old*alpha,1);
}
