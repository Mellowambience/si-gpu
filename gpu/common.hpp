#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>
using Microsoft::WRL::ComPtr;

void check(HRESULT result, const char* action) {
    if (FAILED(result)) throw std::runtime_error(std::string(action)+" HRESULT="+std::to_string(result));
}
struct Event {
    HANDLE handle=CreateEvent(nullptr,FALSE,FALSE,nullptr);
    Event() { if (!handle) throw std::runtime_error("CreateEvent failed"); }
    ~Event() { CloseHandle(handle); }
};
std::vector<float> readFloats(const char* path, size_t count) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if (!file || file.tellg()!=static_cast<std::streamoff>(count*sizeof(float)))
        throw std::runtime_error(std::string("Wrong binary size: ")+path);
    file.seekg(0);
    std::vector<float> data(count);
    if (!file.read(reinterpret_cast<char*>(data.data()),count*sizeof(float))) throw std::runtime_error("Read failed");
    for(float value:data) if(!std::isfinite(value)) throw std::runtime_error("Nonfinite input");
    return data;
}
ComPtr<ID3D12Resource> buffer(ID3D12Device* device, UINT64 size, D3D12_HEAP_TYPE type,
                              D3D12_RESOURCE_STATES state, bool uav=false) {
    D3D12_HEAP_PROPERTIES heap={}; heap.Type=type;
    D3D12_RESOURCE_DESC desc={};
    desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width=size;
    desc.Height=1; desc.DepthOrArraySize=1; desc.MipLevels=1;
    desc.SampleDesc.Count=1; desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (uav) desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> resource;
    check(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,state,nullptr,IID_PPV_ARGS(&resource)),"Create buffer");
    return resource;
}
void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier={}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource=resource; barrier.Transition.StateBefore=before;
    barrier.Transition.StateAfter=after; barrier.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list->ResourceBarrier(1,&barrier);
}
void fill(ID3D12Resource* resource,const std::vector<float>& values) {
    void* pointer=nullptr; D3D12_RANGE empty={0,0};
    check(resource->Map(0,&empty,&pointer),"Map upload");
    std::memcpy(pointer,values.data(),values.size()*sizeof(float)); resource->Unmap(0,nullptr);
}
double percentile(const std::vector<double>& sorted,double fraction) {
    double index=(sorted.size()-1)*fraction; size_t lower=static_cast<size_t>(index);
    size_t upper=std::min(lower+1,sorted.size()-1);
    return sorted[lower]+(sorted[upper]-sorted[lower])*(index-lower);
}
