#pragma once
#include "common.hpp"
#include <d3d12sdklayers.h>

struct Context {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12InfoQueue> info;
    Event event;
    UINT64 fenceValue=0,frequency=0;
    std::string name;
    bool debugEnabled=false;
    Context() {
        char debug[8]={};
        if(GetEnvironmentVariableA("SI_GPU_DEBUG",debug,sizeof(debug)) && debug[0]=='1') {
            ComPtr<ID3D12Debug> layer;
            check(D3D12GetDebugInterface(IID_PPV_ARGS(&layer)),"Debug layer unavailable (install Windows Graphics Tools)");
            layer->EnableDebugLayer(); debugEnabled=true;
        }
        ComPtr<IDXGIFactory4> factory;
        check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"DXGI factory");
        ComPtr<IDXGIAdapter1> adapter,best; DXGI_ADAPTER_DESC1 bestDesc={};
        for(UINT i=0;factory->EnumAdapters1(i,&adapter)!=DXGI_ERROR_NOT_FOUND;i++) {
            DXGI_ADAPTER_DESC1 desc={}; check(adapter->GetDesc1(&desc),"Adapter description");
            if(!(desc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE) &&
               SUCCEEDED(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_11_0,__uuidof(ID3D12Device),nullptr)) &&
               (!best || desc.DedicatedVideoMemory>bestDesc.DedicatedVideoMemory)) { best=adapter; bestDesc=desc; }
            adapter.Reset();
        }
        if(!best) throw std::runtime_error("No hardware D3D12 adapter; no CPU fallback");
        int count=WideCharToMultiByte(CP_UTF8,0,bestDesc.Description,-1,nullptr,0,nullptr,nullptr);
        if(!count) throw std::runtime_error("Adapter name conversion failed");
        std::vector<char> text(count);
        WideCharToMultiByte(CP_UTF8,0,bestDesc.Description,-1,text.data(),count,nullptr,nullptr);
        name=text.data();
        check(D3D12CreateDevice(best.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&device)),"D3D12 device");
        if(debugEnabled) check(device.As(&info),"Debug info queue");
        D3D12_COMMAND_QUEUE_DESC desc={}; desc.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
        check(device->CreateCommandQueue(&desc,IID_PPV_ARGS(&queue)),"Queue");
        check(queue->GetTimestampFrequency(&frequency),"Timestamp frequency");
        check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)),"Allocator");
        check(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)),"List");
        check(list->Close(),"Initial list close");
        check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)),"Fence");
    }
    void begin() {
        check(allocator->Reset(),"Reset allocator");
        check(list->Reset(allocator.Get(),nullptr),"Reset list");
    }
    void submit() {
        check(list->Close(),"Close list");
        ID3D12CommandList* lists[]={list.Get()}; queue->ExecuteCommandLists(1,lists);
        check(queue->Signal(fence.Get(),++fenceValue),"Signal");
        check(fence->SetEventOnCompletion(fenceValue,event.handle),"Fence event");
        if(WaitForSingleObject(event.handle,60000)!=WAIT_OBJECT_0) throw std::runtime_error("GPU wait timeout");
        if(info) {
            for(UINT64 i=0;i<info->GetNumStoredMessages();i++) {
                SIZE_T bytes=0; check(info->GetMessage(i,nullptr,&bytes),"Debug message size");
                std::vector<char> storage(bytes);
                auto message=reinterpret_cast<D3D12_MESSAGE*>(storage.data());
                check(info->GetMessage(i,message,&bytes),"Debug message");
                if(message->Severity<=D3D12_MESSAGE_SEVERITY_ERROR) throw std::runtime_error(message->pDescription);
            }
            info->ClearStoredMessages();
        }
    }
};

ComPtr<ID3D12RootSignature> makeRoot(ID3D12Device* device,UINT constants,UINT srvs) {
    std::vector<D3D12_ROOT_PARAMETER> params(srvs+2);
    params[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues=constants;
    for(UINT i=0;i<srvs;i++) { params[i+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; params[i+1].Descriptor.ShaderRegister=i; }
    params.back().ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
    D3D12_ROOT_SIGNATURE_DESC desc={}; desc.NumParameters=static_cast<UINT>(params.size()); desc.pParameters=params.data();
    ComPtr<ID3DBlob> blob,error;
    check(D3D12SerializeRootSignature(&desc,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&error),"Serialize root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root)),"Create root");
    return root;
}
ComPtr<ID3D12PipelineState> makePipeline(ID3D12Device* device,ID3D12RootSignature* root,const std::string& path) {
    std::wstring wide(path.begin(),path.end()); ComPtr<ID3DBlob> shader,error;
    HRESULT result=D3DCompileFromFile(wide.c_str(),nullptr,D3D_COMPILE_STANDARD_FILE_INCLUDE,"main","cs_5_1",
        D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_ENABLE_STRICTNESS,0,&shader,&error);
    if(FAILED(result) && error) std::cerr<<static_cast<const char*>(error->GetBufferPointer());
    check(result,"Compile shader");
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc={}; desc.pRootSignature=root; desc.CS={shader->GetBufferPointer(),shader->GetBufferSize()};
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateComputePipelineState(&desc,IID_PPV_ARGS(&pipeline)),"Create pipeline");
    return pipeline;
}
