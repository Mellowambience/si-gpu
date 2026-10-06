#include "common.hpp"
int main(int argc,char** argv) {
    try {
        if(argc!=12) throw std::runtime_error("Usage: baseline width height scale learned input.bin weights.bin output.bin metrics.json shader.hlsl samples warmups");
        int width=std::stoi(argv[1]),height=std::stoi(argv[2]),scale=std::stoi(argv[3]);
        int learned=std::stoi(argv[4]),samples=std::stoi(argv[10]),warmups=std::stoi(argv[11]);
        if(width<2 || height<2 || width>4096 || height>4096 || scale<2 || scale>4
           || samples<1 || samples>10000 || warmups<0 || warmups>1000 || (learned!=0 && learned!=1))
            throw std::runtime_error("Invalid dimensions or settings");
        if(static_cast<UINT64>(width)*height*scale*scale>16777216) throw std::runtime_error("Output exceeds 16M pixel lab limit");
        auto input=readFloats(argv[5],static_cast<size_t>(width)*height*4);
        auto weights=readFloats(argv[6],10*scale*scale);
        for(size_t i=0;i<input.size();i++) if(input[i]<0 || input[i]>1) throw std::runtime_error("Input outside [0,1]");
        ComPtr<IDXGIFactory4> factory;
        check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"DXGI factory");
        ComPtr<IDXGIAdapter1> adapter,best;
        DXGI_ADAPTER_DESC1 bestDesc={};
        for(UINT i=0;factory->EnumAdapters1(i,&adapter)!=DXGI_ERROR_NOT_FOUND;i++) {
            DXGI_ADAPTER_DESC1 desc={}; check(adapter->GetDesc1(&desc),"Adapter description");
            if(!(desc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE) &&
               SUCCEEDED(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_11_0,__uuidof(ID3D12Device),nullptr)) &&
               (!best || desc.DedicatedVideoMemory>bestDesc.DedicatedVideoMemory)) { best=adapter; bestDesc=desc; }
            adapter.Reset();
        }
        if(!best) throw std::runtime_error("No hardware D3D12 adapter; CPU/WARP fallback disabled");
        int nameSize=WideCharToMultiByte(CP_UTF8,0,bestDesc.Description,-1,nullptr,0,nullptr,nullptr);
        if(!nameSize) throw std::runtime_error("Adapter name conversion failed");
        std::vector<char> nameBytes(nameSize);
        WideCharToMultiByte(CP_UTF8,0,bestDesc.Description,-1,nameBytes.data(),nameSize,nullptr,nullptr);
        std::string name(nameBytes.data());
        ComPtr<ID3D12Device> device;
        check(D3D12CreateDevice(best.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&device)),"D3D12 device");
        D3D12_COMMAND_QUEUE_DESC queueDesc={}; queueDesc.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
        ComPtr<ID3D12CommandQueue> queue;
        check(device->CreateCommandQueue(&queueDesc,IID_PPV_ARGS(&queue)),"Command queue");
        ComPtr<ID3D12CommandAllocator> allocator;
        check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)),"Command allocator");
        ComPtr<ID3D12GraphicsCommandList> list;
        check(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)),"Command list");
        ComPtr<ID3D12Fence> fence; check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)),"Fence");
        Event event; UINT64 fenceValue=0;
        auto submit=[&]() {
            check(list->Close(),"Close list");
            ID3D12CommandList* lists[]={list.Get()}; queue->ExecuteCommandLists(1,lists);
            check(queue->Signal(fence.Get(),++fenceValue),"Signal fence");
            check(fence->SetEventOnCompletion(fenceValue,event.handle),"Fence event");
            if(WaitForSingleObject(event.handle,60000)!=WAIT_OBJECT_0) throw std::runtime_error("GPU wait timeout");
        };
        D3D12_ROOT_PARAMETER params[4]={};
        params[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants.Num32BitValues=4; params[0].Constants.ShaderRegister=0;
        params[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; params[1].Descriptor.ShaderRegister=0;
        params[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; params[2].Descriptor.ShaderRegister=1;
        params[3].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV; params[3].Descriptor.ShaderRegister=0;
        D3D12_ROOT_SIGNATURE_DESC rootDesc={}; rootDesc.NumParameters=4; rootDesc.pParameters=params;
        ComPtr<ID3DBlob> rootBlob,error;
        check(D3D12SerializeRootSignature(&rootDesc,D3D_ROOT_SIGNATURE_VERSION_1,&rootBlob,&error),"Serialize root");
        ComPtr<ID3D12RootSignature> root;
        check(device->CreateRootSignature(0,rootBlob->GetBufferPointer(),rootBlob->GetBufferSize(),IID_PPV_ARGS(&root)),"Root signature");
        std::string shaderPath(argv[9]); std::wstring shaderWide(shaderPath.begin(),shaderPath.end());
        ComPtr<ID3DBlob> shader;
        HRESULT shaderResult=D3DCompileFromFile(shaderWide.c_str(),nullptr,D3D_COMPILE_STANDARD_FILE_INCLUDE,"main","cs_5_1",
                                               D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_ENABLE_STRICTNESS,0,&shader,&error);
        if(FAILED(shaderResult) && error) std::cerr<<static_cast<const char*>(error->GetBufferPointer());
        check(shaderResult,"Compile shader");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipelineDesc={}; pipelineDesc.pRootSignature=root.Get();
        pipelineDesc.CS={shader->GetBufferPointer(),shader->GetBufferSize()};
        ComPtr<ID3D12PipelineState> pipeline;
        check(device->CreateComputePipelineState(&pipelineDesc,IID_PPV_ARGS(&pipeline)),"Compute pipeline");
        UINT64 inputBytes=input.size()*sizeof(float),weightBytes=weights.size()*sizeof(float);
        UINT64 outputBytes=static_cast<UINT64>(width)*height*scale*scale*4*sizeof(float);
        auto upload=buffer(device.Get(),inputBytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        auto weightUpload=buffer(device.Get(),weightBytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        auto source=buffer(device.Get(),inputBytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST);
        auto weightGPU=buffer(device.Get(),weightBytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST);
        auto target=buffer(device.Get(),outputBytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,true);
        auto readback=buffer(device.Get(),outputBytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        fill(upload.Get(),input); fill(weightUpload.Get(),weights);
        list->CopyBufferRegion(source.Get(),0,upload.Get(),0,inputBytes);
        list->CopyBufferRegion(weightGPU.Get(),0,weightUpload.Get(),0,weightBytes);
        transition(list.Get(),source.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(list.Get(),weightGPU.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        submit();
        upload.Reset(); weightUpload.Reset();
        check(allocator->Reset(),"Reset allocator"); check(list->Reset(allocator.Get(),pipeline.Get()),"Reset list");
        list->SetComputeRootSignature(root.Get());
        UINT settings[]={static_cast<UINT>(width),static_cast<UINT>(height),static_cast<UINT>(scale),static_cast<UINT>(learned)};
        list->SetComputeRoot32BitConstants(0,4,settings,0);
        list->SetComputeRootShaderResourceView(1,source->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(2,weightGPU->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(3,target->GetGPUVirtualAddress());
        D3D12_QUERY_HEAP_DESC queryDesc={}; queryDesc.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP; queryDesc.Count=2*samples;
        ComPtr<ID3D12QueryHeap> query;
        check(device->CreateQueryHeap(&queryDesc,IID_PPV_ARGS(&query)),"Timestamp query heap");
        auto ticks=buffer(device.Get(),static_cast<UINT64>(2)*samples*sizeof(UINT64),D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_RESOURCE_BARRIER barrier={}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; barrier.UAV.pResource=target.Get();
        for(int i=0;i<warmups+samples;i++) {
            if(i>=warmups) list->EndQuery(query.Get(),D3D12_QUERY_TYPE_TIMESTAMP,2*(i-warmups));
            list->Dispatch((width*scale+7)/8,(height*scale+7)/8,1);
            list->ResourceBarrier(1,&barrier);
            if(i>=warmups) list->EndQuery(query.Get(),D3D12_QUERY_TYPE_TIMESTAMP,2*(i-warmups)+1);
        }
        list->ResolveQueryData(query.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,2*samples,ticks.Get(),0);
        transition(list.Get(),target.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(readback.Get(),0,target.Get(),0,outputBytes);
        auto started=std::chrono::steady_clock::now(); submit();
        double batchMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        UINT64 frequency=0; check(queue->GetTimestampFrequency(&frequency),"Timestamp frequency");
        UINT64* timings=nullptr; D3D12_RANGE tickRange={0,static_cast<SIZE_T>(2)*samples*sizeof(UINT64)};
        check(ticks->Map(0,&tickRange,reinterpret_cast<void**>(&timings)),"Map timings");
        std::vector<double> times;
        for(int i=0;i<samples;i++) times.push_back((timings[2*i+1]-timings[2*i])*1000.0/frequency);
        ticks->Unmap(0,nullptr);
        auto unsorted=times; std::sort(times.begin(),times.end());
        void* pixels=nullptr; D3D12_RANGE imageRange={0,static_cast<SIZE_T>(outputBytes)};
        check(readback->Map(0,&imageRange,&pixels),"Map output");
        std::ofstream output(argv[7],std::ios::binary);
        output.write(static_cast<const char*>(pixels),outputBytes);
        if(!output) throw std::runtime_error("Output write failed");
        readback->Unmap(0,nullptr);
        std::ofstream metrics(argv[8]);
        metrics<<"{\n  \"adapter\": "<<std::quoted(name)<<",\n  \"api\": \"D3D12 hardware compute\",\n"
               <<"  \"mode\": \""<<(learned?"learned_linear":"bilinear")<<"\",\n"
               <<"  \"input_size\": ["<<width<<","<<height<<"],\n  \"output_size\": ["<<width*scale<<","<<height*scale<<"],\n"
               <<"  \"warmups\": "<<warmups<<",\n  \"samples\": "<<samples<<",\n  \"gpu_p50_ms\": "<<percentile(times,.5)<<",\n"
               <<"  \"gpu_p95_ms\": "<<percentile(times,.95)<<",\n  \"gpu_p99_ms\": "<<percentile(times,.99)<<",\n"
               <<"  \"gpu_buffer_payload_mib\": "<<(inputBytes+weightBytes+outputBytes)/1048576.0<<",\n"
               <<"  \"batch_submit_wait_ms\": "<<batchMs<<",\n  \"gpu_samples_ms\": [";
        for(size_t i=0;i<unsorted.size();i++) { if(i) metrics<<","; metrics<<unsorted[i]; }
        metrics<<"],\n  \"scope\": \"Repeated fixed GPU-resident input; spatial dispatch plus UAV barrier only. Upload, readback, compilation, engine render and temporal passes excluded. Buffer payload is not peak VRAM.\"\n}\n";
        if(!metrics) throw std::runtime_error("Metrics write failed");
        std::cout<<name<<" "<<(learned?"learned":"bilinear")<<" "<<width*scale<<"x"<<height*scale
                 <<" GPU p50="<<percentile(times,.5)<<" ms p95="<<percentile(times,.95)<<" ms\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
