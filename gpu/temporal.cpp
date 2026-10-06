#include "context.hpp"

struct Header { uint32_t magic,version,width,height,scale,frames; };
struct TemporalSettings { UINT width,height,scale,useHistory; float weight,threshold; };
static_assert(sizeof(Header)==24 && sizeof(TemporalSettings)==24,"Binary protocol layout");

int main(int argc,char** argv) {
    try {
        if(argc!=12) throw std::runtime_error("Usage: temporal sequence.bin weights.bin spatial.hlsl temporal.hlsl output.bin metrics.json repeats warmups capture_all history_weight depth_threshold");
        int repeats=std::stoi(argv[7]),warmups=std::stoi(argv[8]),capture=std::stoi(argv[9]);
        float weight=std::stof(argv[10]),threshold=std::stof(argv[11]);
        if(repeats<1 || repeats>10000 || warmups<0 || warmups>1000 || (capture!=0 && capture!=1)
           || !std::isfinite(weight) || weight<0 || weight>1 || !std::isfinite(threshold) || threshold<0 || threshold>1)
            throw std::runtime_error("Invalid temporal settings");
        std::ifstream sequence(argv[1],std::ios::binary|std::ios::ate);
        if(!sequence) throw std::runtime_error("Cannot open sequence");
        auto sequenceBytes=sequence.tellg(); sequence.seekg(0);
        Header header={}; if(!sequence.read(reinterpret_cast<char*>(&header),sizeof(header))) throw std::runtime_error("Missing sequence header");
        if(header.magic!=0x53494750 || header.version!=1 || header.width<2 || header.height<2
           || header.width>4096 || header.height>4096 || header.scale<2 || header.scale>4 || header.frames<1 || header.frames>3000)
            throw std::runtime_error("Invalid sequence header");
        UINT64 lowPixels=static_cast<UINT64>(header.width)*header.height;
        UINT64 outputPixels=lowPixels*header.scale*header.scale;
        if(outputPixels>16777216 || static_cast<UINT64>(header.frames)*repeats>100000)
            throw std::runtime_error("Sequence exceeds lab limits");
        UINT64 lowBytes=lowPixels*4*sizeof(float),outputBytes=outputPixels*4*sizeof(float);
        UINT64 expectedBytes=sizeof(Header)+header.frames*(4+2*lowBytes);
        if(sequenceBytes!=static_cast<std::streamoff>(expectedBytes)) throw std::runtime_error("Sequence file size mismatch");
        auto weights=readFloats(argv[2],10*header.scale*header.scale);
        bool learned=std::any_of(weights.begin(),weights.end(),[](float value){return value!=0;});
        Context context;
        auto spatialRoot=makeRoot(context.device.Get(),4,2);
        auto temporalRoot=makeRoot(context.device.Get(),6,4);
        auto spatialPipeline=makePipeline(context.device.Get(),spatialRoot.Get(),argv[3]);
        auto temporalPipeline=makePipeline(context.device.Get(),temporalRoot.Get(),argv[4]);
        auto source=buffer(context.device.Get(),lowBytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        ComPtr<ID3D12Resource> metadata[2],history[2];
        for(int i=0;i<2;i++) {
            metadata[i]=buffer(context.device.Get(),lowBytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            history[i]=buffer(context.device.Get(),outputBytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,true);
        }
        auto spatial=buffer(context.device.Get(),outputBytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,true);
        auto weightGPU=buffer(context.device.Get(),weights.size()*sizeof(float),D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST);
        auto weightUpload=buffer(context.device.Get(),weights.size()*sizeof(float),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        auto colorUpload=buffer(context.device.Get(),lowBytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        auto metaUpload=buffer(context.device.Get(),lowBytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        auto readback=buffer(context.device.Get(),outputBytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        fill(weightUpload.Get(),weights); context.begin();
        context.list->CopyBufferRegion(weightGPU.Get(),0,weightUpload.Get(),0,weights.size()*sizeof(float));
        transition(context.list.Get(),weightGPU.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        context.submit(); weightUpload.Reset();
        D3D12_QUERY_HEAP_DESC queryDesc={}; queryDesc.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP; queryDesc.Count=3*repeats;
        ComPtr<ID3D12QueryHeap> query;
        check(context.device->CreateQueryHeap(&queryDesc,IID_PPV_ARGS(&query)),"Query heap");
        auto ticks=buffer(context.device.Get(),static_cast<UINT64>(3)*repeats*sizeof(UINT64),D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        std::vector<float> colors(static_cast<size_t>(lowPixels)*4),meta(colors.size());
        std::vector<double> spatialTimes,temporalTimes,totalTimes,frameTotal;
        std::vector<uint32_t> resets;
        std::ofstream output(argv[5],std::ios::binary);
        if(!output) throw std::runtime_error("Cannot create output");
        auto sequenceStarted=std::chrono::steady_clock::now();
        for(UINT frame=0;frame<header.frames;frame++) {
            uint32_t reset=0;
            if(!sequence.read(reinterpret_cast<char*>(&reset),4)
               || !sequence.read(reinterpret_cast<char*>(colors.data()),lowBytes)
               || !sequence.read(reinterpret_cast<char*>(meta.data()),lowBytes)) throw std::runtime_error("Truncated frame");
            if(reset>1) throw std::runtime_error("Reset must be 0 or 1");
            for(size_t i=0;i<colors.size();i++) if(!std::isfinite(colors[i]) || colors[i]<0 || colors[i]>1)
                throw std::runtime_error("Color must be finite [0,1]");
            for(size_t i=0;i<meta.size();i+=4) {
                for(size_t k=0;k<4;k++) if(!std::isfinite(meta[i+k])) throw std::runtime_error("Nonfinite metadata");
                if(meta[i]<=0 || meta[i+3]<0 || meta[i+3]>1) throw std::runtime_error("Invalid depth/reactive");
            }
            resets.push_back(reset);
            int current=frame%2,previous=1-current;
            fill(colorUpload.Get(),colors); fill(metaUpload.Get(),meta);
            context.begin();
            transition(context.list.Get(),source.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
            transition(context.list.Get(),metadata[current].Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
            context.list->CopyBufferRegion(source.Get(),0,colorUpload.Get(),0,lowBytes);
            context.list->CopyBufferRegion(metadata[current].Get(),0,metaUpload.Get(),0,lowBytes);
            transition(context.list.Get(),source.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition(context.list.Get(),metadata[current].Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition(context.list.Get(),history[current].Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            TemporalSettings settings={header.width,header.height,header.scale,frame>0 && !reset?1u:0u,weight,threshold};
            UINT spatialSettings[]={header.width,header.height,header.scale,learned?1u:0u};
            for(int iteration=0;iteration<warmups+repeats;iteration++) {
                UINT start=static_cast<UINT>(3*(iteration-warmups));
                if(iteration>=warmups) context.list->EndQuery(query.Get(),D3D12_QUERY_TYPE_TIMESTAMP,start);
                context.list->SetPipelineState(spatialPipeline.Get());
                context.list->SetComputeRootSignature(spatialRoot.Get());
                context.list->SetComputeRoot32BitConstants(0,4,spatialSettings,0);
                context.list->SetComputeRootShaderResourceView(1,source->GetGPUVirtualAddress());
                context.list->SetComputeRootShaderResourceView(2,weightGPU->GetGPUVirtualAddress());
                context.list->SetComputeRootUnorderedAccessView(3,spatial->GetGPUVirtualAddress());
                context.list->Dispatch((header.width*header.scale+7)/8,(header.height*header.scale+7)/8,1);
                transition(context.list.Get(),spatial.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                if(iteration>=warmups) context.list->EndQuery(query.Get(),D3D12_QUERY_TYPE_TIMESTAMP,start+1);
                context.list->SetPipelineState(temporalPipeline.Get());
                context.list->SetComputeRootSignature(temporalRoot.Get());
                context.list->SetComputeRoot32BitConstants(0,6,&settings,0);
                context.list->SetComputeRootShaderResourceView(1,spatial->GetGPUVirtualAddress());
                context.list->SetComputeRootShaderResourceView(2,metadata[current]->GetGPUVirtualAddress());
                context.list->SetComputeRootShaderResourceView(3,history[previous]->GetGPUVirtualAddress());
                context.list->SetComputeRootShaderResourceView(4,metadata[previous]->GetGPUVirtualAddress());
                context.list->SetComputeRootUnorderedAccessView(5,history[current]->GetGPUVirtualAddress());
                context.list->Dispatch((header.width*header.scale+7)/8,(header.height*header.scale+7)/8,1);
                D3D12_RESOURCE_BARRIER barrier={}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; barrier.UAV.pResource=history[current].Get();
                context.list->ResourceBarrier(1,&barrier);
                transition(context.list.Get(),spatial.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                if(iteration>=warmups) context.list->EndQuery(query.Get(),D3D12_QUERY_TYPE_TIMESTAMP,start+2);
            }
            context.list->ResolveQueryData(query.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,3*repeats,ticks.Get(),0);
            transition(context.list.Get(),history[current].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            bool save=capture || frame+1==header.frames;
            if(save) {
                transition(context.list.Get(),history[current].Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
                context.list->CopyBufferRegion(readback.Get(),0,history[current].Get(),0,outputBytes);
                transition(context.list.Get(),history[current].Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            context.submit();
            UINT64* values=nullptr; D3D12_RANGE range={0,static_cast<SIZE_T>(3)*repeats*sizeof(UINT64)};
            check(ticks->Map(0,&range,reinterpret_cast<void**>(&values)),"Map timestamps");
            std::vector<double> currentTimes;
            for(int i=0;i<repeats;i++) {
                double a=(values[3*i+1]-values[3*i])*1000.0/context.frequency;
                double b=(values[3*i+2]-values[3*i+1])*1000.0/context.frequency;
                spatialTimes.push_back(a); temporalTimes.push_back(b); totalTimes.push_back(a+b); currentTimes.push_back(a+b);
            }
            ticks->Unmap(0,nullptr); std::sort(currentTimes.begin(),currentTimes.end()); frameTotal.push_back(percentile(currentTimes,.5));
            if(save) {
                void* pixels=nullptr; D3D12_RANGE imageRange={0,static_cast<SIZE_T>(outputBytes)};
                check(readback->Map(0,&imageRange,&pixels),"Map frame output");
                output.write(static_cast<const char*>(pixels),outputBytes); readback->Unmap(0,nullptr);
                if(!output) throw std::runtime_error("Output write failed");
            }
        }
        double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-sequenceStarted).count();
        std::ofstream report(argv[6]);
        report<<"{\n\"adapter\":"<<std::quoted(context.name)<<",\n\"debug_layer\":"<<(context.debugEnabled?"true":"false")
              <<",\n\"input_size\":["<<header.width<<","<<header.height<<"],\n\"output_size\":["<<header.width*header.scale<<","<<header.height*header.scale<<"]"
              <<",\n\"frames\":"<<header.frames<<",\n\"repeats_per_frame\":"<<repeats<<",\n\"warmups_per_frame\":"<<warmups
              <<",\n\"capture_all\":"<<(capture?"true":"false")<<",\n\"learned\":"<<(learned?"true":"false")
              <<",\n\"gpu_buffer_payload_mib\":"<<(3*lowBytes+3*outputBytes+weights.size()*sizeof(float))/1048576.0
              <<",\n\"offline_sequence_wall_ms\":"<<wall<<",\n\"timings\":{";
        auto emit=[&](const char* name,const std::vector<double>& times) {
            auto sorted=times; std::sort(sorted.begin(),sorted.end());
            report<<std::quoted(name)<<":{\"p50_ms\":"<<percentile(sorted,.5)<<",\"p95_ms\":"<<percentile(sorted,.95)<<",\"p99_ms\":"<<percentile(sorted,.99)<<",\"samples_ms\":[";
            for(size_t i=0;i<times.size();i++) { if(i) report<<","; report<<times[i]; } report<<"]}";
        };
        emit("spatial",spatialTimes); report<<","; emit("temporal",temporalTimes); report<<","; emit("combined",totalTimes);
        report<<"},\n\"frame_combined_p50_ms\":[";
        for(size_t i=0;i<frameTotal.size();i++) { if(i) report<<","; report<<frameTotal[i]; }
        report<<"],\n\"scope\":\"GPU spatial+temporal dispatches and intervening/restore barriers. Upload, readback, compilation, CPU fence waits and engine rendering excluded. Repeats hold prior history fixed; history advances once per input frame. Reset and steady history timings must be reported separately. Buffer payload is not peak VRAM.\"\n}";
        if(!report) throw std::runtime_error("Metrics write failed");
        std::cout<<context.name<<" temporal "<<header.width*header.scale<<"x"<<header.height*header.scale<<" "<<header.frames<<" frames, parity output written\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<"\n"; return 1; }
}
