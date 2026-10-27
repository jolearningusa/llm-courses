// L3 educational baseline. Contiguous float32 rows/weights, 256 threads/block.
// Source provided for a CUDA-host exercise; NOT compiled/GPU-validated on the authoring Mac.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_OK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); return 1; } } while(0)

__global__ void row_rmsnorm(const float* x,const float* w,float* y,int cols,float eps){
    __shared__ float partial[256];
    int row=blockIdx.x,t=threadIdx.x;
    float sum=0;
    for(int col=t;col<cols;col+=blockDim.x){float v=x[row*cols+col];sum+=v*v;}
    partial[t]=sum;
    __syncthreads();
    for(int stride=128;stride>0;stride/=2){
        if(t<stride)partial[t]+=partial[t+stride];
        __syncthreads();
    }
    float inv=rsqrtf(partial[0]/cols+eps);
    for(int col=t;col<cols;col+=blockDim.x)y[row*cols+col]=x[row*cols+col]*w[col]*inv;
}

int main(){
    int devices=0;
    cudaError_t status=cudaGetDeviceCount(&devices);
    if(status!=cudaSuccess||devices==0){std::fprintf(stderr,"SKIP: usable CUDA device required\n");return 77;}
    const int rows=3,cols=257;const float eps=1e-5f;
    std::vector<float> x(rows*cols),w(cols),out(rows*cols),reference(rows*cols);
    for(int col=0;col<cols;++col)w[col]=1.f+float(col)/cols;
    for(int row=0;row<rows;++row){
        double sum=0;
        for(int col=0;col<cols;++col){float v=std::sin((row*cols+col)*.13f);x[row*cols+col]=v;sum+=double(v)*v;}
        double inv=1/std::sqrt(sum/cols+eps);
        for(int col=0;col<cols;++col)reference[row*cols+col]=float(x[row*cols+col]*w[col]*inv);
    }
    float *dx=nullptr,*dw=nullptr,*dy=nullptr;
    CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&dx),x.size()*sizeof(float)));
    CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&dw),w.size()*sizeof(float)));
    CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&dy),out.size()*sizeof(float)));
    CUDA_OK(cudaMemcpy(dx,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dw,w.data(),w.size()*sizeof(float),cudaMemcpyHostToDevice));
    row_rmsnorm<<<rows,256>>>(dx,dw,dy,cols,eps);
    CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(out.data(),dy,out.size()*sizeof(float),cudaMemcpyDeviceToHost));
    float max_error=0;
    for(size_t i=0;i<out.size();++i){
        float error=std::fabs(out[i]-reference[i]);max_error=std::fmax(max_error,error);
        if(!std::isfinite(out[i])||error>1e-4f+1e-4f*std::fabs(reference[i])){
            std::fprintf(stderr,"Mismatch at %zu\n",i);return 1;
        }
    }
    CUDA_OK(cudaFree(dx));CUDA_OK(cudaFree(dw));CUDA_OK(cudaFree(dy));
    std::printf("Correctness passed; max absolute error=%g. No timing reported.\n",max_error);
    return 0;
}
