

#include <vector>
#include "pocket-ai/engine/cu/common.hpp"

__global__ void addKernel(int *c, const int *a, const int *b, unsigned int size) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

class GraphBuilder {
public:
    void Init() {
        // Start of Graph Creation
        CUDA_CHECK(cudaGraphCreate(&graph_, 0));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }

    void AddMemcpyNode(void *dst, void *src, size_t num, size_t element_size, enum cudaMemcpyKind kind, std::vector<cudaGraphNode_t> &node_dependencies, cudaGraphNode_t &memcpy_node) {
        cudaMemcpy3DParms memcpy_params = { 0 };
        memset(&memcpy_params, 0, sizeof(memcpy_params));

        memcpy_params.srcArray = NULL;
        memcpy_params.srcPos = make_cudaPos(0, 0, 0);
        memcpy_params.srcPtr = make_cudaPitchedPtr(src, num * element_size, num, 1);
        memcpy_params.dstArray = NULL;
        memcpy_params.dstPos = make_cudaPos(0, 0, 0);
        memcpy_params.dstPtr = make_cudaPitchedPtr(dst, num * element_size, num, 1);
        memcpy_params.extent = make_cudaExtent(num * element_size, 1, 1);
        memcpy_params.kind = kind;
        CUDA_CHECK(cudaGraphAddMemcpyNode(&memcpy_node, graph_, node_dependencies.data(), node_dependencies.size(), &memcpy_params));
    }

    void AddKernelNode(void *kernel_func, dim3 &gridDim, dim3 &blockDim, unsigned int shared_mem_bytes, void **kernel_args, std::vector<cudaGraphNode_t> &node_dependencies, cudaGraphNode_t &kernel_node) {
        // Add a kernel node for launching a kernel on the GPU
        cudaKernelNodeParams kernel_node_params = { 0 };
        memset(&kernel_node_params, 0, sizeof(kernel_node_params));
        kernel_node_params.func = kernel_func;
        kernel_node_params.gridDim = gridDim;
        kernel_node_params.blockDim = blockDim;
        kernel_node_params.sharedMemBytes = shared_mem_bytes;
        kernel_node_params.kernelParams = kernel_args;
        kernel_node_params.extra = NULL;
        CUDA_CHECK(cudaGraphAddKernelNode(&kernel_node, graph_, node_dependencies.data(), node_dependencies.size(), &kernel_node_params));
    }

    void Instantiate() {
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
    }

    void GetGraphExec(cudaGraphExec_t *graph_exec, cudaStream_t *stream) {
        *graph_exec = graph_exec_;
        *stream = stream_;
    }

    void Uninit() {
        CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
        CUDA_CHECK(cudaGraphDestroy(graph_));
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }

private:
    cudaStream_t stream_;
    cudaGraph_t graph_;

    cudaGraphExec_t graph_exec_;
};

void AddWithCuda(int* c, const int* a, const int* b, unsigned int size, int loop_count, bool use_graph) {
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Choose which GPU to run on, change this on a multi-GPU system. Then allocate GPU memory.
    CUDA_CHECK(cudaMalloc((void**)&dev_c, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dev_a, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dev_b, size * sizeof(int)));

    if (!use_graph) {
        pai::cu::GpuTimer timer;
        timer.Start();
        for (int i = 0; i < loop_count; ++i) {
            // Copy input vectors from host memory to GPU buffers.
            CUDA_CHECK(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));
            // Launch a kernel on the GPU with one thread for each element.
            addKernel << <blocks, threads >> > (dev_c, dev_a, dev_b, size);
            // cudaDeviceSynchronize waits for the kernel to finish, and returns
            // any errors encountered during the launch.
            // NOTE: Below in the graph implementation this sync is included via graph dependencies 
            CUDA_CHECK(cudaDeviceSynchronize());
            // Copy output vector from GPU buffer to host memory.
            CUDA_CHECK(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));
        }
        timer.Stop();
        printf("Normal: Looped %d time(s) in %f microseconds\n", loop_count, timer.ElapsedMillis());
    }
    else {
        GraphBuilder graph_builder;
        graph_builder.Init();

        std::vector<cudaGraphNode_t> no_dependence;
        std::vector<cudaGraphNode_t> kernel_dependencies;
        std::vector<cudaGraphNode_t> output_dependencies;

        // Two memcpy node for Input.
        cudaGraphNode_t memcpy_node;
        graph_builder.AddMemcpyNode(dev_a, (void*)a, size, sizeof(float), cudaMemcpyHostToDevice, no_dependence, memcpy_node);
        kernel_dependencies.push_back(memcpy_node);

        graph_builder.AddMemcpyNode(dev_b, (void*)b, size, sizeof(float), cudaMemcpyHostToDevice, no_dependence, memcpy_node);
        kernel_dependencies.push_back(memcpy_node);

        // A kernel node depends on two memcpy nodes.
        cudaGraphNode_t kernel_node;
        void* kernel_args[4] = { (void*)&dev_c, (void*)&dev_a, (void*)&dev_b, &size };
        dim3 threads_per_block(threads, 1, 1);
        dim3 blocks_per_grid(blocks, 1, 1);
        graph_builder.AddKernelNode((void*)addKernel, blocks_per_grid, threads_per_block, 0, kernel_args, kernel_dependencies, kernel_node);
        output_dependencies.push_back(kernel_node);

        // A memcpy node for output.
        graph_builder.AddMemcpyNode(c, dev_c, size, sizeof(int), cudaMemcpyDeviceToHost, output_dependencies, memcpy_node);

        // Graph instantiation.
        graph_builder.Instantiate();

        cudaGraphExec_t graph_exec;
        cudaStream_t stream;
        graph_builder.GetGraphExec(&graph_exec, &stream);

        // Run the graph
        pai::cu::GpuTimer timer;
        timer.Start();
        for (int i = 0; i < loop_count; ++i) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        timer.Stop();
        printf("Graph: Looped %d time(s) in %f microseconds\n", loop_count, timer.ElapsedMillis());

        // Clean up.
        graph_builder.Uninit();
    }

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}

int main() {
    pai::cu::InitEnvironment(0);

    const int num = 5;
    const int a[num] = { 1, 2, 3, 4, 5 };
    const int b[num] = { 10, 20, 30, 40, 50 };
    int c[num] = { 0 };

    // Add vectors in parallel.
    AddWithCuda(c, a, b, num, 10000, true);
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
      c[0], c[1], c[2], c[3], c[4]);

    printf("\n");
    AddWithCuda(c, a, b, num, 10000, false);
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
      c[0], c[1], c[2], c[3], c[4]);

    pai::cu::CleanUpEnvironment();
    return 0;
}