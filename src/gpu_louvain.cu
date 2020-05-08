#include "cpu_louvain.h"

#include <algorithm>
#include <stdint.h>
#include <string.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include "utils.h"

#define BLOCKS 80
#define THREADS_PER_BLOCK 128
#define ARRAY_SIZE (1 << 28)

struct edge_cmp
{
    __device__ bool operator()(const Edge& a, const Edge& b)
    {
        return a.src < b.src;
    }
};

__global__ void
prepare_data_structures_kernel1(int N, int E, int* degrees, Edge* edges, int* c, float* k, int* order, int* nodes_comm)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < E; i += num_threads)
    {
        atomicAdd(&degrees[edges[i].src], 1);
    }
    for (int i = tid; i < N; i += num_threads)
    {
        c[i] = i;
    }
    for (int i = tid; i < E; i += num_threads)
    {
        atomicAdd(&k[edges[i].dst], edges[i].weight);
    }
    for (int i = tid; i < N; i += num_threads)
    {
        order[i] = i;
    }

    for (int i = tid; i < N; i += num_threads)
    {
        nodes_comm[i] = 1;
    }
}

__global__ void prepare_data_structures_kernel2(int N, int E, Edge* edges, int* e_start, int* e_end)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < E; i += num_threads)
    {
        if (i == 0 || edges[i].src != edges[i-1].src)
        {
            e_start[edges[i].src] = i;
        }
        if (i == E - 1 || edges[i].src != edges[i+1].src)
        {
            e_end[edges[i].src] = i + 1;
        }
    }
}

struct vertex_cmp
{
    int* degrees;
    vertex_cmp(int* d)
    : degrees(d)
    {}
    __device__ bool operator()(int a, int b)
    {
        return degrees[a] < degrees[b];
    }
};

__host__ void prepare_data_structures(int N, int E, Edge* edges, int* degrees, int* e_start, int* e_end, float* k, int* order, int* nodes_comm, int* c, float* ac)
{
    thrust::sort(thrust::device, edges, edges + E, edge_cmp());
    CUDA_CHECK(cudaMemset(degrees, 0, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(e_start, '\0', N * sizeof(int)));
    CUDA_CHECK(cudaMemset(e_end, '\0', N * sizeof(int)));
    CUDA_CHECK(cudaMemset(k, '\0', N * sizeof(float)));
    prepare_data_structures_kernel1<<<BLOCKS, THREADS_PER_BLOCK>>>(N, E, degrees, edges, c, k, order, nodes_comm);
    prepare_data_structures_kernel2<<<BLOCKS, THREADS_PER_BLOCK>>>(N, E, edges, e_start, e_end);
    CUDA_CHECK(cudaMemcpy(ac, k, N * sizeof(float), cudaMemcpyDeviceToDevice));
    thrust::sort(thrust::device, order, order + N, vertex_cmp(degrees));
}

__device__ uint32_t arr_hash(uint64_t key, int seed, uint64_t N)
{
    uint64_t l = N * N * seed + key - 1;
    l = (l << 32) * 1605375019ULL + (l & 0xffffffffULL) * 553437317ULL + 3471094223ULL;
    l = (l << 32) * 2769702083ULL + (l & 0xffffffffULL) * 3924398899ULL + 2998053229ULL;
    return l % ARRAY_SIZE;
}

__device__ uint32_t getpos(uint64_t* owner, uint64_t key, int N)
{
    for (int it = 0; ; ++it)
    {
        uint32_t pos = arr_hash(key, it, N);
        if (owner[pos] == key)
        {
            return pos;
        }
        else if (owner[pos] == 0)
        {
            if (atomicCAS((unsigned long long*)&owner[pos], (unsigned long long)(0), (unsigned long long)key) == 0)
            {
                return pos;
            }
            // TODO if there's more than one thread for each vertex this breaks
        }
    }
}

__device__ float compute_move(int vertex, int N, float* changes, uint64_t* owner, int* e_start, int* e_end, Edge* edges, int* c, float* k, int* new_c, int* nodes_comm, int* new_nodes_comm, float* ac, float m)
{
    for (int j = e_start[vertex]; j < e_end[vertex]; ++j)
    {
        if (edges[j].dst != vertex)
        {
            uint64_t key = (uint64_t)N * vertex + c[edges[j].dst] + 1;
            uint32_t pos = getpos(owner, key, N);
            atomicAdd(&changes[pos], edges[j].weight);
        }
    }
    int resultComm = c[vertex];
    float resultChange = 0;
    for (int e = e_start[vertex]; e < e_end[vertex]; ++e)
    {
        int i = c[edges[e].dst];
        uint32_t pos1 = getpos(owner, (uint64_t)N * vertex + i + 1, N);
        uint32_t pos2 = getpos(owner, (uint32_t)N * vertex + c[vertex] + 1, N);
        float change = 1 / m * (changes[pos1] - changes[pos2]) + k[vertex] * ((ac[c[vertex]] - k[vertex]) - ac[i]) / (2 * m * m);
        if ((change > resultChange || (change == resultChange && i < resultComm)) &&
                (nodes_comm[c[vertex]] > 1 ||
                 nodes_comm[i] > 1 ||
                 i < c[vertex]))
        {
            new_nodes_comm[i]++;
            new_nodes_comm[resultComm]--;
            resultChange = change;
            resultComm = i;
        }
    }
    new_c[vertex] = resultComm;
    return resultChange;
}

__global__ void modularity_optimisation_kernel(int N, int* e_start, int* e_end, Edge* edges, int* c, float* k, int* new_c, int* nodes_comm, int* new_nodes_comm, float* ac, float m, float* gain, float* changes, uint64_t* owner, int* order)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int v = tid; v < N; v += num_threads)
    {
        atomicAdd(gain, compute_move(order[v], N, changes, owner, e_start, e_end, edges, c, k, new_c, nodes_comm, new_nodes_comm, ac, m));
    }
}

__global__ void update_ac_kernel(int N, float* ac, int* c, float* k)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int v = tid; v < N; v += num_threads)
    {
        atomicAdd(&ac[c[v]], k[v]);
    }
}

// return modularity gain
__host__ float modularity_optimisation(int N, int* e_start, int* e_end, Edge* edges, int* c, float* k, int* new_c, int* nodes_comm, int* new_nodes_comm, float* ac, float m, float* changes, uint64_t* owner, int* order)
{
    float* gain;
    CUDA_CHECK(cudaMalloc((void**)&gain, sizeof(float)));
    CUDA_CHECK(cudaMemset(gain, '\0', sizeof(float)));
    CUDA_CHECK(cudaMemset(owner, '\0', sizeof(int) * ARRAY_SIZE));
    CUDA_CHECK(cudaMemset(changes, '\0', sizeof(float) * ARRAY_SIZE));
    modularity_optimisation_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(N, e_start, e_end, edges, c, k, new_c, nodes_comm, new_nodes_comm, ac, m, gain, changes, owner, order);
    std::swap(c, new_c);
    CUDA_CHECK(cudaMemcpy(nodes_comm, new_nodes_comm, N * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(ac, '\0', N * sizeof(float)));
    update_ac_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(N, ac, c, k);
    float result = device_fetch_var(gain);
    CUDA_CHECK(cudaFree(gain));
    return result;
}

__global__ void prepare_reorder_kernel(int N, int* reorder, int* c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += num_threads)
        reorder[c[i]] = 1;
}

__global__ void aggregate_kernel(int E, int orig_N, Edge* edges, int* reorder, int* c, int* final_communities)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < E; i += num_threads)
    {
        edges[i].src = reorder[c[edges[i].src]];
        edges[i].dst = reorder[c[edges[i].dst]];
    }
    for (int i = tid; i < orig_N; i += num_threads)
    {
        final_communities[i] = reorder[c[final_communities[i]]];
    }
}

__host__ void aggregate(int& N, int E, int orig_N, Edge* edges, int* c, int* final_communities, int* degrees, int* e_start, int* e_end, float* k, int* order, int* nodes_comm, float* ac)
{
    int* reorder;
    CUDA_CHECK(cudaMalloc((void**)&reorder, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(reorder, 0, (N + 1) * sizeof(int)));
    prepare_reorder_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(N, reorder, c);
    thrust::exclusive_scan(thrust::device, reorder, reorder + N + 1, reorder);
    aggregate_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(E, orig_N, edges, reorder, c, final_communities);
    N = device_fetch_var(reorder + N);
    CUDA_CHECK(cudaFree(reorder));
    prepare_data_structures(N, E, edges, degrees, e_start, e_end, k, order, nodes_comm, c, ac);
}

__global__ void prepare_final_communities(int* fc, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += num_threads)
    {
        fc[i] = i;
    }
}

void gpu_louvain(int N_, Edge* edges_, int E_, float min_gain, bool verbose)
{
    int N;
    int orig_N;
    int E;
    Edge* edges;
    Edge* orig_edges;
    float m;
    int* e_start;
    int* e_end;
    int* degrees;
    int* final_communities;
    int* c;
    int* new_c;
    float* k;
    float* ac;
    float* changes;
    uint64_t* owner;
    int* order;
    int* nodes_comm;
    int* new_nodes_comm;

    N = N_;
    E = E_;
    orig_N = N_;
    orig_edges = edges_;

    CUDA_CHECK(cudaMalloc((void**)&final_communities, N * sizeof(int)));
    prepare_final_communities<<<BLOCKS, THREADS_PER_BLOCK>>>(final_communities, N);
    CUDA_CHECK(cudaMalloc((void**)&degrees, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&e_start, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&e_end, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&c, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&new_c, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&order, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&nodes_comm, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&new_nodes_comm, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&k, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&ac, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&edges, sizeof(Edge) * E));
    CUDA_CHECK(cudaMalloc((void**)&changes, ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&owner, ARRAY_SIZE * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(edges, orig_edges, sizeof(Edge) * E, cudaMemcpyHostToDevice));

    prepare_data_structures(N, E, edges, degrees, e_start, e_end, k, order, nodes_comm, c, ac);
    // we can compute it on cpu because it's done only once
    for (int i = 0; i < E; ++i)
    {
        m += orig_edges[i].weight;
    }
    m /= 2;
    float modularity_change = 0;
    do
    {
        modularity_change = modularity_optimisation(N, e_start, e_end, edges, c, k, new_c, nodes_comm, new_nodes_comm, ac, m, changes, owner, order);
        std::swap(c, new_c);
        aggregate(N, E, orig_N, edges, c, final_communities, degrees, e_start, e_end, k, order, nodes_comm, ac);
    } while (modularity_change > min_gain);

    int* final_communities_host = (int*)malloc(orig_N * sizeof(int));
    CUDA_CHECK(cudaMemcpy(final_communities_host, final_communities, orig_N * sizeof(int), cudaMemcpyDeviceToHost));

    float* k_host = (float*)malloc(orig_N * sizeof(float));
    memset(k_host, '\0', orig_N * sizeof(float));
    for (int i = 0; i < E; ++i)
    {
        k_host[orig_edges[i].dst] += orig_edges[i].weight;
    }
    float* ac_host = (float*)malloc(orig_N * sizeof(float));
    memset(ac_host, '\0', orig_N * sizeof(float));
    cudaDeviceSynchronize();
    for (int i = 0; i < orig_N; ++i)
    {
        ac_host[final_communities_host[i]] += k_host[i];
    }
    float* e_host = (float*)malloc(orig_N * sizeof(float));
    memset(e_host, '\0', orig_N * sizeof(float));
    for (int i = 0; i < E; ++i)
    {
        if (final_communities_host[orig_edges[i].src] == final_communities_host[orig_edges[i].dst])
        {
            e_host[orig_edges[i].src] += orig_edges[i].weight;
        }
    }
    float q = 0;
    for (int i = 0; i < orig_N; ++i)
    {
        q += e_host[i] / (2 * m);
    }
    for (int i = 0; i < orig_N; ++i)
    {
        q -= ac_host[i] * ac_host[i] / (4 * m * m);
    }

    printf("%f\n", q);
    printf("0 0\n"); // TODO measure times
    printf("%d\n", N);
    if (verbose)
    {
        for (int i = 0; i < N; ++i)
        {
            printf("%d ", i + 1);
            for (int v = 0; v < orig_N; ++v)
            {
                if (final_communities_host[v] == i)
                    printf("%d ", v + 1);
            }
            printf("\n");
        }
    }

    free(final_communities_host);
    free(k_host);
    free(ac_host);
    free(e_host);
    CUDA_CHECK(cudaFree(final_communities));
    CUDA_CHECK(cudaFree(c));
    CUDA_CHECK(cudaFree(new_c));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(ac));
    CUDA_CHECK(cudaFree(order));
    CUDA_CHECK(cudaFree(degrees));
    CUDA_CHECK(cudaFree(e_start));
    CUDA_CHECK(cudaFree(e_end));
    CUDA_CHECK(cudaFree(changes));
    CUDA_CHECK(cudaFree(owner));
    CUDA_CHECK(cudaFree(nodes_comm));
    CUDA_CHECK(cudaFree(new_nodes_comm));
    CUDA_CHECK(cudaFree(edges));
}

