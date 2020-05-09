#include "cpu_louvain.h"

#include <algorithm>
#include <stdint.h>
#include <string.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include "utils.h"

#define BLOCKS 80
#define THREADS_PER_BLOCK 128
#define EPS (1e-12)

uint32_t ARRAY_SIZE = 1LL << 28;

__global__ void
prepare_data_structures_kernel(int N, int E, Edge* edges, int* c, float* k, int* nodes_comm)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
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
        nodes_comm[i] = 1;
    }
}

__host__ void prepare_data_structures(int N, int E, Edge* edges, float* k, int* nodes_comm, int* c, float* ac)
{
    CUDA_CHECK(cudaMemset(k, '\0', N * sizeof(float)));
    prepare_data_structures_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(N, E, edges, c, k, nodes_comm);
    CUDA_CHECK(cudaMemcpy(ac, k, N * sizeof(float), cudaMemcpyDeviceToDevice));
}

__device__ uint32_t arr_hash(uint64_t key, int seed, uint64_t N, uint32_t SIZE)
{
    uint64_t l = N * N * seed + key - 1;
    l = (l >> 32) * 1605375019ULL + (l & 0xffffffffULL) * 553437317ULL + 3471094223ULL;
    l = (l >> 32) * 2769702083ULL + (l & 0xffffffffULL) * 3924398899ULL + 2998053229ULL;
    return l & (SIZE - 1);
}

__device__ uint32_t getpos(uint64_t* owner, uint64_t key, int N, uint32_t SIZE)
{
    for (int it = 0; ; ++it)
    {
        uint32_t pos = arr_hash(key, it, N, SIZE);
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
            else if (owner[pos] == key)
            {
                return pos;
            }
        }
    }
}

__global__  void compute_changes_kernel(int N, int E, float* changes, uint64_t* owner, Edge* edges, int* c, uint32_t SIZE)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int e = tid; e < E; e += num_threads)
    {
        int vertex = edges[e].src;
        if (edges[e].dst != vertex)
        {
            uint64_t key = (uint64_t)N * vertex + c[edges[e].dst] + 1;
            uint32_t pos = getpos(owner, key, N, SIZE);
            atomicAdd(&changes[pos], edges[e].weight);
        }
    }
}

union Magic
{
    unsigned long long encoded;
    struct {
        int comm;
        float change;
    } decoded;
};

static_assert(sizeof(Magic) == 8, "too much magic");

__global__ void prepare_magic_kernel(int N, Magic* magic, int* c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int v = tid; v < N; v += num_threads)
    {
        magic[v].decoded.comm = c[v];
        magic[v].decoded.change = 0;
    }
}

__global__ void modularity_optimisation_kernel(int N, int E, Edge* edges, int* c, float* k, int* nodes_comm, int* new_nodes_comm, float* ac, float m, float* changes, uint64_t* owner, Magic* magic, uint32_t SIZE)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int e = tid; e < E; e += num_threads)
    {
        int i = c[edges[e].dst];
        int vertex = edges[e].src;
        if (nodes_comm[c[vertex]] <= 1 && nodes_comm[i] <= 1 && i >= c[vertex])
        {
            continue;
        }
        uint32_t pos1 = getpos(owner, (uint64_t)N * vertex + i + 1, N, SIZE);
        uint32_t pos2 = getpos(owner, (uint64_t)N * vertex + c[vertex] + 1, N, SIZE);
        float change = 1 / m * (changes[pos1] - changes[pos2]) + k[vertex] * ((ac[c[vertex]] - k[vertex]) - ac[i]) / (2 * m * m);
        Magic new_magic;
        new_magic.decoded.comm = i;
        new_magic.decoded.change = change;
        while (true)
        {
            Magic local_magic = magic[vertex];
            if ((change > local_magic.decoded.change ||
                    (fabs(change - local_magic.decoded.change) < EPS && i < local_magic.decoded.comm)))
            {

                if (atomicCAS((unsigned long long*)(magic + vertex),
                            local_magic.encoded, new_magic.encoded)
                        == local_magic.encoded)
                {
                    atomicAdd(new_nodes_comm + i, 1);
                    atomicAdd(new_nodes_comm + local_magic.decoded.comm, -1);
                    break;
                }
            }
            else break;
        }
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

__global__ void compute_new_c_changes_kernel(int N, Magic* magic, int* new_c, float* result_change)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int v = tid; v < N; v += num_threads)
    {
        new_c[v] = magic[v].decoded.comm;
        result_change[v] = magic[v].decoded.change;
    }
}

// return modularity gain
__host__ float modularity_optimisation(int N, int E, Edge* edges, int* c, float* k, int* new_c, int* nodes_comm, int* new_nodes_comm, float* ac, float m, float* changes, uint64_t* owner, float* result_change, Magic* magic)
{
    float* gain;
    CUDA_CHECK(cudaMalloc((void**)&gain, sizeof(float)));
    CUDA_CHECK(cudaMemset(gain, '\0', sizeof(float)));
    CUDA_CHECK(cudaMemset(owner, '\0', sizeof(int) * ARRAY_SIZE));
    CUDA_CHECK(cudaMemset(changes, '\0', sizeof(float) * ARRAY_SIZE));
    compute_changes_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(N, E, changes, owner, edges, c, ARRAY_SIZE);

    prepare_magic_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(N, magic, c);
    modularity_optimisation_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(N, E, edges, c, k, nodes_comm, new_nodes_comm, ac, m, changes, owner, magic, ARRAY_SIZE);
    compute_new_c_changes_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(N, magic, new_c, result_change);
    float result = thrust::reduce(thrust::device, result_change, result_change + N);


    std::swap(c, new_c);
    CUDA_CHECK(cudaMemcpy(nodes_comm, new_nodes_comm, N * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(ac, '\0', N * sizeof(float)));
    update_ac_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(N, ac, c, k);
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

__host__ void aggregate(int& N, int E, int orig_N, Edge* edges, int* c, int* final_communities, float* k, int* nodes_comm, float* ac)
{
    int* reorder;
    CUDA_CHECK(cudaMalloc((void**)&reorder, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(reorder, 0, (N + 1) * sizeof(int)));
    prepare_reorder_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(N, reorder, c);
    thrust::exclusive_scan(thrust::device, reorder, reorder + N + 1, reorder);
    aggregate_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(E, orig_N, edges, reorder, c, final_communities);
    N = device_fetch_var(reorder + N);
    CUDA_CHECK(cudaFree(reorder));
    prepare_data_structures(N, E, edges, k, nodes_comm, c, ac);
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
    int* final_communities;
    int* c;
    int* new_c;
    float* k;
    float* ac;
    float* changes;
    uint64_t* owner;
    int* nodes_comm;
    int* new_nodes_comm;
    float* result_change;
    Magic* magic;

    N = N_;
    E = E_;
    orig_N = N_;
    orig_edges = edges_;

    while (E * 100 < ARRAY_SIZE)
    {
        ARRAY_SIZE >>= 1;
    }

    CUDA_CHECK(cudaMalloc((void**)&final_communities, N * sizeof(int)));
    prepare_final_communities<<<BLOCKS, THREADS_PER_BLOCK>>>(final_communities, N);
    CUDA_CHECK(cudaMalloc((void**)&c, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&new_c, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&nodes_comm, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&new_nodes_comm, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&k, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&ac, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&edges, sizeof(Edge) * E));
    CUDA_CHECK(cudaMalloc((void**)&changes, ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&owner, ARRAY_SIZE * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc((void**)&result_change, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&magic, N * sizeof(Magic)));
    CUDA_CHECK(cudaMemcpy(edges, orig_edges, sizeof(Edge) * E, cudaMemcpyHostToDevice));

    prepare_data_structures(N, E, edges, k, nodes_comm, c, ac);
    // we can compute it on cpu because it's done only once
    for (int i = 0; i < E; ++i)
    {
        m += orig_edges[i].weight;
    }
    m /= 2;
    float modularity_change = 0;
    do
    {
        modularity_change = modularity_optimisation(N, E, edges, c, k, new_c, nodes_comm, new_nodes_comm, ac, m, changes, owner, result_change, magic);
        std::swap(c, new_c);
        aggregate(N, E, orig_N, edges, c, final_communities, k, nodes_comm, ac);
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
    CUDA_CHECK(cudaDeviceSynchronize());
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
    CUDA_CHECK(cudaFree(changes));
    CUDA_CHECK(cudaFree(owner));
    CUDA_CHECK(cudaFree(nodes_comm));
    CUDA_CHECK(cudaFree(new_nodes_comm));
    CUDA_CHECK(cudaFree(edges));
    CUDA_CHECK(cudaFree(result_change));
    CUDA_CHECK(cudaFree(magic));
}

