#include "cpu_louvain.h"

#include <algorithm>
#include <string.h>

// number of vertices and communities
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
float* changes; // TODO separate for each vertex
int* order;
int* nodes_comm;
int* new_nodes_comm;

struct edge_cmp
{
    __device__ bool operator()(const Edge& a, const Edge& b)
    {
        return a.src < b.src;
    }
};

__global__ void
prepare_data_structures_kernel1(int N, int E, int* degrees, Edge* edges, int* c, float* k, int* order)
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

__host__ void prepare_data_structures()
{
    thrust::sort(thrust::device, edges, edges + E, edge_cmp);
    CUDA_CHECK(cudaMemset(degrees, 0, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(e_start, '\0', N * sizeof(int)));
    CUDA_CHECK(cudaMemset(e_end, '\0', N * sizeof(int)));
    CUDA_CHECK(cudaMemset(k, '\0', N * sizeof(float)));
    prepare_data_structures_kernel1<<<80, 32>>>(N, E, degrees, edges, c, k, order);
    int e_count = 0;
    for (int v = 0; v < N; ++v)
    {
        e_start[v] = e_count;
        while (e_count < E && edges[e_count].src == v)
        {
            e_count++;
        }
        e_end[v] = e_count;
    }
    CUDA_CHECK(cudaMemcpy(ac, k, N * sizeof(float), cudaMemcpyDeviceToDevice));


    std::sort(order, order + N, [](int a, int b){return degrees[a] < degrees[b];});

}

__device__ float compute_move(int vertex)
{
    memset(changes, '\0', N * sizeof(int));
    for (int j = e_start[vertex]; j < e_end[vertex]; ++j)
    {
        if (edges[j].dst != vertex)
            changes[c[edges[j].dst]] += edges[j].weight;
    }
    int resultComm = -1;
    float resultChange = 0;
    for (int i = 0; i < N; ++i) // todo nie iteorwać po wszystkich, tylko po sąsiednich
    {
        float change = 1 / m * (changes[i] - changes[c[vertex]]) + k[vertex] * ((ac[c[vertex]] - k[vertex]) - ac[i]) / (2 * m * m);
        if (change > resultChange &&
                (nodes_comm[c[vertex]] > 1 ||
                 nodes_comm[i] > 1 ||
                 i < c[vertex]))
        {
            resultChange = change;
            resultComm = i;
            new_nodes_comm[i]++;
            new_nodes_comm[c[vertex]]--;
        }
    }
    if (resultChange > 0) // todo czy mozna zmieniac w `c` czy trzeba w innej?
        new_c[vertex] = resultComm;
    else
        new_c[vertex] = c[vertex];
    return resultChange;
}

// return modularity gain
float modularity_optimisation()
{
    float gain = 0;
    for (int v = 0; v < N; ++v) // parallel
    {
        gain += compute_move(order[v]);
    }
    std::swap(c, new_c);
    memcpy(nodes_comm, new_nodes_comm, N * sizeof(int));
    memset(ac, '\0', N * sizeof(float));
    for (int v = 0; v < N; ++v)
    {
        ac[c[v]] += k[v];
    }
    return gain;
}

void aggregate()
{
    int reorder[N];
    for (int i = 0; i < N; ++i)
        reorder[i] = -1;
    int counter = 0;
    for (int i = 0; i < N; ++i)
    {
        if (reorder[c[i]] == -1)
            reorder[c[i]] = counter++;
    }
    for (int i = 0; i < E; ++i)
    {
        edges[i].src = reorder[c[edges[i].src]];
        edges[i].dst = reorder[c[edges[i].dst]];
    }
    for (int i = 0; i < orig_N; ++i)
    {
        final_communities[i] = reorder[c[final_communities[i]]];
    }
    N = counter;
    prepare_data_structures();
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

void cpu_louvain(int N_, Edge* edges_, int E_, float min_gain, bool verbose)
{
    N = N_;
    E = E_;
    orig_N = N_;
    orig_edges = edges_;

    CUDA_CHECK(cudaMalloc((void**)&final_communities, N * sizeof(int)));
    prepare_final_communities<<<80, 32>>>(final_communities, N);
    CUDA_CHECK(cudaMalloc((void**)&degrees, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&e_start, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&e_end, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&c, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&new_c, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&order, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&nodes_comm, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&new_nodes_comm, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&changes, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&k, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&ac, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&edges, sizeof(Edge) * E));
    CUDA_CHECK(cudaMemcpy(edges, orig_edges, sieof(Edge) * E));

    m = 0;
    for (int i = 0; i < E; ++i)
    {
        m += edges[i].weight;
    }
    m /= 2;
    memcpy(orig_edges, edges, E * sizeof(Edge));
    prepare_data_structures();
    float modularity_change = 0;
    do
    {
        modularity_change = modularity_optimisation();
        aggregate();
    } while (modularity_change > min_gain);

    memset(k, '\0', orig_N * sizeof(float));
    for (int i = 0; i < E; ++i)
    {
        k[orig_edges[i].dst] += edges[i].weight;
    }
    memset(ac, '\0', orig_N * sizeof(float));
    for (int i = 0; i < orig_N; ++i)
    {
        ac[final_communities[i]] += k[i];
    }
    memset(changes, '\0', orig_N * sizeof(float));
    for (int i = 0; i < E; ++i)
    {
        if (final_communities[orig_edges[i].src] == final_communities[orig_edges[i].dst])
        {
            changes[orig_edges[i].src] += orig_edges[i].weight;
        }
    }
    float q = 0;
    for (int i = 0; i < orig_N; ++i)
    {
        q += changes[i] / (2 * m);
    }
    for (int i = 0; i < orig_N; ++i)
    {
        q -= ac[i] * ac[i] / (4 * m * m);
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
                if (final_communities[v] == i)
                    printf("%d ", v + 1);
            }
            printf("\n");
        }
    }

    free(final_communities);
    free(c);
    free(new_c);
    free(k);
    free(ac);
    free(order);
    free(degrees);
    free(e_start);
    free(e_end);
    free(changes);
    free(nodes_comm);
    free(new_nodes_comm);
    free(orig_edges);
}

