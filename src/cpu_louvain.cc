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

void prepare_data_structures()
{
    std::sort(edges, edges + E, [](const Edge& a, const Edge& b){return a.src < b.src;});
    memset(degrees, '\0', N * sizeof(int));
    memset(e_start, '\0', N * sizeof(int));
    memset(e_end, '\0', N * sizeof(int));
    for (int i = 0; i < E; ++i)
    {
        degrees[edges[i].src]++;
    }
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
    for (int i = 0; i < N; ++i)
    {
        c[i] = i;
    }
    memset(k, '\0', N * sizeof(float));
    for (int i = 0; i < E; ++i)
    {
        k[edges[i].dst] += edges[i].weight;
    }

    memcpy(ac, k, N * sizeof(float));

    for (int i = 0; i < N; ++i)
    {
        order[i] = i;
    }

    std::sort(order, order + N, [](int a, int b){return degrees[a] < degrees[b];});

    for (int i = 0; i < N; ++i)
    {
        nodes_comm[i] = 1;
    }
}

float compute_move(int vertex)
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
        gain += compute_move(v);
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

void cpu_louvain(int N_, Edge* edges_, int E_, float min_gain, bool verbose)
{
    N = N_;
    E = E_;
    orig_N = N_;
    edges = edges_;

    final_communities = (int*)malloc(N * sizeof(int));
    degrees = (int*)malloc(N * sizeof(int));
    e_start = (int*)malloc(N * sizeof(int));
    e_end = (int*)malloc(N * sizeof(int));
    c = (int*)malloc(N * sizeof(int));
    new_c = (int*)malloc(N * sizeof(int));
    changes = (float*)malloc(N * sizeof(float));
    k = (float*)malloc(N * sizeof(float));
    ac = (float*)malloc(N * sizeof(float));
    order = (int*)malloc(N * sizeof(int));
    nodes_comm = (int*)malloc(N * sizeof(int));
    new_nodes_comm = (int*)malloc(N * sizeof(int));
    orig_edges = (Edge*)malloc(E * sizeof(Edge));
    for (int i = 0; i < N; ++i)
    {
        final_communities[i] = i;
    }

    m = 0;
    for (int i = 0; i < E; ++i)
    {
//        if (edges[i].src <= edges[i].dst)
        {
            m += edges[i].weight;
        }
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

