#include "cpu_louvain.h"

#include <algorithm>
#include <math.h>
#include <string.h>
#include <stdio.h>

#define EPS (1e-12)

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

struct vertex_cmp
{
    int* deg;
    vertex_cmp(int* d)
    : deg(d)
    {}
    bool operator()(int a, int b)
    {
        return deg[a] < deg[b];
    }
};

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

    std::sort(order, order + N, vertex_cmp(degrees));

    for (int i = 0; i < N; ++i)
    {
        nodes_comm[i] = 1;
    }
    for (int i = 0; i < N; ++i)
    {
        new_nodes_comm[i] = 1;
    }

    for (int i = 0; i < N; ++i)
    {
        new_c[i] = i;
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
    int resultComm = c[vertex];
    float resultChange = 0;
    for (int e = e_start[vertex]; e < e_end[vertex]; ++e)
    {
        int i = c[edges[e].dst];
        float change = 1 / m * (changes[i] - changes[c[vertex]]) + k[vertex] * ((ac[c[vertex]] - k[vertex]) - ac[i]) / (2 * m * m);
        //printf("vertex = %d new_comm = %d result_change = %f\n", vertex, i, change);
        if ((change > resultChange || (fabs(change - resultChange) < EPS && i < resultComm)) &&
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
    if (resultChange > 0)
        new_c[vertex] = resultComm;
    else
        new_c[vertex] = c[vertex];
    //printf("(%d %f)\n", vertex, resultChange);
    return resultChange;
}

void debug_function(int N, float* ac)
{
    for (int i = 0; i < N; ++i)
        printf("ac[%d] = %f ", i, ac[i]);
    printf("\n");
}

float compute_modularity()
{
//    fprintf(stderr, "new c: ");
//    for (int i = 0; i < N; ++i)
//    {
//        fprintf(stderr, "(%d %d) ", i, new_c[i]);
//    }
//    fprintf(stderr, "\n");
//    debug_function(N, ac);
    float q = 0;
    memset(changes, 0, N * sizeof(float));
    for (int i = 0; i < E; ++i)
    {
        if (new_c[edges[i].src] == new_c[edges[i].dst])
        {
            changes[edges[i].src] += edges[i].weight;
        }
    }
    for (int i = 0; i < N; ++i)
    {
        q += changes[i] / (2 * m);
    }
    for (int i = 0; i < N; ++i)
    {
        q -= ac[i] * ac[i] / (4 * m * m);
    }
    return q;
}


// return modularity gain
float modularity_optimisation()
{
//    printf("N = %d E = %d m = %f\n", N, E, m);
//    fprintf(stderr, "c in modularity_optimisation: ");
//    for (int i = 0; i < N; ++i)
//    {
//        fprintf(stderr, "(%d %d) ", i, c[i]);
//    }
//    fprintf(stderr, "\n");
//    fprintf(stderr, "ac in modularity_optimisation: ");
//    for (int i = 0; i < N; ++i)
//    {
//        fprintf(stderr, "(%d %f) ", i, ac[i]);
//    }
//    fprintf(stderr, "\n");
//    fprintf(stderr, "(new) nodes_comm: ");
//    for (int i = 0; i < N; ++i)
//    {
//        fprintf(stderr, "(%d %d) ", i, nodes_comm[i]);
//    }
//    fprintf(stderr, "\n");
//    fprintf(stderr, "(new) nodes_comm: ");
//    for (int i = 0; i < N; ++i)
//    {
//        fprintf(stderr, "(%d %d) ", i, new_nodes_comm[i]);
//    }
//    fprintf(stderr, "\n");
    float gain = 0;
    for (int v = 0; v < N; ++v) // parallel
    {
        gain += compute_move(order[v]);
    }
//    fprintf(stderr, "new_c after modularity_optimisation: ");
//    for (int i = 0; i < N; ++i)
//    {
//        fprintf(stderr, "(%d %d) ", i, new_c[i]);
//    }
//    fprintf(stderr, "\n");
    return gain;
}

void aggregate()
{
    int reorder[N];
    for (int i = 0; i < N; ++i)
        reorder[i] = 0;
    for (int i = 0; i < N; ++i)
        reorder[c[i]] = 1;
    int new_N = 0, tmp;
    for (int i = 0; i < N; ++i)
    {
        tmp = reorder[i];
        reorder[i] = new_N;
        new_N += tmp;
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
    N = new_N;
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
        m += edges[i].weight;
    }
    m /= 2;
    memcpy(orig_edges, edges, E * sizeof(Edge));
    prepare_data_structures();

    float modularity_change = 0, sum = 0, old_modularity = compute_modularity();
    do
    {
        sum = 0;
        do
        {
            modularity_optimisation();
            memset(ac, '\0', N * sizeof(float));
            for (int v = 0; v < N; ++v)
            {
                ac[new_c[v]] += k[v];
            }
            modularity_change = compute_modularity() - old_modularity;
//            printf("modularity_change = %.9f\n", modularity_change);
            if (modularity_change > EPS)
            {
                std::swap(c, new_c);
                memcpy(nodes_comm, new_nodes_comm, N * sizeof(int));
                sum += modularity_change;
                old_modularity += modularity_change;
            }
            else break;
        } while (true);
//        printf("aggregating\n");
        aggregate();
    } while (sum > min_gain);

    memset(k, '\0', orig_N * sizeof(float));
    for (int i = 0; i < E; ++i)
    {
        k[orig_edges[i].dst] += orig_edges[i].weight;
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
    if (verbose)
    {
        printf("%d\n", N);
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

