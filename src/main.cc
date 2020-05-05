#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <map>
#include <set>

#include "mmio.h"
#include "cpu_louvain.h"

bool verbose = false;
float min_gain = -1;
char* filename;

void parse_args(int argc, char* argv[])
{
    int c;
    while ((c = getopt (argc, argv, "vf:g:")) != -1)
        switch (c)
        {
            case 'v':
                verbose = true;
                break;
            case 'f':
                filename = optarg;
                break;
            case 'g':
                min_gain = strtof(optarg, NULL);
                break;
            case '?':
                if (optopt == 'f' || optopt == 'g')
                    fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf(stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
                exit(1);
            default:
                exit(1);
        }
    if (optind < argc)
    {
        for (int index = optind; index < argc; index++)
            fprintf(stderr, "Non-option argument %s\n", argv[index]);
        exit(1);
    }
    if (filename == NULL)
    {
        fprintf(stderr, "Option -f is required\n");
        exit(1);
    }
    if (min_gain <= 0)
    {
        fprintf(stderr, "Option -g is required and the argument has to be a float\n");
        exit(1);
    }
}

int main(int argc, char* argv[])
{
    parse_args(argc, argv);
    int retcode;
    int M, N, nz;
    MM_typecode matcode;
    FILE* file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening the file with matrix (%d, %s)\n",
                errno, strerror(errno));
        return 1;
    }

    if (mm_read_banner(file, &matcode) != 0)
    {
        fprintf(stderr, "Could not process Matrix Market banner.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((retcode = mm_read_mtx_crd_size(file, &M, &N, &nz)) !=0)
    {
        fprintf(stderr, "Error occured while reading size of matrix\n");
        return 1;
    }

    if (N != M)
    {
        fprintf(stderr, "Matrix has to be a square\n");
        return 1;
    }

    Edge* edges = (Edge*)malloc(2 * nz * sizeof(Edge));

    int I = 0, J = 0;
    float val = 0.0;
    int E = 0;
    std::set<int> vertices_set;
    std::map<int, int> reorder;
    for (int i = 0; i < nz; i++)
    {
        fscanf(file, " %d %d %f\n", &I, &J, &val);
        vertices_set.insert(I);
        vertices_set.insert(J);
        edges[E++] = Edge{I, J, val};
        if (I != J) // don't double count self-loops
        {
            edges[E++] = Edge{J, I, val};
        }
    }
    int vcount = 0;
    for (const auto& v : vertices_set)
    {
        reorder[v] = vcount++;
    }
    for (int i = 0; i < E; ++i)
    {
        edges[i].src = reorder[edges[i].src];
        edges[i].dst = reorder[edges[i].dst];
    }
    if (file != stdin) fclose(file);

//    detect_cuda_init();

    cpu_louvain(N, edges, E, min_gain, verbose);
    free(edges);
}
