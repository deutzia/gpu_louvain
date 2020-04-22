#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "mmio.h"

bool verbose = false;
float min_gain = -1;

int main(int argc, char*argv[])
{
    int c, retcode;
    char* fvalue = NULL;
    int M, N, nz;
    int *I, *J;
    float* val;
    MM_typecode matcode;
    while ((c = getopt (argc, argv, "vf:g:")) != -1)
    switch (c)
    {
        case 'v':
            verbose = true;
            break;
        case 'f':
            fvalue = optarg;
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
            return 1;
        default:
            return 1;
    }
    if (optind < argc)
    {
        for (int index = optind; index < argc; index++)
            fprintf(stderr, "Non-option argument %s\n", argv[index]);
        return 1;
    }
    if (fvalue == NULL)
    {
        fprintf(stderr, "Option -f is required\n");
        return 1;
    }
    if (min_gain <= 0)
    {
        fprintf(stderr, "Option -g is required and the argument has to be a float\n");
        return 1;
    }
    FILE* file = fopen(fvalue, "r");
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
        fprintf(stderr, "Error occured while reading size of matrix");
        return 1;
    }


    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (float *) malloc(nz * sizeof(float));


    for (int i=0; i<nz; i++)
    {
        fscanf(file, "%d %d %g\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    // TODO

    if (file !=stdin) fclose(file);
    free(I);
    free(J);
    free(val);

}
