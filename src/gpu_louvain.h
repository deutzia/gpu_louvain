#ifndef __GPU_LOUVAIN_H_
#define __GPU_LOUVAIN_H_

#include "utils.h"

void gpu_louvain(int N, Edge* edges, int E, float min_gain, bool verbose);

#endif // __GPU_LOUVAIN_H_
