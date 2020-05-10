#ifndef __GPU_LOUVAIN_H_
#define __GPU_LOUVAIN_H_

#include <map>

#include "utils.h"
#include "common.h"

void gpu_louvain(int N, Edge* edges, int E, float min_gain, bool verbose, std::map<int, int>& reorder);

#endif // __GPU_LOUVAIN_H_
