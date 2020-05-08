CXX = g++
CC = gcc
NVCC = /usr/local/cuda/bin/nvcc
HOMEDIR = ..
CPPFLAGS = -Wall -Wextra -std=c++17 -O3 -g
CFLAGS = -Wall -Wextra -std=c11 -O3 -g
-include $(OBJS:.o=.d)

SRCS = main.cu mmio.c utils.cu cpu_louvain.cc gpu_louvain.cu
OBJS = main.o mmio.o utils.o cpu_louvain.o gpu_louvain.o

main: $(OBJS)
	$(NVCC) $(OBJS) $(LINKFLAGS) -o gpulouvain

# This is some dark magic and I don't want to understand
# http://www.scottmcpeak.com/autodepend/autodepend.html
%.o: %.cc
	$(CXX) -c $(CPPFLAGS) $(DEPFLAGS) $(INCFLAGS) $< -o $@
	$(CXX) -MM $(CPPFLAGS) $(DEPFLAGS) $< -o $*.d
	@mv -f $*.d $*.d.tmp
	@sed -e 's|.*:|$*.o:|' < $*.d.tmp > $*.d
	@sed -e 's/.*://' -e 's/\\$$//' < $*.d.tmp | fmt -1 | \
	  sed -e 's/^ *//' -e 's/$$/:/' >> $*.d
	@rm -f $*.d.tmp

%.o: %.c
	$(CC) -c $(CFLAGS) $(DEPFLAGS) $(INCFLAGS) $< -o $@
	$(CC) -MM $(CFLAGS) $(DEPFLAGS) $< -o $*.d
	@mv -f $*.d $*.d.tmp
	@sed -e 's|.*:|$*.o:|' < $*.d.tmp > $*.d
	@sed -e 's/.*://' -e 's/\\$$//' < $*.d.tmp | fmt -1 | \
	  sed -e 's/^ *//' -e 's/$$/:/' >> $*.d
	@rm -f $*.d.tmp

%.o: %.cu
	$(NVCC) -c -O3 $(DEPFLAGS) $(INCFLAGS) $< -o $@
	$(NVCC) -MM $(DEPFLAGS) $< -o $*.d
	@mv -f $*.d $*.d.tmp
	@sed -e 's|.*:|$*.o:|' < $*.d.tmp > $*.d
	@sed -e 's/.*://' -e 's/\\$$//' < $*.d.tmp | fmt -1 | \
	  sed -e 's/^ *//' -e 's/$$/:/' >> $*.d
	@rm -f $*.d.tmp

clean:
	@rm -f $(OBJS) main
	@rm -rf *.d


