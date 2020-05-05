CXX = g++
CC = gcc
HOMEDIR = ..
CPPFLAGS = -Wall -Wextra -std=c++17 -O0 -g
CFLAGS = -Wall -Wextra -std=c11 -O0 -g
INCFLAGS = -I/opt/cuda/targets/x86_64-linux/include/
LINKFLAGS = -L/opt/cuda/targets/x86_64-linux/lib/
-include $(OBJS:.o=.d)

SRCS = main.cc mmio.c utils.cc cpu_louvain.cc
OBJS = main.o mmio.o utils.o cpu_louvain.o

main: $(OBJS)
	$(CXX) $(CPPFLAGS) $(OBJS) $(LINKFLAGS) -o gpulouvain

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

clean:
	@rm -f $(OBJS) main
	@rm -rf *.d


