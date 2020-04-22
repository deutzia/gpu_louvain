CXX = clang++
CC = clang
HOMEDIR = ..
CPPFLAGS = -Wall -Wextra -std=c++17 -O2
CFLAGS = -Wall -Wextra -std=c11 -O2
-include $(OBJS:.o=.d)

SRCS = main.cc mmio.c
OBJS = main.o mmio.o

main: $(OBJS)
	$(CXX) $(CPPFLAGS) $(OBJS) -o gpulouvain

# This is some dark magic and I don't want to understand
# http://www.scottmcpeak.com/autodepend/autodepend.html
%.o: %.cc
	$(CXX) -c $(CPPFLAGS) $(DEPFLAGS) $< -o $@
	$(CXX) -MM $(CPPFLAGS) $(DEPFLAGS) $< -o $*.d
	@mv -f $*.d $*.d.tmp
	@sed -e 's|.*:|$*.o:|' < $*.d.tmp > $*.d
	@sed -e 's/.*://' -e 's/\\$$//' < $*.d.tmp | fmt -1 | \
	  sed -e 's/^ *//' -e 's/$$/:/' >> $*.d
	@rm -f $*.d.tmp

%.o: %.c
	$(CC) -c $(CFLAGS) $(DEPFLAGS) $< -o $@
	$(CC) -MM $(CFLAGS) $(DEPFLAGS) $< -o $*.d
	@mv -f $*.d $*.d.tmp
	@sed -e 's|.*:|$*.o:|' < $*.d.tmp > $*.d
	@sed -e 's/.*://' -e 's/\\$$//' < $*.d.tmp | fmt -1 | \
	  sed -e 's/^ *//' -e 's/$$/:/' >> $*.d
	@rm -f $*.d.tmp

clean:
	@rm -f $(OBJS) main
	@rm -rf *.d


