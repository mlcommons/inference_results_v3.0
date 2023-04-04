# CXX=icpc
# CXXFLAGS=-fPIC -fopenmp -xHOST -fma -O3 -std=c++14 -Wall -Dusercp

CXX = clang++
CXXFLAGS = -g -fPIC -fopenmp -march=native -mfma -O3 -std=c++14 -Wall -Dusercp

COMMON_SRCS = ../amx_init.cpp helper.cpp helper_test.cpp
SRCS = $(COMMON_SRCS) i_linear_tpp.cpp lstm_postop_tpp.cpp tanh_tpp.cpp sigmoid_tpp.cpp

OBJS = $(SRCS:.cpp=.o)

MAIN = xxx.exe

.PHONY: depend clean

all: $(MAIN)
	@echo Simple compiler name xxx.exe has been compiled
$(MAIN): main.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(MAIN) main.o $(OBJS) $(LFLAGS)

# libtpps.a : $(kernels:.cpp=.o)
# 	ar rcs $@ $^

test_i_gemm: test_i_gemm.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o ${@:%=%.exe} ${@:%=%.o} $(OBJS) $(LFLAGS)

test_i_linear: test_i_linear.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o ${@:%=%.exe} ${@:%=%.o} $(OBJS) $(LFLAGS)

clean :
	$(RM) *.o *~ $(MAIN) test_*.exe

# depend: $(SRCS)
# 	makedepend $(INCLUDES) $^
