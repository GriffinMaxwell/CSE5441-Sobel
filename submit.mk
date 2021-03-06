# Compilers & flags
MPI := mpicc
MPI_FLAGS := -O3 -fopenmp

# Paths
BUILD_DIR := Build
SRC_DIR := Source

# Targets
TARGET_LAB5 := lab5.out

TARGET_EXECUTABLES := \
	$(TARGET_LAB5) \

# Objects
OBJS := \
	maxwell_griffin_lab5.o \
	Sobel.o \
	Stencil.o \

PROTECTED_OBJS := \
	gcc_bmpReader.o \


all: $(TARGET_EXECUTABLES)

$(TARGET_LAB5): $(OBJS)
	$(MPI) $(MPI_FLAGS) -o $@ $(OBJS) $(PROTECTED_OBJS)

%.o: %.c
	$(MPI) $(MPI_FLAGS) -c -o $@ $<

.PHONY: clean
clean:
	@echo Cleaning build files...
	@rm -f $(TARGET_EXECUTABLES) $(OBJS)
