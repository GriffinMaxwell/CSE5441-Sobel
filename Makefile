# Compiler vars
MPI := mpicc
MPI_FLAGS := -O3 -fopenmp

# Paths
BUILD_DIR := Build
SRC_DIR := Source

# Targets
TARGET_LAB5 := lab5.out

TARGET_EXECUTABLES := \
	$(TARGET_LAB5) \

# File lists for Lab 5
SRCS_LAB5 := $(shell find $(SRC_DIR) -name *.c)
OBJ_LAB5 := \
$(SRCS_LAB5:%=$(BUILD_DIR)/%.o) \
	$(SRC_DIR)/bmpReader.o \

all: $(TARGET_EXECUTABLES)

$(TARGET_LAB5): $(OBJ_LAB5)
	$(MPICC) $(MPI_FLAGS) -o $@ $(OBJ_LAB5)

# c source
$(BUILD_DIR)/%.c.o: %.c
	mkdir -p $(dir $@)
	$(MPI) $(MPI_FLAGS) -c $< -o $@

.PHONY: clean package

clean:
	@echo "Cleaning build files..."
	@$(RM) -rf $(BUILD_DIR)
	@$(RM) $(TARGET_PART1)
	@$(RM) $(TARGET_PART2)

package:
	@echo "Packaging up project for submission..."
	@mkdir -p cse5441_lab5
	@cp $(SRC_DIR)/*.c cse5441_lab5
	@cp $(SRC_DIR)/*.h cse5441_lab5
	@cp $(SRC_DIR)/*.o cse5441_lab5
	@cp submit.mk cse5441_lab5
	@mv cse5441_lab4/submit.mk cse5441_lab4/Makefile
