# --- Configuration ---
CC = gcc
SRCDIR = src
APPDIR = app
VERDIR = verification
INCDIR = include
THIRDDIR = third_party
OBJDIR = obj
BINDIR = bin

# --- Target Executable Names ---
TARGET = instrument_detector
VERIFIER = verifier
DATAPREP = dataprep_verifier
INSTNORM = instancenorm_verifier
AGG = agg_verifier

# --- Cross-Platform Architecture Detection ---
ARCH := $(shell uname -m)
ifeq ($(ARCH), aarch64)
    SIMD_FLAGS = -DUSE_NEON
else
    SIMD_FLAGS = -DUSE_SSE
endif

# --- Compiler and Linker Flags ---
CFLAGS = -I$(INCDIR) -I$(THIRDDIR) -O3 -march=native $(SIMD_FLAGS) -Wall -Wextra -std=c99 -fopenmp
LDFLAGS = -lfftw3f -lasound -lm -fopenmp

# --- Object File Discovery ---
# List all source files that are compiled into .o files
C_SOURCES = $(wildcard $(SRCDIR)/*.c) $(wildcard $(APPDIR)/*.c) $(wildcard $(VERDIR)/*.c)
OBJECTS = $(patsubst %.c,$(OBJDIR)/%.o,$(notdir $(C_SOURCES)))

# Sources live in src/, app/ and verification/, but all object files go into one flat obj/ folder
vpath %.c $(SRCDIR) $(APPDIR) $(VERDIR)

# Define the specific object files needed for each executable
TARGET_OBJS = $(OBJDIR)/main_inference.o $(OBJDIR)/utils.o $(OBJDIR)/cpu_layers.o $(OBJDIR)/weight_loader.o $(OBJDIR)/cqt.o $(OBJDIR)/sound_acquisition.o $(OBJDIR)/onset_detector.o
VERIFIER_OBJS = $(OBJDIR)/main_verification.o $(OBJDIR)/utils.o $(OBJDIR)/cpu_layers.o $(OBJDIR)/weight_loader.o
DATAPREP_OBJS = $(OBJDIR)/main_dataprep_verifier.o $(OBJDIR)/utils.o $(OBJDIR)/cqt.o $(OBJDIR)/cpu_layers.o
INSTNORM_OBJS = $(OBJDIR)/main_instancenorm_verifier.o $(OBJDIR)/utils.o $(OBJDIR)/cpu_layers.o $(OBJDIR)/weight_loader.o
AGG_OBJS = $(OBJDIR)/main_aggregation_verifier.o $(OBJDIR)/utils.o

# --- Build Rules ---

# Default rule: build the main application
all: $(BINDIR)/$(TARGET)

# Rule to build the main instrument detector
$(BINDIR)/$(TARGET): $(TARGET_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking instrument detector..."
	$(CC) $^ -o $@ $(LDFLAGS)

# Rule to build the model verifier
verifier: $(BINDIR)/$(VERIFIER)
$(BINDIR)/$(VERIFIER): $(VERIFIER_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking model verifier..."
	$(CC) $^ -o $@ $(LDFLAGS)

# Rule to build the data prep verifier
dataprep: $(BINDIR)/$(DATAPREP)
$(BINDIR)/$(DATAPREP): $(DATAPREP_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking data prep verifier..."
	$(CC) $^ -o $@ $(LDFLAGS)

# Rule to build the instance norm verifier
instancenorm: $(BINDIR)/$(INSTNORM)
$(BINDIR)/$(INSTNORM): $(INSTNORM_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking instance norm verifier..."
	$(CC) $^ -o $@ $(LDFLAGS)

# Rule to build the aggregation verifier
agg: $(BINDIR)/$(AGG)
$(BINDIR)/$(AGG): $(AGG_OBJS)
	@mkdir -p $(BINDIR)
	@echo "Linking aggregation verifier..."
	$(CC) $^ -o $@ $(LDFLAGS)

# Generic Pattern Rule to compile .c files into .o files
# This one rule handles all .c files in the src, app and verification directories (found through vpath).
$(OBJDIR)/%.o: %.c
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to clean up all build files
clean:
	@echo "Cleaning up build artifacts..."
	rm -rf $(OBJDIR) $(BINDIR)

# Tell make that these are not actual files
.PHONY: all clean verifier dataprep instancenorm agg