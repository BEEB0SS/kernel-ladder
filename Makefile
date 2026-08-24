# Convenience wrapper around CMake. `make help` lists everything.
BUILD    ?= build
ARCH     ?= 121
SIZE     ?= 4096
RESULTS  ?= bench/results/sgemm.jsonl
ATTN_SIZE    ?= 4x8x1024x64
ATTN_RESULTS ?= bench/results/attention.jsonl

.PHONY: help
help:
	@echo ""
	@echo "kernel-ladder — SGEMM optimization ladder for DGX Spark (GB10 / sm_121)"
	@echo ""
	@echo "  make check        preflight: driver, CUDA version, profiling perms, clocks"
	@echo "  make probe        ask ptxas which tensor-core instructions sm_121 supports"
	@echo "  make build        configure + compile (ARCH=121|121f|121a)"
	@echo "  make test-host    build & run the no-GPU unit tests (oracle + stats)"
	@echo "  make test-attention  build & run the no-GPU attention oracle tests"
	@echo "  make run          run the full ladder at SIZE=$(SIZE)"
	@echo "  make run-small    run at 512 — fast loop while developing a rung"
	@echo "  make run-ragged   run at 1024x2048x512 — catches M/N transposition bugs"
	@echo "  make run-attn     run the attention ladder at ATTN_SIZE=$(ATTN_SIZE)"
	@echo "  make run-attn-small  attention at 2x3x256x64 — fast loop while developing"
	@echo "  make report       terminal table + charts from $(RESULTS)"
	@echo "  make profile K=.. Nsight Compute on one kernel, right sections preselected"
	@echo "  make sanitize K=..  compute-sanitizer memcheck on one kernel"
	@echo "  make sass         dump SASS and count load/store widths"
	@echo "  make sweep        tile-size sweep for rung 4"
	@echo "  make clean"
	@echo ""

.PHONY: build
build:
	@cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$(ARCH)
	@cmake --build $(BUILD) -j

.PHONY: test-host
test-host:
	@g++ -std=c++17 -O2 -Wall -Wextra -o $(BUILD)/test_host tests/test_host.cpp 2>/dev/null || \
	 (mkdir -p $(BUILD) && g++ -std=c++17 -O2 -Wall -Wextra -o $(BUILD)/test_host tests/test_host.cpp)
	@./$(BUILD)/test_host

.PHONY: test-attention
test-attention:
	@mkdir -p $(BUILD)
	@g++ -std=c++17 -O2 -Wall -Wextra -o $(BUILD)/test_attention_host tests/test_attention_host.cpp
	@./$(BUILD)/test_attention_host

.PHONY: run-attn
run-attn: build
	@./$(BUILD)/attention --size $(ATTN_SIZE) --out $(ATTN_RESULTS)

# B, H, S, D all differ so an index mixup cannot hide.
.PHONY: run-attn-small
run-attn-small: build
	@./$(BUILD)/attention --size 2x3x256x64 --iters 30 --out $(ATTN_RESULTS)

.PHONY: check
check:
	@./scripts/setup_check.sh

.PHONY: probe
probe:
	@./scripts/probe_arch.sh

.PHONY: run
run: build
	@./scripts/run_ladder.sh --size $(SIZE) --out $(RESULTS)

.PHONY: run-small
run-small: build
	@./$(BUILD)/ladder --size 512 --iters 30 --out $(RESULTS)

# Ragged sizes catch M/N transposition bugs that square sizes hide.
.PHONY: run-ragged
run-ragged: build
	@./$(BUILD)/ladder --size 1024x2048x512 --iters 30 --out /tmp/ladder_ragged.jsonl

.PHONY: report
report:
	@python3 bench/report.py $(RESULTS)

.PHONY: report-attn
report-attn:
	@python3 bench/report_attention.py bench/results/attention.jsonl

K ?= 04_2d_blocktile
.PHONY: profile
profile: build
	@./scripts/profile.sh $(K)

.PHONY: sanitize
sanitize: build
	@compute-sanitizer --tool memcheck ./$(BUILD)/ladder --size 512 --only $(K) --iters 3

.PHONY: sass
sass: build
	@echo "load/store instruction widths (want LDG.E.128 / STG.E.128 after rung 5):"
	@cuobjdump -sass $(BUILD)/ladder | grep -oE '\b(LDG|STG|LDS|STS)\.[A-Z0-9.]*' | sort | uniq -c | sort -rn

.PHONY: sweep
sweep: build
	@./scripts/sweep_tiles.sh

.PHONY: clean
clean:
	@rm -rf $(BUILD)
