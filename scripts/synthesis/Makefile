# Makefile -- Targets for common EDA flows.
#
# Usage:
#   make lint              -- Verilator lint
#   make lint-spyglass     -- SpyGlass lint
#   make synth-saed14      -- DC synthesis with SAED14nm
#   make synth-asap7       -- DC synthesis with ASAP7
#   make synth-saed32      -- DC synthesis with SAED32
#   make synth-all         -- DC synthesis with all PDKs
#   make pt                -- PrimeTime analysis
#   make vcs-sim           -- VCS simulation
#   make compare           -- Multi-PDK comparison
#   make clean             -- Remove all output directories

SHELL := /bin/bash

# ---------------------------------------------------------------------------
# Configuration (override on command line or in environment)
# ---------------------------------------------------------------------------
DESIGN_NAME   ?= fabric_top
RTL_DIR       ?= $(CURDIR)/../../src/rtl/design
FILELIST      ?= $(RTL_DIR)/filelist.f
TB_FILE       ?=
CLOCK_PERIOD  ?= 10.0
OUTPUT_BASE   ?= $(CURDIR)/output

# Tool modules
DC_MODULE     ?= synopsys/syn/W-2024.09-SP5
PT_MODULE     ?= synopsys/prime/X-2025.06-SP2
VCS_MODULE    ?= synopsys/vcs/X-2025.06-SP1
SG_MODULE     ?= synopsys/spyglass/W-2024.09-SP1
VER_MODULE    ?= verilator/5.044

SCRIPT_DIR    := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Module loading helper
MODULE_LOAD   = source /etc/profile.d/modules.sh 2>/dev/null && module load

# ---------------------------------------------------------------------------
# Lint targets
# ---------------------------------------------------------------------------
.PHONY: lint lint-spyglass

lint:
	@echo "=== Verilator Lint ==="
	$(SCRIPT_DIR)/lint_verilator.sh "$(RTL_DIR)" "$(DESIGN_NAME)"

lint-spyglass:
	@echo "=== SpyGlass Lint ==="
	$(MODULE_LOAD) $(SG_MODULE) && \
	sg_shell -tcl "set DESIGN_NAME $(DESIGN_NAME); \
	               set RTL_DIR $(RTL_DIR); \
	               set FILELIST $(FILELIST); \
	               set OUTPUT_DIR $(OUTPUT_BASE)/spyglass; \
	               source $(SCRIPT_DIR)/lint_spyglass.tcl"

# ---------------------------------------------------------------------------
# Synthesis targets
# ---------------------------------------------------------------------------
.PHONY: synth-saed14 synth-asap7 synth-saed32 synth-all

synth-saed14:
	@echo "=== DC Synthesis: SAED14nm ==="
	python3 $(SCRIPT_DIR)/run_synthesis.py \
		--design $(DESIGN_NAME) \
		--rtl-dir $(RTL_DIR) \
		--filelist $(FILELIST) \
		--pdk saed14 \
		--clock-period $(CLOCK_PERIOD) \
		--output-dir $(OUTPUT_BASE)/synth_saed14

synth-asap7:
	@echo "=== DC Synthesis: ASAP7 ==="
	python3 $(SCRIPT_DIR)/run_synthesis.py \
		--design $(DESIGN_NAME) \
		--rtl-dir $(RTL_DIR) \
		--filelist $(FILELIST) \
		--pdk asap7 \
		--clock-period $(CLOCK_PERIOD) \
		--output-dir $(OUTPUT_BASE)/synth_asap7

synth-saed32:
	@echo "=== DC Synthesis: SAED32 ==="
	python3 $(SCRIPT_DIR)/run_synthesis.py \
		--design $(DESIGN_NAME) \
		--rtl-dir $(RTL_DIR) \
		--filelist $(FILELIST) \
		--pdk saed32 \
		--clock-period $(CLOCK_PERIOD) \
		--output-dir $(OUTPUT_BASE)/synth_saed32

synth-all: synth-saed14 synth-asap7 synth-saed32

# ---------------------------------------------------------------------------
# PrimeTime
# ---------------------------------------------------------------------------
.PHONY: pt

# Default: analyze SAED14nm synthesis output
PT_PDK        ?= saed14
PT_NETLIST_DIR ?= $(OUTPUT_BASE)/synth_$(PT_PDK)
PT_VCD        ?=
PT_SAIF       ?=

pt:
	@echo "=== PrimeTime Analysis ==="
	$(MODULE_LOAD) $(PT_MODULE) && \
	pt_shell -f $(SCRIPT_DIR)/pt_analyze.tcl \
		-x "set DESIGN_NAME $(DESIGN_NAME); \
		    set NETLIST_DIR $(PT_NETLIST_DIR); \
		    set PDK_TARGET $(PT_PDK); \
		    set OUTPUT_DIR $(OUTPUT_BASE)/pt_$(PT_PDK); \
		    set VCD_FILE {$(PT_VCD)}; \
		    set SAIF_FILE {$(PT_SAIF)}"

# ---------------------------------------------------------------------------
# VCS Simulation
# ---------------------------------------------------------------------------
.PHONY: vcs-sim

vcs-sim:
ifdef TB_FILE
	@echo "=== VCS Simulation ==="
	VCS_MODULE=$(VCS_MODULE) \
	$(SCRIPT_DIR)/vcs_sim.sh "$(RTL_DIR)" "$(TB_FILE)" "$(DESIGN_NAME)" \
		"$(OUTPUT_BASE)/vcs_sim"
else
	@echo "ERROR: TB_FILE not set. Usage: make vcs-sim TB_FILE=<testbench.sv>"
	@exit 1
endif

# ---------------------------------------------------------------------------
# Multi-PDK Comparison
# ---------------------------------------------------------------------------
.PHONY: compare

compare:
	@echo "=== Multi-PDK Comparison ==="
	python3 $(SCRIPT_DIR)/compare_pdks.py \
		--design $(DESIGN_NAME) \
		--rtl-dir $(RTL_DIR) \
		--filelist $(FILELIST) \
		--clock-period $(CLOCK_PERIOD) \
		--output-dir $(OUTPUT_BASE)/pdk_compare

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
.PHONY: clean clean-synth clean-pt clean-sim

clean-synth:
	rm -rf $(OUTPUT_BASE)/synth_*

clean-pt:
	rm -rf $(OUTPUT_BASE)/pt_*

clean-sim:
	rm -rf $(OUTPUT_BASE)/vcs_sim

clean:
	rm -rf $(OUTPUT_BASE)
