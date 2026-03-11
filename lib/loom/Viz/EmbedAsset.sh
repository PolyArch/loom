#!/bin/bash
# EmbedAsset.sh -- Convert a file into a C byte-array source using xxd.
# Usage: EmbedAsset.sh <input_file> <symbol_name> <output.cpp> <output.h>
set -e

INPUT="$1"
SYMBOL="$2"
OUT_C="$3"
OUT_H="$4"

FILE_SIZE=$(stat -c%s "$INPUT" 2>/dev/null || stat -f%z "$INPUT")

# Generate header
cat > "$OUT_H" <<EOF
#pragma once
extern const unsigned char ${SYMBOL}[];
extern const unsigned int ${SYMBOL}_len;
EOF

# Generate source using xxd -i (stdin mode: outputs raw hex bytes)
{
  echo "// Auto-generated from $(basename "$INPUT")"
  echo "#include \"$(basename "$OUT_H")\""
  echo "const unsigned char ${SYMBOL}[] = {"
  xxd -i < "$INPUT"
  echo ", 0x00"
  echo "};"
  echo "const unsigned int ${SYMBOL}_len = ${FILE_SIZE};"
} > "$OUT_C"
