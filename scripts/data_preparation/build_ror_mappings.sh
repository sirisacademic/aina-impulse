#!/usr/bin/env bash
# Build ROR mappings from the latest available ROR dump
# 
# Usage:
#   ./build_ror_mappings.sh [--country COUNTRY_CODE]
#
# Example:
#   ./build_ror_mappings.sh --country ES

set -euo pipefail

# Default values
COUNTRY_CODE="ES"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROR_DIR="$PROJECT_ROOT/data/ror"
OUTPUT_FILE="$PROJECT_ROOT/data/normalization/ror_mappings.json"
SCRIPT_DIR="$PROJECT_ROOT/scripts/data_preparation"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --country)
            COUNTRY_CODE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--country COUNTRY_CODE]"
            echo ""
            echo "Options:"
            echo "  --country CODE    Filter by country code (default: ES)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "ROR Mappings Builder"
echo "========================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "ROR directory: $ROR_DIR"
echo "Country filter: $COUNTRY_CODE"
echo ""

# Check if ROR directory exists
if [ ! -d "$ROR_DIR" ]; then
    echo "❌ Error: ROR directory not found: $ROR_DIR"
    echo "Please create it with: mkdir -p $ROR_DIR"
    exit 1
fi

# Find the latest ROR dump file (schema v2)
echo "🔍 Looking for ROR dump files..."
ROR_DUMP=$(find "$ROR_DIR" -name "*ror-data_schema_v2.json" -type f | sort -r | head -n 1)

if [ -z "$ROR_DUMP" ]; then
    echo "❌ Error: No ROR schema v2 dump file found in $ROR_DIR"
    echo ""
    echo "Expected filename pattern: *ror-data_schema_v2.json"
    echo ""
    echo "Please download the latest ROR dump:"
    echo "1. Visit: https://zenodo.org/records/17468391"
    echo "2. Download: v*.*.***-ror-data.zip"
    echo "3. Extract to: $ROR_DIR"
    echo "4. Re-run this script"
    exit 1
fi

echo "✅ Found ROR dump: $(basename "$ROR_DUMP")"

# Check file size
FILE_SIZE=$(du -h "$ROR_DUMP" | cut -f1)
echo "   File size: $FILE_SIZE"

# Extract version and date from filename
FILENAME=$(basename "$ROR_DUMP")
if [[ $FILENAME =~ v([0-9.]+)-([0-9-]+)-ror-data_schema_v2\.json ]]; then
    VERSION="${BASH_REMATCH[1]}"
    DATE="${BASH_REMATCH[2]}"
    echo "   Version: v$VERSION"
    echo "   Date: $DATE"
fi

echo ""

# Check if Python script exists
PYTHON_SCRIPT="$SCRIPT_DIR/build_ror_mappings.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Run the Python script
echo "🔨 Building organization mappings..."
echo "   Input: $ROR_DUMP"
echo "   Output: $OUTPUT_FILE"
echo "   Filter: $COUNTRY_CODE"
echo ""

python3 "$PYTHON_SCRIPT" \
    --ror-dump "$ROR_DUMP" \
    --output "$OUTPUT_FILE" \
    --filter-country "$COUNTRY_CODE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ Success!"
    echo "========================================"
    echo ""
    echo "Mappings saved to: $OUTPUT_FILE"
    echo ""
    
    # Show file info
    if [ -f "$OUTPUT_FILE" ]; then
        OUTPUT_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "Output file size: $OUTPUT_SIZE"
        
        # Extract statistics from the output file
        if command -v jq &> /dev/null; then
            echo ""
            echo "Statistics:"
            jq -r '.statistics | 
                "  Organizations: \(.total_organizations)\n" +
                "  Name variants: \(.total_name_variants)\n" +
                "  Acronyms: \(.total_acronyms)\n" +
                "  Aliases: \(.total_aliases)\n" +
                "  Avg names/org: \(.avg_names_per_org | tonumber | floor)"' "$OUTPUT_FILE"
        fi
    fi
    
    echo ""
    echo "Next steps:"
    echo "  1. Update src/impulse/normalization.py with ROR functions"
    echo "  2. Restart the API to use new mappings"
    echo "  3. Test organization matching"
else
    echo ""
    echo "========================================"
    echo "❌ Failed to build mappings"
    echo "========================================"
    echo ""
    echo "Exit code: $EXIT_CODE"
    echo "Check the error messages above for details."
    exit $EXIT_CODE
fi
