#!/bin/bash
#
# Standalone script to run ruff and yapf on Python files
# Usage: ./scripts/format_code.sh [files...]
#   If no files provided, formats all Python files in src/ and tests/
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed or not in PATH${NC}"
    exit 1
fi

# Determine which files to format
if [ $# -eq 0 ]; then
    # No arguments: format all Python files in src/ and tests/
    echo -e "${YELLOW}Formatting all Python files in src/ and tests/...${NC}"
    FILES="src/ tests/"
else
    # Arguments provided: format only those files
    FILES="$@"
    echo -e "${YELLOW}Formatting specified files...${NC}"
fi

# Run ruff check and fix
echo -e "\n${YELLOW}Running ruff check...${NC}"
uvx ruff check --fix $FILES

# Run ruff format
echo -e "\n${YELLOW}Running ruff format...${NC}"
uvx ruff format $FILES

# Run yapf
echo -e "\n${YELLOW}Running yapf...${NC}"
uv run yapf --recursive --in-place --parallel $FILES

echo -e "\n${GREEN}Formatting complete!${NC}"

# Optionally check mkdocs build
if [ "$CHECK_MKDOCS" = "true" ] || [ "$1" = "--check-mkdocs" ]; then
    echo -e "\n${YELLOW}Checking mkdocs build...${NC}"
    if uv run mkdocs build --strict; then
        echo -e "${GREEN}MkDocs build successful!${NC}"
    else
        echo -e "${RED}MkDocs build failed!${NC}"
        exit 1
    fi
fi

