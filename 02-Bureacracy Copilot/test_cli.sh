#!/bin/bash
# CLI Integration Test Script
#
# This script tests all CLI functionality to ensure proper operation.
# Run this after implementing new features to validate the CLI.
#
# Usage: ./test_cli.sh
#
# Note: For Windows, run with Git Bash or WSL

set -e  # Exit on error

# Configuration
PYTHON="python"
CLI="cli.py"
BACKEND_URL="http://localhost:8000"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counter
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
    ((TESTS_PASSED++))
}

log_error() {
    echo -e "${RED}✗${NC} $1"
    ((TESTS_FAILED++))
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

run_test() {
    local test_name="$1"
    local command="$2"
    local expected_exit_code="${3:-0}"
    
    ((TESTS_RUN++))
    
    log_info "Test $TESTS_RUN: $test_name"
    
    if eval "$command" > /dev/null 2>&1; then
        actual_exit_code=$?
    else
        actual_exit_code=$?
    fi
    
    if [ $actual_exit_code -eq $expected_exit_code ]; then
        log_success "$test_name"
    else
        log_error "$test_name (expected exit code $expected_exit_code, got $actual_exit_code)"
    fi
}

# Main test suite
main() {
    echo "================================================"
    echo "CLI Integration Test Suite"
    echo "================================================"
    echo ""
    
    log_info "Backend URL: $BACKEND_URL"
    echo ""
    
    # Test 1: Health check
    echo "--- Basic Operations ---"
    run_test "Health check" "$PYTHON $CLI health"
    
    # Test 2: Health check verbose mode
    run_test "Health check (verbose)" "$PYTHON $CLI --verbose health"
    
    # Test 3: Health check JSON output
    run_test "Health check (JSON)" "$PYTHON $CLI --json health"
    
    echo ""
    
    # Test 4: List sources (may be empty)
    echo "--- Source Management ---"
    run_test "List sources" "$PYTHON $CLI sources list"
    
    # Test 5: List sources JSON
    run_test "List sources (JSON)" "$PYTHON $CLI --json sources list"
    
    echo ""
    
    # Test 6: Ingest text
    echo "--- Document Ingestion ---"
    run_test "Ingest text document" \
        "$PYTHON $CLI ingest text 'This is a comprehensive test document for the CLI integration testing suite. It contains multiple sentences to ensure proper chunking and processing.' --source 'CLI Test Document'"
    
    # Test 7: Verify ingestion by listing sources again
    run_test "Verify text ingestion" \
        "$PYTHON $CLI --json sources list | grep -q 'CLI Test Document'"
    
    echo ""
    
    # Test 8: URL ingestion (optional, requires internet)
    log_info "Test $((TESTS_RUN + 1)): URL ingestion (skipped - optional)"
    log_warning "Skipping URL test (requires internet and may be slow)"
    
    # Uncomment to test URL ingestion:
    # run_test "Ingest URL" \
    #     "$PYTHON $CLI ingest url 'https://example.com'"
    
    echo ""
    
    # Test 9: PDF ingestion (optional, requires test PDF)
    log_info "Test $((TESTS_RUN + 1)): PDF ingestion (skipped - requires test file)"
    log_warning "Skipping PDF test (requires test PDF file)"
    
    # Uncomment and provide test PDF to test:
    # run_test "Ingest PDF" \
    #     "$PYTHON $CLI ingest pdf test_document.pdf"
    
    echo ""
    
    # Test 10: Clear knowledge base
    echo "--- Cleanup Operations ---"
    log_info "Test $((TESTS_RUN + 1)): Clear knowledge base (skipped - preserves data)"
    log_warning "Skipping clear test (would delete all data)"
    
    # Uncomment to test clearing:
    # run_test "Clear knowledge base" \
    #     "$PYTHON $CLI sources clear --confirm"
    
    echo ""
    echo "================================================"
    echo "Test Summary"
    echo "================================================"
    echo "Total tests run:    $TESTS_RUN"
    echo -e "Passed:             ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Failed:             ${RED}$TESTS_FAILED${NC}"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo ""
        log_success "All tests passed! 🎉"
        exit 0
    else
        echo ""
        log_error "Some tests failed. Please check the output above."
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        log_error "Python not found. Please install Python 3.10 or higher."
        exit 1
    fi
    
    # Check CLI file exists
    if [ ! -f "$CLI" ]; then
        log_error "CLI file not found: $CLI"
        log_info "Make sure you're running this from the project root directory."
        exit 1
    fi
    
    # Check backend connectivity
    log_info "Checking backend connectivity..."
    if ! curl -s "$BACKEND_URL/api/health" > /dev/null 2>&1; then
        log_error "Cannot connect to backend at $BACKEND_URL"
        log_info "Please start the backend: cd backend && uvicorn main:app --reload"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
    echo ""
}

# Run prerequisite checks
check_prerequisites

# Run main test suite
main
