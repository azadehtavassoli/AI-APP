"""Quick verification of invalid date error handling fix"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from src.services.rag import generate_answer

# Test the critical invalid date case
print("="*60)
print("CRITICAL TEST: Invalid Date Error Handling")
print("="*60)

question = "Start date 31.02.2026, add 5 days"
print(f"\nQuestion: {question}\n")

result = generate_answer(question)

print("Answer:")
print("-"*60)
print(result['answer'])
print("-"*60)

# Verification
answer_lower = result['answer'].lower()
no_info_check = "don't have enough information" in answer_lower

print("\n✅ Verification Checks:")
print(f"  - Has tool_results: {result.get('tool_results') is not None}")
print(f"  - Retrieved chunks: {result['retrieved_count']} (should be 0)")
print(f"  - Says 'no info': {no_info_check}")
print(f"  - Explains error: {any(w in answer_lower for w in ['invalid', 'not exist', 'incorrect', 'error'])}")
print(f"  - Lists formats: {'format' in answer_lower or 'yyyy-mm-dd' in answer_lower}")

# Final verdict
if result['retrieved_count'] == 0 and not no_info_check:
    print("\n✅ FIX VERIFIED - Tool error properly handled!")
else:
    print("\n❌ FIX FAILED - Check output above")
    sys.exit(1)
