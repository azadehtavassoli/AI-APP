"""
Smoke Test Script for Backend API

Tests basic functionality of the backend endpoints:
1. Health check
2. Query endpoint when KB is empty
3. Text ingestion
4. Query endpoint after ingestion
5. Sources listing

Run this after starting the backend to verify everything works.

Usage:
    python test_backend_smoke.py

Prerequisites:
    - Backend running on http://localhost:8000
    - OpenAI API key configured in backend/.env
"""

import requests
import sys
from typing import Dict, Any

BACKEND_URL = "http://localhost:8000"


def test_health() -> bool:
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
        
        if response.status_code != 200:
            print(f"❌ FAILED: Status code {response.status_code}")
            return False
        
        data = response.json()
        print(f"✅ PASSED")
        print(f"   Status: {data.get('status')}")
        print(f"   Version: {data.get('version')}")
        print(f"   OpenAI Configured: {data.get('openai_configured')}")
        
        if not data.get('openai_configured'):
            print("⚠️  WARNING: OpenAI API key not configured!")
        
        return True
    
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_query_empty_kb() -> bool:
    """Test query endpoint when knowledge base is empty"""
    print("\n" + "="*60)
    print("TEST 2: Query with Empty Knowledge Base")
    print("="*60)
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/query",
            json={
                "question": "What is the deadline?",
                "top_k": 5
            },
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ FAILED: Status code {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        data = response.json()
        answer = data.get('answer', '')
        citations = data.get('citations', [])
        
        print(f"✅ PASSED")
        print(f"   Answer length: {len(answer)} chars")
        print(f"   Citations: {len(citations)}")
        print(f"   Answer preview: {answer[:100]}...")
        
        # Should have empty citations when KB is empty
        if len(citations) > 0:
            print(f"⚠️  WARNING: Expected no citations, got {len(citations)}")
        
        return True
    
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_ingest_text() -> bool:
    """Test text ingestion"""
    print("\n" + "="*60)
    print("TEST 3: Ingest Text")
    print("="*60)
    
    test_text = """
    German Visa Application Deadline Notice
    
    Your visa application must be submitted by March 15, 2026.
    Please bring the following documents:
    - Valid passport
    - Completed application form
    - Proof of residence
    - Financial statements
    
    The Ausländerbehörde will review your application within 4 weeks.
    If you have questions, contact us at info@example.de
    """
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/ingest/text",
            json={
                "text": test_text,
                "source_name": "Test Visa Notice"
            },
            timeout=30
        )
        
        if response.status_code != 201:
            print(f"❌ FAILED: Status code {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        data = response.json()
        
        if not data.get('success'):
            print(f"❌ FAILED: Success flag is False")
            return False
        
        source = data.get('source', {})
        print(f"✅ PASSED")
        print(f"   Source ID: {source.get('id', '')[:16]}...")
        print(f"   Chunks: {source.get('chunks', 0)}")
        print(f"   Characters: {source.get('characters', 0)}")
        
        return True
    
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_query_with_docs() -> bool:
    """Test query endpoint after ingestion"""
    print("\n" + "="*60)
    print("TEST 4: Query After Ingestion")
    print("="*60)
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/query",
            json={
                "question": "What is the deadline for the visa application?",
                "top_k": 5
            },
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"❌ FAILED: Status code {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        data = response.json()
        answer = data.get('answer', '')
        citations = data.get('citations', [])
        
        print(f"✅ PASSED")
        print(f"   Answer length: {len(answer)} chars")
        print(f"   Citations: {len(citations)}")
        print(f"   Answer: {answer[:200]}...")
        
        # Should have citations when docs exist
        if len(citations) == 0:
            print(f"⚠️  WARNING: Expected citations, got none")
        else:
            print(f"\n   Citations:")
            for i, citation in enumerate(citations[:2], 1):
                print(f"     {i}. {citation.get('source_uri')} - {citation.get('snippet', '')[:60]}...")
        
        # Check if answer mentions deadline
        if "march" not in answer.lower() and "deadline" not in answer.lower():
            print(f"⚠️  WARNING: Answer doesn't mention deadline")
        
        return True
    
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_list_sources() -> bool:
    """Test sources listing endpoint"""
    print("\n" + "="*60)
    print("TEST 5: List Sources")
    print("="*60)
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/sources", timeout=10)
        
        if response.status_code != 200:
            print(f"❌ FAILED: Status code {response.status_code}")
            return False
        
        data = response.json()
        sources = data.get('sources', [])
        total = data.get('total_count', 0)
        
        print(f"✅ PASSED")
        print(f"   Total sources: {total}")
        
        if total > 0:
            print(f"   Sources:")
            for source in sources[:3]:
                print(f"     - {source.get('source_name')} ({source.get('source_type')})")
                print(f"       Chunks: {source.get('chunk_count')}")
        
        return True
    
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("BACKEND API SMOKE TESTS")
    print("="*60)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Testing endpoints...")
    
    results = []
    
    # Run tests in sequence
    results.append(("Health Check", test_health()))
    results.append(("Query Empty KB", test_query_empty_kb()))
    results.append(("Ingest Text", test_ingest_text()))
    results.append(("Query After Ingestion", test_query_with_docs()))
    results.append(("List Sources", test_list_sources()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:12} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
