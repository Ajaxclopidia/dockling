#!/usr/bin/env python3
"""
Test script for SmolDocling FastAPI application
"""

import requests
import json
import time
import os
from PIL import Image, ImageDraw, ImageFont
import io

API_BASE_URL = "http://localhost:8010"

def create_test_image():
    """Create a simple test image with text"""
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Add some text content
    text_content = [
        "Test Document",
        "",
        "This is a test document for SmolDocling API.",
        "It contains multiple lines of text.",
        "",
        "Section 1: Introduction",
        "This section introduces the document.",
        "",
        "Section 2: Content",
        "This section contains the main content.",
        "",
        "Table Example:",
        "Name    | Age | City",
        "--------|-----|-------",
        "Alice   | 25  | NYC",
        "Bob     | 30  | LA",
        "",
        "Conclusion:",
        "This concludes our test document."
    ]
    
    y_position = 50
    for line in text_content:
        if line.startswith("Test Document"):
            # Title
            try:
                title_font = ImageFont.truetype("arial.ttf", 32)
            except:
                title_font = font
            draw.text((50, y_position), line, fill='black', font=title_font)
            y_position += 40
        elif line.startswith("Section"):
            # Section headers
            try:
                section_font = ImageFont.truetype("arial.ttf", 28)
            except:
                section_font = font
            draw.text((50, y_position), line, fill='blue', font=section_font)
            y_position += 35
        else:
            # Regular text
            draw.text((50, y_position), line, fill='black', font=font)
            y_position += 25
    
    return img

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_process_image():
    """Test image processing endpoint"""
    print("\n📷 Testing image processing...")
    
    # Create test image
    test_img = create_test_image()
    img_buffer = io.BytesIO()
    test_img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    try:
        files = {"file": ("test_image.png", img_buffer, "image/png")}
        data = {"output_format": "markdown"}
        
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/process-image", files=files, data=data)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Image processing completed in {end_time - start_time:.2f}s")
            print(f"📄 Success: {result.get('success')}")
            print(f"⏱️  Processing time: {result.get('processing_time', 0):.2f}s")
            
            if result.get('markdown'):
                print(f"📝 Markdown output preview:")
                print(result['markdown'][:300] + "..." if len(result['markdown']) > 300 else result['markdown'])
            
            return True
        else:
            print(f"❌ Image processing failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        return False

def test_custom_prompt():
    """Test custom prompt functionality"""
    print("\n🎯 Testing custom prompt...")
    
    # Create test image
    test_img = create_test_image()
    img_buffer = io.BytesIO()
    test_img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    try:
        files = {"file": ("test_image.png", img_buffer, "image/png")}
        data = {
            "output_format": "json",
            "custom_prompt": "Extract all section headers and any table data from this document."
        }
        
        response = requests.post(f"{API_BASE_URL}/process-image", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Custom prompt processing completed")
            print(f"📄 Success: {result.get('success')}")
            
            if result.get('doctags'):
                print(f"🏷️  DocTags preview:")
                print(result['doctags'][:200] + "..." if len(result['doctags']) > 200 else result['doctags'])
            
            return True
        else:
            print(f"❌ Custom prompt processing failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Custom prompt error: {e}")
        return False

def test_table_extraction():
    """Test table extraction endpoint"""
    print("\n📊 Testing table extraction...")
    
    # Create test image with more table content
    img = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw a table
    table_data = [
        "Employee Data Table",
        "",
        "| Name     | Department | Salary | Years |",
        "|----------|------------|--------|-------|",
        "| Alice    | Engineering| $75,000| 3     |",
        "| Bob      | Marketing  | $65,000| 2     |",
        "| Charlie  | Sales      | $55,000| 1     |",
        "| Diana    | HR         | $60,000| 4     |"
    ]
    
    y_pos = 50
    for line in table_data:
        if line.startswith("Employee"):
            try:
                title_font = ImageFont.truetype("arial.ttf", 24)
            except:
                title_font = font
            draw.text((50, y_pos), line, fill='black', font=title_font)
        else:
            draw.text((50, y_pos), line, fill='black', font=font)
        y_pos += 30
    
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    try:
        files = {"file": ("table_test.png", img_buffer, "image/png")}
        data = {"output_format": "json"}
        
        response = requests.post(f"{API_BASE_URL}/extract-tables", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Table extraction completed")
            print(f"📄 Success: {result.get('success')}")
            print(f"📊 Tables found: {len(result.get('tables', []))}")
            return True
        else:
            print(f"❌ Table extraction failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Table extraction error: {e}")
        return False

def test_api_documentation():
    """Test API documentation endpoint"""
    print("\n📚 Testing API documentation...")
    try:
        response = requests.get(f"{API_BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ API documentation is accessible")
            return True
        else:
            print(f"❌ API documentation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API documentation error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting SmolDocling API Tests")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("API Documentation", test_api_documentation),
        ("Image Processing", test_process_image),
        ("Custom Prompt", test_custom_prompt),
        ("Table Extraction", test_table_extraction),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
        
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your SmolDocling API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print(f"🌐 API is running at {API_BASE_URL}")
    except:
        print(f"❌ API is not running at {API_BASE_URL}")
        print("Make sure to start the API first with: python main.py")
        exit(1)
    
    # Run tests
    success = run_all_tests()
    exit(0 if success else 1)