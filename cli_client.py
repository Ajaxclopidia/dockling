#!/usr/bin/env python3
"""
CLI Client for SmolDocling FastAPI Application
Usage: python cli_client.py [command] [options]
"""

import argparse
import requests
import json
import os
import sys
from pathlib import Path
import time

class SmolDoclingClient:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Health check failed: {e}")
    
    def process_file(self, file_path, output_format="markdown", custom_prompt=None, max_pages=None):
        """Process a single file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine endpoint based on file type
        if file_path.suffix.lower() == '.pdf':
            endpoint = "/process-pdf"
            content_type = "application/pdf"
        elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            endpoint = "/process-image"
            content_type = f"image/{file_path.suffix.lower().lstrip('.')}"
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Prepare request
        with open(file_path, 'rb') as f:
            files = {"file": (file_path.name, f, content_type)}
            data = {"output_format": output_format}
            
            if custom_prompt:
                data["custom_prompt"] = custom_prompt
            if max_pages and endpoint == "/process-pdf":
                data["max_pages"] = max_pages
            
            response = requests.post(f"{self.base_url}{endpoint}", files=files, data=data)
            response.raise_for_status()
            
            return response.json()
    
    def process_batch(self, file_paths, output_format="markdown", custom_prompt=None):
        """Process multiple files"""
        files = []
        
        try:
            for file_path in file_paths:
                file_path = Path(file_path)
                if not file_path.exists():
                    print(f"Warning: File not found: {file_path}")
                    continue
                
                # Determine content type
                if file_path.suffix.lower() == '.pdf':
                    content_type = "application/pdf"
                elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    content_type = f"image/{file_path.suffix.lower().lstrip('.')}"
                else:
                    print(f"Warning: Skipping unsupported file: {file_path}")
                    continue
                
                files.append(("files", (file_path.name, open(file_path, 'rb'), content_type)))
            
            if not files:
                raise ValueError("No valid files to process")
            
            data = {"output_format": output_format}
            if custom_prompt:
                data["custom_prompt"] = custom_prompt
            
            response = requests.post(f"{self.base_url}/process-batch", files=files, data=data)
            response.raise_for_status()
            
            return response.json()
        
        finally:
            # Close all file handles
            for _, (_, file_obj, _) in files:
                if hasattr(file_obj, 'close'):
                    file_obj.close()
    
    def extract_tables(self, file_path):
        """Extract tables from a document"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine content type
        if file_path.suffix.lower() == '.pdf':
            content_type = "application/pdf"
        elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            content_type = f"image/{file_path.suffix.lower().lstrip('.')}"
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        with open(file_path, 'rb') as f:
            files = {"file": (file_path.name, f, content_type)}
            data = {"output_format": "json"}
            
            response = requests.post(f"{self.base_url}/extract-tables", files=files, data=data)
            response.raise_for_status()
            
            return response.json()

def save_output(result, output_file, format_type):
    """Save processing result to file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine what to save based on format
    if format_type == "markdown" and result.get('markdown'):
        content = result['markdown']
    elif format_type == "html" and result.get('html'):
        content = result['html']
    elif format_type == "json" and result.get('json_output'):
        content = json.dumps(result['json_output'], indent=2)
    elif format_type == "doctags" and result.get('doctags'):
        content = result['doctags']
    else:
        # Fallback to full JSON result
        content = json.dumps(result, indent=2)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="SmolDocling CLI Client")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Health check command
    health_parser = subparsers.add_parser("health", help="Check API health")
    
    # Process file command
    process_parser = subparsers.add_parser("process", help="Process a single file")
    process_parser.add_argument("file", help="Input file path")
    process_parser.add_argument("-f", "--format", default="markdown", 
                               choices=["markdown", "html", "json", "doctags"],
                               help="Output format")
    process_parser.add_argument("-p", "--prompt", help="Custom prompt")
    process_parser.add_argument("-m", "--max-pages", type=int, help="Max pages for PDF")
    process_parser.add_argument("-o", "--output", help="Output file path")
    
    # Batch process command
    batch_parser = subparsers.add_parser("batch", help="Process multiple files")
    batch_parser.add_argument("files", nargs="+", help="Input file paths")
    batch_parser.add_argument("-f", "--format", default="markdown",
                             choices=["markdown", "html", "json", "doctags"],
                             help="Output format")
    batch_parser.add_argument("-p", "--prompt", help="Custom prompt")
    batch_parser.add_argument("-o", "--output-dir", help="Output directory")
    
    # Table extraction command
    tables_parser = subparsers.add_parser("tables", help="Extract tables from document")
    tables_parser.add_argument("file", help="Input file path")
    tables_parser.add_argument("-o", "--output", help="Output file path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    client = SmolDoclingClient(args.url)
    
    try:
        if args.command == "health":
            health = client.health_check()
            print("API Health Status:")
            print(json.dumps(health, indent=2))
            
        elif args.command == "process":
            print(f"Processing file: {args.file}")
            start_time = time.time()
            
            result = client.process_file(
                args.file,
                output_format=args.format,
                custom_prompt=args.prompt,
                max_pages=args.max_pages
            )
            
            end_time = time.time()
            
            if result.get('success'):
                print(f"✅ Processing completed in {end_time - start_time:.2f}s")
                print(f"⏱️  Model processing time: {result.get('processing_time', 0):.2f}s")
                
                if args.output:
                    save_output(result, args.output, args.format)
                else:
                    # Print to stdout
                    if args.format == "markdown" and result.get('markdown'):
                        print("\n" + "="*50)
                        print("MARKDOWN OUTPUT:")
                        print("="*50)
                        print(result['markdown'])
                    elif args.format == "html" and result.get('html'):
                        print("\n" + "="*50)
                        print("HTML OUTPUT:")
                        print("="*50)
                        print(result['html'])
                    elif args.format == "json" and result.get('json_output'):
                        print("\n" + "="*50)
                        print("JSON OUTPUT:")
                        print("="*50)
                        print(json.dumps(result['json_output'], indent=2))
                    elif args.format == "doctags" and result.get('doctags'):
                        print("\n" + "="*50)
                        print("DOCTAGS OUTPUT:")
                        print("="*50)
                        print(result['doctags'])
            else:
                print(f"❌ Processing failed: {result.get('error')}")
                sys.exit(1)
                
        elif args.command == "batch":
            print(f"Processing {len(args.files)} files...")
            start_time = time.time()
            
            result = client.process_batch(
                args.files,
                output_format=args.format,
                custom_prompt=args.prompt
            )
            
            end_time = time.time()
            
            if result.get('success'):
                print(f"✅ Batch processing completed in {end_time - start_time:.2f}s")
                print(f"📄 Total pages: {result.get('total_pages', 0)}")
                print(f"⏱️  Total processing time: {result.get('total_processing_time', 0):.2f}s")
                
                if args.output_dir:
                    output_dir = Path(args.output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    for i, page_result in enumerate(result['results']):
                        if page_result.get('success'):
                            output_file = output_dir / f"page_{i+1}.{args.format}"
                            save_output(page_result, output_file, args.format)
                else:
                    # Print summary
                    successful = sum(1 for r in result['results'] if r.get('success'))
                    print(f"📊 Successfully processed: {successful}/{len(result['results'])} pages")
            else:
                print(f"❌ Batch processing failed: {result.get('error')}")
                sys.exit(1)
                
        elif args.command == "tables":
            print(f"Extracting tables from: {args.file}")
            
            result = client.extract_tables(args.file)
            
            if result.get('success'):
                tables = result.get('tables', [])
                print(f"✅ Found {len(tables)} pages with potential tables")
                
                if args.output:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2)
                    print(f"Tables data saved to: {args.output}")
                else:
                    print(json.dumps(result, indent=2))
            else:
                print(f"❌ Table extraction failed")
                sys.exit(1)
                
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()