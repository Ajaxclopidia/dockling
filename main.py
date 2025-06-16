from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import uvicorn
import tempfile
import os
import json
import base64
import asyncio
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF for PDF handling

# Document processing imports
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

# Initialize FastAPI app
app = FastAPI(
    title="SmolDocling Document Processor",
    description="End-to-end document processing API using SmolDocling",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
processor = None
model = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pydantic models for API responses
class ProcessingResult(BaseModel):
    success: bool
    doctags: Optional[str] = None
    markdown: Optional[str] = None
    html: Optional[str] = None
    json_output: Optional[Dict[Any, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    page_count: Optional[int] = None

class BatchProcessingResult(BaseModel):
    success: bool
    results: List[ProcessingResult]
    total_pages: int
    total_processing_time: float
    error: Optional[str] = None

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str

# Startup event to load model
@app.on_event("startup")
async def startup_event():
    """Load SmolDocling model on startup"""
    global processor, model
    
    try:
        print("Loading SmolDocling model...")
        processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
        model = AutoModelForVision2Seq.from_pretrained(
            "ds4sd/SmolDocling-256M-preview",
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
        ).to(DEVICE)
        print(f"Model loaded successfully on {DEVICE}")
        
        # Warm up the model
        dummy_image = Image.new('RGB', (512, 512), color='white')
        await process_image_with_smoldocling(dummy_image, "Convert this page to docling.")
        print("Model warmed up successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    """Convert PDF pages to PIL Images"""
    images = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(BytesIO(img_data))
            images.append(image)
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")
    
    return images

async def process_image_with_smoldocling(image: Image.Image, prompt: str = "Convert this page to docling.") -> ProcessingResult:
    """Process a single image with SmolDocling"""
    import time
    start_time = time.time()
    
    try:
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Prepare inputs
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=prompt_text,
            images=[image],
            return_tensors="pt"
        ).to(DEVICE)
        
        # Generate DocTags
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False
            )
        
        # Decode output
        generated_text = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]
        
        processing_time = time.time() - start_time
        
        # Try to convert to different formats
        markdown_output = None
        html_output = None
        json_output = None
        
        try:
            # Parse DocTags to DoclingDocument
            doc_tags_doc = DocTagsDocument.model_validate_json(generated_text)
            docling_doc = doc_tags_doc.export_to_document_tokens()
            
            # Export to different formats
            markdown_output = docling_doc.export_to_markdown()
            html_output = docling_doc.export_to_html()
            json_output = json.loads(docling_doc.export_to_json())
            
        except Exception as conversion_error:
            print(f"Format conversion error: {conversion_error}")
            # If conversion fails, still return the DocTags
            pass
        
        return ProcessingResult(
            success=True,
            doctags=generated_text,
            markdown=markdown_output,
            html=html_output,
            json_output=json_output,
            processing_time=processing_time,
            page_count=1
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return ProcessingResult(
            success=False,
            error=str(e),
            processing_time=processing_time,
            page_count=1
        )

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SmolDocling Document Processor</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>SmolDocling Document Processor</h1>
            <p>Upload documents (PDF, images) for AI-powered text extraction and conversion.</p>
            
            <h3>Available Endpoints:</h3>
            <ul>
                <li><strong>POST /process-image</strong> - Process single image</li>
                <li><strong>POST /process-pdf</strong> - Process PDF document</li>
                <li><strong>POST /process-batch</strong> - Process multiple files</li>
                <li><strong>GET /health</strong> - Health check</li>
                <li><strong>GET /docs</strong> - API documentation</li>
            </ul>
            
            <div class="upload-area">
                <h3>Quick Test</h3>
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" id="fileInput" accept=".pdf,.jpg,.jpeg,.png" required><br><br>
                    <select id="outputFormat">
                        <option value="markdown">Markdown</option>
                        <option value="html">HTML</option>
                        <option value="json">JSON</option>
                        <option value="doctags">DocTags</option>
                    </select><br><br>
                    <button type="submit">Process Document</button>
                </form>
            </div>
            
            <div id="result" class="result" style="display:none;">
                <h3>Result:</h3>
                <pre id="resultContent"></pre>
            </div>
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const fileInput = document.getElementById('fileInput');
                const outputFormat = document.getElementById('outputFormat').value;
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select a file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('output_format', outputFormat);
                
                const endpoint = file.type === 'application/pdf' ? '/process-pdf' : '/process-image';
                
                try {
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('resultContent').textContent = JSON.stringify(result, null, 2);
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=DEVICE,
        version="1.0.0"
    )

@app.post("/process-image", response_model=ProcessingResult)
async def process_image(
    file: UploadFile = File(...),
    output_format: str = Form("markdown"),
    custom_prompt: Optional[str] = Form(None)
):
    """Process a single image file"""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Use custom prompt or default
        prompt = custom_prompt or "Convert this page to docling."
        
        # Process with SmolDocling
        result = await process_image_with_smoldocling(image, prompt)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/process-pdf", response_model=BatchProcessingResult)
async def process_pdf(
    file: UploadFile = File(...),
    output_format: str = Form("markdown"),
    custom_prompt: Optional[str] = Form(None),
    max_pages: Optional[int] = Form(50)
):
    """Process a PDF file (all pages)"""
    
    # Validate file type
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        import time
        start_time = time.time()
        
        # Read PDF and convert to images
        pdf_bytes = await file.read()
        images = pdf_to_images(pdf_bytes)
        
        # Limit pages if specified
        if max_pages and len(images) > max_pages:
            images = images[:max_pages]
        
        # Use custom prompt or default
        prompt = custom_prompt or "Convert this page to docling."
        
        # Process each page
        results = []
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}")
            result = await process_image_with_smoldocling(image, prompt)
            results.append(result)
        
        total_time = time.time() - start_time
        
        return BatchProcessingResult(
            success=True,
            results=results,
            total_pages=len(images),
            total_processing_time=total_time
        )
        
    except Exception as e:
        return BatchProcessingResult(
            success=False,
            results=[],
            total_pages=0,
            total_processing_time=0,
            error=str(e)
        )

@app.post("/process-batch", response_model=BatchProcessingResult)
async def process_batch(
    files: List[UploadFile] = File(...),
    output_format: str = Form("markdown"),
    custom_prompt: Optional[str] = Form(None)
):
    """Process multiple files in batch"""
    
    try:
        import time
        start_time = time.time()
        
        all_results = []
        total_pages = 0
        
        prompt = custom_prompt or "Convert this page to docling."
        
        for file in files:
            if file.content_type == 'application/pdf':
                # Process PDF
                pdf_bytes = await file.read()
                images = pdf_to_images(pdf_bytes)
                total_pages += len(images)
                
                for image in images:
                    result = await process_image_with_smoldocling(image, prompt)
                    all_results.append(result)
                    
            elif file.content_type.startswith('image/'):
                # Process image
                image_bytes = await file.read()
                image = Image.open(BytesIO(image_bytes)).convert('RGB')
                total_pages += 1
                
                result = await process_image_with_smoldocling(image, prompt)
                all_results.append(result)
            else:
                # Skip unsupported files
                all_results.append(ProcessingResult(
                    success=False,
                    error=f"Unsupported file type: {file.content_type}",
                    page_count=0
                ))
        
        total_time = time.time() - start_time
        
        return BatchProcessingResult(
            success=True,
            results=all_results,
            total_pages=total_pages,
            total_processing_time=total_time
        )
        
    except Exception as e:
        return BatchProcessingResult(
            success=False,
            results=[],
            total_pages=0,
            total_processing_time=0,
            error=str(e)
        )

@app.post("/extract-tables")
async def extract_tables(
    file: UploadFile = File(...),
    output_format: str = Form("json")
):
    """Specialized endpoint for table extraction"""
    
    try:
        if file.content_type == 'application/pdf':
            pdf_bytes = await file.read()
            images = pdf_to_images(pdf_bytes)
        else:
            image_bytes = await file.read()
            images = [Image.open(BytesIO(image_bytes)).convert('RGB')]
        
        all_tables = []
        
        for i, image in enumerate(images):
            result = await process_image_with_smoldocling(
                image, 
                "Extract all tables from this document with their structure and content."
            )
            
            if result.success and result.json_output:
                # Extract table information from the result
                page_tables = {
                    "page": i + 1,
                    "tables": result.json_output.get("tables", []),
                    "doctags": result.doctags
                }
                all_tables.append(page_tables)
        
        return {"success": True, "tables": all_tables}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Table extraction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Change this to your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )