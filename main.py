from fastapi import FastAPI, UploadFile, File
from typing import List
import shutil
import os
import uuid

from extractor import PDFExtractor  # We will split logic into `extractor.py`

app = FastAPI()

BASE_OUTPUT_DIR = "extracted_api_output"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


@app.post("/extract/pdf")
async def extract_single_pdf(file: UploadFile = File(...)):
    file_id = uuid.uuid4().hex
    pdf_path = os.path.join(BASE_OUTPUT_DIR, f"{file_id}.pdf")
    
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_dir = os.path.join(BASE_OUTPUT_DIR, file_id)
    extractor = PDFExtractor(pdf_path, output_dir=output_dir)
    content = extractor.extract_all_content()
    extractor.save_extracted_content(content, "json")
    extractor.save_extracted_content(content, "txt")
    extractor.close_document()

    return {
        "message": "PDF extracted successfully.",
        "file_id": file_id,
        "output_dir": output_dir
    }


@app.post("/extract/pdfs")
async def extract_multiple_pdfs(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        file_id = uuid.uuid4().hex
        pdf_path = os.path.join(BASE_OUTPUT_DIR, f"{file_id}.pdf")

        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_dir = os.path.join(BASE_OUTPUT_DIR, file_id)
        extractor = PDFExtractor(pdf_path, output_dir=output_dir)
        content = extractor.extract_all_content()
        extractor.save_extracted_content(content, "json")
        extractor.save_extracted_content(content, "txt")
        extractor.close_document()

        results.append({
            "filename": file.filename,
            "file_id": file_id,
            "output_dir": output_dir
        })

    return {
        "message": "Batch PDF extraction completed.",
        "results": results
    }
