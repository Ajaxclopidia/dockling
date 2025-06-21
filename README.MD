# PDF Extractor Docker Setup Guide

## Quick Start

1. **Build and run the application:**
   ```bash
   docker-compose up --build
   ```

2. **Access the API:**
   - API Documentation: http://localhost:8010/docs
   - File Browser: http://localhost:8080 (optional)

## Project Structure
```
your-project/
├── extractor.py
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── extracted_api_output/    # Created automatically
└── input_pdfs/             # Optional: for organizing input files
```

## Usage

### Single PDF Extraction
```bash
curl -X POST "http://localhost:8010/extract/pdf" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your-document.pdf"
```

### Multiple PDF Extraction
```bash
curl -X POST "http://localhost:8010/extract/pdfs" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@document1.pdf" \
     -F "files=@document2.pdf"
```

## Docker Commands

### Build only
```bash
docker-compose build
```

### Run in background
```bash
docker-compose up -d
```

### View logs
```bash
docker-compose logs -f pdf-extractor
```

### Stop services
```bash
docker-compose down
```

### Clean up (remove volumes)
```bash
docker-compose down -v
```

## Volumes

- `./extracted_api_output` - Stores all extracted content (JSON, TXT, images)
- `./input_pdfs` - Optional directory for organizing input PDF files

## Services

1. **pdf-extractor** - Main FastAPI application on port 8010
2. **filebrowser** - Optional web-based file browser on port 8080

## Environment Variables

You can customize the application by adding environment variables to the docker-compose.yml:

```yaml
environment:
  - PYTHONPATH=/app
  - LOG_LEVEL=INFO
  - MAX_FILE_SIZE=100MB  # If you want to limit upload size
```

## Troubleshooting

### OCR Issues
If OCR fails, ensure tesseract is properly installed in the container. The Dockerfile includes tesseract-ocr and English language pack.

### Memory Issues
For large PDFs, you might need to increase Docker's memory limit:
```bash
docker-compose up --build --memory=2g
```

### Permission Issues
If you encounter permission issues with volumes:
```bash
sudo chown -R $USER:$USER extracted_api_output/
```

## Development Mode

For development with hot reload:
```bash
docker-compose up --build
```

The `--reload` flag is already included in the Dockerfile CMD.