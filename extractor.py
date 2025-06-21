import fitz
import os
import json
from pathlib import Path
from PIL import Image
import pytesseract

class PDFExtractor:
    def __init__(self, pdf_path, output_dir="extracted_content"):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.doc = None
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def open_document(self):
        try:
            self.doc = fitz.open(self.pdf_path)
            return True
        except Exception as e:
            print(f"Error opening PDF: {e}")
            return False

    def extract_text(self, page_num=None):
        if not self.doc: return {}

        def clean_dict(d):
            if not isinstance(d, dict): return {}

            def clean_lines(lines): return [
                {k: v for k, v in l.items() if isinstance(v, (str, int, float, list))} |
                {"spans": [{sk: sv for sk, sv in s.items() if isinstance(sv, (str, int, float))}
                           for s in l.get("spans", [])]} if "spans" in l else {}
                for l in lines if isinstance(l, dict)
            ]

            return {
                k: ([{kk: vv if kk != "lines" else clean_lines(vv)
                      for kk, vv in b.items() if isinstance(vv, (str, int, float, list, dict))}
                     for b in v] if k == "blocks" else v)
                for k, v in d.items() if isinstance(v, (str, int, float, list, dict))
            }

        def clean_blocks(blks): return [
            dict(x0=float(b[0]), y0=float(b[1]), x1=float(b[2]), y1=float(b[3]),
                 text=str(b[4]), block_type=int(b[6]) if len(b) > 6 else 0)
            for b in blks if isinstance(b, (list, tuple)) and len(b) >= 5
        ]

        pages = range(len(self.doc)) if page_num is None else [page_num]
        return {
            f"page_{i}": {
                "text": (p := self.doc[i]).get_text(),
                "text_dict": clean_dict(p.get_text("dict")),
                "text_blocks": clean_blocks(p.get_text("blocks"))
            } for i in pages if 0 <= i < len(self.doc)
        }

    def extract_tables(self, page_num=None):
        if not self.doc: return {}

        def likely_table(lines):
            if len(lines) < 3: return False
            delims = ['\t', '|', '  ', ',']
            counts = [max(line.count(d) for d in delims) for line in lines]
            return sum(counts) / len(counts) > 1 and len(set(counts)) <= 3

        def detect(page):
            tbls = []
            for blk in page.get_text("dict").get("blocks", []):
                lines = blk.get("lines", [])
                rows = [" ".join(span["text"] for span in line.get("spans", [])) for line in lines]
                if likely_table(rows): tbls.append(rows)
            return tbls

        pages = range(len(self.doc)) if page_num is None else [page_num]
        return {f"page_{i}": detect(self.doc[i]) for i in pages if 0 <= i < len(self.doc)}

    def extract_images(self, page_num=None):
        if not self.doc: return {}

        def extract(page, idx):
            imgs = []
            for i, img in enumerate(page.get_images(full=True)):
                try:
                    xref, pix = img[0], fitz.Pixmap(self.doc, img[0])
                    if pix.n - pix.alpha >= 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    data = pix.tobytes("png")
                    fname = f"page_{idx}_image_{i}.png"
                    fpath = os.path.join(self.output_dir, fname)
                    with open(fpath, "wb") as f: f.write(data)
                    imgs.append({
                        "filename": fname,
                        "path": fpath,
                        "xref": xref,
                        "width": pix.width,
                        "height": pix.height,
                        "colorspace": pix.colorspace.name if pix.colorspace else "Unknown"
                    })
                    pix = None
                except Exception as e:
                    print(f"Image error on page {idx}: {e}")
            return imgs

        pages = range(len(self.doc)) if page_num is None else [page_num]
        return {f"page_{i}": extract(self.doc[i], i) for i in pages if 0 <= i < len(self.doc)}

    def ocr_extracted_images(self, images_data):
        for page_key, images in images_data.items():
            for img in images:
                try:
                    img_path = img.get("path")
                    if img_path and os.path.exists(img_path):
                        image = Image.open(img_path).convert("L")
                        text = pytesseract.image_to_string(image)
                        img["ocr_text"] = text.strip()
                except Exception as e:
                    print(f"OCR failed for {img.get('filename')}: {e}")
        return images_data

    def extract_all_content(self):
        if not self.open_document(): return {}

        print("Extracting text...")
        text = self.extract_text()

        print("Extracting tables...")
        tables = self.extract_tables()

        print("Extracting images...")
        images = self.extract_images()

        print("Running OCR on images...")
        images = self.ocr_extracted_images(images)

        metadata = {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                    for k, v in (self.doc.metadata or {}).items()}

        return {
            "document_info": {
                "filename": os.path.basename(self.pdf_path),
                "page_count": len(self.doc),
                "metadata": metadata
            },
            "text": text,
            "tables": tables,
            "images": images
        }

    def save_extracted_content(self, content, save_format="json"):
        if save_format == "json":
            path = os.path.join(self.output_dir, "extracted_content.json")
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self._make_json_serializable(content), f, indent=2, ensure_ascii=False)
                print(f"Saved to: {path}")
            except:
                simple = {
                    "document_info": content.get("document_info", {}),
                    "text_summary": {k: {"text": v.get("text", "")}
                                     for k, v in content.get("text", {}).items()},
                    "tables_count": {k: len(v) for k, v in content.get("tables", {}).items()},
                    "images_count": {k: len(v) for k, v in content.get("images", {}).items()},
                }
                fallback = os.path.join(self.output_dir, "extracted_content_simplified.json")
                with open(fallback, "w", encoding="utf-8") as f:
                    json.dump(simple, f, indent=2, ensure_ascii=False)
                print(f"Simplified saved to: {fallback}")
        elif save_format == "txt":
            path = os.path.join(self.output_dir, "extracted_text.txt")
            with open(path, "w", encoding="utf-8") as f:
                for k, v in content.get("text", {}).items():
                    f.write(f"\n--- {k.upper()} ---\n{v.get('text', '')}\n")
            print(f"Text saved to: {path}")

    def _make_json_serializable(self, obj):
        if isinstance(obj, dict): return {k: self._make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list): return [self._make_json_serializable(i) for i in obj]
        if isinstance(obj, tuple): return list(obj)
        if isinstance(obj, bytes): return f"<bytes {len(obj)} bytes>"
        if hasattr(obj, '__dict__'): return str(obj)
        return obj if isinstance(obj, (str, int, float, bool, type(None))) else str(obj)

    def close_document(self):
        if self.doc: self.doc.close()

def main():
    pdf_path = "1_Impact_assessment_Cyber_Resilience_Act_oFz3rACXW0RU8B0TrCo22ErI6Y_89545.pdf"
    extractor = PDFExtractor(pdf_path)
    print("Starting PDF extraction...")

    content = extractor.extract_all_content()
    if content:
        extractor.save_extracted_content(content, "json")
        extractor.save_extracted_content(content, "txt")

        print("\n--- EXTRACTION SUMMARY ---")
        print(f"Document: {content['document_info']['filename']}")
        print(f"Pages: {content['document_info']['page_count']}")
        print(f"Images extracted: {sum(len(i) for i in content['images'].values())}")
        print(f"Tables found: {sum(len(t) for t in content['tables'].values())}")
        print(f"Output directory: {extractor.output_dir}")

    extractor.close_document()

if __name__ == "__main__":
    main()
