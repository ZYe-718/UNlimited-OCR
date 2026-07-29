import os
import sys

import torch
from transformers import AutoModel, AutoTokenizer

model_name = 'baidu/Unlimited-OCR'

print("Loading...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, trust_remote_code=True,
    use_safetensors=True, torch_dtype=torch.bfloat16,
)
model = model.eval().cuda()
print("Ready.\n")

if len(sys.argv) < 2:
    print("Usage: python run_ocr.py <image_or_pdf_path>")
    sys.exit(0)

input_path = sys.argv[1]
ext = os.path.splitext(input_path)[1].lower()
output_dir = './ocr_output'
os.makedirs(output_dir, exist_ok=True)

if ext == '.pdf':
    import fitz

    doc = fitz.open(input_path)
    total = len(doc)
    print(f"PDF: {total} pages")

    completed = 0
    for i in range(total):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        tmp_path = os.path.join(output_dir, f'_page_{i+1:04d}.png')
        pix.save(tmp_path)

        print(f"  Page {i+1}/{total}...", end=" ", flush=True)
        try:
            model.infer(tokenizer,
                prompt='<image>document parsing.',
                image_file=tmp_path,
                output_path=os.path.join(output_dir, f'page_{i+1:04d}'),
                base_size=1024, image_size=1024, crop_mode=False,
                max_length=32768, no_repeat_ngram_size=35, ngram_window=128,
                save_results=True,
            )
            completed += 1
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
        finally:
            os.remove(tmp_path)

    doc.close()
    print(f"\nDone. {completed}/{total} pages")
else:
    print(f"Image: {input_path}")
    model.infer(tokenizer,
        prompt='<image>document parsing.', image_file=input_path,
        output_path=output_dir,
        base_size=1024, image_size=640, crop_mode=True,
        max_length=32768, no_repeat_ngram_size=35, ngram_window=128,
        save_results=True,
    )
    print(f"Done. Output: {output_dir}")
