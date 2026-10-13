import os
import shutil
import sys

output_dir = sys.argv[1] if len(sys.argv) > 1 else './ocr_output'
images_dir = os.path.join(output_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

merged_path = os.path.join(output_dir, 'merged.md')
page_count = 0
with open(merged_path, 'w', encoding='utf-8') as merged:
    for name in sorted(os.listdir(output_dir)):
        page_dir = os.path.join(output_dir, name)
        if not os.path.isdir(page_dir) or not name.startswith('page_'):
            continue

        md_path = os.path.join(page_dir, 'result.md')
        if not os.path.exists(md_path):
            continue

        with open(md_path, 'r', encoding='utf-8') as page_file:
            text = page_file.read()

        src_images = os.path.join(page_dir, 'images')
        if os.path.isdir(src_images):
            for img_name in sorted(os.listdir(src_images)):
                src = os.path.join(src_images, img_name)
                dst_name = f'{name}_{img_name}'
                shutil.copy2(src, os.path.join(images_dir, dst_name))
                text = text.replace(
                    f'images/{img_name}',
                    f'images/{dst_name}',
                )

        if page_count:
            merged.write('\n\n---\n\n')
        merged.write(text.strip())
        page_count += 1

print(f"Merged {page_count} pages → {merged_path}")
print(f"Images → {images_dir}/")
