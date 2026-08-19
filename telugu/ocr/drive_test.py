from pathlib import Path
import pymupdf
import time

pdf = Path(
    r"G:\My Drive\LMA_Individual_Project\data\raw\manual\textbooks\Copy of 10 telugu fl 2025-26.pdf"
)

print("Exists:", pdf.exists())
print("Size:", pdf.stat().st_size / (1024 * 1024), "MB")

start = time.perf_counter()

doc = pymupdf.open(str(pdf))

print("Pages:", len(doc))

page = doc[0]

pix = page.get_pixmap(
    matrix=pymupdf.Matrix(2, 2),
    alpha=False
)

print("Rendered first page:", pix.width, "x", pix.height)

doc.close()

elapsed = time.perf_counter() - start

print(f"Time: {elapsed:.3f} seconds")