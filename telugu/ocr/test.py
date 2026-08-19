import json
from pathlib import Path

archive = Path(
    r"G:\My Drive\LMA_Individual_Project\data\processed\internet_archive\jsonl"
)

textbooks = Path(
    r"G:\My Drive\LMA_Individual_Project\data\processed\manual\textbooks\ocr_final"
)

for name, folder in [("ARCHIVE", archive), ("TEXTBOOK", textbooks)]:
    p = next(folder.glob("*.jsonl"))

    with open(p, "r", encoding="utf-8") as f:
        record = json.loads(f.readline())

    print("\n" + "=" * 30)
    print(name)
    print(p.name)
    print(record)