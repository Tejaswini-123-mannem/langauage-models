import internetarchive

query = 'language:"tel" AND mediatype:texts'

results = internetarchive.search_items(
    query,
    fields=[
        "identifier",
        "title",
    ],
    params={
        "rows": 5,
    },
)

print("=" * 70)
print("ARCHIVE SEARCH OK")
print("=" * 70)

for item in results:
    print(
        item.get("identifier"),
        "|",
        item.get("title")
    )