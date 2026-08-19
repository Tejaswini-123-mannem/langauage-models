import internetarchive
import time


IDENTIFIER = "c42729008785f0108bcf6132d1a1ad9b3f270b00"

print("=" * 80)
print("INTERNET ARCHIVE CONNECTION TEST")
print("=" * 80)

print("Identifier:", IDENTIFIER)

for attempt in range(1, 4):

    print()
    print(
        f"Attempt {attempt}/3..."
    )

    try:

        item = internetarchive.get_item(
            IDENTIFIER,
            request_kwargs={
                "timeout": (
                    10,
                    60
                )
            }
        )

        print()
        print("SUCCESS")
        print(
            "Title:",
            item.metadata.get(
                "title",
                ""
            )
        )

        print()
        print("Files:")

        count = 0

        for file_info in item.get_files():

            name = str(
                file_info.name
            )

            if (
                name.lower().endswith(".pdf")
                or "_djvu.txt" in name.lower()
                or "_djvu.xml" in name.lower()
                or "jp2" in name.lower()
            ):

                print(
                    " ",
                    name
                )

                count += 1

                if count >= 20:
                    break

        print()
        print(
            "Connection test complete."
        )

        break

    except Exception as exc:

        print(
            "FAILED:",
            type(exc).__name__
        )

        print(
            exc
        )

        if attempt < 3:

            print(
                "Waiting 5 seconds..."
            )

            time.sleep(5)

        else:

            print()
            print(
                "Archive metadata request "
                "did not succeed after 3 attempts."
            )