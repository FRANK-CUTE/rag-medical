from pprint import pprint

from query_processor import process_query


# =========================
# main
# =========================

def main():

    print("\n==== Week4 QUERY DEMO ====")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("Enter query: ").strip()

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        result = process_query(query)

        print()
        pprint(result, sort_dicts=False)
        print()


if __name__ == "__main__":
    main()
