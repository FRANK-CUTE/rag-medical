from pprint import pprint

from query_processor import process_query


# =========================
# test queries
# =========================

TEST_QUERIES = [
    "MI treatment with aspirin",
    "effect of metformin on CVD after 2020",
    "RCT studies on warfarin in elderly patients",
    "Does insulin reduce stroke risk in diabetes?",
]


# =========================
# main
# =========================

def main():

    print("\n==== Week4 QUERY TEST ====\n")

    for i, query in enumerate(TEST_QUERIES, start=1):

        print(f"Test {i}")
        print("Query:", query)

        result = process_query(query)
        pprint(result, sort_dicts=False)

        print("\n--------------------\n")


if __name__ == "__main__":
    main()
