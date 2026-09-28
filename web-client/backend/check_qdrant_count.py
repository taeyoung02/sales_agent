"""

Qdrant collection 갯수 체크용 스크립트(디버깅용)

"""

from rag.rag import get_rag_pipeline


def main():
    rag = get_rag_pipeline()
    client = rag.client
    collection = rag.collection_name

    print(f"📚 Collection: {collection}")

    # rag query test
    query = "소나타 옵션"
    results = rag.search(query, n_results=3)
    print(f"검색 결과 개수: {len(results)}")
    for result in results:
        print(f"검색 결과: {result['document']}")
        print(f"검색 결과 메타데이터: {result['metadata']}")
        print(f"검색 결과 거리: {result['distance']}")
        print(f"검색 결과 ID: {result['id']}")
        print("-" * 50)

    try:
        # Qdrant v1.x
        info = client.get_collection(collection_name=collection)
        print("✅ Collection status:", info.status)
        print("   Vector size:", info.config.params.vectors.size)
    except Exception as e:
        print("⚠️ Could not get collection info:", e)

    try:
        # 포인트(문서) 개수 조회
        count = client.count(collection_name=collection, exact=True)
        print(f"🔢 Point count: {count.count}")
    except Exception as e:
        print("⚠️ Could not count points:", e)


if __name__ == "__main__":
    main()
