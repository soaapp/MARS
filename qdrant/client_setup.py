from qdrant_client import QdrantClient, models
client = QdrantClient(":memory:")  # or point to localhost

collection_name = "documents"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
)
