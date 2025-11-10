import logging
import time
from sentence_transformers import SentenceTransformer

from src.splade_easy import Index

# Enable logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

def debug_memory_loading(index_dir: str):
    """Debug what happens during memory loading"""

    print("=== Testing Memory Loading ===")

    # Load model once
    print("0. Loading model...")
    start = time.time()
    model = SentenceTransformer("naver/splade-v3")
    print(f"   Model load: {(time.time() - start) * 1000:.1f}ms")

    # Test the explicit approach
    print("1. Creating Index (memory=True)...")
    start = time.time()
    index = Index(index_dir, memory=True)
    print(f"   Index creation: {(time.time() - start) * 1000:.1f}ms")
    print(f"   Cache after creation: {len(index._cache)} shards")

    print("1.5. Loading Index into memory...")
    start = time.time()
    index.load()  # <-- Much cleaner!
    print(f"   Index load: {(time.time() - start) * 1000:.1f}ms")
    print(f"   Cache after load: {len(index._cache)} shards")
    
    # Debug: check if shards exist
    print(f"   Available shards: {list(index.iter_shards())}")
    print(f"   Index opened: {index._opened}")

    print("2. Creating IndexReader...")
    start = time.time()
    reader = index.reader()
    print(f"   Reader creation: {(time.time() - start) * 1000:.1f}ms")
    print(f"   Cache after reader creation: {len(index._cache)} shards")

    # Remove this step since IndexReader shouldn't need open()
    # print("3. Opening reader...")
    # start = time.time()
    # reader.open()
    # print(f"   Reader open: {(time.time() - start) * 1000:.1f}ms")
    # print(f"   Cache after reader open: {len(index._cache)} shards")
    
    # Show cache contents
    if index._cache:
        for shard_path, data in index._cache.items():
            print(f"   {shard_path}: {len(data)} docs")

    print("3. First query...")
    start = time.time()
    results = reader.search_text("test query", model=model, top_k=5)
    first_query_time = time.time() - start
    print(f"   First query: {first_query_time * 1000:.1f}ms")
    print(f"   Cache after first query: {len(index._cache)} shards")

    print("4. Second query...")
    start = time.time()
    results = reader.search_text("another test query", model=model, top_k=5)
    second_query_time = time.time() - start
    print(f"   Second query: {second_query_time * 1000:.1f}ms")

    print("5. Third query...")
    start = time.time()
    results = reader.search_text("another another test query", model=model, top_k=5)
    second_query_time = time.time() - start
    print(f"   Third query: {second_query_time * 1000:.1f}ms")

    # No reader.close() needed since no context manager


if __name__ == "__main__":
    debug_memory_loading("./profile_index")
