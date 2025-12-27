#!/usr/bin/env python3
"""
Migrate RAG data from ChromaDB to Qdrant

This script:
1. Connects to existing ChromaDB
2. Exports all documents
3. Imports them into Qdrant with optimized settings
"""

import asyncio
import sys
from pathlib import Path

# Add router to path
sys.path.insert(0, str(Path(__file__).parent.parent / "router"))

from qdrant_rag import QdrantRAGManager


async def migrate():
    print("=== ChromaDB to Qdrant Migration ===\n")

    # Initialize Qdrant
    qdrant = QdrantRAGManager(host="localhost", port=6333)

    if not await qdrant.initialize():
        print("ERROR: Failed to initialize Qdrant")
        return 1

    # Check if Qdrant already has data
    stats = qdrant.get_stats()
    if stats.get("count", 0) > 0:
        print(f"Qdrant already has {stats['count']} documents")
        response = input("Clear and re-migrate? [y/N]: ")
        if response.lower() == 'y':
            await qdrant.clear_collection()
            print("Collection cleared")
        else:
            print("Skipping migration")
            return 0

    # Try to import from ChromaDB
    chroma_path = Path("/home/user/benchai/data/chromadb")

    if chroma_path.exists():
        print(f"Found ChromaDB at {chroma_path}")
        try:
            migrated = await qdrant.migrate_from_chromadb(chroma_path)
            print(f"Migrated {migrated} documents from ChromaDB")
        except Exception as e:
            print(f"ChromaDB migration failed: {e}")
            print("Will index from scratch instead")
    else:
        print(f"No ChromaDB found at {chroma_path}")

    # Also index the codebase if needed
    stats = qdrant.get_stats()
    if stats.get("count", 0) < 100:
        print("\nIndexing BenchAI codebase...")

        # Index router code
        router_path = Path("/home/user/benchai/router")
        if router_path.exists():
            indexed = await qdrant.index_directory(router_path, extensions=['.py'])
            print(f"Indexed {indexed} chunks from router/")

        # Index docs
        docs_path = Path("/home/user/benchai/docs")
        if docs_path.exists():
            indexed = await qdrant.index_directory(docs_path, extensions=['.md'])
            print(f"Indexed {indexed} chunks from docs/")

    # Final stats
    stats = qdrant.get_stats()
    print(f"\n=== Migration Complete ===")
    print(f"Total documents: {stats.get('count', 0)}")
    print(f"Vectors indexed: {stats.get('indexed_vectors_count', 0)}")
    print(f"Status: {stats.get('status', 'unknown')}")

    return 0


if __name__ == "__main__":
    exit(asyncio.run(migrate()))
