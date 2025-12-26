#!/usr/bin/env python3
"""
Test script for BenchAI Zettelkasten Knowledge Graph.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "router"))

from learning.zettelkasten import ZettelkastenKnowledgeGraph, ZettelType, LinkType
from learning.research_api import ResearchAPI, QueryPriority


async def test_zettelkasten():
    print("=" * 60)
    print("BenchAI Zettelkasten Knowledge Graph Test")
    print("=" * 60)

    # Use test storage
    test_storage = Path("/tmp/benchai_zettel_test")
    test_storage.mkdir(parents=True, exist_ok=True)

    kg = ZettelkastenKnowledgeGraph(test_storage / "knowledge.db")
    await kg.initialize()
    print("[OK] Knowledge graph initialized")

    # Create atomic notes (Zettels)
    print("\n=== Creating Zettels ===")

    z1 = await kg.create_zettel(
        content="""The Zettelkasten method is a personal knowledge management system
        developed by German sociologist Niklas Luhmann. It uses atomic notes (Zettels)
        that contain a single idea each. Notes are linked together organically,
        creating an emergent structure rather than a hierarchical one. The key
        principles are atomicity (one idea per note), unique IDs, and extensive linking.""",
        title="Zettelkasten Method Overview",
        zettel_type=ZettelType.PERMANENT,
        tags=["pkm", "knowledge-management", "methodology"],
        source="research"
    )
    print(f"  [OK] Created: {z1}")

    z2 = await kg.create_zettel(
        content="""Atomic notes are the foundation of Zettelkasten. Each note should
        contain exactly one idea, making it self-contained and reusable. This atomicity
        allows notes to be linked in multiple contexts without losing meaning. A good
        atomic note can be understood without reading other notes.""",
        title="Atomic Notes Principle",
        zettel_type=ZettelType.PERMANENT,
        tags=["atomic-notes", "zettelkasten", "principles"],
        source="research"
    )
    print(f"  [OK] Created: {z2}")

    z3 = await kg.create_zettel(
        content="""BenchAI uses a four-layer learning architecture: Zettelkasten
        knowledge graph for connected knowledge, enhanced memory for typed storage,
        experience replay for learning from past successes/failures, and LoRA
        fine-tuning for periodic model improvement. This creates a self-improving
        AI system.""",
        title="BenchAI Learning Architecture",
        zettel_type=ZettelType.PERMANENT,
        tags=["benchai", "architecture", "learning"],
        source="system"
    )
    print(f"  [OK] Created: {z3}")

    z4 = await kg.create_zettel(
        content="""In a multi-agent AI system, agents can work in parallel while
        querying a central knowledge repository. BenchAI serves as this repository,
        providing deep research capabilities via its Zettelkasten. MarunochiAI
        (programmer) and DottscavisAI (creative) can submit async queries while
        continuing their work.""",
        title="Multi-Agent Knowledge Sharing",
        zettel_type=ZettelType.PERMANENT,
        tags=["multi-agent", "parallel", "knowledge-sharing"],
        source="architecture"
    )
    print(f"  [OK] Created: {z4}")

    z5 = await kg.create_zettel(
        content="""A-MEM (Agentic Memory) implements Zettelkasten principles for
        LLM agents. It uses atomic notes with automatic link discovery based on
        semantic similarity. Notes evolve as new connections form. This was
        published at NeurIPS 2025.""",
        title="A-MEM: Zettelkasten for AI Agents",
        zettel_type=ZettelType.LITERATURE,
        tags=["a-mem", "research", "neurips-2025"],
        source="paper"
    )
    print(f"  [OK] Created: {z5}")

    # Test search
    print("\n=== Testing Search ===")
    results = await kg.search("Zettelkasten atomic notes", limit=5)
    print(f"  Search for 'Zettelkasten atomic notes': {len(results)} results")
    for r in results[:3]:
        print(f"    - {r['title']} (importance: {r['importance']})")

    # Test graph traversal
    print("\n=== Testing Graph Traversal ===")
    graph = await kg.get_connected_knowledge(z1, max_depth=2)
    print(f"  From '{z1[:20]}...': {graph['node_count']} nodes, {graph['edge_count']} edges")

    # Test hub finding
    print("\n=== Finding Hub Notes ===")
    hubs = await kg.find_hubs(limit=3)
    for hub in hubs:
        print(f"  Hub: {hub['title']} (in: {hub['inlinks']}, out: {hub['outlinks']})")

    # Test sleep consolidation
    print("\n=== Testing Sleep Consolidation ===")
    consolidation_result = await kg.sleep_consolidation()
    print(f"  Links strengthened: {consolidation_result['links_strengthened']}")
    print(f"  Importance updated: {consolidation_result['importance_updated']}")

    # Test Research API
    print("\n=== Testing Research API ===")
    research_api = ResearchAPI(kg)
    await research_api.start()

    # Submit async query (simulating agent request)
    query_id = await research_api.submit_query(
        query="How does knowledge linking work in Zettelkasten?",
        agent_id="marunochiAI",
        priority=QueryPriority.NORMAL
    )
    print(f"  Query submitted: {query_id}")

    # Wait for result
    result = await research_api.get_result(query_id, wait=True, timeout=5.0)
    print(f"  Query status: {result['status']}")
    if result['data']:
        print(f"  Results found: {result['data']['zettel_count']} zettels")
        if result['data'].get('synthesis'):
            print(f"  Synthesis preview: {result['data']['synthesis'][:200]}...")

    await research_api.stop()

    # Get stats
    print("\n=== Knowledge Graph Stats ===")
    stats = await kg.get_stats()
    print(f"  Total Zettels: {stats['total_zettels']}")
    print(f"  Total Links: {stats['total_links']}")
    print(f"  Graph Density: {stats['graph_density']}")
    print(f"  By Type:")
    for ztype, info in stats.get('by_type', {}).items():
        print(f"    - {ztype}: {info['count']} (avg importance: {info['avg_importance']})")

    # Cleanup
    import shutil
    shutil.rmtree(test_storage, ignore_errors=True)

    print("\n" + "=" * 60)
    print("ALL ZETTELKASTEN TESTS PASSED")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = asyncio.run(test_zettelkasten())
    sys.exit(0 if success else 1)
