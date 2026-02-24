# knowledge_mcp

FastMCP server for **local knowledge base ingestion + retrieval**.

This server is designed for a two-server MCP setup:
- External webscraper MCP: fetches/cleans remote content.
- `knowledge_mcp` (this package): ingests provided text/markdown and queries the local Chroma KB.

`knowledge_mcp` does **not** fetch URLs or perform internet scraping.

## Tools

- `search(query, filters, top_k)`
  - Semantic search over local Chroma KB.
  - Returns stable `id` (prefers metadata `chunk_id`) and short snippet.
- `fetch(id)`
  - Returns full chunk text and citation metadata.
- `ingest_text(text, source_type, source_id?, source_url?, metadata, options?)`
  - Chunks text, embeds, and upserts into Chroma.
  - Stores stable deterministic `chunk_id` in metadata when missing.
- `upsert_metadata(source_id, patch)`
  - Attempts metadata patch updates for all chunks in a source.
- `health()` / `stats()`
  - Runtime and collection statistics.

## Run locally (stdio)

From repository root:

```bash
python -m knowledge_mcp.server
```

Optional Streamable HTTP mode:

```bash
python -m knowledge_mcp.server --transport http --host 127.0.0.1 --port 8765
```

## Example Python client snippet

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_example():
    server = StdioServerParameters(
        command="python",
        args=["-m", "knowledge_mcp.server"],
    )

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            ingest_result = await session.call_tool(
                "ingest_text",
                {
                    "text": "Anmeldung deadline is 14 days after moving.",
                    "source_type": "text",
                    "source_id": "berlin-registration-guide",
                    "metadata": {"city": "Berlin", "procedure": "Anmeldung", "language": "en"},
                },
            )

            search_result = await session.call_tool(
                "search",
                {
                    "query": "How many days after moving must I register?",
                    "filters": {"city": "Berlin", "procedure": "Anmeldung"},
                    "top_k": 3,
                },
            )

            first_id = search_result.content[0].json["hits"][0]["id"]
            fetch_result = await session.call_tool("fetch", {"id": first_id})
            print(ingest_result, search_result, fetch_result)
```

## Integration note (LangChain MCP / MultiServer)

In a multi-server MCP architecture, route remote web content gathering to the webscraper MCP first,
then pass cleaned text into `knowledge_mcp.ingest_text`. For retrieval + grounding, use
`knowledge_mcp.search` followed by `knowledge_mcp.fetch` for citation-ready chunk payloads.
