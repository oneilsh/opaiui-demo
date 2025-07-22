from mcp.server.fastmcp import FastMCP
from arxiv import Search, Client

mcp = FastMCP("Semantic Scholar MCP Server")

client = Client()

@mcp.tool()
def search_arxiv(search_query: str) -> list[dict]:
    """Search for papers using ArXiv."""
    search_params = Search(query = search_query, max_results=5)
    results = []
    for entry in client.results(search_params):
        results.append({
            "title": entry.title,
            "summary": entry.summary,
            "authors": [author.name for author in entry.authors],
            "published": entry.published.strftime("%Y-%m-%d"),
            "id": entry.entry_id,
            "url": entry.links[0].href if entry.links else None
        })
    return results



if __name__ == "__main__":
    mcp.run()
