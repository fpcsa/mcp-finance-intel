▶️ Run the MCP server

Option A (FastMCP CLI stdio transport):

uv pip install fastmcp
fastmcp run server.py


Option B (plain Python):

python server.py

Option C (http transport)

fastmcp run server.py:mcp --transport http --port 8000

(Configure your MCP client to connect via stdio/SSE depending on your environment.)

TODO: add toml file (if want to release this as package aka pip install mcp-finance-intel)
TODO: verify Dockerfile
TODO: improve time candle (allow more than the one defined now)