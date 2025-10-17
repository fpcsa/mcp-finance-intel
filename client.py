# mcp_stdio_test.py
import json, subprocess, sys
p = subprocess.Popen([sys.executable, "server.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

def send(obj): p.stdin.write(json.dumps(obj) + "\n"); p.stdin.flush()
def read():    return json.loads(p.stdout.readline())

send({"jsonrpc":"2.0","id":1,"method":"initialize",
      "params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"stdio-test","version":"0.1"}}})
print("init ->", read())

send({"jsonrpc":"2.0","method":"notifications/initialized"})
# may produce no immediate response

send({"jsonrpc":"2.0","id":2,"method":"tools/call",
      "params":{"name":"quote","arguments":{"input":{"symbols":["BTC/USDT","AAPL"]}}}})
print("tool ->", read())

p.stdin.close(); p.terminate()
