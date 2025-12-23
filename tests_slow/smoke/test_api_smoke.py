import httpx
import sys
import pytest

@pytest.mark.slow
@pytest.mark.smoke
@pytest.mark.manual
def test_api_filters_smoke():
    """Manual smoke test: check API filters on a live server.
    
    This test is intended to be run against a live server (default http://127.0.0.1:8000/).
    """
    check_meta()

def check_meta(port="8000"):
    base_url = f"http://127.0.0.1:{port}/api"
    print("--- Fetching all meta ---")
    r = httpx.get(f"{base_url}/catalog/meta")
    data = r.json()
    rows = data.get("rows", [])
    print(f"Total rows: {len(rows)}")
    
    if not rows:
        print("Empty catalog index!")
        return

    first = rows[0]
    lib = first.get("library")
    sym = first.get("symbol")
    ent = first.get("entity")
    print(f"First row: lib={lib}, sym={sym}, ent={ent}")
    
    print(f"--- Querying specifically for lib={lib}, sym={sym} ---")
    params = {"library": lib, "symbol": sym}
    r2 = httpx.get(f"{base_url}/catalog/meta", params=params)
    data2 = r2.json()
    rows2 = data2.get("rows", [])
    print(f"Filtered rows: {len(rows2)}")
    if rows2:
        match = rows2[0].get('symbol') == sym
        print(f"Match: {match}")
        assert match, f"Expected symbol {sym}, got {rows2[0].get('symbol')}"
    else:
        print("FAILED TO FIND BY LIB/SYM")
        assert False, "Failed to find row by library and symbol"

if __name__ == "__main__":
    try:
        p = sys.argv[1] if len(sys.argv) > 1 else "8000"
        check_meta(p)
    except Exception as e:
        print(f"Error: {e}")
