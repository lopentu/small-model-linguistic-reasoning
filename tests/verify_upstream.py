"""Verify original final-source blobs and executable syntax against a supplied copy."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def executable(text):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
    return ast.dump(tree, include_attributes=False)
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("upstream", type=Path)
    args = ap.parse_args()
    source = json.loads((ROOT / "final_4b/source.json").read_text(encoding="utf-8"))
    for name, expected in source["git_blob_sha1"].items():
        original = (args.upstream / name).read_bytes()
        blob = b"blob " + str(len(original)).encode() + bytes([0]) + original
        if hashlib.sha1(blob).hexdigest() != expected:
            raise ValueError(f"{name}: upstream blob mismatch")
        release = (ROOT / "final_4b" / name).read_text(encoding="utf-8")
        if executable(original.decode("utf-8")) != executable(release):
            raise ValueError(f"{name}: executable syntax differs")
        print("PASS:", name)
if __name__ == "__main__":
    main()
