from pathlib import Path
import ast

p = Path("paper_trading_tracker.py")
source = p.read_text(encoding="utf-8")
tree = ast.parse(source)

functions = {
    node.name: node
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}

required = {
    "model_version",
    "training_cutoff_time",
    "training_rows",
    "feature_set",
    "prediction_created_at",
}

generate = functions.get("generate_predictions")

if generate is None:
    raise SystemExit("FAIL: generate_predictions function not found")

found = {
    node.value
    for node in ast.walk(generate)
    if isinstance(node, ast.Constant)
    and isinstance(node.value, str)
}

missing = required - found

print("Tracker file exists:", p.exists())
print("Audit fields expected:", sorted(required))
print("Audit fields found:", sorted(required & found))

if missing:
    print("RESULT: Missing fields:", sorted(missing))
    raise SystemExit(1)

print("RESULT: All five audit fields appear in prediction-generation code.")
print("Note: This checks source code only, not actual prediction execution.")
