import sys
from bwa_backend import app
from datetime import date

state = {
    "topic": "Agentic AI",
    "mode": "closed_book",
    "as_of": str(date.today()),
    "recency_days": 30,
    "sections": []
}
res = app.invoke(state)
for s in res.get("sections", []):
    print(f"SECTION {s[0]}:")
    print(s[1][:100])
    print("---")
print("FINAL LENGTH:", len(res.get("final", "")))
print(res.get("final", "")[-500:])
