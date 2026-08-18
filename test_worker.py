import sys
from bwa_backend import worker_node
payload = {
    "task": {"id": 3, "title": "Future Outlook", "goal": "Wrap up with future prospects", "bullets": ["Summary", "Implications", "Conclusion"], "target_words": 200, "tags": [], "requires_research": False},
    "plan": {"blog_title": "Agentic AI", "audience": "General", "tone": "Informative", "blog_kind": "explainer", "constraints": [], "tasks": []},
    "topic": "Agentic AI",
    "mode": "closed_book"
}
try:
    res = worker_node(payload)
    print("SUCCESS:")
    print(res)
except Exception as e:
    print("FAILED:")
    print(e)
