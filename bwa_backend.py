from __future__ import annotations

import operator
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated, Any, Dict, Union, cast

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (ported from your image flow) ----
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: str = "512x512"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    generated_images: dict[str, bytes]  # Stores raw image bytes mapped by filename

    final: str
    llm_fallback_active: Annotated[bool, lambda x, y: x or y]  # Sticky flag


# -----------------------------
# 2) LLM
# -----------------------------
from openai import RateLimitError, APIError
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None


primary_llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=0,       # Fail immediately on 429/Error
    timeout=5,           # 5 seconds timeout to switch fast
)
puter_llm = ChatOpenAI(
    base_url="https://api.puter.com/puterai/openai/v1/",
    model="qwen/qwen-2.5-72b-instruct",
    api_key=os.getenv("PUTER_AUTH_TOKEN"),
    max_retries=3,
    timeout=60,
)
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    max_retries=3,
)

def get_llm_chain(schema=None, static_fallback=None):
    """
    The 'Unstoppable' Chain: Groq (Llama 3.3 70B) -> Puter (Qwen) -> Gemini (Flash 2.0)
    If every single API fails and we are generating text (no schema), it resorts to 'Pseudo-Text' expansion.
    """
    from langchain_core.runnables import RunnableLambda
    import time
    import random

    def _invoke_with_fallback(input_params):
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        errors: List[str] = []
        fallback_used = False
        
        # 1. Groq Layer (Llama 3.3 70B - Ultra Fast)
        try:
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key and groq_key.startswith("gsk_"):
                print("⚡ Trying Groq (llama-3.3-70b) as primary...")
                llm_g = ChatOpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    model="llama-3.3-70b-versatile",
                    api_key=groq_key,
                    timeout=10,
                    max_retries=0
                )
                if schema:
                    hint = "Respond ONLY with valid JSON matching the requested schema. Ensure all bullet lists contain at least 3 items."
                    params = list(input_params) + [HumanMessage(content=hint)] if isinstance(input_params, list) else f"{input_params}\n\n{hint}"
                    chain = llm_g.with_structured_output(schema, method="json_mode")
                else:
                    params = input_params
                    chain = llm_g
                return chain.invoke(params)
        except Exception as e:
            errors.append(f"Groq (llama-3.3): {e}")

        fallback_used = True

        # 2. Puter Layer (Qwen) - First Fallback
        try:
            print("🤖 Trying Puter (Qwen) fallback...")
            llm_p = ChatOpenAI(
                base_url="https://api.puter.com/puterai/openai/v1/",
                model="qwen/qwen-2.5-72b-instruct",
                api_key=os.getenv("PUTER_AUTH_TOKEN"),
                timeout=60,
                max_retries=0
            )
            if schema:
                hint = "Respond ONLY with valid JSON."
                params = list(input_params) + [HumanMessage(content=hint)] if isinstance(input_params, list) else f"{input_params}\n\n{hint}"
                chain = llm_p.with_structured_output(schema, method="json_mode")
            else:
                params = input_params
                chain = llm_p
            res = chain.invoke(params)
            if hasattr(res, "additional_kwargs"):
                res.additional_kwargs["llm_fallback_active"] = True
            return res
        except Exception as e:
            errors.append(f"Puter (Qwen): {e}")

        fallback_used = True

        # 3. Gemini Layer (Flash) - Second Fallback
        try:
            if not ChatGoogleGenerativeAI:
                raise ImportError("langchain_google_genai not installed")
            print("🤖 Trying Gemini (flash-2.0) fallback...")
            g_llm = ChatGoogleGenerativeAI(
                model="gemini-3.6-flash",
                google_api_key=os.getenv("GEMINI_API_KEY"),
                max_retries=0,
                timeout=60
            )
            chain = g_llm.with_structured_output(schema) if schema else g_llm
            res = chain.invoke(input_params)
            if hasattr(res, "additional_kwargs"):
                res.additional_kwargs["llm_fallback_active"] = True
            return res
        except Exception as e:
            errors.append(f"Gemini (flash-2.0): {e}")

        # 4. Emergency Fallbacks
        if static_fallback is not None:
            print("🚨 ALL AI FAILED. Using Emergency Static Object.")
            # We can't easily tag a Pydantic object with fallback_active unless we modify it
            # But the nodes will set it if they catch an exception or see this
            return static_fallback
            
        if not schema:
            # Pseudo-Writer: Extract and format content from the prompt metadata
            print("🚨 ALL AI FAILED. Using Pseudo-Writer Fallback.")
            
            # 1. Extract raw content string from messages if needed
            if isinstance(input_params, list):
                from langchain_core.messages import HumanMessage
                content_sources = [m.content for m in input_params if isinstance(m, HumanMessage)]
                content_str = "\n".join(content_sources) if content_sources else str(input_params)
            else:
                content_str = str(input_params)
            
            # 2. Extract Title
            title_search = re.search(r"Section title: (.*)", content_str)
            title_text = title_search.group(1).strip() if title_search else "Introduction"
            
            # 3. Extract content bullets (distinguish from metadata and evidence)
            # Find the "Bullets:" section and capture everything until "Evidence"
            bullets_part = ""
            bullets_match = re.search(r"Bullets:(.*?)(?:Evidence|$)", content_str, re.DOTALL)
            if bullets_match:
                bullets_part = bullets_match.group(1)
            
            # If "Bullets:" header not found, fall back to general bullet search but exclude metadata-like lines
            raw_bullets = re.findall(r"^- (.*)", bullets_part or content_str, re.MULTILINE)
            content_bullets = []
            for b in raw_bullets:
                b = b.strip()
                # Skip evidence lines (contain http or |) and metadata-like keys
                if "http" in b or "|" in b or b.endswith(":") or not b:
                    continue
                content_bullets.append(b)
            
            # 4. Format into readable text
            if content_bullets:
                sentences = []
                for b in content_bullets:
                    s = b[0].upper() + b[1:] if len(b) > 0 else b
                    if not s.endswith(('.', '!', '?')): s += "."
                    sentences.append(s)
                body = " ".join(sentences)
                msg = AIMessage(content=f"## {title_text}\n\n{body}")
            else:
                msg = AIMessage(content=f"## {title_text}\n\nContent generation failed due to API limits. Summary: {title_text} is important for this topic.")
            
            msg.additional_kwargs["llm_fallback_active"] = True
            return msg

        errors_trace = "\n".join(cast(Any, errors)[-5:])
        raise Exception(f"Total Quota Exhaustion across all available model providers (Groq, Puter, Gemini). Trace:\n{errors_trace}")

    return RunnableLambda(_invoke_with_fallback)

# Default fallback chain for simple completions
llm = get_llm_chain()

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a JSON routing module. Return ONLY valid JSON matching this schema:
{
  "needs_research": boolean,
  "mode": "closed_book" | "hybrid" | "open_book",
  "reason": string,
  "queries": string[],
  "max_results_per_query": number
}

CRITICAL: The 'mode' field MUST be exactly one of: 'closed_book', 'hybrid', 'open_book'.

Modes:
- closed_book: evergreen concepts.
- hybrid: evergreen + needs up-to-date examples.
- open_book: volatile news/latest updates.
"""

def router_node(state: State) -> dict:
    decider = get_llm_chain(
        RouterDecision,
        static_fallback=RouterDecision(
            needs_research=False,
            mode="closed_book",
            reason="LLM quota exhausted fallback",
            queries=[],
            max_results_per_query=5
        )
    )
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}\nReturn JSON."),
        ]
    )

    is_fallback = False
    if hasattr(decision, "additional_kwargs") and decision.additional_kwargs.get("llm_fallback_active"):
        is_fallback = True
    elif isinstance(decision, RouterDecision) and decision.reason == "LLM quota exhausted fallback":
        is_fallback = True

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
        "llm_fallback_active": is_fallback
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a JSON research synthesizer. Return ONLY valid JSON matching this schema:
{
  "evidence": [
    {
      "title": string,
      "url": string,
      "published_at": string | null,
      "snippet": string,
      "source": string | null
    }
  ]
}

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null.
"""

def research_node(state: State) -> dict:
    queries = cast(Any, state.get("queries") or [])[:2] # Limit to max 2 queries
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=3)) # Limit to max 3 results per query

    if not raw:
        return {"evidence": []}
        
    raw_text = str(raw)[:8000] # Hard truncate to prevent Groq 12k TPM rate limit crashes

    try:
        extractor = get_llm_chain(EvidencePack)
        pack = extractor.invoke(
            [
                SystemMessage(content=RESEARCH_SYSTEM),
                HumanMessage(
                    content=(
                        f"As-of date: {state['as_of']}\n"
                        f"Recency days: {state['recency_days']}\n\n"
                        f"Raw results:\n{raw_text}\n\n"
                        "Return JSON."
                    )
                ),
            ]
        )
    except Exception as e:
        print(f"⚠️ Research extraction failed (all LLMs down), skipping research: {e}")
        return {"evidence": []}

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a JSON technical blog planner. Return ONLY valid JSON matching this schema:
{
  "blog_title": string,
  "audience": string,
  "tone": string,
  "blog_kind": "explainer" | "tutorial" | "news_roundup" | "comparison" | "system_design",
  "constraints": string[],
  "tasks": [
    {
      "id": number,
      "title": string,
      "goal": string,
      "bullets": string[],
      "target_words": number,
      "tags": string[],
      "requires_research": boolean,
      "requires_citations": boolean,
      "requires_code": boolean
    }
  ]
}

CRITICAL SECTION STRUCTURE: You MUST create EXACTLY 3 tasks, no more, no less, with these EXACT titles:
- Task 1: "Introduction To [Topic]" with bullets: ["Definition", "Context", "Significance"]
- Task 2: "Core Principles" with bullets: ["Key concept 1", "Key concept 2", "Key concept 3"]
- Task 3: "Future Outlook" with bullets: ["Summary", "Implications", "Conclusion"]

CRITICAL: Each task's 'bullets' array MUST contain AT LEAST 3 bullet items (min 3 items).
CRITICAL: WORD COUNT BUDGET: Sum of 'target_words' MUST match 'TOTAL WORD BUDGET' exactly. Distribute as: Introduction ~30%, Core Principles ~50%, Future Outlook ~20%.
CRITICAL: ALWAYS create exactly 3 sections regardless of MAX SECTIONS.
"""

def orchestrator_node(state: State) -> dict:
    planner = get_llm_chain(
        Plan,
        static_fallback=Plan(
            blog_title=state['topic'],
            audience="General",
            tone="Informative",
            blog_kind="explainer",
            constraints=["Standard blog overview"],
            tasks=[
                Task(id=1, title="Introduction to " + state['topic'], goal="Provide an overview", bullets=["Definition", "Context", "Significance"], target_words=300),
                Task(id=2, title="Core Principles", goal="Explain the main topic", bullets=["Key concept 1", "Key concept 2", "Key concept 3"], target_words=500),
                Task(id=3, title="Future Outlook", goal="Wrap up", bullets=["Summary", "Implications", "Conclusion"], target_words=200)
            ]
        )
    )
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    # Programmatic word count extraction
    topic = state['topic']
    word_match = re.search(r"(\d+)\s*words?", topic, re.IGNORECASE)
    total_budget = int(word_match.group(1)) if word_match else 1000
    
    max_sections = 5
    if total_budget < 400:
        max_sections = 2
    elif total_budget < 700:
        max_sections = 3

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {topic}\n"
                    f"TOTAL WORD BUDGET: {total_budget}\n"
                    f"MAX SECTIONS: {max_sections}\n"
                    f"Mode: {mode}\n"
                    f"Return JSON."
                )
            ),
        ]
    )
    
    # --- HARD ENFORCEMENT ---
    # 1. Force correct section titles
    topic_clean = re.sub(r"\d+\s*words?", "", state['topic'], flags=re.IGNORECASE).strip()
    section_defs = [
        {"title": f"Introduction To {topic_clean}", "goal": "Provide an overview", "bullets": ["Definition", "Context", "Significance"]},
        {"title": "Core Principles", "goal": "Explain the main topic in depth", "bullets": ["Key concept 1", "Key concept 2", "Key concept 3"]},
        {"title": "Future Outlook", "goal": "Wrap up with future prospects", "bullets": ["Summary", "Implications", "Conclusion"]},
    ]
    
    # 2. Trim excess tasks
    if len(plan.tasks) > 3:
        plan.tasks = plan.tasks[:3]
    
    # 3. Add missing tasks if LLM returned fewer than 3
    while len(plan.tasks) < 3:
        idx = len(plan.tasks)
        plan.tasks.append(Task(
            id=idx + 1,
            title=section_defs[idx]["title"],
            goal=section_defs[idx]["goal"],
            bullets=section_defs[idx]["bullets"],
            target_words=200,
        ))
    
    # 4. Force correct section titles on all tasks
    for i, t in enumerate(plan.tasks):
        t.title = section_defs[i]["title"]
        t.id = i + 1
    
    # 5. Scale Word Budgets to match Total Budget (30% / 50% / 20%)
    ratios = [0.30, 0.50, 0.20]
    for i, t in enumerate(plan.tasks):
        t.target_words = int(total_budget * ratios[i])

    is_fallback = state.get("llm_fallback_active", False)
    if hasattr(plan, "additional_kwargs") and plan.additional_kwargs.get("llm_fallback_active"):
        is_fallback = True
    elif plan.blog_title == state['topic'] and "LLM quota exhausted fallback" in str(plan.constraints): # Simple heuristic if static fallback used
        is_fallback = True

    return {"plan": plan, "llm_fallback_active": is_fallback}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are an expert blog writer who produces vivid, engaging, and deeply informative prose.
ONLY output the requested section markdown starting with a level 2 header (e.g. "## Introduction To Kedarnath").

FORMAT RULES (FOLLOW EXACTLY):
1. Start with a level 2 heading: "## [Section Title]"
2. Write 1-2 opening paragraphs of flowing, rich prose about the topic.
3. Then include a line: "Some key points about [topic] include:" or "The key concepts that define [topic] include:" or "[Topic]'s future outlook includes:"
4. Follow with a structured key-points list formatted as:
   - For Introduction sections: "Definition: ...", "Context: ...", "Significance: ..."
   - For Core Principles sections: "Key concept 1: [Name], ...", "Key concept 2: [Name], ...", "Key concept 3: [Name], ..."
   - For Future Outlook sections: "Summary: ...", "Implications: ...", "Conclusion: ..."
5. End with 1-2 closing paragraphs of flowing prose that add additional depth.

WRITING STYLE RULES:
1. **WRITE LIKE A JOURNALIST**: Use vivid storytelling, sensory details, and specific facts/numbers.
2. **BE SPECIFIC**: Include real names, dates, numbers, and concrete details.
3. **NO REPETITION**: Never repeat information already stated.
4. **STAY WITHIN ±5% of the 'Target words'**.

STRICT CONSTRAINTS:
- **NO INTROS/OUTROS**: NEVER say "Hello", "Welcome", "In this section", "Thank you".
- **NO LINKS/CODE**: Pure text ONLY.
- **NO GENERIC FILLER**: Remove phrases like "it is worth noting", "it is important to understand".
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    # We add a static fallback for the worker too
    worker_llm = get_llm_chain(
        static_fallback=AIMessage(content="Content generation failed for this section due to total LLM quota exhaustion. Please try again later.")
    )

    res = worker_llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: False\n"
                    f"requires_code: False\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Grounding Data (DO NOT cite or list these):\n{evidence_text}\n"
                )
            ),
        ]
    )
    # Handle Gemini returning content as a list of parts instead of a string
    raw_content = res.content
    if isinstance(raw_content, list):
        # Extract text from list of content parts like [{'type': 'text', 'text': '...'}]
        text_parts = []
        for part in raw_content:
            if isinstance(part, dict) and 'text' in part:
                text_parts.append(part['text'])
            elif isinstance(part, str):
                text_parts.append(part)
        section_md = "\n".join(text_parts).strip()
    else:
        section_md = str(raw_content).strip()
    
    is_fallback = payload.get("llm_fallback_active", False)
    if hasattr(res, "additional_kwargs") and res.additional_kwargs.get("llm_fallback_active"):
        is_fallback = True

    return {"sections": [(task.id, section_md)], "llm_fallback_active": is_fallback}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    
    # Post-processing to ensure NO links are in the output as requested by user
    # 1. Convert markdown links [text](url) to just 'text'
    body = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body)
    # 2. Remove raw http/https URLs (avoiding those inside code blocks might be tricky, 
    # but the user was very clear: "dont add links")
    body = re.sub(r"https?://\S+", "", body)
    
    # 3. Aggressive Header/Meta Cleanup
    bad_headers = [
        "Evidence", "References?", "Further Reading", "Grounding Data", "Thank You", 
        "Requirements", "Knowledge Base", "Business Applications", "Source Code",
        "Blog Post", "Background Information", "Advantages for Industries", "Disadvantages",
        "Code snippet", "Example", "Bullets", "Roadmap"
    ]
    for bh in bad_headers:
        body = re.sub(rf"(?i)^#*\s*{bh}:?.*$", "", body, flags=re.MULTILINE)
    
    # 4. Remove meta-commentary, research mentions, and placeholders
    meta_patterns = [
        r"(?i)\(DO NOT cite or list these\)",
        r"(?i)Please note:.*",
        r"(?i)Stay tuned for.*",
        r"(?i)I (am proud to say|have done|researched|consulted).*?research.*",
        r"(?i)This (blog post|section) (provides|contains|is significantly).*?overview.*",
        r"(?i)The focus here is on.*?rather than.*",
        r"(?i)Remember, technology is advancing.*",
        r"(?i)I hope this summary.*",
        r"(?i)Leaving a comment while you're working.*"
    ]
    for pattern in meta_patterns:
        body = re.sub(pattern, "", body)
    
    # Remove bracketed link placeholders [Link], [Reference Link 1], etc.
    body = re.sub(r"\[[A-Za-z\s]*\d?\](\s*\(\s*[^)]*\s*\))?", "", body)
    
    # 5. Cross-Section Deduplication (Paragraph level)
    paragraphs = body.split("\n\n")
    unique_paragraphs = []
    seen_content = set()
    for p in paragraphs:
        p = p.strip()
        if not p: continue
        # Clean paragraph for comparison
        clean_p = re.sub(r"\W+", "", p).lower()
        if not clean_p or len(clean_p) < 40: # Allow short headers or transitions
            unique_paragraphs.append(p)
            continue
        if clean_p in seen_content:
            continue
        seen_content.add(clean_p)
        unique_paragraphs.append(p)
    body = "\n\n".join(unique_paragraphs).strip()

    # 6. Deduplicate Headers
    lines = body.splitlines()
    new_lines = []
    seen_headers = set()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            header = stripped[3:].lower().strip(" :")
            # Remove redundant "in 300 words" from headers
            header = re.sub(r"in \d+ words", "", header).strip()
            if header in seen_headers:
                continue 
            seen_headers.add(header)
            new_lines.append(f"## {header.title()}")
        else:
            new_lines.append(line)
    body = "\n".join(new_lines).strip()

    # 7. Final Hard Pruning
    topic = state.get("topic", "")
    word_match = re.search(r"(\d+)\s*words?", str(topic), re.IGNORECASE)
    if word_match:
        budget = int(word_match.group(1))
        words_only = body.split()
        if len(words_only) > budget:
            target_words = int(budget * 1.1)
            matches = list(re.finditer(r'\S+', body))
            if len(matches) > target_words:
                cutoff_index = matches[target_words].end()
                body = body[:cutoff_index]
                last_period = body.rfind(".")
                if last_period > len(body) * 0.8:
                    body = body[:last_period + 1]
    
    merged_md = f"{body}\n"
    return {"merged_md": merged_md, "final": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor. Return ONLY valid JSON matching this schema:
{
  "md_with_placeholders": string,
  "images": [
    {
      "placeholder": string,
      "filename": string,
      "alt": string,
      "caption": string,
      "prompt": string,
      "size": "256x256" | "512x512" | "1024x1024" | "1024x1792" | "1792x1024",
      "quality": "low" | "medium" | "high"
    }
  ]
}

CRITICAL: EXACTLY 2 images total. Placeholders must be exactly: [[IMAGE_1]], [[IMAGE_2]].
CRITICAL: Place [[IMAGE_1]] on its own line immediately after the very first paragraph of the entire blog. Place [[IMAGE_2]] after the "## Core Principles" heading.
CRITICAL: Placeholders MUST be on their own line, separated from other text by a blank line. Do NOT wrap them in Markdown link or image syntax.
CRITICAL: Preferred size is 1024x576 (exactly 16:9 aspect ratio) to match the blog banner format. Use this for all images.
"""

def decide_images(state: State) -> dict:
    merged_md = state["merged_md"]
    topic = state.get("topic", "blog")
    topic_clean = re.sub(r"\d+\s*words?", "", topic, flags=re.IGNORECASE).strip()
    slug = _safe_slug(topic_clean)

    # 1. Programmatically place [[IMAGE_1]] and [[IMAGE_2]] safely without altering/truncating blog text
    lines = merged_md.split("\n")
    new_lines = []
    img1_placed = False
    img2_placed = False

    for i, line in enumerate(lines):
        if not img1_placed and i > 0 and line.strip().startswith("## "):
            new_lines.append("\n[[IMAGE_1]]\n")
            img1_placed = True
        elif img1_placed and not img2_placed and i > 0 and line.strip().startswith("## "):
            new_lines.append("\n[[IMAGE_2]]\n")
            img2_placed = True
        new_lines.append(line)

    if not img1_placed:
        new_lines.insert(2, "\n[[IMAGE_1]]\n")
    if not img2_placed:
        new_lines.append("\n[[IMAGE_2]]\n")

    md_with_placeholders = "\n".join(new_lines)

    # 2. Propose image prompts via LLM or fallback
    try:
        planner = get_llm_chain(GlobalImagePlan)
        plan = state["plan"]
        assert plan is not None

        image_plan = planner.invoke(
            [
                SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                HumanMessage(
                    content=(
                        f"Blog kind: {plan.blog_kind}\n"
                        f"Topic: {state['topic']}\n\n"
                        "Propose image prompts for [[IMAGE_1]] and [[IMAGE_2]].\n\n"
                        f"{merged_md[:1500]}\n\n"
                        "Return JSON."
                    )
                ),
            ]
        )
        image_specs = [img.model_dump() for img in image_plan.images]
    except Exception as e:
        print(f"⚠️ decide_images LLM prompt proposal failed ({e}). Using default prompts...")
        image_specs = [
            {
                "placeholder": "[[IMAGE_1]]",
                "filename": f"{slug}_banner.webp",
                "alt": f"Illustration of {topic_clean}",
                "caption": f"A representation of {topic_clean}",
                "prompt": f"A high-quality professional 16:9 banner image representing {topic_clean}, clean and modern design",
                "size": "1024x576",
                "quality": "high"
            },
            {
                "placeholder": "[[IMAGE_2]]",
                "filename": f"{slug}_detail.webp",
                "alt": f"Details of {topic_clean}",
                "caption": f"Visual details related to {topic_clean}",
                "prompt": f"A beautiful detailed 16:9 wide photograph or illustration showcasing concepts related to {topic_clean}",
                "size": "1024x576",
                "quality": "high"
            }
        ]

    return {
        "md_with_placeholders": md_with_placeholders,
        "image_specs": image_specs,
    }


def _resize_image_bytes(img_bytes: bytes, target_size_str: str, quality: int = 80) -> bytes:
    """
    Resizes image bytes to the target size and applies quality compression.
    target_size_str: e.g. "512x512"
    quality: 1-100 (for WebP/JPEG)
    """
    from PIL import Image
    import io

    try:
        w, h = map(int, target_size_str.split("x"))
    except Exception:
        w, h = 512, 512

    img = Image.open(io.BytesIO(img_bytes))
    
    # Use LANCZOS for high-quality downsampling
    img = img.resize((w, h), Image.Resampling.LANCZOS)
    
    out_io = io.BytesIO()
    # Save as WebP for best size/quality ratio. Fallback to JPEG if needed.
    # WebP is widely supported in modern browsers (Streamlit).
    try:
        img.save(out_io, format="WEBP", quality=quality, method=6)
    except Exception:
        # Fallback to JPEG if WEBP is not available in the Pillow installation
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(out_io, format="JPEG", quality=quality, optimize=True)
        
    return out_io.getvalue()
# 
# 
def _fetch_fallback_image_bytes(keywords: str, size: str = "512x512") -> bytes:
    """
    Fetches a fallback image from a public service (LoremFlickr).
    """
    import urllib.request
    import urllib.parse
    
    # Clean keywords: take first 3 words, remove non-alphanumeric
    clean_kws = re.sub(r"[^a-zA-Z0-9 ]", "", keywords)
    search_term = ",".join(clean_kws.split()[:3]) or "technology"
    
    try:
        w, h = map(int, size.split("x"))
    except Exception:
        w, h = 512, 512

    url = f"https://loremflickr.com/{w}/{h}/{urllib.parse.quote(search_term)}"
    
    try:
        # Use a user-agent to avoid being blocked
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.read()
    except Exception as e:
        raise RuntimeError(f"Fallback image fetch failed: {e}")
# 
# 
def _search_real_image_url(query: str) -> Optional[str]:
    """
    Searches for a real image URL using Tavily.
    """
    api_key = os.environ.get("TAVILY_API_KEY") or os.environ.get("TVLY_API_KEY")
    if not api_key:
        return None
    
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        # Search for images specifically
        result = client.search(query=query, search_depth="advanced", include_images=True)
        images = result.get("images", [])
        if images:
            # Return the first image URL
            return images[0]
    except Exception as e:
        print(f"⚠️ Tavily image search failed: {e}")
    return None
# 
# 
def _download_image_bytes(url: str) -> bytes:
    """
    Downloads image bytes from a URL.
    """
    import requests
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    return response.content

def _gemini_generate_image_bytes(prompt: str, size: str = "1024x1024") -> bytes:
    """
    Returns raw image bytes generated by Google Gemini (Imagen 3).
    Env var: GEMINI_API_KEY
    """
    from google import genai
    from google.genai import types
    import io

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    try:
        # Map aspect ratios for Imagen 3
        # Imagen 3 supports: "1:1", "4:3", "3:4", "16:9", "9:16"
        try:
            target_w, target_h = map(int, size.split("x"))
            ratio = target_w / target_h
            if 0.9 <= ratio <= 1.1:
                aspect_ratio = "1:1"
            elif 1.2 <= ratio <= 1.4:
                aspect_ratio = "4:3"
            elif 0.7 <= ratio <= 0.8:
                aspect_ratio = "3:4"
            elif 1.7 <= ratio <= 1.8:
                aspect_ratio = "16:9"
            elif 0.5 <= ratio <= 0.6:
                aspect_ratio = "9:16"
            else:
                aspect_ratio = "1:1"
        except Exception:
            aspect_ratio = "1:1"

        print(f"🎨 Generating {aspect_ratio} image with Gemini Imagen 3")
        
        response = client.models.generate_images(
            model='imagen-3.0-generate-001',
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
                output_mime_type='image/png'
            )
        )
        
        if not response.generated_images:
            raise RuntimeError("Gemini returned no images.")
            
        gen_img = response.generated_images[0]
        
        # gen_img.image contains the bytes if we used output_mime_type
        # or it might be a PIL Image object depending on the SDK version
        if hasattr(gen_img.image, 'image_bytes'):
            return gen_img.image.image_bytes
        else:
            # Fallback for PIL-like object
            img_byte_arr = io.BytesIO()
            gen_img.image.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()

    except Exception as e:
        print(f"⚠️ Gemini image generation failed: {e}")
        raise e

def _nvidia_generate_image_bytes(prompt: str, model: str = "flux.2-klein-4b") -> bytes:
    """
    Returns raw image bytes generated by an NVIDIA-hosted model.
    Env var: NVIDIA_API_KEY
    """
    import requests
    import base64
    
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set.")
        
    invoke_url = "https://integrate.api.nvidia.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "response_format": "b64_json"
    }
    
    print(f"🎨 Generating image with NVIDIA ({model})")
    response = requests.post(invoke_url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    if "data" in data and len(data["data"]) > 0 and "b64_json" in data["data"][0]:
        b64_str = data["data"][0]["b64_json"]
        return base64.b64decode(b64_str)
    else:
        raise RuntimeError(f"NVIDIA API returned unexpected format: {data}")

def _puter_generate_image_bytes(prompt: str, size: str = "1024x1024") -> bytes:
    """
    Returns raw image bytes generated by Puter (dall-e-3).
    Env var: PUTER_AUTH_TOKEN
    """
    import requests
    import base64
    
    api_key = os.environ.get("PUTER_AUTH_TOKEN")
    if not api_key:
        raise RuntimeError("PUTER_AUTH_TOKEN is not set.")
        
    invoke_url = "https://api.puter.com/puterai/openai/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "dall-e-3",
        "prompt": prompt,
        "size": size,
        "response_format": "b64_json"
    }
    
    print(f"🎨 Generating image with Puter (dall-e-3)")
    response = requests.post(invoke_url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    if "data" in data and len(data["data"]) > 0 and "b64_json" in data["data"][0]:
        b64_str = data["data"][0]["b64_json"]
        return base64.b64decode(b64_str)
    else:
        raise RuntimeError(f"Puter API returned unexpected format: {data}")

def _pollinations_generate_image_bytes(prompt: str, size: str = "512x512") -> bytes:
    """
    Generates raw AI image bytes using Pollinations.ai (100% Free, FLUX AI model).
    No API Key required.
    """
    import urllib.request
    import urllib.parse
    
    try:
        w, h = map(int, size.split("x"))
    except Exception:
        w, h = 512, 512
        
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={w}&height={h}&nologo=true"
    
    print(f"🌸 Generating AI image with Pollinations.ai (FLUX)")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=20) as response:
        return response.read()

def _fetch_picsum_image_bytes(size: str = "512x512") -> bytes:
    """
    Bulletproof fallback image fetcher from Picsum Photos. Never fails.
    """
    import urllib.request
    try:
        w, h = map(int, size.split("x"))
    except Exception:
        w, h = 512, 512
    url = f"https://picsum.photos/{w}/{h}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.read()

def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    import io
    from PIL import Image

    md = state.get("md_with_placeholders") or state["merged_md"]
    
    plan = state.get("plan")
    image_specs = state.get("image_specs", []) or []
    generated_images = {}

    def _process_single_image(spec):
        placeholder = spec["placeholder"]
        orig_filename = spec["filename"]
        filename = Path(orig_filename).stem + ".webp"
        target_size = spec.get("size", "512x512")
        img_bytes = None

        # 1. Try Gemini first (Best Quality)
        try:
            print(f"🎨 Gemini Image Gen: {spec['prompt']}")
            img_bytes = _gemini_generate_image_bytes(spec["prompt"], size=spec.get("size", "1024x1024"))
        except Exception as e:
            print(f"⚠️ Gemini failed for {placeholder}: {e}")

        # 2. Try Pollinations AI if Gemini fails
        if not img_bytes:
            try:
                print(f"🌸 Pollinations AI Image Gen: {spec['prompt']}")
                img_bytes = _pollinations_generate_image_bytes(spec["prompt"], size=target_size)
            except Exception as e:
                print(f"⚠️ Pollinations AI failed for {placeholder}: {e}")

        # 3. Try real image search (Tavily) if AI failed
        if not img_bytes:
            try:
                print(f"🔍 Searching Real Images: {spec['prompt']}")
                img_url = _search_real_image_url(spec["prompt"])
                if img_url:
                    img_bytes = _download_image_bytes(img_url)
            except Exception as e:
                print(f"⚠️ Search failed for {placeholder}: {e}")

        # 4. Ultimate Safety Net: Picsum Photos (NEVER FAILS!)
        if not img_bytes:
            try:
                print(f"🖼️ Picsum Fallback for {placeholder}")
                img_bytes = _fetch_picsum_image_bytes(size=target_size)
            except Exception as fe:
                print(f"❌ All sources failed for {placeholder}: {fe}")
                return placeholder, None, filename

        # 4. Resize and Optimize
        if img_bytes:
            try:
                # Use a default quality of 80 since it's most common
                img_bytes = _resize_image_bytes(img_bytes, target_size, quality=80)
                return placeholder, img_bytes, filename
            except Exception as ree:
                print(f"⚠️ Resize failed for {filename}: {ree}")
                return placeholder, None, filename
        
        return placeholder, None, filename

    # Run processing in parallel
    print(f"🚀 Processing {len(image_specs)} images in parallel...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(_process_single_image, image_specs))

    # Apply results back to MD and state
    final_md = str(md)
    for placeholder, img_bytes, filename in results:
        # Find the original spec to get alt/caption
        target_spec = next((s for s in image_specs if s["placeholder"] == placeholder), {})
        
        if img_bytes:
            generated_images[filename] = img_bytes
            alt_text = target_spec.get('alt', 'image')
            caption_text = target_spec.get('caption', '') or alt_text
            img_md = f"\n\n![{alt_text}](images/{filename})\n*{caption_text}*\n\n"
        else:
            # Image failed — skip it silently (no ugly error message in the blog)
            img_md = "\n\n"
        
        pattern = rf"\[* ?{re.escape(placeholder)} ?\]*"
        final_md = re.sub(pattern, img_md, final_md)

    return {"final": final_md, "generated_images": generated_images}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app
