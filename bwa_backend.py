from __future__ import annotations

import operator
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated, Any, Dict, Union, cast

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv
from prompts import ROUTER_SYSTEM, RESEARCH_SYSTEM, ORCH_SYSTEM, WORKER_SYSTEM, DECIDE_IMAGES_SYSTEM

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
    source_preference: Literal["ai", "search"] = Field(default="ai", description="Use 'search' for technical diagrams/charts/UI, 'ai' for aesthetic banners.")


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
def get_llm_chain(schema=None, static_fallback=None, max_tokens_limit=4000, force_provider=None):
    """
    Text LLM Chain: Gemini directly as requested by the user.
    """
    from langchain_core.runnables import RunnableLambda
    from langchain_google_genai import ChatGoogleGenerativeAI
    import time
    import os

    def _invoke_with_fallback(input_params):
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            return AIMessage(content="Error: GEMINI_API_KEY is not set.")

        try:
            print(" Generating with Gemini (gemini-2.5-flash)...")
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=gemini_key,
                max_output_tokens=max_tokens_limit,
                temperature=0.7
            )
            
            if schema:
                llm = llm.with_structured_output(schema)
                
            res = llm.invoke(input_params)
            return res
        except Exception as e:
            print(f" Gemini API failed: {e}")
            
            # 2. Fallback to OpenRouter
            or_key = os.getenv("OPENROUTER_API_KEY")
            if or_key:
                try:
                    from langchain_openai import ChatOpenAI
                    print(" Trying OpenRouter (nvidia/nemotron-3-ultra-550b-a55b:free) as first fallback...")
                    or_llm = ChatOpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        model="nvidia/nemotron-3-ultra-550b-a55b:free",
                        api_key=or_key,
                        timeout=45,
                        max_retries=1,
                        max_tokens=max_tokens_limit
                    )
                    
                    if schema:
                        hint = "Respond ONLY with valid JSON matching the requested schema."
                        params = list(input_params) + [HumanMessage(content=hint)] if isinstance(input_params, list) else f"{input_params}\n\n{hint}"
                        chain = or_llm.with_structured_output(schema, method="json_mode")
                    else:
                        params = input_params
                        chain = or_llm
                        
                    res = chain.invoke(params)
                    if hasattr(res, 'content') and isinstance(res.content, str):
                        import re
                        res.content = re.sub(r'<think>.*?(</think>|$)', '', res.content, flags=re.DOTALL).strip()
                    if hasattr(res, "content") and not schema:
                        res.additional_kwargs["llm_fallback_active"] = True
                    return res
                except Exception as oe:
                    print(f" OpenRouter Gemma failed: {oe}")

            
            if static_fallback is not None:
                return static_fallback
            if schema:
                return schema()
            return AIMessage(content="Content generation failed.")
            
    return RunnableLambda(_invoke_with_fallback)

# -----------------------------
# 3) Router
# -----------------------------
# ROUTER_SYSTEM prompt → see prompts.py

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

# RESEARCH_SYSTEM prompt → see prompts.py

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
        print(f" Research extraction failed (all LLMs down), skipping research: {e}")
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
# ORCH_SYSTEM prompt → see prompts.py

def orchestrator_node(state: State) -> dict:
    orchestrator_llm = get_llm_chain(
        Plan,
        max_tokens_limit=4000,
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

    plan = orchestrator_llm.invoke(
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
        {"title": f"Introduction", "goal": "Provide an overview", "bullets": ["Definition", "Context", "Significance"]},
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
# 6) Fanout — SEQUENTIAL execution to avoid Groq rate limits
# -----------------------------
def fanout(state: State):
    """Run workers SEQUENTIALLY with delays to stay within Groq's 8,000 TPM free-tier limit."""
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
# WORKER_SYSTEM prompt → see prompts.py

def _is_section_truncated(section_md: str) -> bool:
    """Detect if a section was truncated mid-sentence by the LLM's token limit."""
    stripped = section_md.rstrip()
    if not stripped:
        return True
    last_char = stripped[-1]
    # A properly finished section ends with punctuation
    if last_char in '.!?"\u201d':
        return False
    # Check if last line is a bullet item (can end without period)
    last_line = stripped.split('\n')[-1].strip()
    if last_line.startswith('- ') and len(last_line) > 20:
        return False
    return True

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload['topic']
    topic_clean = re.sub(r"\d+\s*words?", "", topic, flags=re.IGNORECASE).strip()

    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    # Dynamic Task-Specific Formatting Instructions
    task_title_lower = task.title.lower()
    if "intro" in task_title_lower:
        section_heading = "## Introduction"
        transition_line = f"Some key points about {topic_clean} include:"
        bullet_format = "- Definition: [Clear definition]\n- Context: [Historical & current context]\n- Significance: [Key importance]"
    elif "core" in task_title_lower or "principle" in task_title_lower:
        section_heading = "## Core Principles"
        transition_line = f"The key concepts that define {topic_clean} include:"
        bullet_format = "- Key concept 1: [Name] – [Explanation]\n- Key concept 2: [Name] – [Explanation]\n- Key concept 3: [Name] – [Explanation]"
    else:
        section_heading = "## Future Outlook"
        transition_line = f"Some key points about {topic_clean}'s future outlook include:"
        bullet_format = "- Summary: [Overview of future trajectory]\n- Implications: [Impact on industry & society]\n- Conclusion: [Final perspective]"

    # Sequential stagger: wait for Groq's TPM quota to reset between sections
    if "core" in task_title_lower or "principle" in task_title_lower or "concept" in task_title_lower:
        time.sleep(8.0)
    elif "future" in task_title_lower or "outlook" in task_title_lower or "conclusion" in task_title_lower:
        time.sleep(12.0)

    prompt_messages = [
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(
            content=(
                f"Blog title: {plan.blog_title}\n"
                f"Topic: {topic_clean}\n"
                f"Target words: {task.target_words}\n\n"
                f"EXACT SECTION STRUCTURE TO GENERATE:\n"
                f"1. Start with exact header: {section_heading}\n"
                f"2. Write 1-2 opening paragraphs of vivid, flowing prose about {topic_clean}.\n"
                f"3. Include exact transition line: {transition_line}\n"
                f"4. Include 3 bullet items formatted exactly as:\n{bullet_format}\n"
                f"5. End with 1-2 closing paragraphs of flowing prose.\n\n"
                f"Grounding Data (DO NOT cite or list these):\n{evidence_text}\n"
            )
        ),
    ]

    # Try up to 2 attempts — if first attempt is truncated, retry with Pollinations
    section_md = ""
    for attempt in range(2):
        provider = "pollinations" if attempt > 0 else None
        worker_llm = get_llm_chain(
            static_fallback=AIMessage(content=f"{section_heading}\n\nContent generation failed for this section due to total LLM quota exhaustion. Please try again later."),
            force_provider=provider
        )
        res = worker_llm.invoke(prompt_messages)

        # Handle Gemini returning content as a list of parts instead of a string
        raw_content = res.content
        if isinstance(raw_content, list):
            text_parts = []
            for part in raw_content:
                if isinstance(part, dict) and 'text' in part:
                    text_parts.append(part['text'])
                elif isinstance(part, str):
                    text_parts.append(part)
            section_md = "\n".join(text_parts).strip()
        else:
            section_md = str(raw_content).strip()

        # Strip any leaked reasoning text before the actual section header
        header_idx = section_md.find("## ")
        if header_idx > 0:
            section_md = section_md[header_idx:]
        elif header_idx == -1:
            section_md = f"{section_heading}\n\n{section_md}"

        # Check if section was truncated — if so, retry once
        if _is_section_truncated(section_md) and attempt == 0:
            print(f" Section '{task.title}' appears truncated. Retrying with fallback provider...")
            time.sleep(3)
            continue
        break

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
        "Code snippet", "Example", "Bullets", "Roadmap", "User Safety", "Safety Rating", "Safety", "Moderation"
    ]
    for bh in bad_headers:
        body = re.sub(rf"(?i)^#*\s*{bh}:?.*$", "", body, flags=re.MULTILINE)
    
    # 4. Remove meta-commentary, research mentions, drafting notes, and placeholders
    meta_patterns = [
        r"(?i)User Safety:.*",
        r"(?i)Safety Rating:.*",
        r"(?i)\(DO NOT cite or list these\)",
        r"(?i)Please note:.*",
        r"(?i)Stay tuned for.*",
        r"(?i)I (am proud to say|have done|researched|consulted).*?research.*",
        r"(?i)This (blog post|section) (provides|contains|is significantly).*?overview.*",
        r"(?i)The focus here is on.*?rather than.*",
        r"(?i)Remember, technology is advancing.*",
        r"(?i)I hope this summary.*",
        r"(?i)Leaving a comment while you're working.*",
        r"(?i)^Draft:?.*$",
        r"(?i)^Now count words.*$",
        r"(?i)^Count:?.*$",
        r"(?i)^Sentence \d+:.*$",
        r"(?i)^Paragraph\d+.*$",
        r"(?i)^Let's (draft|count|compute|outline).*$",
        r"(?i)^First line:.*$",
        r"(?i)^We'll (start|write|draft|count).*$",
        r"(?i)^Target words:.*$",
        r"(?i)^Word count target:.*$",
        r"(?i)^Structure:.*$",
        r"(?i)^Must follow structure:.*$",
        r"(?i)^Then (end|include|follow|transition).*$",
        r"(?i)^We need to.*$",
        r"(?i)^Proposed text:.*$",
        r"(?i)^First, compute.*$",
        r"(?i)^I'll (write|count|number|draft).*$",
        r"(?i)^\"?[A-Za-z]+\"?\d+\s+\"?[A-Za-z]+\"?\d+.*$"
    ]
    for pattern in meta_patterns:
        body = re.sub(pattern, "", body, flags=re.MULTILINE)
    
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

    merged_md = f"{body}\n"
    return {"merged_md": merged_md, "final": merged_md}


def _get_image_prompts(topic_clean: str):
    """Shared image prompt generation logic — used by both try and except paths in decide_images."""
    abstract_keywords = [
        "seo", "marketing", "business", "strategy", "architecture", "system", "management", 
        "linkedin", "software", "data", "cloud", "ai", "finance", "economics",
        "python", "programming", "code", "coding", "java", "javascript", "react", "developer", 
        "devops", "api", "database", "sql", "html", "css", "c++", "rust", "golang", "web"
    ]
    brand_keywords = [
        "samsung", "apple", "google", "microsoft", "sony", "tesla", "technology", "tech", 
        "company", "brand", "smartphone", "phone", "device"
    ]
    
    topic_lower = topic_clean.lower()
    is_abstract = any(kw in topic_lower for kw in abstract_keywords)
    is_brand = any(kw in topic_lower for kw in brand_keywords)

    if is_brand:
        img1_prompt = f"{topic_clean} flagship product official photo"
        img1_pref = "search"
        img2_prompt = f"{topic_clean} technology company logo or product lineup"
        img2_pref = "search"
    elif is_abstract:
        topic_visual = f"{topic_clean} software programming" if any(kw in topic_lower for kw in ["python", "code", "coding", "java", "javascript"]) else topic_clean
        img1_prompt = f"{topic_visual} concept mind map or diagram"
        img1_pref = "search"
        img2_prompt = f"{topic_visual} architecture diagram or infographic"
        img2_pref = "search"
    else:
        img1_prompt = f"{topic_clean} landmark or real photograph"
        img1_pref = "search"
        img2_prompt = f"{topic_clean} high quality photo"
        img2_pref = "search"
    
    return img1_prompt, img1_pref, img2_prompt, img2_pref


# DECIDE_IMAGES_SYSTEM prompt → see prompts.py

def decide_images(state: State) -> dict:
    merged_md = state["merged_md"]
    topic = state.get("topic", "blog")
    topic_clean = re.sub(r"\d+\s*words?", "", topic, flags=re.IGNORECASE).strip()
    slug = _safe_slug(topic_clean)

    # 1. Programmatically place [[IMAGE_1]] after Introduction section and [[IMAGE_2]] after Core Principles section
    lines = merged_md.split("\n")
    new_lines = []
    img1_placed = False
    img2_placed = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            header_lower = stripped[3:].lower()
            if any(k in header_lower for k in ["core", "principle"]) and not img1_placed:
                new_lines.append("\n[[IMAGE_1]]\n")
                img1_placed = True
            elif any(k in header_lower for k in ["future", "outlook", "conclusion"]) and not img2_placed:
                new_lines.append("\n[[IMAGE_2]]\n")
                img2_placed = True
        new_lines.append(line)

    if not img1_placed:
        new_lines.append("\n[[IMAGE_1]]\n")
    if not img2_placed:
        new_lines.append("\n[[IMAGE_2]]\n")

    md_with_placeholders = "\n".join(new_lines)

    # 2. Propose image prompts via LLM or fallback
    try:
        planner = get_llm_chain(GlobalImagePlan, max_tokens_limit=4000)
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
        
        # Guarantee exactly 2 images are generated
        if len(image_specs) < 2:
            print(f" LLM returned only {len(image_specs)} images. Supplementing missing images...")
        # Use shared helper for fallback prompts
        img1_prompt, img1_pref, img2_prompt, img2_pref = _get_image_prompts(topic_clean)

        fallback_specs = [
            {
                "placeholder": "[[IMAGE_1]]",
                "filename": f"{slug}_banner.webp",
                "alt": f"Illustration of {topic_clean}",
                "caption": f"A representation of {topic_clean}",
                "prompt": img1_prompt,
                "size": "1024x576",
                "quality": "high",
                "source_preference": img1_pref
            },
            {
                "placeholder": "[[IMAGE_2]]",
                "filename": f"{slug}_detail.webp",
                "alt": f"Details of {topic_clean}",
                "caption": f"Visual details related to {topic_clean}",
                "prompt": img2_prompt,
                "size": "1024x576",
                "quality": "high",
                "source_preference": img2_pref
            }
        ]
        
        placeholders_found = {img.get("placeholder") for img in image_specs}
        for fallback in fallback_specs:
            if fallback["placeholder"] not in placeholders_found and len(image_specs) < 2:
                image_specs.append(fallback)
    except Exception as e:
        print(f" decide_images LLM prompt proposal failed ({e}). Using default prompts...")
        img1_prompt, img1_pref, img2_prompt, img2_pref = _get_image_prompts(topic_clean)

        image_specs = [
            {
                "placeholder": "[[IMAGE_1]]",
                "filename": f"{slug}_banner.webp",
                "alt": f"Illustration of {topic_clean}",
                "caption": f"A representation of {topic_clean}",
                "prompt": img1_prompt,
                "size": "1024x576",
                "quality": "high",
                "source_preference": img1_pref
            },
            {
                "placeholder": "[[IMAGE_2]]",
                "filename": f"{slug}_detail.webp",
                "alt": f"Details of {topic_clean}",
                "caption": f"Visual details related to {topic_clean}",
                "prompt": img2_prompt,
                "size": "1024x576",
                "quality": "high",
                "source_preference": img2_pref
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
    from PIL import Image, ImageOps
    import io

    try:
        w, h = map(int, target_size_str.split("x"))
    except Exception:
        w, h = 512, 512

    img = Image.open(io.BytesIO(img_bytes))
    
    # Convert to RGB to ensure compatibility
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    # Resize to fit within max dimensions, preserving original aspect ratio (no cropping)
    img.thumbnail((w, h), Image.Resampling.LANCZOS)
    
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
def _search_real_image_url(query: str) -> Optional[list]:
    """
    Searches for a real image URL using Tavily.
    """
    api_key = os.environ.get("TAVILY_API_KEY") or os.environ.get("TVLY_API_KEY")
    if not api_key:
        return None
    
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        
        # Add anti-watermark and anti-stock filters to ensure high-quality perfect images
        # Keep query concise (max 4-5 words) to get cleaner search results
        words = query.split()
        concise_query = " ".join(words[:5])
        safe_query = concise_query + " -stock -watermark -alamy -shutterstock -gettyimages -template -dreamstime -123rf -vector"
        
        # Search for images specifically
        result = client.search(query=safe_query, search_depth="advanced", include_images=True)
        images = result.get("images", [])
        if images:
            # Return all image URLs so we can try the next one if a download fails
            return images
    except Exception as e:
        print(f" Tavily image search failed: {e}")
    return None
# 
# 
def _download_image_bytes(url: str) -> bytes:
    """
    Downloads image bytes from a URL with a tight 5-second timeout.
    """
    import requests
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=5)
    response.raise_for_status()
    return response.content


def _search_outscraper_image_urls(query: str) -> Optional[list]:
    """
    Searches for image URLs using Outscraper API.
    """
    import requests
    
    api_key = os.environ.get("OUTSCRAPER_API_KEY")
    if not api_key:
        return None
        
    try:
        url = "https://api.app.outscraper.com/google-search-images"
        headers = {
            'X-API-KEY': api_key,
        }
        
        words = query.split()
        concise_query = " ".join(words[:5])
        
        params = {
            "query": concise_query,
            "limit": 10,
            "async": "false"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Outscraper typically returns a 'data' array with arrays of results
        # Or a list of objects if async=false
        if "data" in data and len(data["data"]) > 0:
            query_results = data["data"][0]
            if isinstance(query_results, list) and len(query_results) > 0:
                return [img.get("image_url") for img in query_results if img.get("image_url")]
                
    except Exception as e:
        print(f" Outscraper image search failed: {e}")
        
    return None

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

        source_pref = spec.get("source_preference", "ai")
        
        def _try_search():
            img = None
            try:
                print(f" Searching Real Images (Tavily): {spec['prompt']}")
                img_urls = _search_real_image_url(spec["prompt"])
                if img_urls:
                    for img_url in img_urls:
                        try:
                            img = _download_image_bytes(img_url)
                            if img:
                                break
                        except Exception as dl_e:
                            print(f"   Failed to download {img_url}: {dl_e}")
            except Exception as e:
                print(f" Tavily Search failed for {placeholder}: {e}")
            return img

        def _try_outscraper_search():
            img = None
            try:
                print(f" Searching Real Images (Outscraper): {spec['prompt']}")
                img_urls = _search_outscraper_image_urls(spec["prompt"])
                if img_urls:
                    for img_url in img_urls:
                        try:
                            img = _download_image_bytes(img_url)
                            if img:
                                break
                        except Exception as dl_e:
                            print(f"   Failed to download {img_url}: {dl_e}")
            except Exception as e:
                print(f" Outscraper Search failed for {placeholder}: {e}")
            return img

        # Try web search first for real, relevant images (Tavily)
        print(f" Routing: Trying Tavily web search for '{placeholder}'...")
        img_bytes = _try_search()

        # If Tavily failed, try Outscraper
        if not img_bytes:
            print(f" Tavily web search failed, trying Outscraper web search for '{placeholder}'...")
            img_bytes = _try_outscraper_search()

        # 4. Resize and Optimize
        if img_bytes:
            try:
                img_bytes = _resize_image_bytes(img_bytes, target_size, quality=80)
                return placeholder, img_bytes, filename
            except Exception as ree:
                print(f" Resize failed for {filename}: {ree}.")
                return placeholder, None, filename
        
        return placeholder, None, filename

    # Process all image searches and downloads in parallel for maximum speed
    print(f" Processing {len(image_specs)} images in parallel...")
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
