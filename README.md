# Blog-Creation Agent

## Description

The Blog Writing Agent (BWA) is an autonomous, AI-powered blog generation platform designed to replicate the workflow of a human editorial team. It transforms a simple topic prompt into a fully structured, research-backed, and media-enriched blog post. Operating as an agentic state machine, the system decouples the writing process into distinct automated roles: a researcher to gather context, an orchestrator to plan the narrative and assign word budgets, parallel workers to draft individual sections, and a publisher to merge the content and integrate AI-generated images.

## Features

- **Autonomous Agentic Pipeline:** A directed acyclic graph (DAG) workflow that routes tasks through specialized AI agents (Research, Orchestration, Drafting, and Merging).
- **Dynamic Research Integration:** Automatically queries external search indexes to ground the content in real-world, up-to-date information when topics demand it.
- **Multi-Tier LLM Resilience:** A cascading failover system that gracefully degrades across multiple LLM providers (e.g., from primary reasoning models to local/offline models) to ensure continuous operation despite API rate limits or failures.
- **Algorithmic Constraint Enforcement:** Strict programmatic enforcement of narrative flow, deduplication, and exact word count budgets during the content merging phase.
- **Intelligent Media Sourcing:** An automated image planning system that proposes, generates, and places contextually relevant visual assets within the document structure.
- **Rich-Text Workspace:** A seamless transition from AI generation to human editing, providing a full WYSIWYG interface for final editorial polishing.

## Tech Stack

- **Agentic Orchestration:** State graph engine for managing complex, multi-agent workflows (LangGraph).
- **Backend Application Layer:** Asynchronous REST API framework for non-blocking task execution (FastAPI).
- **Client Interface:** Reactive single-page application framework for handling dynamic UI states and rich-text editing (React, TipTap).
- **Persistence Layer:** Relational database for ACID-compliant storage of user identities, document metadata, and binary media assets (PostgreSQL / SQLAlchemy).
- **Inference & Search:** Abstraction layers connecting to various Large Language Models and external web search APIs (LangChain, OpenAI, Gemini, Tavily).

## Usage

1. **Initiate Generation:** The user enters a high-level topic or prompt into the command center.
2. **Agentic Processing:** The system autonomously determines the research needs, drafts an outline, and delegates section writing to parallel AI workers.
3. **Review & Edit:** Once the pipeline completes, the fully assembled and formatted document is presented on a rich-text canvas.
4. **Human Polish:** The user refines the generated content, adjusting formatting or text using the integrated WYSIWYG editor.
5. **Export:** The finalized document can be copied or exported as standard Markdown or DOCX for external publication.
