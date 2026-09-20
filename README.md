# PlutoAI

A personal agentic AI assistant built from scratch in Python, to understand how LLM-based systems are engineered beyond simple prompting.

PlutoAI was built in seven versions. Each version solves one problem the previous version couldn't, and adds exactly one capability. Starting from a single API call, it grows into an agent that holds a conversation, remembers it across restarts, answers from a private knowledge base, runs deterministic Python tools, searches the web, and orchestrates several of these at once.

The goal was never just to make it work. It was to learn **where the LLM should reason, and where ordinary software should take over.**

---

## The progression at a glance

| Version | Capability added | Problem it solves |
|---|---|---|
| **V1** | Basic LLM Q&A | How do I call an LLM from Python? |
| **V2** | System prompt | How do I control how it behaves? |
| **V3** | Multi-turn conversation | How does it remember the last question? |
| **V4** | Persistent memory | How does it remember after I close the app? |
| **V5** | RAG over private knowledge | How does it answer from *my* documents? |
| **V6** | Tools — calculator + web search | How does it do things an LLM is bad at? |
| **V7** | Agentic orchestration | How does it combine several capabilities in one answer? |

---

## V1 — A basic LLM application

The simplest thing that works.

```
User
  ↓
Python application
  ↓
LLM
  ↓
Response
```

**What it does:** takes a question, sends it to the LLM, prints the answer.

**What it taught:** the mechanics of the API call — request structure, model parameters, and how the response comes back. Nothing more, and that's the point. Every later version is a modification of this loop.

**Limitation:** the assistant has no personality, no memory, and no knowledge beyond what the model was trained on.

---

## V2 — System intelligence and behaviour

V1 treats the user's question as the entire input. But an assistant needs to know *how* to behave, separately from *what* it's being asked.

V2 adds a **system prompt**:

```
System prompt  ──┐
                 ├──→  LLM  ──→  Response
User question  ──┘
```

**What it does:** defines the assistant's role, tone, and constraints once, and applies it to every question.

**What it taught:** the separation between **what the user asks** and **how the assistant should behave**. This is the first piece of controllable LLM behaviour — the same question now produces different answers depending on the system prompt, and that's a design lever rather than an accident.

**Limitation:** each question is still answered in isolation.

---

## V3 — Conversation context

Ask "What is RAG?" then "Give me an example" and V2 has no idea what *it* refers to. Each call is independent:

```
Question 1 → Answer 1
Question 2 → Answer 2     (no connection between them)
Question 3 → Answer 3
```

V3 sends the accumulated history with every request:

```
Question 1
Answer 1
Question 2
Answer 2
Question 3
    ↓
   LLM
    ↓
Contextual answer
```

**What it does:** maintains a running list of messages and passes the whole list on each turn.

**What it taught:** LLMs are stateless. The "memory" in a chat interface is not in the model — it is the application resending the conversation every single time. Understanding this makes the context window a real engineering constraint rather than an abstract number.

**Limitation:** the history lives in memory. Close the program and it's gone.

---

## V4 — Persistent memory

The next question was simply: *what happens when I close the application?*

```
Conversation
     ↓
conversation.json
     ↓
Application restart
     ↓
Conversation restored
```

**What it does:** serialises the message history to JSON on each turn, and reloads it on startup.

**What it taught:** the distinction between two things that are easy to conflate —

- **conversation context** — what gets sent to the LLM in this request
- **persistent memory** — what survives between sessions

They are related but not the same, and as a conversation grows, they stop being the same thing entirely. You cannot send an unbounded history to the model, so persistence and context become separate design decisions.

**Limitation:** the assistant still only knows what the model was trained on.

---

## V5 — RAG over a private knowledge base

This is the transition from a generic chatbot to a domain-aware assistant.

The assistant needed to answer from my own documents — material the model has never seen. That means retrieving the relevant parts of those documents and giving them to the LLM as context.

```
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS vector index
    ↓
Similarity search
    ↓
Relevant context
    ↓
   LLM
    ↓
Grounded answer
```

**What it does:**

1. Splits source documents into chunks
2. Converts each chunk into an embedding vector
3. Stores the vectors in a FAISS index
4. Embeds the user's question and retrieves the nearest chunks
5. Passes those chunks to the LLM as context alongside the question

**What it taught:** the core idea behind retrieval-augmented generation —

> The LLM provides reasoning and language generation. Retrieval provides access to controlled knowledge.

Also, that chunking strategy matters more than it looks. Chunk too large and the retrieved context is mostly noise; too small and it loses the meaning that made it relevant.

**Limitation:** the assistant can retrieve and reason, but it can't *do* anything.

---

## V6 — Tools

LLMs are unreliable at things ordinary code does perfectly, arithmetic being the obvious example. V6 gives the assistant tools it can call.

```
              ┌── Calculator
User → Agent ─┼── RAG
              └── Web Search
```

The LLM decides *when* a capability is needed. It does not do the work itself:

```
LLM     → identify that a calculation is needed, extract the operands
  ↓
Python  → compute the result deterministically
  ↓
LLM     → explain the result in context
```

**What it does:** exposes a calculator and a web search function as callable tools, and lets the model choose when to invoke them.

**What it taught:** the principle the whole project ended up organised around —

> **Code for certainty, LLM for interpretation.**

The calculator isn't asking the model to guess arithmetic. It routes the deterministic part to deterministic software and keeps the model for the part it's actually good at. Web search does the same for recency: the model is not asked to know today's facts, only to interpret what the search returned.

**Limitation:** one capability at a time.

---

## V7 — Agentic orchestration

V7 brings the pieces together. A single question can now require more than one capability:

> *"According to my knowledge base, what is RAG, and calculate 840 × 25%?"*

That needs retrieval **and** a calculation, and the results of both need to be combined into one coherent answer.

```
                    ┌── Knowledge Search
User question → LLM ┤
                    └── Calculator
                         ↓
                    Tool results
                         ↓
                        LLM
                         ↓
                   Final response
```

**What it does:** the agent plans which capabilities the question requires, invokes them, collects the results, and synthesises a single answer. It can also combine conversation context with retrieval, so a follow-up question still knows both what was said earlier and what's in the knowledge base.

**What it taught:** this is the actual transition from an *LLM application* to an *LLM-powered agentic system*. The difference is the orchestration layer — the LLM stops being the whole program and becomes one component inside it.

---

## Architecture

```
                    ┌──────────────────────┐
                    │         User         │
                    └───────────┬──────────┘
                                ↓
                    ┌──────────────────────┐
                    │    PlutoAI Agent     │
                    │  LLM + orchestration │
                    └───────────┬──────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ↓                 ↓                 ↓
       Conversation            RAG              Tools
          Memory           Private KB     Calculator / Web Search
              │                 │                 │
              └─────────────────┼─────────────────┘
                                ↓
                               LLM
                                ↓
                          Final Answer
```

The architecture is deliberately modular. The knowledge base, the tools and the agent logic can each evolve without touching the others — new tools can be registered, the retrieval layer can be swapped, and the orchestration logic stays the same.

---

## Design principles

**Code for certainty, LLM for interpretation.** Anything with a correct answer — arithmetic, parsing, filtering, data processing — belongs in Python. The LLM decides what needs doing and explains what came back.

**Retrieval over recall.** Where the answer must come from specific source material, retrieve it and pass it in. Don't rely on what the model happens to have memorised.

**Context and memory are different problems.** What you send to the model in one request, and what the system remembers over time, need separate designs.

**One capability per version.** Each version was kept small enough that when something broke, it was obvious what had caused it.

---

## Stack

- **Python** — application and tool layer
- **LLM API** — reasoning and generation
- **FAISS** — vector index for similarity search
- **Embeddings** — chunk and query vectorisation
- **JSON** — conversation persistence
- **Web search API** — live retrieval beyond the local corpus

---

## Why it was built this way

Building this as seven versions rather than one application was the point of the exercise. Each step forced one specific question — *why doesn't this remember? why can't it answer from my documents? why is it bad at arithmetic?* — and each answer turned out to be an architectural decision rather than a coding problem.

The broader conclusion: an LLM-based system is not just a model. It's an orchestration layer connecting the model to controlled knowledge, deterministic tools, and persistent state, with the reasoning and the certainty kept firmly in separate places.
