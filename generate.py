'''
generate ai content if no other cases are found
uses a LOCAL Ollama model to generate said AI contnet

Usage:
python generate.py --model MODEL_NAME --prompt "Your prompt here"

make sure to pull a model from ollama
prefered:

ollama pull llama3 
ollama pull mistral
ollama pull phi3
ollama pull gemma3

'''

import argparse
import json
import random
import sys
import subprocess
import time
from pathlib import Path

import ollama
from tqdm import tqdm

MODEL = "llama3"
COUNT = 300
OUTPUT = "gen.jsonl"
RETRIES = 3
TEMP = 0.85
PROMPT_TEMPLATES = [
 
    # ── Essays ──────────────────────────────────────────────────────────────
    ("essay", "You are a thoughtful essay writer. Write clearly and professionally.",
     "Write a 3-paragraph essay about: {topic}. "
     "Do not include a title. Start directly with the first paragraph."),
 
    ("essay_opinion", "You are an opinionated writer crafting a persuasive piece.",
     "Write a persuasive 3-paragraph essay arguing that {opinion}. "
     "Use evidence and logical reasoning. No title, start immediately."),
 
    # ── News / Articles ─────────────────────────────────────────────────────
    ("news_article", "You are a professional news journalist.",
     "Write a short news article (3-4 paragraphs) reporting on: {topic}. "
     "Use an inverted pyramid structure. Include a dateline (e.g. 'NEW YORK —'). "
     "Do not write a headline."),
 
    ("blog_post", "You are a tech blogger writing for a general audience.",
     "Write a blog post section (2-3 paragraphs, no title, no headings) "
     "about: {topic}. Write in first person, conversational tone."),
 
    # ── Code ────────────────────────────────────────────────────────────────
    ("code_snippet", "You are a senior software engineer.",
     "Write a working {lang} code snippet that {code_task}. "
     "Include brief inline comments explaining what each section does. "
     "Output only the code block, no explanation outside of it."),
 
    ("code_explanation", "You are a software engineering tutor.",
     "Explain how {code_concept} works in {lang}. "
     "Include a short code example. "
     "Write 2-3 paragraphs of explanation, then show the code."),
 
    # ── Summaries ───────────────────────────────────────────────────────────
    ("summary", "You are a professional summarizer.",
     "Write a concise 2-paragraph summary of the following topic as if "
     "summarizing a research paper: {topic}. "
     "Use formal academic language."),
 
    # ── Emails ──────────────────────────────────────────────────────────────
    ("professional_email", "You are a business professional.",
     "Write a professional email about: {email_topic}. "
     "Include a subject line, greeting, body (2 short paragraphs), and sign-off. "
     "Keep it concise and polite."),
 
    # ── Q&A ─────────────────────────────────────────────────────────────────
    ("qa_response", "You are a helpful assistant providing thorough answers.",
     "Answer this question in detail (3-5 sentences): {question}"),
 
    # ── Creative / Story ────────────────────────────────────────────────────
    ("short_story_opening", "You are a creative fiction writer.",
     "Write the opening 3 paragraphs of a short story about: {fiction_topic}. "
     "Do not include a title. Establish setting and character immediately."),
 
    # ── Product descriptions ─────────────────────────────────────────────────
    ("product_description", "You are a professional copywriter.",
     "Write a compelling product description (2 paragraphs) for: {product}. "
     "Highlight benefits, not just features. No bullet points."),
 
    # ── Social media posts ───────────────────────────────────────────────────
    ("social_post", "You are a social media manager.",
     "Write 3 different LinkedIn posts about {topic}. "
     "Each should be 2-3 sentences. Separate them with '---'."),
 
    # ── Technical documentation ──────────────────────────────────────────────
    ("tech_doc", "You are a technical writer.",
     "Write a technical documentation section (no headings, 3 paragraphs) "
     "explaining {tech_concept}. Target audience: intermediate developers."),
]
 
 
# ─── TOPIC POOLS ──────────────────────────────────────────────────────────────
 
TOPICS = [
    "the impact of social media on mental health",
    "renewable energy adoption challenges",
    "the future of remote work",
    "artificial intelligence in healthcare",
    "climate change mitigation strategies",
    "cryptocurrency and the future of finance",
    "the ethics of gene editing",
    "space colonization feasibility",
    "urban planning and walkable cities",
    "the decline of print journalism",
    "privacy in the age of big data",
    "e-commerce vs. traditional retail",
    "the role of government in innovation",
    "digital education and MOOCs",
    "food security and vertical farming",
    "autonomous vehicles and public safety",
    "the gig economy and worker rights",
    "antibiotic resistance as a global threat",
    "the psychology of conspiracy theories",
    "open source software development",
    "ocean plastic pollution solutions",
    "the economics of universal basic income",
    "cultural appropriation in fashion",
    "the future of nuclear energy",
    "mental health awareness in workplaces",
]
 
OPINIONS = [
    "remote work is more productive than office work",
    "social media does more harm than good",
    "college degrees are overrated in today's economy",
    "electric vehicles will dominate roads within 10 years",
    "artificial intelligence will create more jobs than it destroys",
    "four-day work weeks should be the global standard",
    "public transportation should be free in all cities",
    "standardized testing is an inadequate measure of intelligence",
    "cryptocurrency will replace traditional banking",
    "zoos do more harm than good for wildlife conservation",
]
 
PROGRAMMING_LANGS = [
    "Python", "JavaScript", "TypeScript", "Rust", "Go",
    "Java", "C++", "Kotlin", "Swift", "Ruby",
]
 
CODE_TASKS = [
    "reads a CSV file and computes column statistics",
    "implements a binary search tree with insert and search",
    "fetches data from a REST API and parses JSON",
    "implements a simple rate limiter using a sliding window",
    "validates an email address using regex",
    "implements a LRU cache",
    "reverses a linked list in place",
    "sorts a list of dictionaries by a nested key",
    "implements debounce and throttle functions",
    "generates a Fibonacci sequence using memoization",
    "parses command line arguments",
    "connects to a SQLite database and runs a query",
    "implements a retry decorator with exponential backoff",
    "flattens a deeply nested dictionary",
    "computes the Levenshtein distance between two strings",
]
 
CODE_CONCEPTS = [
    "async/await and event loops",
    "dependency injection",
    "the observer design pattern",
    "garbage collection",
    "database indexing",
    "JWT authentication",
    "REST vs GraphQL APIs",
    "containerization with Docker",
    "the CAP theorem",
    "memoization and dynamic programming",
    "type inference",
    "coroutines",
    "the SOLID principles",
    "monads in functional programming",
    "WebSockets vs polling",
]
 
EMAIL_TOPICS = [
    "requesting a project deadline extension",
    "following up on an unanswered proposal",
    "announcing a new company policy",
    "asking a colleague for feedback on a report",
    "declining a meeting invitation politely",
    "introducing yourself to a new team",
    "requesting a salary review",
    "thanking a client after a successful project",
    "notifying customers of a service disruption",
    "proposing a collaboration with another department",
]
 
QUESTIONS = [
    "What is the difference between machine learning and deep learning?",
    "How does HTTPS encryption work?",
    "What causes inflation and how can it be controlled?",
    "What is the difference between RAM and storage?",
    "How do vaccines train the immune system?",
    "What is the Dunning-Kruger effect?",
    "How does a blockchain prevent data tampering?",
    "What is the difference between a virus and a bacterium?",
    "How does compound interest work?",
    "What is quantum entanglement?",
    "What is the difference between supervised and unsupervised learning?",
    "How do search engines rank web pages?",
    "What causes recessions?",
    "How does DNS resolution work?",
    "What is cognitive behavioral therapy?",
]
 
FICTION_TOPICS = [
    "a detective who can read emotions",
    "the last library on a colonized Mars",
    "a chef who discovers their food has healing powers",
    "two strangers trapped in an elevator during a blackout",
    "a historian who can enter old photographs",
    "an AI that develops a fear of being turned off",
    "a cartographer mapping a city that changes every night",
    "a musician who lost their hearing and is rediscovering music",
]
 
PRODUCTS = [
    "a smart water bottle that tracks your hydration",
    "an AI-powered journaling app for mental wellness",
    "noise-cancelling headphones designed for open offices",
    "a biodegradable phone case made from bamboo",
    "a standing desk with built-in posture sensors",
    "a portable espresso maker for travelers",
    "a subscription box for indie board games",
    "solar-powered outdoor speakers",
]
 
TECH_CONCEPTS = [
    "how HTTP/2 multiplexing improves web performance",
    "the difference between SQL and NoSQL databases",
    "how load balancers distribute traffic",
    "the purpose of a reverse proxy",
    "how OAuth 2.0 authorization works",
    "the difference between threads and processes",
    "how content delivery networks (CDNs) work",
    "what happens during a TCP handshake",
    "how containerization differs from virtualization",
    "the purpose of a message queue in distributed systems",
]
 
def pick(lst):
    return random.choice(lst)

def fill_prompt(template: str) -> str:
    """Fill in {placeholders} in a prompt template."""
    return (template
        .replace("{topic}",        pick(TOPICS))
        .replace("{opinion}",      pick(OPINIONS))
        .replace("{lang}",         pick(PROGRAMMING_LANGS))
        .replace("{code_task}",    pick(CODE_TASKS))
        .replace("{code_concept}", pick(CODE_CONCEPTS))
        .replace("{email_topic}",  pick(EMAIL_TOPICS))
        .replace("{question}",     pick(QUESTIONS))
        .replace("{fiction_topic}",pick(FICTION_TOPICS))
        .replace("{product}",      pick(PRODUCTS))
        .replace("{tech_concept}", pick(TECH_CONCEPTS))
    )

def generate_one(model: str, system: str, user: str) -> str | None:
    for attempt in range(RETRIES):
        try:
            resp = ollama.chat(
                model=model,
                options={"temperature": TEMP},
                messages=[
                    {"role": "system",    "content": system},
                    {"role": "user",      "content": user},
                ]
            )
            text = resp["message"]["content"].strip()
            return text if len(text) > 80 else None
        except Exception as e:
            if attempt < RETRIES - 1:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                print(f"\n  [warn] Generation failed after {RETRIES} attempts: {e}")
                return None

def check_model(model: str) -> bool:
    """Check if model is available locally; offer to pull it."""
    try:
        models = ollama.list()
        names  = [m["name"].split(":")[0] for m in models.get("models", [])]
        if model in names:
            print(f"[setup] Model '{model}' is available ✓")
            return True
 
        print(f"[setup] Model '{model}' not found locally.")
        print(f"        Available: {', '.join(names) if names else 'none'}")
        ans = input(f"        Pull '{model}' now? [y/N]: ").strip().lower()
        if ans == "y":
            print(f"[setup] Pulling {model}...")
            ollama.pull(model)
            return True
        return False
    except Exception as e:
        print(f"[ERROR] Cannot connect to Ollama: {e}")
        print("        Start Ollama with: ollama serve")
        return False
 
def main():
    parser = argparse.ArgumentParser(description="Generate AI text training data via Ollama")
    parser.add_argument("--model",  default=MODEL,  help=f"Ollama model name (default: {MODEL})")
    parser.add_argument("--count",  type=int, default=COUNT, help=f"Number of samples to generate (default: {COUNT})")
    parser.add_argument("--output", default=OUTPUT, help=f"Output JSONL file (default: {OUTPUT})")
    args = parser.parse_args()
 
    if not check_model(args.model):
        print("[ERROR] Cannot proceed without a model. Exiting.")
        sys.exit(1)
    existing = 0
    if Path(args.output).exists():
        with open(args.output) as f:
            existing = sum(1 for _ in f)
        print(f"[resume] Found {existing} existing samples in {args.output} — appending.")
 
    needed = args.count - existing
    if needed <= 0:
        print(f"[done] Already have {existing} samples. Nothing to generate.")
        return
 
    # Generate
    generated_count = 0
    failed_count    = 0
 
    with open(args.output, "a", encoding="utf-8") as out_f:
        with tqdm(total=needed, desc="Generating", unit="sample") as pbar:
            while generated_count < needed:
                # Cycle through all template categories evenly
                category, system_tmpl, user_tmpl = random.choice(PROMPT_TEMPLATES)
                user_prompt = fill_prompt(user_tmpl)
 
                text = generate_one(args.model, system_tmpl, user_prompt)
 
                if text and len(text) > 80:
                    record = {
                        "text":        text,
                        "generated":   1,           # label: 1 = AI
                        "source_name": f"ollama-{args.model}",
                        "category":    category,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    generated_count += 1
                    pbar.update(1)
                else:
                    failed_count += 1
                    if failed_count > 50:
                        print("\n[ERROR] Too many consecutive failures. Is Ollama still running?")
                        break
    
    print(f"   Total in {args.output}: {generated_count} samples")
    print(f"\nTo merge with your training data:")
    print(f"  The JSONL format matches what train.py expects.")
    print(f"  Each line: {{\"text\": \"...\", \"generated\": 1, ...}}")

if __name__ == "__main__":
    main()