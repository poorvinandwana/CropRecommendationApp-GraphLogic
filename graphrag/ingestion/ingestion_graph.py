from neo4j import GraphDatabase
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from dotenv import load_dotenv
import uuid
import json
import os
import time


# JSON extraction helper (ROBUST)

def extract_json(text: str):
    """
    Extract the FIRST valid JSON object from LLM output.
    Handles extra text, multiple JSON blocks, and trailing garbage.
    """
    decoder = json.JSONDecoder()
    text = text.strip()

    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue

    raise ValueError("No valid JSON object found in LLM output:\n" + text)


# Environment & Neo4j setup

load_dotenv()

NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASS]):
    raise ValueError("Neo4j environment variables not set")
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASS)
)


# LLM (BOUNDED)

llm = ChatOllama(
    model="mistral",
    temperature=0,
    num_predict=512  


# Prompt (STRICT JSON, ESCAPED)

prompt = ChatPromptTemplate.from_template("""
You are a knowledge extraction system.

Your task:
- Extract entities and relationships from the given text.
- Respond with ONLY valid JSON.
- Do NOT include explanations.
- Do NOT include markdown.
- Do NOT include text outside the JSON object.

The JSON format MUST be exactly:

{{
  "entities": [
    {{ "name": "EntityName", "type": "EntityType" }}
  ],
  "relations": [
    {{ "source": "A", "relation": "RELATION", "target": "B" }}
  ]
}}

Text:
{text}
""")


# Document loader

DOCS_DIR = Path("docs")

def load_documents():
    documents = []
    for file in DOCS_DIR.rglob("*.txt"):
        documents.append({
            "id": str(uuid.uuid4()),
            "source": file.name,
            "text": file.read_text(encoding="utf-8")
        })
    return documents


# Knowledge extraction

def extract_knowledge(text: str) -> dict:
    start = time.time()

    response = llm.invoke(
        prompt.format_messages(text=text)
    )

    output_text = response.content if hasattr(response, "content") else str(response)
    kg = extract_json(output_text)

    print(f"  ✓ Extraction completed in {time.time() - start:.2f}s")
    return kg


# Neo4j ingestion

def ingest_document(doc: dict):
    print(f"→ Extracting knowledge from {doc['source']}")

    # 🔑 Input cap (GraphRAG best practice)
    MAX_CHARS = 2000
    safe_text = doc["text"][:MAX_CHARS]

    kg = extract_knowledge(safe_text)

    with driver.session() as session:
        # Document node
        session.run(
            """
            MERGE (d:Document {id: $id})
            SET d.source = $source,
                d.text   = $text
            """,
            id=doc["id"],
            source=doc["source"],
            text=doc["text"]
        )

        # Entities
        for e in kg.get("entities", []):
            session.run(
                """
                MERGE (ent:Entity {name: $name})
                SET ent.type = $type
                WITH ent
                MATCH (d:Document {id: $doc_id})
                MERGE (ent)-[:MENTIONED_IN]->(d)
                """,
                name=e["name"],
                type=e["type"],
                doc_id=doc["id"]
            )

        # Relationships
        for r in kg.get("relations", []):
            session.run(
                """
                MATCH (a:Entity {name: $src})
                MATCH (b:Entity {name: $tgt})
                MERGE (a)-[:RELATED_TO {type: $rel}]->(b)
                """,
                src=r["source"],
                tgt=r["target"],
                rel=r["relation"]
            )


# Main pipeline

if __name__ == "__main__":
    docs = load_documents()
    print(f"Found {len(docs)} documents")

    for doc in docs:
        ingest_document(doc)
        print(f"✓ Ingested: {doc['source']}\n")

    driver.close()
    print("Graph ingestion complete.")
