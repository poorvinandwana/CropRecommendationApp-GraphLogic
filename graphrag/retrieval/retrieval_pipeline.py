from dotenv import load_dotenv
from neo4j import GraphDatabase
import os


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    raise ValueError("Neo4j environment variables are not set properly.")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)



def graph_retrieve(user_query):
    uq = user_query.lower()
    nutrient = None

    if "potassium" in uq:
        nutrient = "Potassium"
    elif "nitrogen" in uq:
        nutrient = "Nitrogen"
    elif "phosphorus" in uq:
        nutrient = "Phosphorus"

    with driver.session() as session:

        # Case 1: nutrient-specific query
        if nutrient:
            result = session.run(
                """
                MATCH (c:Crop)-[r:REQUIRES]->(n:Nutrient {name:$nutrient}),
                      (c)-[:GROWS_IN]->(s:Soil)
                WHERE
                    s.salinity_dsm <= 3
                    AND s.moisture_percent >= 55
                RETURN
                    c.name AS crop,
                    r.mgkg AS value
                """,
                nutrient=nutrient
            )

        # Case 2: general suitability query
        else:
            result = session.run(
                """
                MATCH (c:Crop)-[:GROWS_IN]->(s:Soil)
                WHERE
                    c.temperature_c >= 18 AND c.temperature_c <= 26
                    AND s.ph >= 6.0 AND s.ph <= 7.0
                    AND s.moisture_percent >= 55
                    AND s.salinity_dsm <= 3
                RETURN
                    c.name AS crop
                """
            )

        rows = [dict(r) for r in result]
        print("DEBUG graph rows:", rows)
        return rows




# Context Builder

def build_graph_context(graph_results):
    if not graph_results:
        return ""

    lines = []

    for r in graph_results:
        crop = r.get("crop")

        if "value" in r:
            lines.append(
                f"{crop} has a required value of {r['value']}."
            )
        else:
            lines.append(
                f"{crop} is a suitable crop under the given conditions."
            )

    return "\n".join(lines)




# LLM Setup

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(model="mistral")

def generate_answer(user_query: str, graph_context: str):
    messages = [
        SystemMessage(
            content=(
                "You are an agronomy expert."
                "Answer the question using ONLY the data below."
                "Do NOT assume values that are not present."
                "If the graph knowledge is insufficient, say so clearly."
            )
        ),
        HumanMessage(
            content=f"""Question:
{user_query}

Graph Knowledge:
{graph_context}

Answer:"""
        )
    ]

    response = llm.invoke(messages)
    return response.content


# GraphRAG Pipeline

def graph_rag_pipeline(user_query: str):
    graph_results = graph_retrieve(user_query)
    graph_context = build_graph_context(graph_results)
    answer = generate_answer(user_query, graph_context)

    return {
        "query": user_query,
        "graph_context": graph_context,
        "answer": answer
    }


# Run Test Query

if __name__ == "__main__":
    response = graph_rag_pipeline(
        "I have nitrogen-rich soil. What crops benefit most from it?"
    )

    print("GRAPH CONTEXT:")
    print(response["graph_context"])
    print("\nFINAL ANSWER:")
    print(response["answer"])



def query_neo4j_for_recommendation(soil_data):
    with driver.session() as session:
        result = session.run(
            """
MATCH (c:Crop)-[r:REQUIRES]->(n:Nutrient),
      (c)-[:GROWS_IN]->(s:Soil)
WHERE toLower(s.type) CONTAINS toLower($soil_type)

WITH
  c, s,
  collect({name: n.name, mgkg: r.mgkg}) AS nutrients

WITH
  c, s,
  [x IN nutrients WHERE x.name = 'Nitrogen'][0] AS N,
  [x IN nutrients WHERE x.name = 'Phosphorus'][0] AS P,
  [x IN nutrients WHERE x.name = 'Potassium'][0] AS K

RETURN
  c.name AS crop,
  (
    CASE WHEN N IS NULL THEN 0 ELSE abs($N - N.mgkg) END +
    CASE WHEN P IS NULL THEN 0 ELSE abs($P - P.mgkg) END +
    CASE WHEN K IS NULL THEN 0 ELSE abs($K - K.mgkg) END +
    abs($pH - s.ph) +
    abs($M - s.moisture_percent) +
    abs($salinity - s.salinity_dsm) +
    abs($T - c.temperature_c)
  ) AS score
ORDER BY score ASC
LIMIT 3

            """,
            **soil_data
        )

        return [dict(r) for r in result]



def recommend_crop(soil_data: dict) -> dict:
    """
    Input: soil parameters from form
    Output: top crops + explanation
    """

    # 1. Query Neo4j
    ranked_crops = query_neo4j_for_recommendation(soil_data)

    if not ranked_crops:
        return {
            "crop": None,
            "alternatives": [],
            "explanation": "No crops are suitable for the given soil conditions."
        }

    # 2. Extract results
    top_crop = ranked_crops[0]["crop"]
    alternatives = [r["crop"] for r in ranked_crops[1:]]

    # 3. Build explanation prompt (grounded)
    prompt = f"""
    Soil conditions:
    Nitrogen: {soil_data['N']} mg/kg
    Phosphorus: {soil_data['P']} mg/kg
    Potassium: {soil_data['K']} mg/kg
    Temperature: {soil_data['T']} °C
    Soil pH: {soil_data['pH']}
    Moisture: {soil_data['M']} %
    Salinity: {soil_data['salinity']} dSm
    Soil Type: {soil_data['soil_type']}

    Recommended crop: {top_crop}
    Alternative crops: {', '.join(alternatives)}

    Explain clearly why the recommended crop is the best choice
    based strictly on the soil conditions.
    """

    explanation = llm.invoke(prompt).content

    return {
        "crop": top_crop,
        "alternatives": alternatives,
        "explanation": explanation
    }
