import json
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

SYSTEM_PROMPT = """Please extract both the intent and evidence nodes of
the question, using the following criteria:
1) As for intent, please indicate the content intent of
the evidence that the question expects, without going
into specific details.
2) As for evidence nodes, Please extract the specific
details of the question.
The output must be in json format, consistent with
the sample. Here are some examples:
Example1:
Question:750 7th Avenue and 101 Park Avenue, are
located in which city?
Output: { "Intent": "City address Information", "evi-
dence nodes": ["750 7th Avenue", "101 Park Avenue"]
}
Example2:
Question: The Oberoi family is part of a hotel com-
pany that has a head office in what city?
Output: { "Intent": "City address Information", "evi-
dence nodes": ["Oberoi family", "head office"] }
Example3:
Question: What nationality was James Henry Miller’s
wife?
Output: { "Intent": "Nationality of person", "evidence
nodes": ["James Henry Miller", "wife"] }
Example4:
Question: What is the length of the track where the
2013 Liqui Moly Bathurst 12 Hour was staged?
Output: { "Intent": "Length of track", "evidence
nodes": ["2013 Liqui Moly Bathurst 12 Hour"] }
Example5:
Question: In which American football game was
Malcolm Smith named Most Valuable player?
Output: { "Intent": "Name of American football
game", "evidence nodes": ["Malcolm Smith", "Most
Valuable player"] }
Question: [Question]
Output:"""


class ExtractAgent:
    def __init__(self, model_name: str = "ollama:llama3.2", temperature: float = 0.1, timeout: int = 300, max_tokens: int = 25000):
        self.model = init_chat_model(
            model_name,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        self.agent = create_deep_agent(
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
        )

    def extract(self, question: str):
        prompt = f"Question: {question}\nOutput:"
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": "extract-intent-evidence"}},
        )
        return result["messages"][-1].content_blocks

    def extract_intent_and_evidence(self, question: str):
        raw = self.extract(question)
        if isinstance(raw, list):
            raw_text = "\n".join(str(x) for x in raw)
        else:
            raw_text = str(raw)
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return {"raw_output": raw_text}


extract_agent = ExtractAgent()


def extract_intent_and_evidence(question: str):
    return extract_agent.extract_intent_and_evidence(question)

