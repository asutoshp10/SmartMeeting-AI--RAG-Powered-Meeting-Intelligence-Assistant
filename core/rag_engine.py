import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.vector_store import build_vector_store, load_vector_store, get_retriever
from core.citations import format_source_citation, format_timestamp_range

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )

def format_docs(docs):
    formatted_docs = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        source_file = metadata.get("source_file") or "Unknown source"
        timestamp_range = format_timestamp_range(metadata.get("start_seconds"), metadata.get("end_seconds"))
        formatted_docs.append(
            f"[{index}] {source_file} | {timestamp_range}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted_docs)


def format_sources(docs):
    seen = set()
    sources = []

    for doc in docs:
        citation = format_source_citation(doc.metadata or {})
        if citation not in seen:
            seen.add(citation)
            sources.append(citation)

    return "\n\n".join(sources)


class TimestampedRAGChain:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.answer_chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise. If the user asks about a decision, action item, or open question, answer in a short direct sentence.

Context from meeting transcript:
{context}""",
                    ),
                    ("human", "{question}"),
                ]
            )
            | llm
            | StrOutputParser()
        )

    def invoke(self, question: str) -> str:
        docs = self.retriever.invoke(question)
        context = format_docs(docs)
        answer = self.answer_chain.invoke({"context": context, "question": question}).strip()
        sources = format_sources(docs)

        if sources:
            return f"{answer}\n\nSources:\n{sources}"
        return answer

def build_rag_chain(transcript:str):

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store, k = 4)

    llm = get_llm()
    return TimestampedRAGChain(retriever, llm)


def load_rag_chain():
    vector_store = load_vector_store()
    retriver = get_retriever(vector_store, k = 4)

    llm = get_llm()
    return TimestampedRAGChain(retriver, llm)


def ask_question(rag_chain, question:str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"answer :{answer}")
    return answer