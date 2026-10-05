import os 
from langchain_chroma import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from core.citations import seconds_to_timestamp

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device" : 'cpu'}
    )

def _build_documents_from_segments(transcript_data: dict, max_chars: int = 500) -> list[Document]:
    segments = transcript_data.get("segments", [])
    source_file = transcript_data.get("source_file") or "meeting_transcript"
    documents = []

    current_segments = []
    current_text_parts = []
    current_length = 0

    def flush_chunk(chunk_index: int) -> None:
        nonlocal current_segments, current_text_parts, current_length
        if not current_segments:
            return

        first_segment = current_segments[0]
        last_segment = current_segments[-1]
        documents.append(
            Document(
                page_content=" ".join(current_text_parts).strip(),
                metadata={
                    "source_file": first_segment.get("source_file") or source_file,
                    "chunk_id": f"chunk-{chunk_index}",
                    "chunk_index": chunk_index - 1,
                    "segment_start_index": first_segment.get("segment_index"),
                    "segment_end_index": last_segment.get("segment_index"),
                    "start_seconds": first_segment.get("start"),
                    "end_seconds": last_segment.get("end"),
                    "start_timestamp": seconds_to_timestamp(first_segment.get("start")) if first_segment.get("start") is not None else None,
                    "end_timestamp": seconds_to_timestamp(last_segment.get("end")) if last_segment.get("end") is not None else None,
                },
            )
        )

        current_segments = []
        current_text_parts = []
        current_length = 0

    chunk_index = 1
    for segment in segments:
        segment_text = segment.get("text", "").strip()
        if not segment_text:
            continue

        if current_text_parts and current_length + len(segment_text) > max_chars:
            flush_chunk(chunk_index)
            chunk_index += 1

        current_segments.append(segment)
        current_text_parts.append(segment_text)
        current_length += len(segment_text)

    flush_chunk(chunk_index)

    return documents


def build_vector_store(transcript : str | dict)->Chroma:
    print("Building vector Store")

    docs = []
    if isinstance(transcript, dict) and transcript.get("segments"):
        docs = _build_documents_from_segments(transcript)

    if not docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 50
        )
        transcript_text = transcript if isinstance(transcript, str) else transcript.get("text", "")
        chunks = splitter.split_text(transcript_text)
        source_file = transcript.get("source_file") if isinstance(transcript, dict) else "meeting_transcript"

        docs = [
            Document(
                page_content=chunk,
                metadata={
                    'chunk_index' : i,
                    'chunk_id': f'chunk-{i + 1}',
                    'source_file': source_file,
                    'start_seconds': None,
                    'end_seconds': None,
                    'start_timestamp': None,
                    'end_timestamp': None,
                },
            )
            for i,chunk in enumerate(chunks)
        ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents= docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    return vector_store



def load_vector_store() ->Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function= embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store

def get_retriever(vector_store : Chroma, k :int = 4):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":k}
    )