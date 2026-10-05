import os
from dotenv import load_dotenv

load_dotenv()
from pathlib import Path
from utils.audio_processor import process_input
from core.transcriber import transcribe_all_with_segments
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question



def run_pipeline(source :str, language :str = "english") -> dict:
    print("starting AI Video Assistant")

    chunks = process_input(source)
    source_file = Path(chunks[0]).stem.rsplit("_chunk_", 1)[0] if chunks else Path(source).name

    transcription = transcribe_all_with_segments(chunks,language)
    transcription["source_file"] = source_file
    transcript = transcription["text"]
    timestamped_transcript = transcription["timestamped_transcript"]
    print(f"raw transcription (first 300 characters ) {transcript[:300]}")

    title = generate_title(transcription)

    summary = summarize(transcription)

    action_item = extract_action_items(timestamped_transcript)

    decisions = extract_key_decisions(timestamped_transcript)
    questions = extract_questions(timestamped_transcript)
    
    rag_chain = build_rag_chain(transcription)

    return {
        "title": title,
        "transcript": transcript,
        "timestamped_transcript": timestamped_transcript,
        "segments": transcription["segments"],
        "source_file": source_file,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }

if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f" Title: {result['title']}")
    print(f"\n Summary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\n Key Decisions:\n{result['key_decisions']}")
    print(f"\n Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print(" Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n Assistant: {answer}\n")