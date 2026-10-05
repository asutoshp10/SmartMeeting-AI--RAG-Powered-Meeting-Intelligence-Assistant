# SmartMeeting AI: RAG-Powered Meeting Intelligence Assistant

SmartMeeting AI is an AI-powered meeting intelligence platform that transforms meeting recordings into actionable insights. The system leverages Whisper for speech-to-text transcription, LangChain for orchestration, and ChromaDB for Retrieval-Augmented Generation (RAG) to enable semantic search and conversational querying over meeting recordings.


## Features

### Meeting Transcription

* Supports audio and video meeting recordings
* Powered by OpenAI Whisper for accurate speech-to-text conversion
* Preserves segment-level timestamps for traceable retrieval

### Executive Summaries

* Generates concise meeting summaries
* Highlights key discussion points and outcomes

### Action Item Extraction

* Identifies tasks assigned during meetings
* Extracts responsible stakeholders and associated context

### Decision Tracking

* Automatically detects decisions made during discussions
* Maintains timestamp references for verification

### Open Question Identification

* Extracts unresolved questions and discussion points
* Helps teams track pending decisions

### Timestamp-Aware RAG

* Stores transcript chunks with timestamp metadata
* Enables source-grounded responses with precise citations
* Provides traceable evidence for generated answers

### Conversational Meeting Search

Ask questions such as:

* What decisions were made regarding deployment?
* What action items were assigned to the backend team?
* What concerns were raised about infrastructure costs?

### Export Reports

* PDF export
* TXT export
* Structured meeting reports

---

## System Architecture

```text
Audio / Video Recording
            │
            ▼
     Whisper Transcription
            │
            ▼
 Timestamped Transcript Segments
            │
            ▼
     Chunking & Embeddings
            │
            ▼
          ChromaDB
            │
            ▼
       Retrieval Layer
            │
            ▼
        LangChain RAG
            │
            ▼
     Answer + Citations
```

---

## Tech Stack

### AI & LLM

* LangChain
* Mistral AI
* OpenAI Whisper

### Retrieval

* ChromaDB
* Sentence Transformers
* HuggingFace Embeddings

### Backend

* Python
* FastAPI

### Frontend

* Streamlit

### Utilities

* FFmpeg
* yt-dlp
* ReportLab

---

## Timestamp-Aware Citations

Every retrieved answer is backed by source citations.

Example:

```text
Decision:
The team agreed to migrate from MongoDB to PostgreSQL.

Sources:
Meeting_12.mp4
22:14 - 22:48
```

This improves transparency and reduces hallucinations by allowing users to verify responses directly against the original meeting recording.

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd SmartMeetingAI
```

### Create Virtual Environment

```bash
python -m venv myenv
source myenv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
MISTRAL_API_KEY=your_api_key_here
```

---

## Run Application

```bash
streamlit run streamlit_app.py
```

---

## Example Workflow

1. Upload a meeting recording.
2. Generate transcript using Whisper.
3. Extract:

   * Executive Summary
   * Action Items
   * Decisions
   * Open Questions
4. Store transcript embeddings in ChromaDB.
5. Chat with your meeting using RAG.
6. View timestamp-aware source citations.
7. Export meeting reports.

---

## Future Improvements

* Multi-meeting knowledge base
* Cross-meeting decision tracking
* Speaker diarization
* Hybrid retrieval (BM25 + Vector Search)
* Confidence scoring
* Meeting analytics dashboard

---