import whisper
import os
import requests
from pydub import AudioSegment

from core.citations import format_timestamp_range, seconds_to_timestamp

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None


def format_segment_timeline(segments: list) -> str:
    lines = []
    for segment in segments:
        start = seconds_to_timestamp(segment.get("start", 0))
        end = seconds_to_timestamp(segment.get("end", 0))
        text = segment.get("text", "").strip()
        if not text:
            continue
        lines.append(f"[{start} - {end}] {text}")
    return "\n".join(lines)


def load_model():

    global _model  

    if _model is None: 
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL) 
        print("Whisper model loaded.")
    return _model 


def transcribe_chunk_whisper(chunk_path: str, start_offset_seconds: float = 0.0) -> dict:

    model = load_model()  

    result = model.transcribe(
        chunk_path,
        task="transcribe",
        fp16=False,
        verbose=False,
    )  
    segments = []
    for index, segment in enumerate(result.get("segments", [])):
        segment_text = segment.get("text", "").strip()
        if not segment_text:
            continue
        segments.append(
            {
                "chunk_segment_id": f"{os.path.basename(chunk_path)}:{index}",
                "start": round(start_offset_seconds + float(segment.get("start", 0.0)), 2),
                "end": round(start_offset_seconds + float(segment.get("end", 0.0)), 2),
                "text": segment_text,
                "engine": "whisper",
            }
        )

    return {
        "text": result.get("text", "").strip(),
        "segments": segments,
    }


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str, start_offset_seconds: float = 0.0) -> dict:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    segments = []
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            piece_text = _send_to_sarvam(piece_path).strip()
            if piece_text:
                full_text += piece_text + " "
                segments.append(
                    {
                        "chunk_segment_id": f"{os.path.basename(chunk_path)}:piece_{i}",
                        "start": round(start_offset_seconds + (start / 1000.0), 2),
                        "end": round(start_offset_seconds + ((start + len(piece)) / 1000.0), 2),
                        "text": piece_text,
                        "engine": "sarvam",
                    }
                )
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return {
        "text": full_text.strip(),
        "segments": segments,
    }

   



def transcribe_chunk(chunk_path: str, language: str = "english", start_offset_seconds: float = 0.0) -> dict:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model)
    - hinglish → Sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path, start_offset_seconds=start_offset_seconds)
    return transcribe_chunk_whisper(chunk_path, start_offset_seconds=start_offset_seconds)


def transcribe_all_with_segments(chunks: list, language: str = "english") -> dict:

    full_transcript_parts = []
    all_segments = []
    current_offset_seconds = 0.0

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):  

        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        chunk_audio = AudioSegment.from_wav(chunk)
        chunk_duration_seconds = len(chunk_audio) / 1000.0
        transcription = transcribe_chunk(
            chunk,
            language=language,
            start_offset_seconds=current_offset_seconds,
        )  

        chunk_text = transcription.get("text", "").strip()
        if chunk_text:
            full_transcript_parts.append(chunk_text)
        all_segments.extend(transcription.get("segments", []))
        current_offset_seconds += chunk_duration_seconds

    print("Transcription complete.")

    return {
        "text": " ".join(full_transcript_parts).strip(),
        "segments": all_segments,
        "timestamped_transcript": format_segment_timeline(all_segments),
    }


def transcribe_all(chunks: list, language: str = "english") -> str:
    return transcribe_all_with_segments(chunks, language=language)["text"]