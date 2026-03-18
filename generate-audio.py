#!/usr/bin/env python3
"""Generate MP3 audio narrations from text files using OpenAI's TTS API.

Usage:
    python generate-audio.py                    # Generate audio for all .txt files in content/
    python generate-audio.py sample-article     # Generate audio for a single file (without .txt extension)

Requirements:
    export OPENAI_API_KEY="your-key-here"
    (No third-party packages needed — uses only the standard library.)
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

CONTENT_DIR = Path(__file__).parent / "content"
AUDIO_DIR = Path(__file__).parent / "audio"
API_URL = "https://api.openai.com/v1/audio/speech"
MODEL = "gpt-4o-mini-tts"
VOICE = "coral"
INSTRUCTIONS = "Read this article in a calm, clear, natural newscaster voice."
CHUNK_LIMIT = 4096


def split_into_chunks(text: str, limit: int = CHUNK_LIMIT) -> list:
    """Split text into chunks that respect the character limit, breaking at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > limit:
            sentences = para.replace(". ", ".\n").split("\n")
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(current) + len(sentence) + 1 > limit:
                    if current:
                        chunks.append(current.strip())
                    current = sentence
                else:
                    current = current + " " + sentence if current else sentence
        elif len(current) + len(para) + 2 > limit:
            if current:
                chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


def call_tts_api(text: str, api_key: str) -> bytes:
    """Call OpenAI TTS API and return raw MP3 bytes."""
    payload = json.dumps({
        "model": MODEL,
        "voice": VOICE,
        "input": text,
        "instructions": INSTRUCTIONS,
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    # Build an opener that bypasses any system proxy
    no_proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(no_proxy_handler, urllib.request.HTTPSHandler())

    with opener.open(req) as resp:
        return resp.read()


def generate_audio(text_file: Path, api_key: str) -> None:
    """Generate an MP3 file from a text file using OpenAI TTS."""
    stem = text_file.stem
    output_path = AUDIO_DIR / f"{stem}.mp3"
    text = text_file.read_text().strip()

    if not text:
        print(f"  Skipping {text_file.name}: empty file")
        return

    chunks = split_into_chunks(text)
    print(f"  {text_file.name} -> {output_path.name} ({len(chunks)} chunk{'s' if len(chunks) != 1 else ''})")

    audio_parts = []
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            print(f"    Generating chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)...")
        audio_parts.append(call_tts_api(chunk, api_key))

    with open(output_path, "wb") as f:
        for part in audio_parts:
            f.write(part)

    print(f"  Done: {output_path}")


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("  export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    AUDIO_DIR.mkdir(exist_ok=True)

    # Determine which files to process
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if not name.endswith(".txt"):
            name += ".txt"
        text_file = CONTENT_DIR / name
        if not text_file.exists():
            print(f"Error: {text_file} not found.")
            sys.exit(1)
        files = [text_file]
    else:
        files = sorted(CONTENT_DIR.glob("*.txt"))
        if not files:
            print(f"No .txt files found in {CONTENT_DIR}/")
            sys.exit(1)

    print(f"Generating audio for {len(files)} file{'s' if len(files) != 1 else ''}...\n")
    for text_file in files:
        generate_audio(text_file, api_key)

    print("\nAll done.")


if __name__ == "__main__":
    main()
