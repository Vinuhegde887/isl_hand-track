import os
import requests
import json
import time

# Primary and Secondary Keys
# You can set these via environment variables or hardcode them here.
KEY_1 = os.getenv("GOOGLE_API_KEY") or "AIzaSyCaK4VAFFrqdZFxBQU4htLDb0uq2SzY2KE"
KEY_2 = os.getenv("GOOGLE_API_KEY_2") or "AIzaSyDq3FPprG_mdp6tJl-enlJxtHsKsSLFarc" # User to fill or set env var

API_KEYS = [KEY_1, KEY_2]
MODEL = "gemini-2.5-flash"

def call_gemini(tokens, confidences=None, temperature=0.05, max_tokens=120):
    if confidences is None:
        confidences = ["NA"] * len(tokens)

    prompt = f"""
You are a strict sign-gloss → English converter.
Input is a list of gloss tokens and optional confidences.
Rules:
- Return ONLY one fluent English sentence.
- Do NOT include explanations or reasoning.
- Fix repetition, ordering, and missing small words (is, the, a, to).
- Merge compound gloss forms into the correct English meaning.
- Important compound mappings include:
    • PLACE + WHAT → "where"
    • PERSON + WHAT → "who"
    • TIME + WHAT → "when"
    • REASON + WHAT → "why"
    • THING + WHAT → "what"
- If compound logic applies, convert directly to the natural English question/phrase.

Examples:
TOKENS: [HELLO, WHAT, PERSON]
OUTPUT: Hello — which person?

TOKENS: [NAME, WHAT]
OUTPUT: What is the name?

TOKENS: [PLACE, WHAT]
OUTPUT: Where?


Now convert:
TOKENS: {tokens}
CONFIDENCES: {confidences}
OUTPUT:
"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature
        }
    }

    last_error = None
    last_response_text = ""

    for i, api_key in enumerate(API_KEYS):
        # Skip if key is placeholder
        if "YOUR_SECOND_KEY" in api_key:
            continue

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={api_key}"
        
        try:
            response = requests.post(endpoint, json=payload)
            
            # Check for generic success
            if response.status_code == 200:
                data = response.json()
                # Double check unrelated errors in 200 OK body if any (Gemini usually returns non-200 for errors, but safety filters returned in 200)
                # But here we care about Quota.
                try:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    return text.strip().replace("\n", " ")
                except (KeyError, IndexError) as ignored:
                    # Could be safety block or other issue. 
                    # If it's not quota related, maybe we shouldn't retry? 
                    # But for now assume we return whatever we have if it's not an HTTP error.
                    return json.dumps(data, indent=2)

            # If we are here, status code is not 200.
            # Check for 429 (Too Many Requests) or 403 (Quota)
            if response.status_code in [429, 503, 403]:
                print(f"Key {i+1} failed with status {response.status_code}. Retrying with next key...")
                last_error = f"Status {response.status_code}"
                last_response_text = response.text
                continue
            
            # Other errors (400, etc) might be permanent, but let's treat them as failures and raise/return.
            response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            print(f"Key {i+1} connection error: {e}. Retrying...")
            last_error = str(e)
            continue

    # If all keys failed
    return f"Error: All API keys failed. Last error: {last_error} | {last_response_text}"
