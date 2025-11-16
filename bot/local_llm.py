# bot/local_llm.py
import requests
import json
import traceback
from . import config # Import config for URL and model

def get_llm_decision(market_data: dict) -> dict | None:
    """
    Sends market data to local LLM (Ollama) and gets a decision.
    Returns a dictionary with the decision or None on error.
    """
    # Form prompt for LLM
    prompt = f"""You are a crypto trading analyst. Based on the following market data for {market_data.get('pair', 'the asset')} on the {market_data.get('timeframe', 'current')} timeframe, provide a trading suggestion.

Market Data:
{json.dumps(market_data, indent=2)}

Analyze the data considering the model prediction, technical indicators, and overall trend context.

Respond ONLY with a JSON object containing:
- "suggested_action": "LONG", "SHORT", or "HOLD"
- "confidence": A float between 0.0 and 1.0 indicating your confidence in the action.
- "reason": A brief explanation (max 2 sentences).

Example JSON response:
{{
  "suggested_action": "LONG",
  "confidence": 0.83,
  "reason": "Strong bullish signal from model confirmed by EMA crossover and moderate RSI. Volume supports the move."
}}

Your JSON response:"""

    payload = {
        "model": config.LLM_MODEL_NAME,
        "prompt": prompt,
        "format": "json", # Ask Ollama to return JSON
        "stream": False   # Non-streaming response
    }

    print(f"Sending request to LLM ({config.LLM_MODEL_NAME})...")
    try:
        response = requests.post(config.OLLAMA_API_URL, json=payload, timeout=30) # 30 sec timeout
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

        # Ollama may return JSON inside the 'response' field
        response_data = response.json()
        
        # Try to parse JSON from LLM response
        try:
             # Look for JSON inside response string
             json_response_str = response_data.get('response', '{}').strip()
             # Remove possible ```json ... ``` wrappers
             if json_response_str.startswith("```json"):
                  json_response_str = json_response_str[7:]
             if json_response_str.endswith("```"):
                  json_response_str = json_response_str[:-3]
             
             llm_decision = json.loads(json_response_str)
             
             # Check for required keys
             if all(k in llm_decision for k in ["suggested_action", "confidence", "reason"]):
                 print(f"✅ LLM response received: {llm_decision}")
                 return llm_decision
             else:
                 print(f"⚠️ LLM response does not contain all keys: {llm_decision}")
                 return None
                 
        except json.JSONDecodeError as json_err:
             print(f"❌ Error decoding LLM JSON response: {json_err}")
             print(f"   Received text: {response_data.get('response', 'N/A')}")
             return None
        except Exception as parse_err:
             print(f"❌ Unknown error parsing LLM response: {parse_err}")
             print(f"   Response data: {response_data}")
             return None


    except requests.exceptions.RequestException as req_err:
        print(f"❌ Error connecting to Ollama API ({config.OLLAMA_API_URL}): {req_err}")
        return None
    except Exception as e:
        print(f"❌ Unknown error when requesting LLM: {e}")
        traceback.print_exc(limit=2)
        return None

if __name__ == '__main__':
    # Example usage
    dummy_data = {
        "pair": "SOLUSDT", "timeframe": "1h",
        "model_prediction": {"LONG": 0.78, "SHORT": 0.22},
        "indicators": {"EMA_9": 190.4, "EMA_55": 189.9, "EMA_100": 189.2, "RSI_14": 57.1, "ATR_14": 0.45, "Volume": 10000.0, "AvgVolume": 8000.0},
        "filter_result": {"long_confirmed": True, "short_confirmed": False},
        "open_position": False
    }
    decision = get_llm_decision(dummy_data)
    if decision:
        print("\nFinal LLM decision:")
        print(decision)
    else:
        print("\nFailed to get decision from LLM.")