import httpx
import json
from app.core import config

class OllamaClient:
    def __init__(self):
        self.chat_url = config.LLM_CHAT_ENDPOINT
        self.model = config.ROUTER_MODEL

        self.schema = {
            "type": "object",
            "properties": {
                "confidence_score": {"type": "integer"},
                "probabilities": {
                    "type": "object",
                    "properties": {
                        "General": {"type": "number"},
                        "Reasoning": {"type": "number"},
                        "Coding": {"type": "number"},
                        "Vision": {"type": "number"}
                    },
                    "required": ["General", "Reasoning", "Coding", "Vision"]
                },
                "response": {"type": "string"}
            },
            "required": ["confidence_score", "probabilities", "response"]
        }

    def generate_route(self, messages: list) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": self.schema,
            "keep_alive": -1  # Lock the 3B router in VRAM permanently
        }
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(self.chat_url, json=payload)
                response.raise_for_status()
                return response.json().get("message", {}).get("content", "{}")
        except Exception as e:
            return f'{{"error": "Failed to reach LLM endpoint at {self.chat_url}: {str(e)}"}}'
    
    def generate_worker_response(self, target_model: str, messages: list) -> str:
        """Queries the specialized worker models and streams the output in real-time."""
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": True, # Enabled streaming
            "keep_alive": "5m", # Let heavy models unload after 5 mins to save VRAM
            "options": {
                "num_ctx": 4096,       # Expand context window to 8k tokens
                "num_predict": 2048    # Allow the model to generate up to 4k tokens
            }
        }
        
        full_response = ""
        
        print(f"\n\033[36m[{target_model} Output]\033[0m")
        try:
            with httpx.Client(timeout=120.0) as client:
                # Use httpx's streaming connection
                with client.stream("POST", self.chat_url, json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            content = chunk.get("message", {}).get("content", "")
                            
                            # Print to terminal instantly without a newline
                            print(content, end="", flush=True)
                            
                            # Accumulate the string for the history history
                            full_response += content
            
            print("\n") # Add a final newline when the stream finishes
            return full_response
            
        except Exception as e:
            error_msg = f"Worker model ({target_model}) failed to respond: {str(e)}"
            print(error_msg)
            return error_msg