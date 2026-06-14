import json
from app.core.llm import OllamaClient
from app.router.prompts import ROUTER_SYSTEM_PROMPT

class RoutingEngine:
    def __init__(self):
        self.client = OllamaClient()
        self.sessions = {}

    def get_or_create_session(self, session_id: str) -> list:
        if session_id not in self.sessions:
            self.sessions[session_id] = [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT}
            ]
        return self.sessions[session_id]

    def _is_route_trusted(self, parsed_response: dict) -> bool:
        """
        Validates whether the 3B model's own answer can be trusted.
        Requires a confidence score of 85+ and at least 70% probability
        assigned to the winning category.
        """
        confidence = parsed_response.get("confidence_score", 0)
        recommended = parsed_response.get("recommended", "")
        probabilities = parsed_response.get("probabilities", {})
        
        primary_prob = probabilities.get(recommended, 0.0)
        
        if confidence >= 85 and primary_prob >= 70.0:
            return True
        return False

    def process_query(self, session_id: str, user_input: str) -> dict:
        """
        Processes queries using an efficient escalation architecture:
        1. Exploit: If the fast 3B model is highly confident, serve its response instantly.
        2. Escalate: If confidence falls below the threshold, dynamically route the
           context to a heavier, specialized worker model.
        """
        from app.core import config

        # 1. Input Guardrail
        if not user_input or not user_input.strip():
            return self._fallback_error("Input was empty. Please provide a query.")

        # Establish session history and append current user payload
        history = self.get_or_create_session(session_id)
        history.append({"role": "user", "content": user_input})
        
        # Invoke the fast classification/response model
        raw_json_output = self.client.generate_route(history)
        
        try:
            parsed_response = json.loads(raw_json_output)
            
            # 2. Deterministic Inference: Find the highest probability category
            probabilities = parsed_response.get("probabilities", {})
            if probabilities:
                highest_category = max(probabilities, key=probabilities.get)
                parsed_response["recommended"] = highest_category
            else:
                raise ValueError("Model failed to populate the probabilities matrix.")

            if not parsed_response.get("response"):
                raise ValueError("Model generated an empty response string.")
                
            # 3. Dynamic Escalation Logic
            target_category = parsed_response["recommended"]
            is_trusted = self._is_route_trusted(parsed_response)

            if is_trusted and target_category == "General":
                # FAST PATH: 3B handles casual chat completely
                parsed_response["target_model"] = config.ROUTER_MODEL
                final_answer = parsed_response["response"]
                
            elif not is_trusted:
                # CLARIFICATION PATH: The router is deeply confused
                parsed_response["target_model"] = config.ROUTER_MODEL
                parsed_response["recommended"] = "Clarification Needed"
                final_answer = "I am not entirely confident how to handle this request. Could you clarify if you are asking for coding help, general reasoning, or something else?"
                
            else:
                # ESCALATION PATH: High confidence, but it requires a specialized worker
                target_model = config.WORKER_MODELS.get(target_category, config.ROUTER_MODEL)
                parsed_response["target_model"] = target_model
                
                print(f"\n\033[33m[Escalation]\033[0m Routing specialized {target_category} task to {target_model}...")
                
                # Clean the history context for the worker
                worker_history = [msg for msg in history if msg["role"] != "system"]
                
                worker_system_prompts = {
                    "Coding": "You are an expert software engineer. Provide clean, optimized, well-commented code.",
                    "Reasoning": "You are an advanced logic engine. Break down problems step-by-step to find exact solutions.",
                    "Vision": "You are an advanced computer vision assistant analyzing image data alongside text context."
                }
                
                chosen_prompt = worker_system_prompts.get(target_category, "You are a helpful assistant.")
                worker_history.insert(0, {"role": "system", "content": chosen_prompt})
                
                # Fetch the streamed response from Qwen
                final_answer = self.client.generate_worker_response(target_model, worker_history)
                
            # Overwrite the weak 3B response with the final payload
            parsed_response["response"] = final_answer
            
            # --- ADD THESE TWO LINES BACK ---
            # Commit the final conversational response to session memory
            history.append({"role": "assistant", "content": final_answer})
            
            return parsed_response
            # --------------------------------
            
        except (json.JSONDecodeError, ValueError) as e:
            # History Rollback: Remove corrupt context to protect subsequent queries
            if history and history[-1]["role"] == "user":
                history.pop() 
            return self._fallback_error(f"Failed to parse generation: {str(e)}")

    def _fallback_error(self, message: str) -> dict:
        from app.core import config
        return {
            "confidence_score": 0,
            "probabilities": {"General": 100.0, "Reasoning": 0.0, "Coding": 0.0, "Vision": 0.0},
            "recommended": "Error",
            "target_model": config.ROUTER_MODEL,
            "response": message
        }