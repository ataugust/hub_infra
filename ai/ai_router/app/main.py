import sys
import json
from app.router.engine import RoutingEngine

def run_local_simulator():
    print("====================================================")
    # Using raw ANSI sequences for bold cyan escape styling natively on Arch
    print("\033[1;36mCixio AI Router Engine - Local Terminal Simulator\033[0m")
    print("Type your prompt below to see the JSON routing response.")
    print("Type 'exit' or 'quit' to stop the simulation.")
    print("====================================================\n")

    engine = RoutingEngine()
    session_id = "local_dev_session"

    while True:
        try:
            # Capture user input
            user_input = input("\033[1;32mUser > \033[0m")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("\nShutting down simulator. Goodbye!")
                break
                
            if not user_input.strip():
                continue

            # Process through the core engine
            print("\n\033[33mThinking (Inference running via local Ollama)...\033[0m")
            result = engine.process_query(session_id, user_input)
            
            # Print the exact structured JSON output
            print("\033[1;34mLlama3.2:3b Reply:\033[0m")
            print(json.dumps(result, indent=2))
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n\nExiting simulator safely.")
            sys.exit(0)
        except Exception as e:
            print(f"\033[1;31mError processing request:\033[0m {e}\n")

if __name__ == "__main__":
    run_local_simulator()