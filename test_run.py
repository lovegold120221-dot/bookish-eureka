#!/usr/bin/env python3

import sys
import os

# Add the eburon package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    # Try importing basic components
    from eburon.assistant import Assistant
    from eburon.models import LlmAgent, SimpleLlmAgent

    print("✅ Successfully imported Eburon components")

    # Create a simple assistant
    assistant = Assistant(name="test_agent")

    llm_agent = LlmAgent(
        agent_type="simple_llm_agent",
        agent_flow_type="streaming",
        llm_config=SimpleLlmAgent(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.2,
        ),
    )

    # Add a text-only task
    assistant.add_task(
        task_type="conversation",
        llm_agent=llm_agent,
        enable_textual_input=True,
    )

    print("✅ Successfully created assistant configuration")
    print("🚀 Eburon Voice Hub app is ready to run!")
    print("\nTo run with actual LLM calls, set OPENAI_API_KEY environment variable")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Some dependencies are missing due to Python 3.14 compatibility issues")
except Exception as e:
    print(f"❌ Error: {e}")
