#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the eburon package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


async def test_eburon():
    try:
        print("🚀 Testing Eburon Voice Hub...")

        # Test basic imports without audio components
        from eburon.llms import LiteLLM

        print("✅ Successfully imported LiteLLM")

        # Test LLM connectivity (requires OPENAI_API_KEY)
        llm = LiteLLM(model="gpt-4o-mini", provider="openai", temperature=0.2)
        print("✅ Successfully created LLM instance")

        # Test basic LLM call
        response = await llm.generate("Hello, can you tell me what Eburon Voice Hub is?")
        print(f"✅ LLM Response: {response[:100]}...")

        print("\n🎉 Eburon Voice Hub is working! You can now:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Run the full examples in the examples/ directory")
        print("3. Use Docker setup for complete telephony features")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Some dependencies are missing. Try the Docker setup instead.")
    except Exception as e:
        print(f"❌ Error: {e}")
        if "OPENAI_API_KEY" in str(e):
            print("💡 Set OPENAI_API_KEY environment variable to test LLM functionality")


if __name__ == "__main__":
    asyncio.run(test_eburon())
