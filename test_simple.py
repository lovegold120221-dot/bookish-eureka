#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the eburon package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


async def test_simple():
    try:
        print("🚀 Testing Eburon Voice Hub...")

        # Test basic LLM functionality
        from eburon.llms import LiteLLM

        print("✅ Successfully imported LiteLLM")

        # Test LLM connectivity (requires OPENAI_API_KEY)
        llm = LiteLLM(model="gpt-4o-mini", provider="openai", temperature=0.2)
        print("✅ Successfully created LLM instance")

        # Test basic LLM call
        response = await llm.generate(
            [{"role": "user", "content": "Hello, can you tell me what Eburon Voice Hub is in one sentence?"}]
        )
        print(f"✅ LLM Response: {response}")

        print("\n🎉 Eburon LLM component is working!")
        print("📝 Next steps:")
        print("1. Set OPENAI_API_KEY environment variable if not already set")
        print("2. Use Docker setup for complete voice AI features")
        print("3. Check examples/ directory for more usage patterns")

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        if "OPENAI_API_KEY" in str(e) or "api key" in str(e).lower():
            print("💡 Set OPENAI_API_KEY environment variable to test LLM functionality")
        else:
            print("💡 Check the error and ensure all dependencies are properly installed")


if __name__ == "__main__":
    asyncio.run(test_simple())
