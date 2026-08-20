"""The SpeechLab research agent: an LLM assistant with acoustic tools.

Requires an OpenAI-compatible API endpoint:

    export SPEECHLAB_API_KEY=sk-...
    export SPEECHLAB_BASE_URL=https://api.openai.com/v1   # optional
    export SPEECHLAB_MODEL=gpt-4o-mini                    # optional

Usage: python examples/research_agent.py "your research question"
"""

import sys

from speechlab.agent import AgentConfig, SpeechResearchAgent


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    question = " ".join(sys.argv[1:])

    agent = SpeechResearchAgent(AgentConfig())
    answer = agent.ask(question)
    print(answer)


if __name__ == "__main__":
    main()
