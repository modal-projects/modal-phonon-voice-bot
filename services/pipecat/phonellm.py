"""Phonellm Mini on a Modal Endpoint.

Pipecat already speaks this server's OpenAI chat API. The subclass exists so
the console shows this service by name.
"""

from pipecat.services.openai.llm import OpenAILLMService


class ModalPhoneLLMMiniService(OpenAILLMService):
    """Phonellm Mini served from the phonellm-mini-alpha-2 Endpoint."""
