class EburonComponentError(Exception):
    """Base exception that carries component context for error attribution."""

    def __init__(self, message, component, provider=None, model=None):
        self.component = component
        self.provider = provider
        self.model = model
        super().__init__(message)


class LLMError(EburonComponentError):
    def __init__(self, message, provider=None, model=None):
        super().__init__(message, component="llm", provider=provider, model=model)


class SynthesizerError(EburonComponentError):
    def __init__(self, message, provider=None, model=None):
        super().__init__(message, component="synthesizer", provider=provider, model=model)


class TranscriberError(EburonComponentError):
    def __init__(self, message, provider=None, model=None):
        super().__init__(message, component="transcriber", provider=provider, model=model)
