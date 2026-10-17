"""Explicit exception types so callers can react to specific failures."""


class RagError(Exception):
    """Base class for all application errors."""


class ConfigError(RagError):
    """Invalid or missing configuration."""


class DocumentError(RagError):
    """A PDF could not be used (corrupt, encrypted, empty, scanned, too large)."""


class KnowledgeBaseError(RagError):
    """Base class for knowledge-base index problems."""


class IndexMissingError(KnowledgeBaseError):
    """No index has been built yet."""


class IndexIncompatibleError(KnowledgeBaseError):
    """The stored index was built with settings that cannot be used safely."""


class IndexCorruptError(KnowledgeBaseError):
    """Index files are unreadable or inconsistent."""


class IndexBusyError(KnowledgeBaseError):
    """Another process or session is currently writing the index."""


class LLMError(RagError):
    """Base class for language-model failures."""


class LLMUnavailableError(LLMError):
    """The Ollama server cannot be reached."""


class ModelNotFoundError(LLMError):
    """The configured model is not pulled on the Ollama server."""


class LLMTimeoutError(LLMError):
    """The model did not respond within the configured timeout."""


class LLMStreamError(LLMError):
    """The response stream broke off part-way through."""
