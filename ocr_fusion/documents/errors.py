"""Document loading errors.

These are the failures the UI is expected to explain in plain language, so each
one carries a message written for a non-technical reader (spec s13).
"""

from __future__ import annotations


class DocumentError(Exception):
    """Base class for every document-loading failure."""

    #: Short, client-safe explanation shown in the UI.
    user_message = "The document could not be processed."

    def __init__(self, message: str = "", *, user_message: str | None = None) -> None:
        super().__init__(message or self.user_message)
        if user_message is not None:
            self.user_message = user_message
        elif message:
            self.user_message = message


class UnsupportedFormatError(DocumentError):
    user_message = "That file type is not supported yet."


class CorruptDocumentError(DocumentError):
    user_message = "The file appears to be damaged and could not be opened."


class DocumentTooLargeError(DocumentError):
    user_message = "The file is larger than the configured limit."


class EmptyDocumentError(DocumentError):
    user_message = "The document contains no pages to process."


class PdfRenderError(DocumentError):
    user_message = "The PDF could not be converted to images."


__all__ = [
    "CorruptDocumentError",
    "DocumentError",
    "DocumentTooLargeError",
    "EmptyDocumentError",
    "PdfRenderError",
    "UnsupportedFormatError",
]
