
import pytest
from models.utils import validate_text_input, clean_polish_text

# --- Tests for validate_text_input ---

def test_validate_text_input_valid():
    """
    Tests that a valid text string passes validation.
    """
    text = "This is a sufficiently long and valid text for processing. It has more than fifty characters."
    is_valid, result = validate_text_input(text)
    assert is_valid is True
    assert result == text

def test_validate_text_input_too_short():
    """
    Tests that text shorter than the minimum length fails validation.
    """
    text = "Too short."
    is_valid, result = validate_text_input(text)
    assert is_valid is False
    assert result == "Tekst jest zbyt krótki (minimum 50 znaków)."

def test_validate_text_input_too_long():
    """
    Tests that text longer than the maximum length fails validation.
    """
    long_text = "a" * 50001
    is_valid, result = validate_text_input(long_text)
    assert is_valid is False
    assert result == "Tekst jest zbyt długi (maksimum 50000 znaków)."

def test_validate_text_input_empty():
    """
    Tests that an empty string fails validation.
    """
    is_valid, result = validate_text_input("")
    assert is_valid is False
    assert result == "Tekst jest pusty."

def test_validate_text_input_none():
    """
    Tests that None input fails validation.
    """
    is_valid, result = validate_text_input(None)
    assert is_valid is False
    assert result == "Tekst jest pusty."

def test_validate_text_input_whitespace():
    """
    Tests that a string with only whitespace fails validation.
    """
    is_valid, result = validate_text_input("     ")
    assert is_valid is False
    assert result == "Tekst jest zbyt krótki (minimum 50 znaków)."

def test_validate_text_input_at_boundary():
    """
    Tests that text with exactly the minimum required length passes.
    """
    boundary_text = "a" * 50
    is_valid, result = validate_text_input(boundary_text)
    assert is_valid is True
    assert result == boundary_text

# --- Tests for clean_polish_text ---

def test_clean_polish_text_removes_url():
    """
    Tests that URLs are correctly removed from the text.
    """
    text = "Please visit https://example.com for more info."
    cleaned = clean_polish_text(text)
    assert "https://example.com" not in cleaned
    assert cleaned == "Please visit for more info."

def test_clean_polish_text_handles_multiple_spaces():
    """
    Tests that multiple whitespace characters are collapsed into a single space.
    """
    text = "This   has    extra   spaces."
    cleaned = clean_polish_text(text)
    assert cleaned == "This has extra spaces."

def test_clean_polish_text_strips_whitespace():
    """
    Tests that leading/trailing whitespace is removed.
    """
    text = "  Some text with spaces.   "
    cleaned = clean_polish_text(text)
    assert cleaned == "Some text with spaces."

def test_clean_polish_text_no_changes():
    """
    Tests that text without URLs or extra spaces remains unchanged.
    """
    text = "This is a clean sentence."
    cleaned = clean_polish_text(text)
    assert cleaned == text
