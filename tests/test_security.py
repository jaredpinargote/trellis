"""
Security-focused tests for the sanitize_text function.
"""
import pytest
from app.security import sanitize_text


class TestSanitizeText:
    def test_normal_text_unchanged(self):
        text = "The court ruled in favor of the defendant after a lengthy trial."
        assert sanitize_text(text) == text

    def test_strips_script_tags(self):
        text = "<script>alert('xss')</script>Hello World"
        result = sanitize_text(text)
        assert "<script>" not in result
        assert "Hello World" in result

    def test_strips_nested_html(self):
        text = "<div><b>Bold</b> <a href='evil.com'>click</a></div> Normal text"
        result = sanitize_text(text)
        assert "<div>" not in result
        assert "<b>" not in result
        assert "Normal text" in result

    def test_removes_null_bytes(self):
        text = "Hello\x00World"
        result = sanitize_text(text)
        assert "\x00" not in result
        assert "HelloWorld" in result

    def test_removes_control_characters(self):
        text = "Hello\x07World\x0eTest"
        result = sanitize_text(text)
        assert "\x07" not in result
        assert "\x0e" not in result

    def test_preserves_newlines_and_tabs(self):
        text = "Line 1\nLine 2\tTabbed"
        result = sanitize_text(text)
        assert "\n" in result
        assert "\t" in result

    def test_path_traversal_stripped(self):
        text = "../../../../etc/passwd important document"
        result = sanitize_text(text)
        assert "../" not in result

    def test_collapses_excessive_whitespace(self):
        text = "Hello     World     Test"
        result = sanitize_text(text)
        # Should collapse runs of 3+ spaces to 2
        assert "     " not in result

    def test_sql_injection_detected_not_stripped(self):
        """SQL patterns are logged but NOT stripped (could be legit legal text)."""
        text = "SELECT * FROM documents WHERE id = 1"
        result = sanitize_text(text)
        # Text should still contain the SQL since we only log, don't strip
        assert "SELECT" in result

    def test_mixed_attacks(self):
        """Multiple attack vectors in one input."""
        text = "<script>alert(1)</script>\x00../../etc\x07 Normal content here"
        result = sanitize_text(text)
        assert "<script>" not in result
        assert "\x00" not in result
        assert "\x07" not in result
        assert "Normal content here" in result
