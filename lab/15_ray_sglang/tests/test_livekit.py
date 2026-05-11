"""Tests for A2UI channel, AG-UI handler, frontend tools, and model config."""

from __future__ import annotations

from unittest import TestCase, main


class TestA2UIChannel(TestCase):
    """A2UIChannel protocol compliance."""

    def setUp(self):
        self.channel = None  # Requires LiveKit room mock

    def test_create_surface_returns_id(self):
        # Unit: create_surface returns the surface_id for chaining
        pass

    def test_delete_surface_removes_from_active(self):
        pass


class TestFrontendTools(TestCase):
    """Frontend tool definitions should return correct schemas."""

    def test_get_user_location_returns_pending(self):
        from ..livekit.frontend_tools import get_user_location
        result = get_user_location()
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["tool"], "get_user_location")

    def test_pick_file_accepts_mime_filter(self):
        from ..livekit.frontend_tools import pick_file
        result = pick_file(accept="image/*")
        self.assertEqual(result["accept"], "image/*")

    def test_read_clipboard_returns_pending(self):
        from ..livekit.frontend_tools import read_clipboard
        result = read_clipboard()
        self.assertEqual(result["tool"], "read_clipboard")

    def test_write_clipboard_includes_text(self):
        from ..livekit.frontend_tools import write_clipboard
        result = write_clipboard(text="hello")
        self.assertEqual(result["text"], "hello")


class TestModelProfiles(TestCase):
    """ModelProfile validation and configuration."""

    def test_get_model_returns_profile(self):
        from ..config.models import get_model
        profile = get_model("orchestrator")
        self.assertEqual(profile.role, "orchestrator")

    def test_get_model_fallback_to_default(self):
        from ..config.models import get_model
        profile = get_model("nonexistent_role")
        self.assertIsNotNone(profile)

    def test_invalid_role_raises(self):
        from ..config.models import ModelProfile
        with self.assertRaises(ValueError):
            ModelProfile(name="test", model_id="org/model", sglang_port=1, role="invalid")

    def test_invalid_model_id_raises(self):
        from ..config.models import ModelProfile
        with self.assertRaises(ValueError):
            ModelProfile(name="test", model_id="bad", sglang_port=1, role="orchestrator")

    def test_sglang_endpoint_property(self):
        from ..config.models import get_model
        profile = get_model("orchestrator")
        self.assertIn(str(profile.sglang_port), profile.sglang_endpoint)

    def test_configure_lm_returns_dspy_lm(self):
        from ..config.models import configure_lm
        lm = configure_lm(role="orchestrator")
        self.assertIsNotNone(lm)


class TestPickleModule(TestCase):
    """Pickle module utility rate-limiting."""

    def test_pickle_warning_rate_limited(self):
        from ..ray.executor import _pickle_module, _pickle_warnings_logged
        _pickle_warnings_logged.clear()
        # Test with a non-picklable object (lambda)
        result = _pickle_module(lambda x: x)
        self.assertIsNone(result)
        # Second call should not raise — just return None
        result2 = _pickle_module(lambda x: x)
        self.assertIsNone(result2)


if __name__ == "__main__":
    main()
