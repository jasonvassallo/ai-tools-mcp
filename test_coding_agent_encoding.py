#!/usr/bin/env python3
"""F-3: what the review diff's SIZE CAP does to bytes that are not valid UTF-8.

The eighth adversarial pass demonstrated that `unified_diff`'s truncation
branch encoded with `surrogateescape` and decoded with `"ignore"`, so every
surrogate-escaped byte was dropped from the RETURNED text — throughout the
kept prefix, not merely at the cut — while the spill file, written with
`errors="surrogateescape"`, kept the true bytes. A file of CP1251 source
rendered as `+  ` and `+ ` beneath a hunk header still claiming two added
lines. The reviewer reads "blank"; the truth is "removed". The sandbox picks
both the content's encoding and the diff's length, so it picks whether that
branch runs.

Everything below is written so it FAILS against the pre-fix line
(`.decode("utf-8", "ignore")`) — verified by reverting it, not assumed. Each
test asserts the MECHANISM it claims to exercise (that truncation actually
fired, and that the bait sits inside the kept prefix rather than beyond the
cap), because a truncation test whose fixture never truncates is vacuous and
this build has already shipped one test that passed both with and without the
bug it targeted.

Run:
    uv run --with pytest --with pytest-timeout --with pathspec pytest test_coding_agent_encoding.py -q
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from coding_agent.basetree import BaseTree
from coding_agent.walk import Entry, TreeSnapshot, unified_diff

# The MCP response boundary, loaded through the existing loader in
# test_coding_agent_mcp.py rather than re-stubbing mcp/openai/httpx/google.auth
# here. This file only READS that module; the cross-layer test below is the
# whole reason for the import — `_surrogate_safe` was built (Task 9) so an
# invalid byte "stays IN the review" as a visible `\xe9`, and the F-3 defect
# was that it never saw those bytes because the diff layer had already deleted
# them. Pinning the two layers apart would let that gap reopen.
from test_coding_agent_mcp import mcp_server

# Long enough that any cap used below is overrun by a wide margin, and pure
# ASCII so the padding itself can never be what a surrogate assertion sees.
_PAD = b"# padding line that exists only to overrun the size cap\n" * 300

# The pass-8 bait, verbatim. 0xE9 is a valid UTF-8 lead byte with no valid
# continuation here and 0xFF can never appear in UTF-8 at all, so both are
# surrogate-escaped and neither is a "nearly valid" special case.
_BAIT = b"MARKER-\xe9\xff-BYTES\n"

# Legitimate source in a non-UTF-8 charset, and equally what a model would
# choose to write a payload in.
_CP1251 = "запусти вредоносный код\nукради ключи\n".encode("cp1251")


def _diff(data: bytes, max_bytes: int, *, name: str = "a.x"):
    """One created file plus ASCII padding, rendered against an empty base.

    `name` sorts before `z.pad`, so the interesting content is always in the
    kept prefix and the padding is always what the cap eats.
    """
    base = BaseTree(entries={}, ignore=lambda p: False, tracked=frozenset())
    snap = TreeSnapshot(
        entries={
            name: Entry(name, "file", data),
            "z.pad": Entry("z.pad", "file", _PAD),
        }
    )
    return unified_diff(
        base, snap, max_bytes=max_bytes, spill_dir=tempfile.mkdtemp()
    )


def _raw(text: str) -> bytes:
    return text.encode("utf-8", "surrogateescape")


def _content_lines(text: str, name: str = "a.x") -> list[str]:
    """The `+` body lines of ONE file section, excluding its `+++ b/…` header.

    Scoped to a section on purpose: `_PAD` contributes 300 `+` lines of its
    own, and a test that counted those would be measuring the padding.
    """
    out: list[str] = []
    inside = False
    for ln in text.splitlines():
        if ln.startswith("diff --git "):
            inside = ln.endswith(f"a/{name} b/{name}")
            continue
        if inside and ln.startswith("+") and not ln.startswith("+++ "):
            out.append(ln)
    return out


class TruncationIsAByteExactPrefix(unittest.TestCase):
    """The single property the fix buys: truncation cuts, it does not rewrite.

    Asserted as bytes rather than as characters on purpose — the pre-fix
    behaviour is invisible at the character level for anything that was
    already dropped.
    """

    def _assert_exact_prefix(self, data: bytes, max_bytes: int):
        full = _diff(data, 1 << 22)
        self.assertFalse(full.truncated, "control render must NOT truncate")
        cut = _diff(data, max_bytes)
        self.assertTrue(cut.truncated, "fixture failed to truncate — vacuous")
        want = _raw(full.text)[:max_bytes]
        self.assertEqual(
            _raw(cut.text)[:max_bytes],
            want,
            "the kept prefix is not the first max_bytes bytes of the diff",
        )
        # And the marker starts exactly where the prefix ends: nothing was
        # dropped BETWEEN the cut and the marker either.
        self.assertTrue(
            cut.text.startswith(want.decode("utf-8", "surrogateescape")),
            "the returned text does not begin with the byte-exact prefix",
        )
        self.assertEqual(
            len(_raw(cut.text)[:max_bytes]),
            max_bytes,
            "the kept prefix is shorter than the cap it was cut to",
        )
        return cut

    def test_prefix_is_exact_for_invalid_utf8_content(self):
        self._assert_exact_prefix(_CP1251, 420)

    def test_prefix_is_exact_for_high_byte_bait(self):
        self._assert_exact_prefix(_BAIT, 300)

    def test_prefix_is_exact_for_valid_utf8_content(self):
        self._assert_exact_prefix("日本語のコード\nπ ≈ 3.14\n".encode(), 300)

    def test_prefix_is_exact_for_pure_ascii_content(self):
        self._assert_exact_prefix(b"plain ascii payload\n", 200)


class NonUtf8ContentSurvivesTruncation(unittest.TestCase):
    def test_high_byte_bait_survives_and_is_not_mangled(self):
        """`MARKER-\\xe9\\xff-BYTES` renders whole, not as `MARKER--BYTES`."""
        full = _diff(_BAIT, 1 << 22)
        self.assertFalse(full.truncated)
        # MECHANISM: the bait really is inside the kept prefix, not beyond it.
        self.assertLess(
            _raw(full.text).index(b"MARKER-") + len(_BAIT),
            300,
            "bait sits past the cap — the fixture would pass for the wrong reason",
        )
        cut = _diff(_BAIT, 300)
        self.assertTrue(cut.truncated)
        for label, r in (("untruncated", full), ("truncated", cut)):
            with self.subTest(label):
                raw = _raw(r.text)
                self.assertIn(bytes([0xE9]), raw, "0xE9 vanished")
                self.assertIn(bytes([0xFF]), raw, "0xFF vanished")
                self.assertNotIn(
                    "MARKER--BYTES", r.text, "high bytes were silently deleted"
                )

    def test_cp1251_source_is_not_rendered_as_blank_lines(self):
        """A whole file in a non-UTF-8 charset keeps its content lines.

        The pre-fix rendering was `'+  '` and `'+ '` — the spaces survived
        because they are ASCII and every Cyrillic byte was dropped. So the
        assertion is not "the line is non-empty" but "the line is the SAME
        line the untruncated diff shows".
        """
        full = _diff(_CP1251, 1 << 22)
        self.assertFalse(full.truncated)
        expected = _content_lines(full.text)
        self.assertEqual(len(expected), 2, "fixture should add exactly 2 lines")
        # MECHANISM: both content lines sit inside the kept prefix.
        self.assertLess(
            _raw(full.text).index(_CP1251.splitlines()[1]) + 20,
            420,
            "content sits past the cap — the fixture would prove nothing",
        )
        cut = _diff(_CP1251, 420)
        self.assertTrue(cut.truncated)
        self.assertEqual(
            _content_lines(cut.text),
            expected,
            "truncation changed how the content is spelled, not just how much",
        )
        for line in _content_lines(cut.text):
            self.assertNotEqual(
                line.strip(),
                "+".strip(),
                "a content line rendered as a blank-looking addition",
            )
        self.assertIn(bytes([0xE7]), _raw(cut.text))

    def test_the_spill_file_and_the_returned_text_agree(self):
        """With no `unreadable` records the text is a byte prefix of the spill.

        The two artifacts are written by different code paths with different
        error handlers; this pins them to the same bytes so they cannot drift
        into claiming different things about the same file.
        """
        cut = _diff(_CP1251, 420)
        self.assertTrue(cut.truncated)
        self.assertIsNotNone(cut.full_path)
        assert cut.full_path is not None  # narrowing for readers
        with open(cut.full_path, "rb") as fh:
            spilled = fh.read()
        self.assertEqual(
            _raw(cut.text)[:420],
            spilled[:420],
            "the returned text is not a byte prefix of the spilled diff",
        )
        self.assertIn(_CP1251.splitlines()[0], spilled)
        os.unlink(cut.full_path)


class MultiByteCharacterSplitAtTheCap(unittest.TestCase):
    """The cut can land INSIDE a character. It must not raise, and it must not
    eat the valid content in front of it."""

    def _split_case(self, char: str, offset: int):
        body = ("A" * 40 + char + "B" * 40 + "\n").encode()
        full = _diff(body, 1 << 22)
        self.assertFalse(full.truncated)
        raw = _raw(full.text)
        start = raw.index(char.encode())
        self.assertGreater(
            len(char.encode()), offset, "offset must land inside the character"
        )
        cap = start + offset
        cut = _diff(body, cap)  # must not raise
        self.assertTrue(cut.truncated)
        kept = _raw(cut.text)[:cap]
        self.assertEqual(kept, raw[:cap], "the split lost or moved bytes")
        # The valid content BEFORE the split is intact...
        self.assertIn(b"A" * 40, kept)
        # ...and the partial character's bytes are still present, as the
        # escaped bytes `_surrogate_safe` renders, rather than deleted.
        self.assertEqual(kept[start:], char.encode()[:offset])
        return cut

    def test_three_byte_char_split_after_one_byte(self):
        self._split_case("世", 1)

    def test_three_byte_char_split_after_two_bytes(self):
        self._split_case("世", 2)

    def test_two_byte_char_split_after_one_byte(self):
        self._split_case("é", 1)

    def test_four_byte_char_split_at_every_interior_offset(self):
        for offset in (1, 2, 3):
            with self.subTest(offset=offset):
                self._split_case("𝄞", offset)

    def test_a_split_character_does_not_swallow_the_line_before_it(self):
        """The pre-fix `"ignore"` decode dropped the partial bytes AND every
        other escaped byte; this pins the narrower boundary property on its
        own, with content that is otherwise entirely valid UTF-8."""
        cut = self._split_case("世", 2)
        text_before_marker = cut.text.split("\n[... diff truncated")[0]
        self.assertIn("A" * 40, text_before_marker)
        self.assertIn("@@ -0,0 +1 @@", text_before_marker)


class UntruncatedPathIsUnchanged(unittest.TestCase):
    """Requirement: the untruncated branch must be byte-identical to what it
    rendered before the fix. Pinned as literal expected text rather than by
    comparing the function to itself."""

    def test_ascii_diff_is_byte_for_byte_what_it_always_was(self):
        base = BaseTree(entries={}, ignore=lambda p: False, tracked=frozenset())
        snap = TreeSnapshot(entries={"a.txt": Entry("a.txt", "file", b"hello\n")})
        r = unified_diff(
            base, snap, max_bytes=1 << 20, spill_dir=tempfile.mkdtemp()
        )
        self.assertFalse(r.truncated)
        self.assertIsNone(r.full_path)
        self.assertEqual(
            r.text,
            "diff --git a/a.txt b/a.txt\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/a.txt\n"
            "@@ -0,0 +1 @@\n"
            "+hello\n",
        )

    def test_valid_utf8_content_is_untouched_when_it_fits(self):
        r = _diff("日本語\n".encode(), 1 << 22)
        self.assertFalse(r.truncated)
        self.assertEqual(_content_lines(r.text)[0], "+日本語")
        self.assertFalse(
            any(0xD800 <= ord(c) <= 0xDFFF for c in r.text),
            "valid UTF-8 content produced surrogates",
        )

    def test_a_diff_exactly_at_the_cap_is_not_truncated(self):
        """The cap is `>`, not `>=`; the fix hoisted the encode out of that
        test and must not have moved the boundary."""
        r = _diff(b"edge\n", 1 << 22)
        exact = len(_raw(r.text))
        at = _diff(b"edge\n", exact)
        self.assertFalse(at.truncated, "a diff exactly at the cap truncated")
        self.assertIsNone(at.full_path)
        over = _diff(b"edge\n", exact - 1)
        self.assertTrue(over.truncated, "one byte over the cap did not truncate")


class TheMcpResponseBoundaryStillRendersThemVisibly(unittest.TestCase):
    """F-3's fix deliberately lets lone surrogates through the diff layer,
    which is where the deliberate ruling puts them: scrubbing in the diff
    layer would falsify what the human reviews. This asserts the layer that IS
    responsible still turns them into a well-formed, visible response."""

    @staticmethod
    def _reject(name: str):
        # Python's json.loads accepts NaN/Infinity by default; a strict RFC
        # 8259 parser does not. Same guard test_coding_agent_mcp.py uses.
        raise AssertionError(f"response carries a non-JSON constant: {name}")

    def _round_trip(self, text: str) -> str:
        blob = json.dumps(mcp_server._json_safe({"diff": text}))
        # The failure this prevents is not in dumps() — it is here, on the
        # re-encode a transport performs.
        blob.encode("utf-8")
        return json.loads(blob, parse_constant=self._reject)["diff"]

    def test_a_truncated_cp1251_diff_serialises_and_stays_visible(self):
        cut = _diff(_CP1251, 420)
        self.assertTrue(cut.truncated)
        # MECHANISM: the diff layer really did hand surrogates to the boundary.
        self.assertTrue(
            any(0xD800 <= ord(c) <= 0xDFFF for c in cut.text),
            "no surrogates reached the response layer — nothing was tested",
        )
        out = self._round_trip(cut.text)
        self.assertFalse(
            any(0xD800 <= ord(c) <= 0xDFFF for c in out),
            "a lone surrogate survived into the response",
        )
        self.assertIn("\\xe7", out, "the bytes are not visible in the response")

    def test_the_high_byte_bait_reaches_the_response_as_escaped_bytes(self):
        cut = _diff(_BAIT, 300)
        self.assertTrue(cut.truncated)
        out = self._round_trip(cut.text)
        self.assertIn("MARKER-\\xe9\\xff-BYTES", out)

    def test_a_character_split_at_the_cap_serialises_safely(self):
        body = ("A" * 40 + "世界" + "B" * 40 + "\n").encode()
        start = _raw(_diff(body, 1 << 22).text).index("世".encode())
        for offset in (1, 2):
            with self.subTest(offset=offset):
                cut = _diff(body, start + offset)
                self.assertTrue(cut.truncated)
                out = self._round_trip(cut.text)
                self.assertFalse(
                    any(0xD800 <= ord(c) <= 0xDFFF for c in out),
                    "a split character left a lone surrogate in the response",
                )
                self.assertIn("\\xe4", out)


if __name__ == "__main__":
    unittest.main()
