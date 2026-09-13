"""
PRESET LOADER -- the shipped presets, read out of ``static/app.js`` at run time
==============================================================================

``static/app.js`` is the ONLY place a model preset is defined. Anything that wants
to test a preset used to keep a hand-copied Python transcription of it, and the two
drifted silently: the EGF/EGFR preset was fixed in app.js and its behavioural test
kept measuring the stale copy, so a red test named the wrong model and a green test
proved nothing about what the app ships.

This module removes the copy. It extracts the ``Presets`` and ``PaperModels`` object
literals from app.js and returns them as plain Python dicts.

WHY NOT json.loads
------------------
The preset tables are JavaScript object literals, not JSON:

    const Presets = {
        egfr: {                                  // unquoted (identifier) keys
            text: `EGF binds to EGFR ...`,       // template literal, spans lines
            targets: [
                { species: "ERK", type: "peak_time", min: 5.0, max: 15.0 },
                { species: "EGFR", type: "decay_ratio", max: 0.2 }  // line comment
            ],                                   // trailing commas
            blueprint: { ... }
        },
    };

So there is a small recursive-descent reader below (``_Reader``) covering exactly
that dialect: identifier / string / numeric keys, single- double- and back-quoted
strings, ``//`` and ``/* */`` comments, trailing commas, ``true`` / ``false`` /
``null``, and numbers. It is a READER, not an evaluator -- there is no ``eval``, no
``json.loads``, no JS engine, and no third-party dependency. Standard library only.

WHAT IT REFUSES vs WHAT IT SKIPS
--------------------------------
The distinction matters more than the parsing does. A loader that returns ``{}``
when it gets confused would make every behavioural test vacuously pass, which is
strictly worse than the drift it replaces. So:

  * SKIPPED, silently and by design -- values this module is documented not to
    need and that carry no model content: function expressions and arrow functions
    (``thermal: (v) => {...}``) and ``undefined``. The key is omitted from the
    result and recorded in ``skipped_keys``.
  * REFUSED with ``PresetLoadError`` -- everything else that is not understood: a
    missing table, a table defined twice, an unterminated string, an unbalanced
    brace, a missing colon, a bare identifier used as a value (that would mean
    model content lives somewhere this reader cannot see), a preset with no
    blueprint and no text, a blueprint with no nodes or no edges, a required
    preset that has disappeared, and end-of-input in the middle of anything.

Every error names the offending line and column in app.js.

USAGE
-----
    import agent, preset_loader

    presets = preset_loader.load_presets()            # 8 records, as shipped
    blueprints = preset_loader.load_blueprints(text_compiler=agent.rule_based_parse)

``turing`` is the one preset that ships ``text`` + ``targets`` and no ``blueprint``
(the user's blueprint is whatever the deterministic parser makes of the text), so
``load_blueprints`` needs ``text_compiler`` to produce it. Passing no compiler
raises rather than dropping the preset.
"""

import copy
import os
import re

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
APP_JS = os.path.join(REPO_DIR, "static", "app.js")

#: The object literals in app.js that hold presets, in load order.
TABLE_NAMES = ("Presets", "PaperModels")

#: Presets that must exist. A shipped preset disappearing is a bug in this loader's
#: pattern or a rename that has to be dealt with deliberately -- never silently.
REQUIRED_PRESETS = (
    "egfr", "turing", "oscillator", "bistable", "foldchange",   # Presets
    "berridge", "zhabotinsky", "lyashenko",                     # PaperModels
)

#: Values that are understood but deliberately not represented in Python.
_SKIP = object()

_IDENT_RE = re.compile(r"[A-Za-z_$][A-Za-z_$0-9]*")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_KEYWORDS = {"true": True, "false": False, "null": None}
_UNSUPPORTED_KEYWORDS = {"undefined", "function"}


class PresetLoadError(RuntimeError):
    """static/app.js could not be found, located within, or parsed.

    Always raised loudly -- this module never degrades to an empty or partial
    preset table, because a silently empty table makes preset tests pass by
    measuring nothing.
    """


# ==========================================================================
# THE READER
# ==========================================================================

class _Reader:
    """Recursive-descent reader for the JS object-literal dialect app.js uses."""

    def __init__(self, src, origin="<string>"):
        self.src = src
        self.n = len(src)
        self.i = 0
        self.origin = origin
        self.skipped_keys = []

    # -- diagnostics -------------------------------------------------------
    def _where(self, index=None):
        index = self.i if index is None else index
        index = max(0, min(index, self.n))
        line = self.src.count("\n", 0, index) + 1
        col = index - (self.src.rfind("\n", 0, index) + 1) + 1
        return "%s:%d:%d" % (self.origin, line, col)

    def fail(self, message, index=None):
        near = self.src[(self.i if index is None else index):][:40].replace("\n", "\\n")
        raise PresetLoadError("%s: %s (near %r)" % (self._where(index), message, near))

    # -- lexical helpers ---------------------------------------------------
    def skip_dead_space(self):
        """Advance past whitespace, ``// line`` and ``/* block */`` comments."""
        while self.i < self.n:
            ch = self.src[self.i]
            if ch in " \t\r\n\f\v":
                self.i += 1
            elif self.src.startswith("//", self.i):
                end = self.src.find("\n", self.i)
                self.i = self.n if end < 0 else end + 1
            elif self.src.startswith("/*", self.i):
                end = self.src.find("*/", self.i + 2)
                if end < 0:
                    self.fail("unterminated /* block comment")
                self.i = end + 2
            else:
                return

    def peek(self):
        self.skip_dead_space()
        if self.i >= self.n:
            self.fail("unexpected end of input")
        return self.src[self.i]

    def expect(self, ch):
        if self.peek() != ch:
            self.fail("expected %r" % ch)
        self.i += 1

    # -- values ------------------------------------------------------------
    def value(self):
        ch = self.peek()
        if ch == "{":
            return self.object_literal()
        if ch == "[":
            return self.array_literal()
        if ch in "\"'`":
            return self.string_literal()
        if ch == "(":
            return self._maybe_arrow_function()
        if ch.isdigit() or ch in "+-." :
            return self.number_literal()
        match = _IDENT_RE.match(self.src, self.i)
        if match:
            word = match.group(0)
            if word in _KEYWORDS:
                self.i = match.end()
                return _KEYWORDS[word]
            if word in ("Infinity", "NaN"):
                self.i = match.end()
                return float(word.replace("Infinity", "inf").replace("NaN", "nan"))
            if word == "undefined":
                self.i = match.end()
                return _SKIP
            if word == "function":
                self.i = match.end()
                self._skip_function_body()
                return _SKIP
            if self._is_arrow_from(match.end()):
                self.i = match.end()
                self._skip_arrow_body()
                return _SKIP
            # A bare reference means the real value lives somewhere this reader
            # cannot see. Refuse -- dropping it would lose model content.
            self.fail("value is the identifier %r; this reader only reads literals "
                      "(move the value inline in app.js, or extend this loader)" % word)
        self.fail("unexpected character %r" % ch)

    def object_literal(self):
        start = self.i
        self.expect("{")
        out = {}
        while True:
            ch = self.peek()
            if ch == "}":
                self.i += 1
                return out
            key = self.object_key()
            self.expect(":")
            value = self.value()
            if value is _SKIP:
                self.skipped_keys.append(key)
            else:
                out[key] = value
            ch = self.peek()
            if ch == ",":
                self.i += 1
                continue
            if ch == "}":
                self.i += 1
                return out
            self.fail("expected ',' or '}' in object literal opened at %s"
                      % self._where(start))

    def object_key(self):
        ch = self.peek()
        if ch in "\"'`":
            return self.string_literal()
        match = _IDENT_RE.match(self.src, self.i)
        if match:
            self.i = match.end()
            return match.group(0)
        match = _NUMBER_RE.match(self.src, self.i)
        if match:
            self.i = match.end()
            return match.group(0)
        self.fail("expected an object key")

    def array_literal(self):
        start = self.i
        self.expect("[")
        out = []
        while True:
            ch = self.peek()
            if ch == "]":
                self.i += 1
                return out
            value = self.value()
            if value is not _SKIP:
                out.append(value)
            ch = self.peek()
            if ch == ",":
                self.i += 1
                continue
            if ch == "]":
                self.i += 1
                return out
            self.fail("expected ',' or ']' in array opened at %s" % self._where(start))

    def string_literal(self):
        quote = self.src[self.i]
        start = self.i
        self.i += 1
        chunks = []
        while True:
            if self.i >= self.n:
                self.fail("unterminated %s string started at %s"
                          % ("template" if quote == "`" else "quoted", self._where(start)),
                          index=start)
            ch = self.src[self.i]
            if ch == "\\":
                self.i += 1
                if self.i >= self.n:
                    self.fail("string ends with a dangling backslash", index=start)
                chunks.append(self._escape(self.src[self.i]))
                self.i += 1
                continue
            if ch == quote:
                self.i += 1
                return "".join(chunks)
            if ch == "\n" and quote != "`":
                self.fail("unterminated %r string (newline inside it)" % quote, index=start)
            if quote == "`" and self.src.startswith("${", self.i):
                self.fail("template literal interpolates ${...}; this reader returns "
                          "text verbatim and cannot evaluate it", index=self.i)
            chunks.append(ch)
            self.i += 1

    _SIMPLE_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f",
                       "v": "\v", "0": "\0"}

    def _escape(self, ch):
        if ch in self._SIMPLE_ESCAPES:
            return self._SIMPLE_ESCAPES[ch]
        if ch == "u":
            if self.src.startswith("{", self.i + 1):
                end = self.src.find("}", self.i)
                if end < 0:
                    self.fail(r"unterminated \u{...} escape")
                code = self.src[self.i + 2:end]
                self.i = end
            else:
                code = self.src[self.i + 1:self.i + 5]
                self.i += 4
            try:
                return chr(int(code, 16))
            except ValueError:
                self.fail(r"bad \u escape %r" % code)
        if ch == "x":
            code = self.src[self.i + 1:self.i + 3]
            self.i += 2
            try:
                return chr(int(code, 16))
            except ValueError:
                self.fail(r"bad \x escape %r" % code)
        if ch == "\n":          # line continuation
            return ""
        return ch               # \\  \'  \"  \`  \/ ...

    def number_literal(self):
        match = _NUMBER_RE.match(self.src, self.i)
        if not match or not match.group(0).strip("+-"):
            self.fail("expected a number")
        raw = match.group(0)
        self.i = match.end()
        if _IDENT_RE.match(self.src, self.i):
            self.fail("number %r is followed by %r; this reader does not evaluate "
                      "expressions" % (raw, _IDENT_RE.match(self.src, self.i).group(0)))
        if "." in raw or "e" in raw or "E" in raw:
            return float(raw)
        return int(raw)

    # -- function values (recognised, skipped) -----------------------------
    def _is_arrow_from(self, index):
        probe = _Reader(self.src, self.origin)
        probe.i = index
        probe.skip_dead_space()
        return probe.src.startswith("=>", probe.i)

    def _maybe_arrow_function(self):
        """``(`` starts either an arrow-function parameter list or an expression."""
        close = self._match_bracket(self.i, "(", ")")
        if self._is_arrow_from(close + 1):
            self.i = close + 1
            self._skip_arrow_body()
            return _SKIP
        self.fail("parenthesised expression; this reader only reads literals")

    def _skip_arrow_body(self):
        self.skip_dead_space()
        if not self.src.startswith("=>", self.i):
            self.fail("expected '=>'")
        self.i += 2
        self.skip_dead_space()
        if self.peek() == "{":
            self.i = self._match_bracket(self.i, "{", "}") + 1
            return
        self._skip_expression()

    def _skip_function_body(self):
        self.skip_dead_space()
        match = _IDENT_RE.match(self.src, self.i)          # optional name
        if match:
            self.i = match.end()
        self.skip_dead_space()
        if self.peek() != "(":
            self.fail("expected a function parameter list")
        self.i = self._match_bracket(self.i, "(", ")") + 1
        self.skip_dead_space()
        if self.peek() != "{":
            self.fail("expected a function body")
        self.i = self._match_bracket(self.i, "{", "}") + 1

    def _skip_expression(self):
        """Skip a concise arrow body: up to the ',' / '}' / ']' that closes it."""
        depth = 0
        while self.i < self.n:
            self.skip_dead_space()
            if self.i >= self.n:
                break
            ch = self.src[self.i]
            if ch in "([{":
                self.i = self._match_bracket(self.i, ch, ")]}"["([{".index(ch)]) + 1
                continue
            if ch in "\"'`":
                self.string_literal()
                continue
            if depth == 0 and ch in ",}])":
                return
            self.i += 1
        self.fail("unexpected end of input inside a function value")

    def _match_bracket(self, index, opener, closer):
        """Index of the ``closer`` matching the ``opener`` at ``index``.

        Strings and comments inside are consumed properly, so a brace in a comment
        or a bracket in a string can never move the match.
        """
        if self.src[index] != opener:
            self.fail("expected %r" % opener, index=index)
        probe = _Reader(self.src, self.origin)
        probe.i = index + 1
        depth = 1
        while probe.i < probe.n:
            probe.skip_dead_space()
            if probe.i >= probe.n:
                break
            ch = probe.src[probe.i]
            if ch in "\"'`":
                probe.string_literal()
                continue
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth == 0:
                    return probe.i
            probe.i += 1
        self.fail("unbalanced %r opened here" % opener, index=index)


# ==========================================================================
# LOCATING THE TABLES
# ==========================================================================

def read_app_js(path=None):
    """Return the text of app.js, or raise ``PresetLoadError`` if it is not there."""
    path = APP_JS if path is None else path
    if not os.path.exists(path):
        raise PresetLoadError("static/app.js not found at %s -- the presets under test "
                              "are defined there and cannot be loaded" % path)
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    if not source.strip():
        raise PresetLoadError("%s is empty" % path)
    return source


def parse_object_literal(source, origin="<string>"):
    """Read one standalone JS object literal (``{...}``) from ``source``."""
    reader = _Reader(source, origin)
    reader.skip_dead_space()
    if reader.peek() != "{":
        reader.fail("expected an object literal")
    value = reader.object_literal()
    return value


def extract_table(source, name, origin="static/app.js"):
    """Read the ``const <name> = { ... };`` object literal out of ``source``."""
    pattern = re.compile(r"(?m)^[ \t]*(?:const|let|var)[ \t]+%s[ \t]*=[ \t]*\{"
                         % re.escape(name))
    matches = list(pattern.finditer(source))
    if not matches:
        raise PresetLoadError(
            "no `const %s = {` declaration in %s -- the preset table was renamed, "
            "moved or deleted; this loader will not guess" % (name, origin))
    if len(matches) > 1:
        lines = [source.count("\n", 0, m.start()) + 1 for m in matches]
        raise PresetLoadError("`%s` is declared %d times in %s (lines %s) -- ambiguous"
                              % (name, len(matches), origin,
                                 ", ".join(str(line) for line in lines)))
    reader = _Reader(source, origin)
    reader.i = matches[0].end() - 1          # sit on the '{'
    table = reader.object_literal()
    reader.skip_dead_space()
    if reader.i < reader.n and reader.src[reader.i] != ";":
        reader.fail("expected ';' after the %s object literal" % name)
    if not table:
        raise PresetLoadError("`const %s` in %s parsed to an EMPTY table; refusing to "
                             "return it, because empty presets make preset tests pass "
                             "without measuring anything" % (name, origin))
    return table, reader.skipped_keys


def load_tables(source=None, path=None):
    """Return ``{table name: {preset name: record}}`` for every table in app.js."""
    origin = "static/app.js" if source is not None and path is None else (path or APP_JS)
    if source is None:
        source = read_app_js(path)
    tables = {}
    for name in TABLE_NAMES:
        table, _skipped = extract_table(source, name, origin=origin)
        tables[name] = table
    return tables


# ==========================================================================
# PRESET RECORDS
# ==========================================================================

def _validate_blueprint(name, blueprint, how):
    if not isinstance(blueprint, dict):
        raise PresetLoadError("preset %r: %s is %s, not an object"
                              % (name, how, type(blueprint).__name__))
    nodes = blueprint.get("nodes")
    edges = blueprint.get("edges")
    if not nodes:
        raise PresetLoadError("preset %r: %s has no nodes" % (name, how))
    if not edges:
        raise PresetLoadError("preset %r: %s has no edges" % (name, how))
    for node in nodes:
        if not isinstance(node, dict) or not node.get("id"):
            raise PresetLoadError("preset %r: %s has a node with no id: %r"
                                  % (name, how, node))
    for edge in edges:
        if not isinstance(edge, dict) or not edge.get("source") or not edge.get("target"):
            raise PresetLoadError("preset %r: %s has an edge with no source/target: %r"
                                  % (name, how, edge))
    return blueprint


def load_presets(source=None, path=None):
    """Every shipped preset, keyed by the name its button uses.

    Each record is the preset's own object from app.js (``text`` / ``description``,
    ``targets``, ``blueprint`` when it ships one, ``title``, ``hallmark``) plus a
    ``table`` key saying which literal it came from.
    """
    tables = load_tables(source=source, path=path)
    presets = {}
    for table_name in TABLE_NAMES:
        for preset_name, record in tables[table_name].items():
            if not isinstance(record, dict):
                raise PresetLoadError("%s.%s is %s, not an object"
                                      % (table_name, preset_name, type(record).__name__))
            if preset_name in presets:
                raise PresetLoadError("preset %r is defined in both %s and %s -- "
                                      "ambiguous" % (preset_name,
                                                     presets[preset_name]["table"],
                                                     table_name))
            record = dict(record)
            record["table"] = table_name
            if "blueprint" in record:
                _validate_blueprint(preset_name, record["blueprint"],
                                    "its app.js blueprint")
            elif not (record.get("text") or "").strip():
                raise PresetLoadError(
                    "preset %r has neither a `blueprint` nor a `text` to compile one "
                    "from; nothing about it can be tested" % preset_name)
            presets[preset_name] = record
    missing = [name for name in REQUIRED_PRESETS if name not in presets]
    if missing:
        raise PresetLoadError(
            "app.js no longer defines these presets: %s (found %s). Either they were "
            "renamed -- update REQUIRED_PRESETS and the behavioural tests -- or this "
            "loader is reading the wrong table."
            % (", ".join(missing), ", ".join(sorted(presets))))
    return presets


def preset_names(source=None, path=None):
    """Sorted names of every preset app.js ships."""
    return sorted(load_presets(source=source, path=path))


def load_blueprints(text_compiler=None, source=None, path=None):
    """``{preset name: blueprint}`` for every shipped preset.

    A preset that ships an explicit ``blueprint`` yields a deep copy of it. A
    text-only preset (``turing``) is compiled by ``text_compiler`` -- pass
    ``agent.rule_based_parse``, the same deterministic, LLM-off path the backend
    uses behind ``POST /api/blueprint``. Without a compiler such a preset raises,
    so it can never be quietly missing from the result.
    """
    presets = load_presets(source=source, path=path)
    blueprints = {}
    for name, record in presets.items():
        if "blueprint" in record:
            blueprints[name] = copy.deepcopy(record["blueprint"])
            continue
        if text_compiler is None:
            raise PresetLoadError(
                "preset %r ships text only, so it needs a text_compiler to become a "
                "blueprint; call load_blueprints(text_compiler=agent.rule_based_parse)"
                % name)
        compiled = text_compiler(record["text"])
        blueprints[name] = _validate_blueprint(
            name, compiled, "the blueprint %s compiled from its text"
            % getattr(text_compiler, "__name__", "text_compiler"))
    return blueprints
