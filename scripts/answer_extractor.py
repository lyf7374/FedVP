# Final-answer-v3 parser used in FedVP; adapted from the FedCoT answer interface.
import re
import unicodedata
from typing import Dict, Optional, Tuple


ANSWER_PARSER_VERSION = "final-answer-v3-2026-10-21"
_ANSWER_LABEL = re.compile(
    r"^(?:(?:therefore|thus|hence|so)[,:]?\s+)?"
    r"(?:the\s+)?(?:(?:final|correct)\s+)?answer\b"
    r"(?:\s+is)?\s*[:\-]?\s*", re.I)
_LETTER_CONTENT = re.compile(r"^\(?([A-Za-z])\s*(?:\)\s*\.?|[.:\-])\s*(.+)$", re.S)
_LETTER_ONLY = re.compile(r"^\(?([A-Za-z])\)?[.]?$")
_INLINE_ANSWER = re.compile(
    r"(?:\b(?:(?:the\s+)?(?:final|correct)\s+answer(?:\s+is)?|the\s+answer\s+is)\b\s*[:\-]?\s*|####\s*)", re.I)
_EXPLANATION = re.compile(
    r"^(?:because\b|since\b|as\b|which\b|explanation\s*:|reasoning\s*:|"
    r"this (?:is|means|indicates|suggests|shows)\b|"
    r"the (?:study|evidence|results|findings)\b|based on\b)", re.I)
_MULTIPLE_LETTERS = re.compile(
    r'^\(?[A-Za-z]\)?\s*(?:or\b|and\b|/)\s*\(?[A-Za-z]\)?(?:\s|$)', re.I)


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"',
                                      "\u201d": '"', "\u2013": "-", "\u2014": "-"}))
    return text.replace("**", "").replace("__", "").replace("`", "").strip()


def _marked_answers(text: str):
    """Find explicit answer boundaries; ordinary rationale lines are not answers."""
    lines = text.splitlines(keepends=True)
    nonempty = [i for i, line in enumerate(lines) if line.strip()]
    last = nonempty[-1] if nonempty else -1
    markers = []
    offset = 0
    for index, line in enumerate(lines):
        clean = _clean_text(line)
        heading = re.match(r"^#{2,6}\s*", clean)
        start_shift = 0
        if not heading and not _ANSWER_LABEL.match(clean) and '####' in line:
            # The author's hash delimiter also worked inline after reasoning.
            start_shift = line.index('####')
            clean = _clean_text(line[start_shift:])
            heading = re.match(r"^#{2,6}\s*", clean)
        content = clean[heading.end():] if heading else clean
        label = _ANSWER_LABEL.match(content)
        if label:
            payload = content[label.end():]
        elif heading and (clean.startswith('####') and (
                _LETTER_CONTENT.match(content) or _LETTER_ONLY.fullmatch(content)
                or (index == last and not re.match(r'^(?:step\b|conclusion\b|reasoning\b|explanation\b)', content, re.I)))):
            payload = content
        else:
            offset += len(line)
            continue
        markers.append((offset + start_shift, offset + len(line), payload, bool(label)))
        offset += len(line)
    result = []
    for index, (start, after_line, payload, labeled) in enumerate(markers):
        end = markers[index + 1][0] if index + 1 < len(markers) else len(text)
        continuation = text[after_line:end]
        body = (payload + '\n' + continuation).strip()
        if index + 1 < len(markers):
            # "Final answer:\n#### B. ..." is one heading and one answer.
            if not body and labeled and not markers[index + 1][3]:
                continue
            # Hash headings also introduce per-option rationale sections in the
            # saved responses. A bullet analysis / explicit rejection followed
            # by another answer boundary is not a competing final selection.
            if (not labeled and _LETTER_CONTENT.match(payload)
                    and re.match(r'^(?:[-*]\s|This is incorrect\b)',
                                 _clean_text(continuation), re.I)):
                continue
        result.append((start, body))
    return result


def extract_answer(response_text: str) -> str:
    blocks = _marked_answers(response_text)
    if len(blocks) == 1:
        return blocks[0][1]
    if blocks:
        # Keep multiple explicit answers visible to the option-aware matcher.
        return '\n'.join('Final answer: ' + body for _, body in blocks)
    lines = [line for line in response_text.splitlines() if line.strip()]
    terminal = _clean_text(lines[-1]) if lines else ''
    # A bare letter/text answer at the end is usable; a bare unmarked letter is not.
    return terminal if _LETTER_CONTENT.fullmatch(terminal) else ''


def split_final_answer(response_text: str) -> Optional[Tuple[str, str]]:
    """Share an explicit, unambiguous final-answer boundary with NLI preparation."""
    blocks = _marked_answers(response_text)
    if not blocks or any(not body for _, body in blocks):
        return None
    answers = [(start, part) for start, body in blocks for part in _split_inline_answers(body)]
    if any(not body or _MULTIPLE_LETTERS.match(_clean_text(body)) for _, body in answers):
        return None
    letter_answers = []
    for start, body in answers:
        core = _answer_core(body)
        match = _LETTER_CONTENT.fullmatch(core) or _LETTER_ONLY.fullmatch(core)
        if match:
            letter_answers.append((start, body, match.group(1).upper(),
                                   _normalize_text(match.group(2)) if len(match.groups()) > 1 else ''))
    if letter_answers:
        if len({entry[2] for entry in letter_answers}) != 1:
            return None
        contents = {entry[3] for entry in letter_answers if entry[3]}
        if len(contents) > 1:
            return None
        for _, body in answers:
            core = _answer_core(body)
            if _LETTER_CONTENT.fullmatch(core) or _LETTER_ONLY.fullmatch(core):
                continue
            if _normalize_text(core) not in contents and not _narrative_restatement(core):
                return None
        start, answer = letter_answers[-1][:2]
    else:
        if len({_normalize_text(body) for _, body in answers}) != 1:
            return None
        start, answer = answers[-1]
    return response_text[:start].strip(), answer


def extract_hash_content(text: str) -> Optional[str]:
    """Compatibility helper for callers requesting the last hash-delimited line."""
    index = text.rfind("####")
    if index < 0:
        return None
    content = text[index + 4:].strip()
    return content.split('\n', 1)[0].strip() or None


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = _clean_text(text).casefold().strip(" . ,;:!?\'\"\t\r\n")
    return " ".join(text.split())


def _matches_option(content: str, expected: str) -> bool:
    if _normalize_text(content) == _normalize_text(expected):
        return True
    clean = _clean_text(content)
    first, separator, remainder = clean.partition('\n\n')
    if separator and _normalize_text(first) == _normalize_text(expected):
        # Preserve the author's canonical standalone answer before a separate
        # explanatory paragraph. Another standalone option is not explanation.
        if not any(_LETTER_CONTENT.fullmatch(line.strip()) for line in remainder.splitlines()):
            return True
    # An option must end before an explicitly explanatory continuation. There is
    # no fuzzy substring search and no fallback that mines arbitrary prose.
    for boundary in re.finditer(r"[,.;:\-]\s+|\n+", clean):
        if (_normalize_text(clean[:boundary.start()]) == _normalize_text(expected)
                and _EXPLANATION.match(clean[boundary.end():].strip())):
            return True
    return False


def _split_inline_answers(block: str):
    """Retain subsequent explicit answer statements even on the same line."""
    matches = [match for match in _INLINE_ANSWER.finditer(block)
               if not re.search(r'\b(?:this|that|it)\s+is\s*$', block[:match.start()], re.I)]
    if not matches:
        return [block]
    parts = [block[:matches[0].start()].strip()]
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(block)
        parts.append(block[match.end():end].strip())
    # An empty final answer remains unresolved; only an empty prefix before the
    # first marker is structural, rather than an answer.
    return parts[1:] if not parts[0] else parts


def _answer_core(block: str) -> str:
    """Separate an explicit answer from its following explanatory paragraph."""
    clean = _clean_text(block).split('\n\n', 1)[0].strip()
    for boundary in re.finditer(r'[,.;:\-]\s+|\n+', clean):
        if _EXPLANATION.match(clean[boundary.end():].strip()):
            return clean[:boundary.start()].strip()
    return clean


def _narrative_restatement(text: str) -> bool:
    """Recognize prose restatement, without interpreting its answer or meaning."""
    return bool(re.match(
        r'^(?:that\b|because\b|based on\b|(?:yes|no|maybe)\s*[,;]|'
        r'(?:\S+\s+){0,12}(?:is|are|was|were|does|do|have|has|can|could|would|should)\b)',
        text, re.I))


def _match_answer_block(block: str, options: Dict[str, str]) -> str:
    clean = _clean_text(block)
    letter_only = _LETTER_ONLY.fullmatch(clean)
    if letter_only:
        letter = letter_only.group(1).upper()
        return letter if letter in options else ''
    match = _LETTER_CONTENT.fullmatch(clean)
    if match:
        letter, content = match.groups()
        letter = letter.upper()
        # If letter and text disagree, neither is allowed to override the other.
        return letter if letter in options and _matches_option(content, options[letter]) else ''
    matches = [letter for letter, content in options.items()
               if _normalize_text(clean) == _normalize_text(content)]
    return matches[0] if len(matches) == 1 else ''


def match_choice_answer(extracted_text: str, options_dict: Dict[str, str]) -> str:
    if not extracted_text or not options_dict:
        return ""
    marked = _marked_answers(extracted_text)
    blocks = [body for _, body in marked] if marked else [extracted_text]
    blocks = [part for block in blocks for part in _split_inline_answers(block)]
    answers = []
    for block in blocks:
        answer = _match_answer_block(block, options_dict)
        if answer:
            answers.append(answer)
        elif (not _clean_text(block) or _LETTER_CONTENT.fullmatch(_clean_text(block))
              or _LETTER_ONLY.fullmatch(_clean_text(block))
              or _MULTIPLE_LETTERS.match(_clean_text(block))):
            # An explicit malformed/conflicting choice cannot be overruled by
            # another answer. Unmatched narrative prose makes no choice claim.
            return ''
    return answers[0] if answers and len(set(answers)) == 1 else ''


def compute_reward(extracted_answer: str, ground_truth_item: Dict) -> float:
    if extracted_answer and extracted_answer.upper() == ground_truth_item["answer_letter"].upper():
        return 1.0
    return 0.0
