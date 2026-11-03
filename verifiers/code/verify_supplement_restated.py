#!/usr/bin/env python3
"""The supplement restates each paper statement it proves; check the restatement is the paper's statement.

Every `restated` environment of the supplement (any file it reaches by \\input) repeats a theorem, lemma or
proposition of the paper (any file the paper reaches by \\input), which it names by TeX label. This script compares each body with the paper's statement of that label
(labels, comments and \\internal blocks dropped, starred and unstarred displays identified, whitespace ignored), and the
restated kind (Theorem, Lemma, ...) with the paper's environment, and fails on any difference, so the two
documents cannot drift apart. It checks text only; the proofs are those of the supplement.
"""
import re
import sys
from pathlib import Path

# The TeX sources are beside code/ in the repository and in ../paper/ in the companion tree.
ROOT = Path(__file__).resolve().parents[1]
NOTE = next(p for p in (ROOT, ROOT.parent / 'paper') if (p / 'supplement.tex').exists())
ENVS = r'theorem|lemma|proposition|corollary|conjecture'


def drop_braced(text, macro):
    """Remove every \\macro{...} block, balanced."""
    out, i = [], 0
    while (j := text.find(macro + '{', i)) >= 0:
        out.append(text[i:j])
        depth, k = 1, j + len(macro) + 1
        while depth:
            depth += {'{': 1, '}': -1}.get(text[k], 0) if text[k - 1] != '\\' else 0
            k += 1
        i = k
    return ''.join(out) + text[i:]


def norm(s):
    s = drop_braced(s, '\\internal')
    s = re.sub(r'\\label\{[^}]*\}|(?<!\\)%[^\n]*', '', s)
    # a restated display may be unnumbered: drop \notag and the star of equation/align
    s = re.sub(r'\\(?:notag|nonumber)\b', '', s)
    s = re.sub(r'\\(begin|end)\{(equation|align)\*\}', r'\\\1{\2}', s)
    return re.sub(r'\s+', '', s)


def sources(root):
    """The files root reaches by \\input, root first, each once."""
    out, todo = [], [NOTE / root]
    while todo:
        f = todo.pop(0)
        if f in out:
            continue
        out.append(f)
        text = re.sub(r'(?<!\\)%[^\n]*', '', f.read_text())
        for name in re.findall(r'\\input\{([^}]*)\}', text):
            g = (NOTE / name).with_suffix('.tex').resolve()  # TeX resolves \input from the main file's directory
            assert g.exists(), f'{f.name}: missing \\input {name}'
            todo.append(g)
    return out


def paper_statements():
    out = {}
    for f in sources('four_qubit_entropy.tex'):
        for m in re.finditer(rf'\\begin\{{({ENVS})\}}(.*?)\\end\{{\1\}}', f.read_text(), re.S):
            lab = re.search(r'\\label\{([^}]*)\}', m.group(2))
            if lab:
                body = re.sub(r'^\[[^\n]*\]', '', m.group(2))
                out[lab.group(1)] = (m.group(1), body)
    return out


def main():
    paper = paper_statements()
    bad = n = 0
    for f in sources('supplement.tex'):
        for m in re.finditer(r'\\begin\{restated\}(?:\[[^\n]*?\])?\{(\w+)\}\{([^}]*)\}(.*?)\\end\{restated\}', f.read_text(), re.S):
            kind, lab, body = m.groups()
            n += 1
            env, text = paper.get(lab, (None, None))
            same_kind = env == kind.lower()
            ok = env is not None and same_kind and norm(body) == norm(text)
            note = '' if env is None or same_kind else f' (paper environment: {env})'
            print(f'{"ok  " if ok else "FAIL"} {f.name}: {kind} {lab}{note}')
            bad += not ok
    print(f'{n} restated statements, {bad} differ from the paper (text or kind)')
    assert n > 0, 'no restated statements found'
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
