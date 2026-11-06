#!/usr/bin/env python3
"""Summarize a macOS `sample` call graph for one thread: inclusive and self samples per function, ranked (devtools/vkbench).

  prof.py <sample.txt> [--thread main] [--top 40] [--under prepare_draw]

Inclusive counts take each function once per path (recursion counted at its outermost frame); self counts are a node's samples less its
children's. --under: only the subtree of that function (its share of the thread)."""
import argparse, collections, re, sys

LINE = re.compile(r'^(?P<ind>[ +!:|]*)(?P<n>\d+) (?P<rest>.*)$')


def parse(path, thread):
    lines = open(path, errors='replace').read().splitlines()
    out, on = [], False
    for l in lines:
        if l.startswith('Call graph:'): on = True; continue
        if not on: continue
        if l.startswith('Total number in stack') or l.startswith('Sort by top of stack'): break
        m = LINE.match(l)
        if not m: continue
        depth = len(m.group('ind')); n = int(m.group('n')); rest = m.group('rest')
        if 'Thread_' in rest.split(' ')[0] or rest.startswith('Thread_'):
            cur = thread in rest or (thread == 'main' and 'main-thread' in rest)
            if cur: out = []; base = depth
            on_thread = cur; continue
        if 'on_thread' in dir() and on_thread:
            name = rest.split('  (in ')[0].split(' (in ')[0].strip()
            out.append((depth, n, name))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('file'); ap.add_argument('--thread', default='main'); ap.add_argument('--top', type=int, default=40); ap.add_argument('--under'); ap.add_argument('--callers', help='samples of this function by its immediate caller')
    a = ap.parse_args()
    nodes = parse(a.file, a.thread)
    if not nodes: sys.exit('no thread ' + a.thread)
    incl, selfc = collections.Counter(), collections.Counter()
    stack = []   # (depth, name, count, child sum)
    total = nodes[0][1] if nodes else 0
    under_total = 0
    def close(upto):
        while stack and stack[-1][0] >= upto:
            d, name, n, ch = stack.pop(); selfc[name] += n - ch
    for depth, n, name in nodes:
        close(depth)
        names = [s[1] for s in stack]
        in_under = a.under is None or a.under in names or name == a.under
        if a.under and name == a.under and a.under not in names: under_total += n
        if in_under and name not in names: incl[name] += n
        if stack: stack[-1][3] += n
        stack.append([depth, name, n, 0])
    close(-1)
    if a.callers:
        by = collections.Counter(); st = []
        for depth, n, name in nodes:
            while st and st[-1][0] >= depth: st.pop()
            if name == a.callers and a.callers not in [x[1] for x in st]: by[st[-1][1] if st else '?'] += n
            st.append((depth, name))
        tot = sum(by.values()); print(f'{a.callers}: {tot} samples ({100.0 * tot / total:.1f}% of {a.thread}), by caller:')
        for k, v in by.most_common(15): print(f'{v:8d} {100.0 * v / total:6.1f}%  {k}')
        return
    base = under_total if a.under else total
    print(f'{a.thread}: {total} samples' + (f'; under {a.under}: {under_total}' if a.under else ''))
    print(f'{"inclusive":>9} {"%":>6}  {"self":>6} {"%":>6}  function')
    for name, n in incl.most_common(a.top):
        print(f'{n:9d} {100.0 * n / base:6.1f}  {selfc[name]:6d} {100.0 * selfc[name] / base:6.1f}  {name}')


if __name__ == '__main__':
    main()
