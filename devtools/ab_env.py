#!/usr/bin/env python3
# A/B of the running app's switches: each variant's KEY=VALUE settings go to the app as DM_CTL "env" lines (live switches, ctl.txt) or into env.txt with a restart
# (switches read at start-up), then the next timing lines of the log (perf: "timing up to frame N", 300 frames each) are averaged; the base settings are measured
# again beside every variant, so a scene or a clock that drifts shows as the base moving.   ab_env.py <variants file> [--lines 2] [--repeat 1] [--out <dir>]
# Variants file, one per line (# comments):   name: KEY=VALUE KEY=VALUE      a live variant
#                                             name @start: KEY="A VALUE"     read at start-up: env.txt gets the line, the app restarts, Enter once back in the scene
# The first line is the base. Every key a live variant names is set to the base's value (or to the value after "reset:" on a line of its own) before the next variant,
# as an empty "env KEY=" is not the switch's default. Keep the scene steady (a free drive, a menu that keeps drawing): a race's start or a loading screen moves the numbers more
# than most switches. Outputs: <out>/ab.tsv (a row per measure), <out>/<n>-<name>.log (the log while it was measured), env.txt's original in <out>/env.txt.orig
import argparse, re, shlex, statistics, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device import ADB, PKG, FILES   # noqa: E402  devtools/device.json: the device and the app

TIMING = re.compile(r'timing up to frame (\d+): frame time p50 ([\d.]+) / p95 ([\d.]+).*?NVN submit\+present p50 ([\d.]+).*?GPU p50 ([\d.]+)')
REPLAY = re.compile(r'replay per frame \(last \d+\): ([\d.]+) draws, replay ([\d.]+) ms \(descriptor sets ([\d.]+)')


def sh(cmd, check=False, timeout=60):
    r = subprocess.run(ADB + ['shell', cmd], capture_output=True, timeout=timeout)
    if check and r.returncode: sys.exit(f'ab_env: adb shell {cmd!r} failed: {r.stderr.decode(errors="replace").strip()}')
    return r.stdout.decode(errors='replace')


def parse(path):
    variants, reset = [], {}
    for raw in Path(path).read_text(encoding='utf-8').splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line: continue
        head, _, body = line.partition(':'); head = head.strip(); kv = dict(p.split('=', 1) for p in shlex.split(body) if '=' in p)
        if head == 'reset': reset.update(kv); continue
        variants.append((head.replace('@start', '').strip(), '@start' in head, kv))
    if not variants: sys.exit(f'ab_env: no variants in {path}')
    return variants, reset


def log_size(): return int((sh(f'stat -c %s {FILES}/log.txt') or '0').strip() or 0)


def ctl(lines): sh(f"printf '%s\\n' {' '.join(repr(l) for l in lines)} >> {FILES}/ctl.txt", check=True)


def measure(n_lines, timeout):   # the timing lines that follow the first one (which still holds frames from before the switch)
    off, got, text, t0 = log_size(), [], '', time.time()
    while len(got) < n_lines + 1 and time.time() - t0 < timeout:
        time.sleep(2); size = log_size()
        if size < off: off = 0   # the log started over (the app restarted)
        if size > off: chunk = sh(f'tail -c +{off + 1} {FILES}/log.txt | head -c {size - off}', timeout=120); off = size; text += chunk; got += TIMING.findall(chunk)
        if not sh(f'pidof {PKG}').strip(): print('ab_env: the app is gone', file=sys.stderr); break
    rep = REPLAY.findall(text)
    temp = (re.findall(r'^\s*temperature: (\d+)', sh('dumpsys battery'), re.M) or ['?'])[0]   # (the dump's history lines hold temperature: too)
    rows = got[1:] or got
    med = lambda i: statistics.mean(float(r[i]) for r in rows) if rows else float('nan')
    return {'p50': med(1), 'p95': med(2), 'submit': med(3), 'gpu': med(4), 'lines': len(rows), 'replay': float(rep[-1][1]) if rep else float('nan'),
            'dsets': float(rep[-1][2]) if rep else float('nan'), 'temp': f'{int(temp) / 10:.1f}' if temp.isdigit() else '?'}, text


def put_env(text, out):   # env.txt as given (pushed: its lines may hold anything), or none
    if not text: sh(f'rm -f {FILES}/env.txt'); return
    tmp = out / 'env.txt.next'; tmp.write_text(text, encoding='utf-8')
    r = subprocess.run(ADB + ['push', str(tmp), f'{FILES}/env.txt'], capture_output=True, timeout=60)
    if r.returncode: sys.exit(f'ab_env: env.txt not pushed: {r.stderr.decode(errors="replace").strip()}')


def restart():
    sh(f'am force-stop {PKG}'); time.sleep(1)
    sh(f'am start -W -n {PKG}/android.app.NativeActivity >/dev/null 2>&1; true', timeout=90)
    input('ab_env: the app restarted; get back to the scene, then press Enter ')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('variants'); ap.add_argument('--lines', type=int, default=2, help='timing lines per measure (300 frames each)')
    ap.add_argument('--repeat', type=int, default=1); ap.add_argument('--timeout', type=int, default=120, help='seconds a measure may wait for its lines')
    ap.add_argument('--out', default='ab-' + time.strftime('%Y%m%d-%H%M%S')); a = ap.parse_args()
    variants, reset = parse(a.variants); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not sh(f'pidof {PKG}').strip(): sys.exit(f'ab_env: {PKG} is not running: start it and get to the scene first')
    env0 = sh(f'cat {FILES}/env.txt 2>/dev/null'); (out / 'env.txt.orig').write_text(env0, encoding='utf-8')
    if not re.search(r'^DM_CTL=', env0, re.M): sys.exit(f'ab_env: {FILES}/env.txt has no DM_CTL=<its ctl.txt> line: the app reads no env lines without it (a port rewrites env.txt)')
    base_name, _, base = variants[0]
    live_keys = sorted({k for _, start, kv in variants if not start for k in kv})
    defaults = {k: base.get(k, reset.get(k)) for k in live_keys}
    missing = [k for k, v in defaults.items() if v is None]
    if missing: sys.exit(f'ab_env: no base or reset value for {", ".join(missing)} (an empty env line is not the default)')
    set_live = lambda kv: ctl([f'env {k}={v}' for k, v in {**defaults, **kv}.items()])
    tsv = out / 'ab.tsv'; tsv.write_text('n\tvariant\tframe p50\tframe p95\tGPU p50\tsubmit p50\treplay\tdescriptor sets\tlines\tbattery °C\n', encoding='utf-8'); n = 0

    def run(name, kv, start):
        nonlocal n; n += 1
        if start:
            put_env(env0 + ('' if not env0 or env0.endswith('\n') else '\n') + ''.join(f'{k}={v}\n' for k, v in kv.items()) if kv else env0, out)
            restart(); set_live({})
        else: set_live(kv)
        m, text = measure(a.lines, a.timeout); (out / f'{n:02d}-{re.sub(r"[^A-Za-z0-9_.-]", "_", name)}.log').write_text(text, encoding='utf-8')
        with tsv.open('a', encoding='utf-8') as f: f.write(f'{n}\t{name}\t{m["p50"]:.2f}\t{m["p95"]:.2f}\t{m["gpu"]:.2f}\t{m["submit"]:.2f}\t{m["replay"]:.2f}\t{m["dsets"]:.2f}\t{m["lines"]}\t{m["temp"]}\n')
        print(f'{n:3d} {name:24s} frame p50 {m["p50"]:6.2f} p95 {m["p95"]:6.2f}  GPU {m["gpu"]:6.2f}  submit {m["submit"]:6.2f}  replay {m["replay"]:6.2f} (sets {m["dsets"]:5.2f})  {m["temp"]} °C')
        return m

    try:
        restarted = False
        for _ in range(a.repeat):
            for name, start, kv in variants[1:]:
                b = run(base_name, {}, restarted); restarted = False
                v = run(name, kv, start); restarted = start
                print(f'    {name}: frame {v["p50"] - b["p50"]:+.2f} ms, GPU {v["gpu"] - b["gpu"]:+.2f} ms against the base beside it')
        if restarted: run(base_name, {}, True)
    finally:
        set_live({}); put_env(env0, out)   # the base again, env.txt as it was (the next start reads it)
    print(f'ab_env: {tsv}')


if __name__ == '__main__':
    main()
