# Push a local tree or one file to the device in short transfers (porter/adb.py: small files in batches, big files in pieces appended on the device as they arrive; a rerun resumes).
#   push_chunked.py <local dir or file> <remote dir or file> [--chunk-mb N] [--timeout S]      (devtools/device.json: the device)
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from device import ADB_EXE, SERIAL   # noqa: E402
from porter.adb import Adb, AdbError   # noqa: E402

args = [x for x in sys.argv[1:] if not x.startswith('--') and not x.isdigit()]
opt = lambda k, d: int(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
if len(args) != 2: sys.exit('usage: push_chunked.py <local dir or file> <remote dir or file> [--chunk-mb N] [--timeout S]')
local, remote, chunk = Path(args[0]), args[1].rstrip('/'), opt('--chunk-mb', 64) << 20
adb = Adb(ADB_EXE, SERIAL, timeout=opt('--timeout', 120), log=lambda m: print(m, flush=True))
show = lambda d, t, note: print(f'  {d / 2**20:,.0f}/{t / 2**20:,.0f} MB - {note}', flush=True)
try:
    if local.is_file(): adb.push_file(local, remote, chunk, progress=show)
    else: adb.push_tree(local, remote, chunk, progress=show)
except AdbError as e: sys.exit(str(e))
print('ok', flush=True)
