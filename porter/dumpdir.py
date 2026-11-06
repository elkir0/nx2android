# A game whose files are already extracted (a console dump: folders, not NSPs), in the layout upstream's scripts/pack_dump.py takes (seedds/nxnative 8dd0c98a), → the extraction
# layout the port's later stages read: extract/v<version>/{ExeFS,RomFS} as nsp.extract writes them, and aoc/<index>/ as nsp.extract_aoc does. The dump folder holds:
#   base/exefs/ (main, main.npdm, sdk, rtld, subsdk*), base/romfs/   (base/romfs is optional when an update has its RomFS: that one is whole, and nothing reads the base's for it)
#   updates/<version>/exefs/, updates/<version>/romfs/   (optional; <version> is the title version as a decimal number, 917504 = 0xE0000)
#   dlc/<DLC title ID>/<n>/romfs/   (optional; the DLC title ID in hex)
# An update's RomFS in such a dump is the patched RomFS (base + the update's patch, the tree the game sees), not a patch: it is taken on its own, never laid over the base
# (upstream: Donkey Kong Bananza 458752 drops 42 base files). The title ID is the program ID in base/exefs/main.npdm, so no prod.keys are needed.
# Upstream packs each RomFS into one image (its hosts' DM_ROMFS); the Android host reads a RomFS folder from the data pack, so here the trees are hard links into the dump (copies on a
# volume without hard links), which take no extra space, and the data pack links them again. A dump has no control data (control.nacp, icon.jpg): the app takes the profile's name.
import os, shutil, struct
from pathlib import Path

from porter.upstream import BuildError, PROG, nsp, say


def program_id(npdm):   # the ACI0 program ID of main.npdm
    d = Path(npdm).read_bytes()
    if d[:4] != b'META': raise BuildError(f'{npdm}: not an NPDM')
    off, size = struct.unpack_from('<II', d, 0x70)
    if d[off:off + 4] != b'ACI0' or size < 0x18: raise BuildError(f'{npdm}: no ACI0 section')
    return struct.unpack_from('<Q', d, off + 0x10)[0]


def is_dump(folder):   # whether a folder has the layout above
    f = Path(folder)
    return (f / 'base/exefs/main.npdm').is_file() and ((f / 'base/romfs').is_dir() or any(p.is_dir() for p in f.glob('updates/*/romfs')))


def scan(dump):
    """What a dump folder holds, in the shape of nsp.scan_nsps: (title ID, {'base': base/, 'updates': [(version, updates/<version>/)], 'dlc': [(DLC title ID, None, its RomFS folder)]})"""
    dump = Path(dump)
    if not is_dump(dump): raise BuildError(f'{dump}: no base/exefs/main.npdm, or no RomFS in base/ or updates/<version>/')
    app = program_id(dump / 'base/exefs/main.npdm')
    if app & 0xFFF: raise BuildError(f'{dump}: base program ID {app:016X} is not a base title ID')
    ups = sorted((int(p.name), p) for p in (dump / 'updates').iterdir() if p.is_dir() and p.name.isdigit()) if (dump / 'updates').is_dir() else []
    dlc = []
    for d in sorted(p for p in (dump / 'dlc').iterdir() if p.is_dir()) if (dump / 'dlc').is_dir() else []:
        try: tid = int(d.name, 16)
        except ValueError: raise BuildError(f'{d}: not a DLC title ID') from None
        if not 0 < tid - app - 0x1000 <= 2000: raise BuildError(f'{d}: {tid:016X} is not a DLC of {app:016X}')
        roms = sorted(d.glob('*/romfs')) or ([d / 'romfs'] if (d / 'romfs').is_dir() else [])
        if len(roms) != 1: raise BuildError(f'{d}: expected one <n>/romfs folder, found {len(roms)}')
        dlc.append((tid, None, roms[0]))
    return app, {'base': dump / 'base', 'updates': ups, 'dlc': dlc}


def versions(fam):   # {version: the folder it is taken from}: v0 only when the base has its RomFS, v<N> for each update with one
    out = {'v0': fam['base']} if (fam['base'] / 'romfs').is_dir() else {}
    out.update((f'v{v}', p) for v, p in fam['updates'] if (p / 'romfs').is_dir())
    return out


def can_link(d):   # whether this volume takes hard links (exFAT does not: the trees are then copied, and the port needs their room)
    d = Path(d); d.mkdir(parents=True, exist_ok=True); a, b = d / '.link_test', d / '.link_test2'
    try:
        a.write_bytes(b'x'); nsp.rmtree(b); os.link(a, b); return True
    except OSError: return False
    finally: nsp.rmtree(a); nsp.rmtree(b)


def tree_bytes(d): return sum(f.stat().st_size for f in Path(d).rglob('*') if f.is_file())


def link_tree(src, dst, done=None):   # every file of src at the same place under dst, as a hard link or, failing that, a copy; (files, bytes)
    n = b = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [x for x in dirs if not x.startswith('.')]   # AppleDouble and other hidden folders a copy from a Mac leaves are no game files
        out = Path(dst) / Path(root).relative_to(src); out.mkdir(parents=True, exist_ok=True)
        for f in files:
            if f.startswith('._') or f == '.DS_Store': continue
            s = Path(root) / f
            try: os.link(s, out / f)
            except OSError: shutil.copy2(s, out / f)
            n += 1; b += s.stat().st_size
            if done and n % 1000 == 0: done(n)
    return n, b


def _linked(src, dst, what):
    """src's tree as dst, made in dst.partial and swapped in; reused when the files under src (paths, sizes, times) are as at the last run, keyed in <dst>.dump.key"""
    kf = dst.with_name(dst.name + '.dump.key'); key = nsp.fingerprint(('stat', src))
    if dst.is_dir() and (note := nsp.reused(kf, key)) is not None:
        PROG.stage(f'link {what}'); say(f'reusing the linked {what}: {dst}'); PROG.end(f'{note}, reused (dump unchanged)'); return dst
    nsp.rmtree(kf); tmp = dst.with_name(dst.name + '.partial'); nsp.rmtree(tmp)
    PROG.stage(f'link {what}', unit='files')
    n, b = link_tree(src, tmp, lambda k: PROG.set(done=k))
    nsp.rmtree(dst); tmp.rename(dst)
    note = f'{n:,} files, {nsp.amount(b, "B")}'
    kf.write_text(f'{key}\n{note}\n'); PROG.end(note)
    return dst


def link_version(src, app, dst):   # one version's folder (base/ or updates/<version>/) → dst/ExeFS and dst/RomFS (nsp.extract's layout)
    src, dst = Path(src), Path(dst)
    if not (src / 'exefs/main').is_file() or not (src / 'exefs/main.npdm').is_file() or not (src / 'romfs').is_dir(): raise BuildError(f'{src}: no exefs/main, exefs/main.npdm and romfs')
    if (pid := program_id(src / 'exefs/main.npdm')) != app: raise BuildError(f'{src}: program ID {pid:016X}, the base is {app:016X}')
    dst.mkdir(parents=True, exist_ok=True)
    say(f'{dst.name}: {src}')
    _linked(src / 'exefs', dst / 'ExeFS', f'{dst.name} ExeFS')
    _linked(src / 'romfs', dst / 'RomFS', f'{dst.name} RomFS')
    return dst


def link_dlc(app, dlc, aoc):   # [(DLC title ID, None, its RomFS folder)] → aoc/<index>/ (index = DLC title ID − base title ID − 0x1000); folders of DLC no longer in the dump go; None without DLC
    aoc = Path(aoc)
    if not dlc: nsp.rmtree(aoc); return None
    want = {str(tid - app - 0x1000): roms for tid, _, roms in dlc}
    if aoc.is_dir():
        for f in aoc.iterdir():
            if f.name.split('.')[0] not in want: nsp.rmtree(f)
    aoc.mkdir(parents=True, exist_ok=True)
    for name, roms in want.items(): _linked(roms, aoc / name, f'DLC {name}')
    return aoc
