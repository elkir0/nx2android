"""The shader split on a RomFS image (dm/tools/romfs_image.py, extract/v<version>/RomFS.img) gives what it gives on the same RomFS as a folder, byte for byte; NVN2 (Switch 2) shaders
are keyed as the host keys them (the control section's first 0x48 bytes with +0x30 zeroed, plus the code area: hle_nvn.c nvn_ctl_lens), task and mesh shaders take their stage from the
control section, and DLC RomFS roots given as DM_SPLIT_AOC (the porter's aoc/) split as the DLC argument does. Synthetic bytes only (nx2android: the port hands Mario Kart World's
image to the split)."""
import hashlib
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / 'dm/tools'), str(REPO / 'dm/tools/fuzz'), str(REPO / 'dm/gfx/tools')]
import romfs_image   # noqa: E402
import fuzz_shader_split as fz   # noqa: E402
ss = fz.ss
import trinity   # noqa: E402

GAME = '010015100B514000'   # any profile: the tables' package names follow its shader_groups


def nvn2(stage_byte, sph_type, own_len, tag):
    """(control section, code area) of one NVN2 stage: version 0xc9 at +4, +0x30 the section's own length in a BNSH file and 0 in agl's BAHS (own_len), +0x44 the code area's
    length, +0x56 the stage byte (5 compute, 6 task, 7 mesh; 0 otherwise, where the SPH decides); the code a 0x80-byte SPH (word 0 the stage) and instructions, no code header"""
    code = struct.pack('<I', sph_type) + bytes(0x7c) + struct.pack('<I', tag) + bytes.fromhex('0f00000000000000') * 8
    ctl = bytearray(0xd8); ctl[:4] = ss.CTL_MAGIC; struct.pack_into('<I', ctl, 4, 0xc9); struct.pack_into('<I', ctl, 0x30, own_len)
    struct.pack_into('<I', ctl, 0x44, len(code)); ctl[0x56] = stage_byte; ctl[0x60:0x64] = b'tail'   # bytes past 0x48 are not in the key
    return bytes(ctl), code


def host_key(ctl, code):   # the host's rule, written out: the first 0x48 bytes with +0x30 zeroed, then the code area
    return hashlib.sha256(ctl[:0x30] + bytes(4) + ctl[0x34:0x48] + code).hexdigest()[:16]


VS, PS = nvn2(0, 0x0461, 0xd8, 1), nvn2(0, 0x1462, 0xd8, 2)            # in a BNSH: +0x30 = the section's length
TS, MS = nvn2(6, 0x0461, 0, 3), nvn2(7, 0x0c61, 0, 4)                  # task (its SPH reads as vertex) and mesh (as domain), in a BAHS: +0x30 = 0
VS_BAHS = (VS[0][:0x30] + bytes(4) + VS[0][0x34:], VS[1])              # the BNSH's vertex shader again, as agl's BAHS holds it


def bnsh(stages):   # fz.bnsh's layout with these (control, code) stages in the vertex and pixel slots
    d = bytearray(0x300); d[:8] = b'BNSH\0\0\0\0'; d[0x60:0x64] = b'grsc'
    struct.pack_into('<IQ', d, 0x7c, 1, 0x100); struct.pack_into('<Q', d, 0x110, 0x140)
    for si, desc, (ctl, code) in zip((0, 4), (0x200, 0x240), stages):
        struct.pack_into('<Q', d, 0x148 + 8 * si, desc); at = len(d); d += ctl + code
        struct.pack_into('<QQII', d, desc + 8, at, at + len(ctl), len(code), len(ctl))
    struct.pack_into('<I', d, 0x1c, len(d))
    return bytes(d)


def bahs(stages):
    """agl's BAHS: a binary array at 0x40 (records of 0x20 bytes, each with its 16-byte entry {code offset, 0, control length, control - entry} at +0x10), a program array after it,
    the control sections, then the code region (+0x18 its length, +0x1c its start) with each code area at a 0x100 boundary"""
    n = len(stages); d = bytearray(0x200); d[:4] = b'BAHS'
    struct.pack_into('<II', d, 0x40, 8 + 0x20 * n, n)
    ctl_at = [0x200 + 0x100 * i for i in range(n)]
    for i, (ctl, _) in enumerate(stages):
        rec = 0x48 + 0x20 * i; ent = rec + 0x10
        struct.pack_into('<II', d, rec, 0x20, i); struct.pack_into('<IIIi', d, ent, 0x100 * i, 0, len(ctl), ctl_at[i] - ent)
    parr = 0x48 + 0x20 * n; struct.pack_into('<II', d, parr, 8 + 16, 1); struct.pack_into('<IIII', d, parr + 8, 16, 0, 0x11, 0)
    for i, (ctl, _) in enumerate(stages): d += bytes(ctl_at[i] - len(d)) + ctl
    region = 0x200 + 0x100 * n; d += bytes(region - len(d))
    code = bytearray(0x100 * n)
    for i, (_, c) in enumerate(stages): code[0x100 * i:0x100 * i + len(c)] = c
    struct.pack_into('<II', d, 0x18, len(code), region); d += code
    return bytes(d)


def flatbuffer(head, roots):
    """Just enough of a FlatBuffers writer for trinity.py's reader. Objects: ('table', {field: ('Q' or 'b', value) or an object}), ('u64', [values]), ('u8', bytes), ('str', text),
    ('refs', [objects]). Each reference points forward, to its object written after it, as FlatBuffers' uoffsets do. roots: [(position, object)], each position a u32 root offset that
    the caller has kept free in head (in order, every root after the head). → (the bytes, the position of each root)"""
    d = bytearray(head)
    def put(fmt, *v): at = len(d); d.extend(struct.pack('<' + fmt, *v)); return at
    def ref(slot, target): struct.pack_into('<I', d, slot, target - slot)
    def emit(o):
        kind, v = o
        if kind == 'u64': return put(f'I{len(v)}Q', len(v), *v)
        if kind == 'u8': at = put('I', len(v)); d.extend(v); return at
        if kind == 'str': at = put('I', len(v)); d.extend(v.encode() + b'\0'); return at
        if kind == 'refs':
            at = put('I', len(v)); slots = [put('I', 0) for _ in v]
            for slot, child in zip(slots, v): ref(slot, emit(child))
            return at
        n = max(v) + 1 if v else 0; vt = put(f'{2 + n}H', *([0] * (2 + n))); t = put('i', 0); struct.pack_into('<i', d, t, t - vt); slots = {}
        for i, f in sorted(v.items()):
            slots[i] = put('I', 0) if f[0] not in ('Q', 'b') else put(f[0], f[1])
        struct.pack_into('<HH', d, vt, 4 + 2 * n, len(d) - t)
        for i, slot in slots.items(): struct.pack_into('<H', d, vt + 4 + 2 * i, slot - t)
        for i, f in sorted(v.items()):
            if f[0] not in ('Q', 'b'): ref(slots[i], emit(f))
        return t
    out = []
    for at, o in roots:
        if at < 0: at = put('I', 0)
        ref(at, emit(o)); out.append(at)
    return d, out


def trinity_fs(files):
    """A Trinity file system of one pack 'pack0' holding files [(file hash, stored bytes)]: data.trpfd (0 file hashes, 1 pack names, 2 file infos {0 pack index}, 3 pack infos {0 size,
    1 file count}) and data.trpfs ('ONEPACK\\0', u64 position of its root offset; root {1 pack offsets}; the pack {0 file hashes, 1 files {3 length, 4 data}}). → (trpfs, trpfd, pack position)"""
    hashes = ('u64', [h for h, _ in files])
    fd, _ = flatbuffer(bytes(4), [(0, ('table', {0: hashes, 1: ('refs', [('str', 'pack0')]), 2: ('refs', [('table', {0: ('Q', 0)}) for _ in files]),
                                                 3: ('refs', [('table', {0: ('Q', 0x1000), 1: ('Q', len(files))})])}))])
    pack = ('table', {0: hashes, 1: ('refs', [('table', {3: ('Q', len(b)), 4: ('u8', b)}) for _, b in files])})
    fs, (pack_at, root_at) = flatbuffer(b'ONEPACK\0' + bytes(12), [(16, pack), (-1, ('table', {0: ('u64', [0x77]), 1: ('u64', [16])}))])
    struct.pack_into('<Q', fs, 8, root_at)
    return bytes(fs), bytes(fd), pack_at


def romfs(root, files):
    for rel, data in files.items(): p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    return root


def split(rom, main, out, extra=(), env=None):
    out.mkdir(parents=True)
    names = [out / n for n in ('pkg', 'shader_map.tsv', 'shader_facts.tsv', 'shader_pairs.tsv', 'pso_pairs.tsv')]
    proc = subprocess.run([sys.executable, REPO / 'dm/gfx/tools/shader_split.py', rom, main, *names, *extra], capture_output=True, text=True,
                          env=dict({k: v for k, v in os.environ.items() if k not in ('DM_SPLIT_AOC', 'DM_GAMES_DIR', 'DM_GLSLC_CAPTURE', 'DM_UE_SHADERS')}, DM_GAME=GAME, J='2', **(env or {})))
    if proc.returncode or '[RED]' in proc.stdout + proc.stderr: raise AssertionError(proc.stdout + proc.stderr)
    return proc.stdout, {str(p.relative_to(out)): p.read_bytes() for p in out.rglob('*') if p.is_file()}


def rows(tables):   # shader_map.tsv's data rows: (package, index, stage, key, source)
    return [ln.split('\t') for ln in tables['shader_map.tsv'].decode().splitlines()[3:]]


class RomfsImage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.t = Path(self.tmp.name)
        self.main = self.t / 'main'; self.main.write_bytes(b'NSO0' + bytes(0xfc))
        zstd = ss.zstd_mod()
        self.files = {'Shader/Model.Nin_Ounce_NVN2.bfsha.zs': zstd.compress(bnsh([VS, PS])),   # zstd by its magic, read with the module from inside the image
                      'Shader/Drive.Nin_Ounce_NVN2.release.sarc.zs': zstd.compress(fz.sarc([(b'Pass.sharcb', bahs([VS_BAHS, TS, MS]))])),
                      'Lib/agl/agl_resource.sarc': fz.yaz0(fz.sarc([(b'agl_common.bnsh', fz.bnsh())])),   # Switch 1 shaders beside them
                      'Movie/a.webm': b'not a shader'}

    def test_image_and_folder_give_the_same_package_and_tables(self):
        folder = romfs(self.t / 'RomFS', self.files)
        image = self.t / 'v917504/RomFS.img'; image.parent.mkdir(); romfs_image.pack(folder, image)
        log_f, a = split(folder, self.main, self.t / 'from-folder')
        log_i, b = split(image, self.main, self.t / 'from-image')
        self.assertEqual(a, b); self.assertEqual(log_f, log_i)
        got = {r[4].split('#', 1)[0]: None for r in rows(a)}
        self.assertEqual(sorted(got), ['/Lib/agl/agl_resource.sarc/agl_common.bnsh', '/Shader/Drive.Nin_Ounce_NVN2.release.sarc.zs/Pass.sharcb', '/Shader/Model.Nin_Ounce_NVN2.bfsha.zs'])
        stage = {r[3]: r[2] for r in rows(a)}
        self.assertEqual(stage[host_key(*TS)], 'task'); self.assertEqual(stage[host_key(*MS)], 'mesh')
        self.assertEqual(stage[host_key(*VS)], 'vertex'); self.assertEqual(stage[host_key(*PS)], 'pixel')
        # the vertex shader in the BNSH (+0x30 0xd8) and in the BAHS (+0x30 0) is one key, kept once with the source it first appears in
        self.assertEqual(host_key(*VS), host_key(*VS_BAHS))
        self.assertEqual(sum(r[3] == host_key(*VS) for r in rows(a)), 1)
        self.assertIn('shaders: 6 (mesh 1, pixel 2, task 1, vertex 2)', log_f)

    def test_dlc_roots_from_the_argument_or_dm_split_aoc(self):
        base = romfs(self.t / 'RomFS', {'Movie/a.webm': b'base'})
        aoc = self.t / 'aoc'; romfs(aoc / '1', {'Shader/Dlc.Nin_Ounce_NVN2.bfsha': bnsh([VS, PS])})
        romfs_image.pack(romfs(self.t / 'dlc2', {'Shader/More.bnsh': fz.bnsh()}), aoc / '2.img')   # a DLC as an image, as the NSP script extracts them
        _, by_arg = split(base, self.main, self.t / 'by-arg', [f'{aoc / "1"},{aoc / "2.img"}'])
        _, by_env = split(base, self.main, self.t / 'by-env', env={'DM_SPLIT_AOC': str(aoc)})
        self.assertEqual(by_arg, by_env)
        self.assertEqual(sorted({r[4].split('#', 1)[0] for r in rows(by_env)}), ['/aoc/1/Shader/Dlc.Nin_Ounce_NVN2.bfsha', '/aoc/2/Shader/More.bnsh'])

    def test_trinity_packs_inside_an_image(self):
        # trinity.py reads through romfs.py: with the builtin os and open, the descriptor beside a .trpfs inside the image did not exist, and Pokémon Legends: Z-A's packs were scanned
        # as one loose file, their shaders left out
        trpfs, trpfd, pack_at = trinity_fs([(0xabc, bnsh([VS, PS])), (0xdef, b'no shader')])
        folder = romfs(self.t / 'RomFS', {'arc/data.trpfs': trpfs, 'arc/data.trpfd': trpfd})
        image = self.t / 'v1/RomFS.img'; image.parent.mkdir(); romfs_image.pack(folder, image)
        for rom in (folder, image):
            with self.subTest(rom=rom.name): self.assertEqual(trinity.packs(str(rom / 'arc/data.trpfs')), [(pack_at, 'pack0', 0x1000)])
        _, a = split(folder, self.main, self.t / 'from-folder'); _, b = split(image, self.main, self.t / 'from-image')
        self.assertEqual(a, b)
        self.assertEqual({r[4] for r in rows(a)}, {'/arc/data.trpfs#pack0#0000000000000abc#BNSH@0x0#var0'})


if __name__ == '__main__': unittest.main()
