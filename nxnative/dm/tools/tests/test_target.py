"""The rewriter's output per target (DM_TARGET, patch_nso.py emit_tp): the Android rewrite of a module is the apple one with each thread slot access reaching the slot block
through bionic's TLS_SLOT_APP (mrs xT, tpidr_el0; ldr xT, [xT, #16]) instead of Darwin's TSD (mrs xT, tpidrro_el0), the same sites and gates, and the target named in the fixups
table, which the host checks. Real rewrites of a synthetic module (no game files; temporary files only in a temporary folder)."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))   # its sibling test module, also when the file runs alone
from test_nso_match import image, RET   # noqa: E402

TOOLS = Path(__file__).resolve().parents[1]
MRS_TPIDRRO, MRS_TPIDR = 0xD53BD060, 0xD53BD040


def module():
    """An NRO whose one function has a site of each slot kind: mov x18, x0 (K18); mrs x1, tpidr_el0 (KTLS); mrs x2, tpidrro_el0 (KTLR); msr tpidr_el0, x3 (KTLS); svc #0x1 (a gate,
    KGATE), as test_nro_header.py lays an NRO out"""
    raw = image({0x100: [0xaa0003f2, 0xd53bd041, 0xd53bd062, 0xd51bd043, 0xd4000021, RET]})
    raw[0x1100:0x111c] = raw[0x20:0x3c]
    struct.pack_into('<I', raw, 4, 0x1100)
    struct.pack_into('<ii', raw, 0x1110, 0x3000 - 0x1100, 0x3014 - 0x1100)
    raw[0x10:0x14] = b'NRO0'
    struct.pack_into('<I', raw, 0x18, len(raw))
    struct.pack_into('<6I', raw, 0x20, 0, 0x1000, 0x1000, 0x2000, 0x3000, 0x1000)
    struct.pack_into('<I', raw, 0x38, 0)
    return raw


class Target(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); t = Path(cls.tmp.name)
        src = t / 'input.nro'; src.write_bytes(module())
        cls.out = {}
        for target in ('apple', 'android', 'ios'):
            d = t / target; d.mkdir()
            env = dict(os.environ, DM_GAME='', DM_TARGET=target, DM_REWRITE_J='1')
            cls.out[target] = (d, subprocess.run([sys.executable, str(TOOLS / 'patch_nso.py'), str(src), str(d), 'nro0'], env=env, capture_output=True, text=True))

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def files(self, target):
        d, run = self.out[target]
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        fix = {l[2:].split('\t')[0]: l.split('\t')[1] for l in (d / 'nro0.fixups.tsv').read_text().splitlines() if l.startswith('# ')}
        img = (d / 'nro0.patched.bin').read_bytes(); po = int(fix['PATCH_OFF'], 0)
        return d, fix, [struct.unpack_from('<I', img, o)[0] for o in range(po, len(img), 4)]

    def test_slot_access_per_target(self):
        _, fa, pa = self.files('apple'); _, fb, pb = self.files('android')
        self.assertEqual((fa.pop('TARGET'), fb.pop('TARGET')), ('apple', 'android'))   # the host refuses a module rewritten for the other one
        self.assertEqual(fa, fb)
        n = sum(w & 0xFFFFFFE0 == MRS_TPIDRRO for w in pa)
        self.assertGreaterEqual(n, 5)   # one per site at least: the four slot sites and the gate
        self.assertFalse([w for w in pb if w & 0xFFFFFFE0 == MRS_TPIDRRO], 'tpidrro_el0 reads 0 for a 64-bit Linux task')
        pairs = [(w, pb[k + 1]) for k, w in enumerate(pb[:-1]) if w & 0xFFFFFFE0 == MRS_TPIDR]
        self.assertEqual(len(pairs), n)
        for w, ldr in pairs: t = w & 31; self.assertEqual(ldr, 0xF9400800 | t << 5 | t)   # ldr xT, [xT, #16]: bionic's TLS_SLOT_APP, the guest thread's slot block
        self.assertEqual(len(pb), len(pa) + n)

    def test_same_sites_and_gates(self):
        da, _, _ = self.files('apple'); db, _, _ = self.files('android')
        for ext in ('sites.tsv', 'cuts.tsv', 'obs.tsv', 'changed.tsv'): self.assertEqual((da / f'nro0.{ext}').read_text(), (db / f'nro0.{ext}').read_text(), ext)
        ga, gb = [(da / 'nro0.gates.tsv').read_text().splitlines(), (db / 'nro0.gates.tsv').read_text().splitlines()]
        self.assertEqual([l.split('\t')[1:] for l in ga], [l.split('\t')[1:] for l in gb])   # the return points move with the longer stubs
        self.assertEqual(len(ga), 1)

    def test_unknown_target_stops(self):
        _, run = self.out['ios']
        self.assertNotEqual(run.returncode, 0); self.assertIn('DM_TARGET ios: apple or android', run.stderr)


if __name__ == '__main__': unittest.main()
