# Unit tests of the port pipeline's own parts (python -m unittest discover tests). They need no game files, toolchains or device.
import contextlib, hashlib, importlib.util, io, json, os, random, re, shutil, struct, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ['NX2A_HOME'] = tempfile.mkdtemp(prefix='nx2a_test_home_')

from porter import config, datapack, pipeline, profiles   # noqa: E402
from porter.upstream import android_profile, game          # noqa: E402

# The pipeline's messages hold "→" (mem_profile's alias region line): stdout as porter.cli.main sets it, or the Windows release runner's cp1252 pipe fails the test
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')

NEO = '010043B013C5C000'


def profile_dir(rows, gid='0100000000AAA000'):
    """A throwaway game profile with an android.tsv of these rows, found through DM_GAMES_DIR"""
    root = Path(tempfile.mkdtemp(prefix='nx2a_test_profiles_')); d = root / gid; d.mkdir()
    (d / 'game.tsv').write_text(f'title_id\t{gid}\nprogram_id\t{gid}\nname\tTest Game\nipa\tTestGame.ipa\nmodules\tmain sdk\n', encoding='utf-8')
    (d / 'android.tsv').write_text(''.join(f'{k}\t{v}\n' for k, v in rows), encoding='utf-8')
    os.environ['DM_GAMES_DIR'] = str(root)
    return d


class AndroidProfiles(unittest.TestCase):
    def tearDown(self): os.environ.pop('DM_GAMES_DIR', None)

    def test_neo(self):
        p = profiles.load(NEO)
        self.assertEqual(p.status, 'playable'); self.assertEqual(p.package, 'org.nx2a.neo'); self.assertTrue(p.landscape)
        self.assertEqual(p.settings, {'docked': '0', 'overlay': '1', 'rumble': '1'}); self.assertEqual(p.requires, ['bcn'])

    def test_only_android_profiles_listed(self):
        rows = profiles.all_entries()
        self.assertIn(NEO, [r.id for r in rows]); self.assertTrue(all(r.android for r in rows))
        self.assertGreater(len(profiles.all_entries(android_only=False)), len(rows))

    def test_fps_setting(self):   # the pause menu's Show FPS (front_android.c settings_load fps=)
        d = profile_dir([('status', 'boots'), ('setting', 'fps=1')])
        self.assertEqual(android_profile.load(d.name).settings, {'fps': '1'})
        with self.assertRaises(android_profile.ProfileError): android_profile.load(profile_dir([('status', 'boots'), ('setting', 'fps=2')]).name)

    def test_opacity_setting(self):   # the gamepad's opacity, 20..80 percent by 5 (the menu's slider, overlay_layout.h OV_OPACITY_MIN..MAX)
        self.assertEqual(android_profile.load(profile_dir([('status', 'boots'), ('setting', 'opacity=45')]).name).settings, {'opacity': '45'})
        for v in ('10', '85', '42'):
            with self.assertRaises(android_profile.ProfileError, msg=v): android_profile.load(profile_dir([('status', 'boots'), ('setting', f'opacity={v}')]).name)

    def test_default_package(self):
        d = profile_dir([('status', 'boots')])
        self.assertEqual(android_profile.load(d.name).package, 'org.nx2a.t0100000000aaa000')

    def test_bad_keys_are_refused(self):
        for rows in ([('status', 'great')], [('colour', 'red')], [('setting', 'volume=3')], [('overlay', 'tablet q 0.1 0.2')], [('env', 'lower=1')], [('package', 'Not A Package')],
                     [('status', 'boots'), ('status', 'playable')]):
            d = profile_dir(rows)
            with self.assertRaises(android_profile.ProfileError, msg=rows): android_profile.load(d.name)

    def test_header_default_layout(self):   # the ergonomic layouts (the themes' design), the tablet's then the phone's, the same controls in each
        h = android_profile.header(android_profile.load(NEO))
        rx = r'\{(OV_\w+), (-?[\d.]+)f, (-?[\d.]+)f, ([\d.]+)f, ([\d.]+)f, ([\d.]+)f, (\w+), "([^"]*)", "(\w+)"\}'
        tablet, phone = (re.findall(rx, h.split(f'#define DM_AOV_{n}')[1].split('#define')[0]) for n in ('TABLET', 'PHONE'))
        self.assertEqual((len(tablet), len(phone)), (16, 16)); self.assertIn('#define DM_AOV_N 16', h)
        self.assertEqual([r[0] for r in tablet], [r[0] for r in phone]); self.assertEqual([r[6] for r in tablet], [r[6] for r in phone])
        self.assertEqual(tablet[0], ('OV_STICK_L', '0.0', '0.2105', '0.5789', '0.2421', '0.2421', '0', '', 'stick_l'))   # 184 px of the 1216x760 design, 160 px from the left edge
        a = next(r for r in tablet if r[6] == 'DM_BTN_A'); self.assertEqual(a[1:3], ('1.0', '-0.1421'))   # from the right edge
        self.assertEqual(next(r for r in tablet if r[6] == 'DM_BTN_MINUS')[-2], '-')   # the minus button's label, not "no label"

    def test_header_overlay_move(self):   # a moved control: x and y fractions of the window, in the one layout named
        d = profile_dir([('status', 'boots'), ('overlay', 'phone a 0.5 0.5')])
        h = android_profile.header(android_profile.load(d.name))
        self.assertIn('{OV_BUTTON, 0.5f, 0.0f, 0.5f, 0.1456f, 0.1456f, DM_BTN_A, "A", "a"}', h.split('#define DM_AOV_PHONE')[1])
        self.assertNotIn('0.5f, 0.0f, 0.5f', h.split('#define DM_AOV_PHONE')[0])
        with self.assertRaises(android_profile.ProfileError): android_profile.load(profile_dir([('status', 'boots'), ('overlay', 'a 0.5 0.5')]).name)

    def test_theme_mapping(self):   # porter packs the game's own pack, the ergo pack for a game the table does not name
        self.assertEqual(android_profile.theme(NEO).name, 'neotwewy'); self.assertEqual(android_profile.theme(NEO.lower()).name, 'neotwewy')
        self.assertEqual(android_profile.theme('0100000000AAA000').name, 'ergo')
        rows = [l.split('\t') for l in (android_profile.THEMES / 'themes.tsv').read_text(encoding='utf-8').splitlines() if l and not l.startswith('#')]
        self.assertEqual(len({r[0] for r in rows}), len(rows))   # one pack per game
        for r in rows: self.assertRegex(r[0], r'^0100[0-9A-F]{12}$'); self.assertTrue((android_profile.THEMES / r[1] / 'theme.tsv').is_file(), r)

    def test_theme_packs(self):   # every pack: both layouts' atlases, and sprites only of the layout's controls, within the atlas, every control drawn idle (a button held too)
        from PIL import Image
        names = {r['name'] for r in android_profile.default_layout()}
        for pack in sorted(p for p in android_profile.THEMES.iterdir() if p.is_dir()):
            rows = [l.split('\t') for l in (pack / 'theme.tsv').read_text(encoding='utf-8').splitlines() if l and not l.startswith('#')]
            lays = {r[1]: r for r in rows if r[0] == 'layout'}
            self.assertEqual(set(lays), set(android_profile.LAYOUTS), pack.name)
            for lay, row in lays.items():
                im = Image.open(pack / row[4]); self.assertIn(im.mode, ('P', 'RGBA'), pack.name)   # what overlay_theme.c decodes
                sp = [r for r in rows if r[0] == 'sprite' and r[1] == lay]
                for r in sp:
                    self.assertIn(r[2], names, r); self.assertIn(r[4], ('idle', 'held', 'knob', 'decor'), r)
                    x, y, w, h = map(int, r[5:9]); self.assertTrue(x >= 0 and y >= 0 and x + w <= im.width and y + h <= im.height, (pack.name, r))
                drawn = {(r[2], r[3], r[4]) for r in sp}
                for n in names - {'dpad'}:
                    self.assertIn((n, '-', 'idle'), drawn, (pack.name, lay, n))
                    if n not in ('stick_l', 'stick_r'): self.assertIn((n, '-', 'held'), drawn, (pack.name, lay, n))
                for d in ('up', 'down', 'left', 'right'): self.assertTrue({('dpad', d, 'idle'), ('dpad', d, 'held')} <= drawn, (pack.name, lay, d))
                self.assertTrue({('stick_l', '-', 'knob'), ('stick_r', '-', 'knob')} <= drawn, (pack.name, lay))

    def test_apk_assets(self):   # mkapk.py --asset: the theme's files as assets/overlay/<name>, nothing of the other packs
        t = Path(tempfile.mkdtemp(prefix='nx2a_test_apk_'))
        try:
            so = t / 'libdmhost.so'; so.write_bytes(b'\x7fELF' + b'\0' * 60); pack = android_profile.theme(NEO)
            args = [a for f in ('theme.tsv', 'tablet.png', 'phone.png') for a in ('--asset', f'overlay/{f}={pack / f}')]
            import subprocess, zipfile
            subprocess.run([sys.executable, str(REPO / 'nxnative/dm/android/apk/mkapk.py'), str(so), str(t / 'a.apk'), str(t / 'k.json'), '--lib', 'dmhost'] + args,
                           check=True, capture_output=True)
            with zipfile.ZipFile(t / 'a.apk') as z:
                self.assertEqual(sorted(n for n in z.namelist() if n.startswith('assets/')), ['assets/overlay/phone.png', 'assets/overlay/tablet.png', 'assets/overlay/theme.tsv'])
                self.assertEqual(z.read('assets/overlay/tablet.png'), (pack / 'tablet.png').read_bytes())
        finally:
            shutil.rmtree(t, ignore_errors=True)


class DataPack(unittest.TestCase):
    def test_build(self):
        t = Path(tempfile.mkdtemp(prefix='nx2a_test_pack_'))
        try:
            (t / 'patched').mkdir(); (t / 'patched/main.patched.bin').write_bytes(b'x' * 10)
            (t / 'romfs/Data').mkdir(parents=True); (t / 'romfs/Data/a.bin').write_bytes(b'y' * 20)
            (t / 'aoc/1').mkdir(parents=True); (t / 'aoc/1/d.bin').write_bytes(b'z')
            (t / 'shaders.dmsv').write_bytes(b'DMSV')
            p = profiles.load(NEO)
            man = datapack.build(t / 'out-data', profile=p, version='v196608', display='1.03', patched=t / 'patched', romfs=t / 'romfs', shaders=t / 'shaders.dmsv',
                                 aoc=t / 'aoc', language='ja')
            o = t / 'out-data'
            self.assertTrue((o / 'mods/main.patched.bin').is_file()); self.assertTrue((o / 'RomFS/Data/a.bin').is_file()); self.assertTrue((o / 'aoc/1/d.bin').is_file())
            env = (o / 'env.txt').read_text()
            self.assertIn('DM_LANG=ja', env); self.assertIn('DM_AOC_DIR=/sdcard/Android/data/org.nx2a.neo/files/aoc', env)
            self.assertEqual((o / 'settings.txt').read_text(), 'docked=0\noverlay=1\nrumble=1\n')
            self.assertEqual((man['files'], man['bytes'], man['dlc'], man['requires']), (4, 35, [1], ['bcn']))   # the installer checks requires
            self.assertEqual(datapack.read_manifest(o)['package'], 'org.nx2a.neo')
            datapack.build(t / 'touch-data', profile=p, version='v196608', display='1.03', patched=t / 'patched', romfs=t / 'romfs', shaders=t / 'shaders.dmsv',
                           settings=pipeline.TOUCH_SETTINGS)   # --touch: the gamepad starts hidden, the profile's other settings stay
            self.assertEqual((t / 'touch-data/settings.txt').read_text(), 'docked=0\noverlay=0\nrumble=1\n')
            bare = android_profile.load(profile_dir([('status', 'experimental')]).name); os.environ.pop('DM_GAMES_DIR', None)   # no setting rows: --touch alone still writes the file
            datapack.build(t / 'bare-data', profile=bare, version='v196608', display='1.03', patched=t / 'patched', romfs=t / 'romfs', shaders=t / 'shaders.dmsv', settings=pipeline.TOUCH_SETTINGS)
            self.assertEqual((t / 'bare-data/settings.txt').read_text(), 'overlay=0\n')
        finally:
            shutil.rmtree(t, ignore_errors=True)

    @unittest.skipUnless(any(Path(p).is_file() for p in pipeline.LETTER_FONTS), 'no letter font on this machine')
    def test_button_font(self):   # the Nintendo extension stand-in: made once, in the pack, named in env.txt (Super Mario Bros. Wonder's "Press L+R" drew '?' without it)
        try: from fontTools.ttLib import TTFont
        except ImportError: self.skipTest('no fontTools')
        from types import SimpleNamespace
        t = Path(tempfile.mkdtemp(prefix='nx2a_test_font_'))
        try:
            port = SimpleNamespace(aw=t / 'android', logs=t / 'logs', env=dict(os.environ, PYTHONUTF8='1'), package='org.nx2a.test')
            env, files = pipeline.Port.button_font(port)
            self.assertEqual(env, ['DM_SHARED_FONT_EXT=/sdcard/Android/data/org.nx2a.test/files/nintendo_ext.ttf']); self.assertEqual(list(files), ['nintendo_ext.ttf'])
            cmap = TTFont(files['nintendo_ext.ttf']).getBestCmap()
            self.assertTrue({0xE0E0, 0xE0E1, 0xE0E4, 0xE0E5, 0xE0EF, 0xE100} <= set(cmap))   # A, B, L, R, +, stick
            made = files['nintendo_ext.ttf'].stat().st_mtime_ns
            self.assertEqual(pipeline.Port.button_font(port)[1]['nintendo_ext.ttf'].stat().st_mtime_ns, made)   # reused while the tool and the letters are unchanged
        finally:
            shutil.rmtree(t, ignore_errors=True)


class Experimental(unittest.TestCase):
    def test_ios_profile_copied_with_experimental_android_tsv(self):   # an iOS-only profile is copied out of the repository, never edited in place
        ios_only = '010028600EBDA000'   # Super Mario 3D World + Bowser's Fury: an iOS profile without android.tsv or patches.py
        t = Path(tempfile.mkdtemp(prefix='nx2a_test_exp_')); environ = dict(os.environ)
        try:
            port = pipeline.Port(pipeline.Options(folder=t, experimental=True)); port.env = {}; port.work = t / f'{ios_only}_build'; port.o.keys = None
            g, ap = port.experimental_profile(game.load(ios_only), ios_only)
            self.assertEqual(g.dir, (t / f'{ios_only}_build/profile/{ios_only}').resolve())
            self.assertEqual((ap.status, ap.package), ('experimental', f'org.nx2a.t{ios_only.lower()}'))
            self.assertFalse((REPO / f'nxnative/dm/games/{ios_only}/android.tsv').exists())
            self.assertFalse((g.dir / 'patches.py').exists())   # patch_nso.py finds the per-build cuts in dm/games
            self.assertEqual(os.environ.get('DM_GAMES_FIRST'), '1')
        finally:
            os.environ.clear(); os.environ.update(environ); shutil.rmtree(t, ignore_errors=True)


def _npdm(pid):   # an NPDM whose ACI0 section names this program ID (porter.dumpdir.program_id reads only that)
    d = bytearray(0x100); d[:4] = b'META'; struct.pack_into('<II', d, 0x70, 0x80, 0x40); d[0x80:0x84] = b'ACI0'; struct.pack_into('<Q', d, 0x90, pid); return bytes(d)


def _dump(root, app=0x0100000000AAA000, base_romfs=True, updates=(), dlc=()):   # a dump folder in upstream pack_dump.py's layout, with a few files in each tree
    def version(d, pid, files):
        (d / 'exefs').mkdir(parents=True); (d / 'exefs/main').write_bytes(b'NSO0' + bytes(pid & 0xff for _ in range(12))); (d / 'exefs/main.npdm').write_bytes(_npdm(pid))
        for rel, data in files.items(): p = d / 'romfs' / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    version(root / 'base', app, {'Movie/a.webm': b'base', 'x.bin': b'x'} if base_romfs else {})
    for v in updates: version(root / f'updates/{v}', app, {'Movie/a.webm': b'patched', '._x.bin': b'appledouble', 'y.bin': b'y'})
    for tid in dlc: p = root / f'dlc/{tid:016X}/0/romfs/d.bin'; p.parent.mkdir(parents=True); p.write_bytes(b'dlc')
    return root


class DumpFolder(unittest.TestCase):   # porter/dumpdir.py: a dump folder (base/exefs, base/romfs, updates/<version>/, dlc/) as upstream's scripts/pack_dump.py takes it
    APP = 0x0100000000AAA000

    def setUp(self): self.t = Path(tempfile.mkdtemp(prefix='nx2a_test_dump_'))
    def tearDown(self): shutil.rmtree(self.t, ignore_errors=True); os.environ.pop('DM_GAMES_DIR', None)

    def test_scan(self):   # the title ID from main.npdm, each update by its decimal version, each DLC by its title ID; the base is v0
        from porter import dumpdir
        d = _dump(self.t, updates=(917504, 65536), dlc=(self.APP + 0x1001,))
        self.assertTrue(dumpdir.is_dump(d))
        app, fam = dumpdir.scan(d)
        self.assertEqual(app, self.APP); self.assertEqual([v for v, _ in fam['updates']], [65536, 917504])
        self.assertEqual(fam['dlc'], [(self.APP + 0x1001, None, d / f'dlc/{self.APP + 0x1001:016X}/0/romfs')])
        self.assertEqual(sorted(dumpdir.versions(fam)), ['v0', 'v65536', 'v917504'])

    def test_update_alone(self):   # no base RomFS: the update's patched RomFS is whole, and the base is no version of its own (Mario Kart World's 1.8.0 NUP holds the update alone)
        from porter import dumpdir
        d = _dump(self.t, base_romfs=False, updates=(917504,))
        self.assertTrue(dumpdir.is_dump(d)); self.assertEqual(list(dumpdir.versions(dumpdir.scan(d)[1])), ['v917504'])
        self.assertFalse(dumpdir.is_dump(_dump(self.t / 'none', base_romfs=False)))   # no RomFS anywhere: not a dump

    def test_refused(self):   # not a base title ID; an update of another program; a DLC of another game
        from porter import dumpdir
        with self.assertRaises(pipeline.BuildError): dumpdir.scan(_dump(self.t / 'a', app=self.APP + 0x800))
        d = _dump(self.t / 'b', updates=(65536,)); (d / 'updates/65536/exefs/main.npdm').write_bytes(_npdm(self.APP + 0x2000))
        with self.assertRaises(pipeline.BuildError): dumpdir.link_version(d / 'updates/65536', self.APP, self.t / 'out/v65536')
        with self.assertRaises(pipeline.BuildError): dumpdir.scan(_dump(self.t / 'c', dlc=(0x0100000000BBB000 + 0x1001,)))

    def test_link_version(self):   # the update's RomFS alone (never laid over the base), as hard links, hidden AppleDouble files left out; a second run reuses the trees
        from porter import dumpdir
        d = _dump(self.t, updates=(917504,)); out = self.t / 'build/extract/v917504'
        dumpdir.link_version(d / 'updates/917504', self.APP, out)
        self.assertEqual(sorted(str(p.relative_to(out / 'RomFS')) for p in (out / 'RomFS').rglob('*') if p.is_file()), ['Movie/a.webm', 'y.bin'])
        self.assertEqual((out / 'RomFS/Movie/a.webm').read_bytes(), b'patched'); self.assertTrue((out / 'ExeFS/main').is_file())
        if dumpdir.can_link(self.t): self.assertTrue(os.path.samefile(out / 'RomFS/y.bin', d / 'updates/917504/romfs/y.bin'))
        ino = (out / 'RomFS').stat().st_ino
        dumpdir.link_version(d / 'updates/917504', self.APP, out); self.assertEqual((out / 'RomFS').stat().st_ino, ino)   # reused
        (d / 'updates/917504/romfs/z.bin').write_bytes(b'z'); dumpdir.link_version(d / 'updates/917504', self.APP, out)
        self.assertTrue((out / 'RomFS/z.bin').is_file())   # a changed dump links again

    def test_link_dlc(self):   # aoc/<index>/ with index = DLC title ID − base − 0x1000; a DLC gone from the dump goes from aoc/; none at all: no aoc/
        from porter import dumpdir
        aoc = self.t / 'build/aoc'; (aoc / '7').mkdir(parents=True)
        dumpdir.link_dlc(self.APP, dumpdir.scan(_dump(self.t / 'd', dlc=(self.APP + 0x1003,)))[1]['dlc'], aoc)
        self.assertEqual(sorted(p.name for p in aoc.iterdir() if p.is_dir()), ['3']); self.assertEqual((aoc / '3/d.bin').read_bytes(), b'dlc')
        self.assertIsNone(dumpdir.link_dlc(self.APP, [], aoc)); self.assertFalse(aoc.exists())

    def test_inspect(self):   # porter inspect on a dump folder needs no prod.keys; a version the profile does not register is packed from a derived profile
        profile_dir([('status', 'experimental')], f'{self.APP:016X}')
        info = pipeline.inspect(_dump(self.t, base_romfs=False, updates=(917504,)))
        self.assertEqual((info['title_id'], info['versions'], info['pack'], info['derived']), (f'{self.APP:016X}', ['v917504'], 'v917504', True))


def _nup(path, packages, standalone=True):
    """packages: [(kind 0 base / 1 update / 2 DLC, title ID, version, {'exefs': {name: bytes}, 'romfs': {path: bytes}})] → a NUP file (SPEC v1, as upstream's
    dm/tools/tests/test_nsp_nup.py writes them), every file of 200 bytes or more zstd-compressed; standalone: the updates' standalone_update flag"""
    from porter.upstream import pack_nup
    dirs, files, slots, pool = [], [], [], bytearray()
    def walk(name, tree):   # folders pre-order, each one's children consecutive and sorted
        d = len(dirs); dirs.append([name, 0, 0]); kids = sorted(tree.items(), key=lambda kv: kv[0].encode())
        first = len(slots); slots.extend([None] * len(kids)); dirs[d][1:] = [first, len(kids)]
        for j, (n, v) in enumerate(kids):
            if isinstance(v, bytes): slots[first + j] = ('f', len(files)); files.append((n, v))
        for j, (n, v) in enumerate(kids):
            if isinstance(v, dict): slots[first + j] = ('d', walk(n, v))
        return d
    def nest(flat):
        t = {}
        for q, v in flat.items():
            *ds, f = q.split('/'); c = t
            for x in ds: c = c.setdefault(x, {})
            c[f] = v
        return t
    recs = []
    for kind, tid, ver, tree in packages:
        t = {k: v if k == 'exefs' else nest(v) for k, v in tree.items()}; root = walk('', t)
        sub = {n: q[1] for n, q in zip(sorted(t), slots[dirs[root][1]:dirs[root][1] + dirs[root][2]])}
        recs.append((tid, root, sub.get('exefs', 0xFFFFFFFF), sub.get('romfs', 0xFFFFFFFF), ver, kind))
    def name(x): b = x.encode(); off = len(pool); pool.extend(b); return off, len(b)
    pool_off = 0x30 + len(recs) * 0x40 + len(dirs) * 0x18 + len(files) * 0x50 + len(slots) * 4
    names_d, names_f = [name(d[0]) for d in dirs], [name(f[0]) for f in files]
    toc_size = (pool_off + len(pool) + 15) // 16 * 16; pos = blob = 0x80 + toc_size; ext, blobs = [], []
    for n, data in files:
        stored, codec = (pack_nup.zstd.compress(data), 1) if len(data) >= 200 else (data, 0)
        if not data: ext.append((0, 0, 0)); continue
        ext.append((pos, len(stored), codec)); blobs.append((pos, stored)); pos = (pos + len(stored) + 15) // 16 * 16
    toc = bytearray(struct.pack('<4sIQQQQQ', b'TOC\x01', len(recs), len(dirs), len(files), pool_off, len(pool), 0))
    for tid, root, ex, ro, ver, kind in recs: toc += struct.pack('<QIIIIBBHIQI20x', tid, root, ex, ro, ver, kind, int(standalone and kind == 1), 0, 0, 0, 0)
    for (o, ln), d in zip(names_d, dirs): toc += struct.pack('<IIIIQ', o, ln, d[1], d[2], 0)
    for (o, ln), (n, data), (eo, es, c) in zip(names_f, files, ext): toc += struct.pack('<IIQQQ32sBBH12x', o, ln, len(data), eo, es, hashlib.sha256(data).digest(), c, 0, 0)
    for k, i in slots: toc += struct.pack('<I', i if k == 'd' else len(dirs) + i)
    toc += pool; toc += b'\0' * (toc_size - len(toc))
    body = bytearray(pos); body[0x80:0x80 + toc_size] = toc
    for o, b in blobs: body[o:o + len(b)] = b
    body[:0x80] = struct.pack('<4sHHQQ32sQQII16s32s', b'NUP\x01', 0x80, 1, 0x80, toc_size, hashlib.sha256(toc).digest(), blob, pos, len(recs), 0, b'\x11' * 16, b'\0' * 32)
    Path(path).write_bytes(body)
    return Path(path)


def _nup_version(pid, marker='Movie/a.webm', main=b'NSO0' + bytes(12)):
    return {'exefs': {'main': main, 'main.npdm': _npdm(pid), '.listing': b'not a module'}, 'romfs': {marker: b'webm' * 100, 'Data/b.bin': b'b', 'empty': b''}}


class NupInput(unittest.TestCase):   # a NUP container (upstream scripts/pack_nup.py): Mario Kart World 1.8.0's holds one standalone update and no base record
    APP, VER = 0x0100000000AAA000, 917504

    def setUp(self): self.t = Path(tempfile.mkdtemp(prefix='nx2a_test_nup_')); (self.t / 'game').mkdir()
    def tearDown(self): shutil.rmtree(self.t, ignore_errors=True); os.environ.pop('DM_GAMES_DIR', None)

    def nup(self, packages=None, **kw):
        return _nup(self.t / 'game/Game [v1.8.1].nup', packages or [(1, self.APP, self.VER, _nup_version(self.APP))], **kw)

    def test_scan_update_alone(self):   # no base record, standalone_update: the update is the whole game, there is no v0
        from porter.upstream import nsp
        p = self.nup()
        self.assertEqual(nsp.nup_file(self.t / 'game'), p); self.assertEqual(nsp.nup_file(p), p)
        app, fam = nsp.scan_nup(p)
        try:
            self.assertEqual((app, fam['base'], [v for v, _ in fam['updates']], fam['dlc']), (self.APP, None, [self.VER], []))
            self.assertEqual(list(nsp.nsp_versions(fam, {})), [f'v{self.VER}'])
        finally: fam['nup'].close()
        with self.assertRaises(pipeline.BuildError): nsp.scan_nup(_nup(self.t / 'b.nup', [(1, self.APP, self.VER, _nup_version(self.APP))], standalone=False))   # an update of a base elsewhere

    def test_nup_file_rules(self):   # an NSP beside it: the NSPs are what the player gave; two NUPs: one game per folder
        from porter.upstream import nsp
        p = self.nup()
        (self.t / 'game/x.nsp').write_bytes(b'PFS0'); self.assertIsNone(nsp.nup_file(self.t / 'game')); (self.t / 'game/x.nsp').unlink()
        shutil.copyfile(p, self.t / 'game/other.nup')
        with self.assertRaises(pipeline.BuildError): nsp.nup_file(self.t / 'game')

    def test_extract(self):   # ExeFS (dump-tool artifacts left out) and RomFS.img, which the second run keeps; the image holds every file of the container's RomFS
        from porter.upstream import nsp, pack_nup
        app, fam = nsp.scan_nup(self.nup())
        try:
            ex = self.t / 'build/extract'; k = fam['updates'][0][1]
            d = nsp.extract_nup(ex, f'v{self.VER}', k, None, app, fam['nup'])
            self.assertEqual(sorted(p.name for p in (d / 'ExeFS').iterdir()), ['main', 'main.npdm'])
            idx = pack_nup.romfs_image.read_index(str(d / 'RomFS.img'))
            self.assertEqual({q: v[2] for q, v in idx.items() if not v[0]}, {'Movie/a.webm': 400, 'Data/b.bin': 1, 'empty': 0})
            ident = pack_nup.romfs_image.file_identity(d / 'RomFS.img')
            nsp.extract_nup(ex, f'v{self.VER}', k, None, app, fam['nup']); self.assertEqual(pack_nup.romfs_image.file_identity(d / 'RomFS.img'), ident)   # reused
            self.assertEqual(pipeline.romfs_of(d), d / 'RomFS.img')
        finally: fam['nup'].close()

    def test_dlc(self):   # aoc/<index>.img, index = DLC title ID − base − 0x1000; the data pack lists it
        from porter.upstream import nsp
        app, fam = nsp.scan_nup(self.nup([(1, self.APP, self.VER, _nup_version(self.APP)), (2, self.APP + 0x1002, 0, {'romfs': {'d.bin': b'dlc'}})]))
        try: aoc = nsp.extract_aoc(None, None, pipeline.NXNATIVE, self.t / 'build', app, fam, self.t / 'logs')
        finally: fam['nup'].close()
        self.assertTrue((aoc / '2.img').is_file())

    def test_inspect(self):   # porter inspect needs no keys for a NUP; no base game: no v0, and the version is packed from a derived profile
        profile_dir([('status', 'experimental')], f'{self.APP:016X}'); self.nup()
        info = pipeline.inspect(self.t / 'game')
        self.assertEqual((info['kind'], info['needs_keys'], info['base'], info['versions'], info['pack'], info['derived']), ('nup', False, None, [f'v{self.VER}'], f'v{self.VER}', True))
        self.assertIn('note', info)

    def test_space(self):   # the image is written once and linked into the data pack: the RomFS once, less what the build folder holds, plus 1 GB
        from types import SimpleNamespace as NS
        from porter.upstream import nsp
        app, fam = nsp.scan_nup(self.nup())
        p = NS(o=NS(folder=self.t / 'game'), work=self.t / 'game/build', dump=False, nup=True, fam=fam, have={f'v{self.VER}': fam['updates'][0][1]}, versions=[f'v{self.VER}'])
        real = shutil.disk_usage
        try:
            shutil.disk_usage = lambda _: NS(free=2**30 + 400); pipeline.Port.check_space(p)
            shutil.disk_usage = lambda _: NS(free=2**30 + 300)
            if pipeline.dumpdir.can_link(p.work):
                with self.assertRaises(pipeline.BuildError): pipeline.Port.check_space(p)
        finally: shutil.disk_usage = real; fam['nup'].close()

    def test_data_pack_image(self):   # RomFS.img linked into the pack, its reuse stamp still holding after the link (a hard link changes the ctime the stamp holds); the stamps stay out
        from porter.upstream import nsp, pack_nup
        app, fam = nsp.scan_nup(self.nup([(1, self.APP, self.VER, _nup_version(self.APP)), (2, self.APP + 0x1001, 0, {'romfs': {'d.bin': b'dlc'}})]))
        try:
            d = nsp.extract_nup(self.t / 'build/extract', f'v{self.VER}', fam['updates'][0][1], None, app, fam['nup'])
            aoc = nsp.extract_aoc(None, None, pipeline.NXNATIVE, self.t / 'build', app, fam, self.t / 'logs')
            (self.t / 'patched').mkdir(); (self.t / 'patched/main.patched.bin').write_bytes(b'x'); (self.t / 'shaders.dmsv').write_bytes(b'DMSV')
            man = datapack.build(self.t / 'out-data', profile=profiles.load(NEO), version=f'v{self.VER}', display='1.8.0', patched=self.t / 'patched', romfs=pipeline.romfs_of(d),
                                 shaders=self.t / 'shaders.dmsv', aoc=aoc)
            o = self.t / 'out-data'
            self.assertEqual(sorted(p.name for p in o.iterdir() if p.name not in ('env.txt', 'settings.txt', datapack.MANIFEST)), ['RomFS.img', 'aoc', 'mods', 'shaders.dmsv'])
            self.assertEqual([p.name for p in (o / 'aoc').iterdir()], ['1.img']); self.assertEqual((man['romfs'], man['dlc']), ('image', [1]))
            if pipeline.dumpdir.can_link(self.t): self.assertTrue(os.path.samefile(o / 'RomFS.img', d / 'RomFS.img'))
            datapack.build(self.t / 'out-data', profile=profiles.load(NEO), version=f'v{self.VER}', display='1.8.0', patched=self.t / 'patched', romfs=pipeline.romfs_of(d),
                           shaders=self.t / 'shaders.dmsv', aoc=aoc)   # again: the older pack's links go, which changes the ctime once more
            img = str(d / 'RomFS.img'); ident = pack_nup.romfs_image.file_identity(img)
            self.assertEqual(json.loads(Path(img + '.inputs.json').read_text())['image'], ident)
            shutil.rmtree(self.t / 'out-data'); datapack.renew_stamp(img)   # the pack deleted by hand: the port renews the stamp before the extraction
            ident = pack_nup.romfs_image.file_identity(img)
            nsp.extract_nup(self.t / 'build/extract', f'v{self.VER}', fam['updates'][0][1], None, app, fam['nup'])
            self.assertEqual(pack_nup.romfs_image.file_identity(img), ident)   # not written again
            os.utime(img, ns=(1, 1)); datapack.renew_stamp(img)   # another mtime: the stamp stays void, and the image is written again
            nsp.extract_nup(self.t / 'build/extract', f'v{self.VER}', fam['updates'][0][1], None, app, fam['nup'])
            self.assertNotEqual(pack_nup.romfs_image.file_identity(img)[1], ident[1])
        finally: fam['nup'].close()


    def test_derived_modules(self):   # a derived profile names the modules its version ships (Mario Kart World 1.0.0 has no subsdk1, 1.8.0 has), each in its build row
        from porter.upstream import nsp
        g0 = profile_dir([('status', 'experimental')], f'{self.APP:016X}')
        (g0 / 'game.tsv').write_text((g0 / 'game.tsv').read_text().replace('modules\tmain sdk', 'modules\tmain subsdk0 subsdk1 sdk') + 'build\tv0\t1.0.0\tmain=' + '11' * 16 + '\n', encoding='utf-8')
        g = game.load(f'{self.APP:016X}')
        for ver, mods in (('v65536', ('main', 'subsdk0', 'sdk')), ('v917504', ('main', 'subsdk0', 'subsdk1', 'sdk'))):
            d = self.t / ver; (d / 'ExeFS').mkdir(parents=True)
            for i, m in enumerate(mods): (d / 'ExeFS' / m).write_bytes(b'NSO0' + bytes(0x3c) + bytes([i + 1]) * 16)
            nsp.derive_profile(g, self.t / 'derived' / ver / g.id, ver, d, g.words('modules'))
            rows = [l.split('\t') for l in (self.t / 'derived' / ver / g.id / 'game.tsv').read_text().splitlines() if l and not l.startswith('#')]
            self.assertEqual(next(r[1:] for r in rows if r[0] == 'modules'), [' '.join(mods)])
            self.assertEqual([c.split('=')[0] for c in next(r for r in rows if r[0] == 'build')[3:]], list(mods))


class PushFile(unittest.TestCase):   # porter/adb.py push_file: pieces appended on the device as they arrive, resumable by the partial copy's size (a fake device: a local folder)
    class FakeAdb(pipeline.Adb):
        def __init__(self, dev): super().__init__('adb', log=lambda s: None); self.dev, self.pushes = Path(dev), 0
        def at(self, p): return self.dev / p.lstrip('/')
        def run(self, *a, **kw):
            assert a[:2] == ('push', '-Z'); q = self.at(a[3]); q.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(a[2], q); self.pushes += 1; return ''
        def shell(self, script, **kw):
            if m := re.match(r'mkdir -p (\S+) .*!= (\S+) \].* stat -c %s (\S+) 2>/dev/null', script):   # the set-up: other partial copies of this file go; the partial copy's size
                d = self.at(m[1]); d.mkdir(parents=True, exist_ok=True)
                for x in d.glob('.dmpart-*'):
                    if x.name != m[2]: shutil.rmtree(x) if x.is_dir() else x.unlink()
                q = self.at(m[3]); return f'{q.stat().st_size}\n' if q.is_file() else 'none\n'
            if m := re.fullmatch(r'truncate -s (\d+) (\S+)', script): os.truncate(self.at(m[2]), int(m[1])); return ''
            if m := re.fullmatch(r'cat (\S+) >> (\S+) && rm -f \S+ && stat -c %s \S+', script):
                with open(self.at(m[2]), 'ab') as f: f.write(self.at(m[1]).read_bytes())
                self.at(m[1]).unlink(); return f'{self.at(m[2]).stat().st_size}\n'
            if m := re.fullmatch(r'mv -f (\S+) (\S+) && stat -c %s \S+', script): os.replace(self.at(m[1]), self.at(m[2])); return f'{self.at(m[2]).stat().st_size}\n'
            raise AssertionError(script)

    def test_push_and_resume(self):
        t = Path(tempfile.mkdtemp(prefix='nx2a_test_push_'))
        try:
            src = t / 'RomFS.img'; src.write_bytes(random.Random(1).randbytes(2500))
            adb = self.FakeAdb(t / 'dev'); dst = '/sdcard/files/RomFS.img'; files = adb.at('/sdcard/files')
            adb.push_file(src, dst, chunk=1000)
            self.assertEqual((adb.at(dst).read_bytes(), adb.pushes), (src.read_bytes(), 3)); self.assertEqual([p.name for p in files.iterdir()], ['RomFS.img'])   # no partial copy or piece left
            # an interrupted push: 1.5 pieces in its partial copy (cut back to 1, then pieces 2 and 3), and a partial copy of another file, which goes
            tag, st = hashlib.sha1(dst.encode()).hexdigest()[:12], src.stat()
            ours = files / f'.dmpart-{tag}-{hashlib.sha1(f"{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:8]}'
            adb.at(dst).unlink(); ours.write_bytes(src.read_bytes()[:1500]); (files / f'.dmpart-{tag}-deadbeef').write_bytes(b'old'); (files / f'.dmpart-{tag}').mkdir()
            adb.pushes = 0; adb.push_file(src, dst, chunk=1000)
            self.assertEqual((adb.at(dst).read_bytes(), adb.pushes), (src.read_bytes(), 2)); self.assertEqual([p.name for p in files.iterdir()], ['RomFS.img'])
            ours.write_bytes(src.read_bytes()); adb.pushes = 0; adb.push_file(src, dst, chunk=1000)   # a whole partial copy: renamed, nothing pushed
            self.assertEqual((adb.at(dst).read_bytes(), adb.pushes), (src.read_bytes(), 0))
            empty = t / 'empty.bin'; empty.write_bytes(b''); adb.push_file(empty, '/sdcard/files/empty.bin'); self.assertEqual(adb.at('/sdcard/files/empty.bin').read_bytes(), b'')
        finally: shutil.rmtree(t, ignore_errors=True)

    def test_push_need(self):   # what the device lacks at its size, plus one piece
        from porter import install
        t = Path(tempfile.mkdtemp(prefix='nx2a_test_need_'))
        try:
            (t / 'RomFS.img').write_bytes(b'x' * 100); (t / 'mods').mkdir(); (t / 'mods/main').write_bytes(b'y' * 10)
            self.assertEqual(install.push_need(t, {'RomFS.img': 100, 'mods/main': 9}), 10 + (64 << 20))
        finally: shutil.rmtree(t, ignore_errors=True)


class Misc(unittest.TestCase):
    def test_config_roundtrip(self):
        config.set_value('device', 'abc:5555'); self.assertEqual(config.load()['device'], 'abc:5555')
        with self.assertRaises(KeyError): config.set_value('nope', 'x')

    def test_version_code_grows_with_updates(self):
        self.assertLess(pipeline.version_code('v0'), pipeline.version_code('v65536'))
        self.assertEqual(pipeline.version_code('v196608'), 196609)

    def test_version_code_offset(self):   # a rebuild lifted above an earlier one, still below the next game version's port
        self.assertEqual(pipeline.version_code('v589824', 1), 589826)
        self.assertLess(pipeline.version_code('v65536', pipeline.VERCODE_OFFSET_MAX), pipeline.version_code('v131072'))
        for bad in (-1, pipeline.VERCODE_OFFSET_MAX + 1):
            with self.assertRaises(pipeline.BuildError): pipeline.version_code('v65536', bad)

    def test_install_downgrade_names_the_offset(self):   # a refused downgrade says what is installed and how to lift the rebuild above it
        from porter.adb import Adb, AdbError
        a = Adb('adb')
        def run(*x, **_):
            if x[0] == 'install': raise AdbError('adb install: Failure [INSTALL_FAILED_VERSION_DOWNGRADE: Downgrade detected: Update version code 589825 is older than current 589826]')
            return '    versionCode=589826 minSdk=29 targetSdk=34\n'
        a.run = run
        with self.assertRaises(AdbError) as e: a.install('x.apk', 'org.nx2a.test')
        self.assertIn('(589826)', str(e.exception)); self.assertIn('--vercode-offset', str(e.exception))

    def test_derived_alias_headroom(self):   # issue #26: a derived profile of a game with virtual address memory gets 8 GB more alias region, the heap unchanged
        d = profile_dir([('status', 'playable')])
        try:
            t = d / 'game.tsv'; base = t.read_text(encoding='utf-8')
            t.write_text(base + '# memory\nmem_reserve\t0x140000000\nmem_alias\t0x118000000\nmem_stack\t0x10000000\nsys_resource\t0x1000000\n', encoding='utf-8')
            self.assertEqual(pipeline.alias_headroom(t), (0x118000000, 0x318000000))
            g = game.load(d.name)
            self.assertEqual((g.int('mem_reserve'), g.int('mem_alias'), g.int('mem_stack')), (0x340000000, 0x318000000, 0x10000000))
            self.assertIn('# porter: 8 GB more alias region', t.read_text(encoding='utf-8'))
            vam = base + 'mem_reserve\t0x140000000\nmem_alias\t0x118000000\nmem_stack\t0x10000000\nsys_resource\t0x1000000\n'
            t.write_text(vam, encoding='utf-8'); self.assertEqual(pipeline.alias_headroom(t, 16), (0x118000000, 0x518000000))   # --alias-headroom 16
            self.assertIn('# porter: 16 GB more alias region', t.read_text(encoding='utf-8'))
            t.write_text(vam, encoding='utf-8'); self.assertIsNone(pipeline.alias_headroom(t, 0)); self.assertEqual(t.read_text(encoding='utf-8'), vam)   # 0: as it is
            for rows in ('sys_resource\t0x0\n', ''):   # no virtual address memory, or a profile older than the key (heap games): as it is
                t.write_text(base + 'mem_reserve\t0x100000000\nmem_alias\t0x10000000\nmem_stack\t0x10000000\n' + rows, encoding='utf-8')
                self.assertIsNone(pipeline.alias_headroom(t)); self.assertIsNone(pipeline.alias_headroom(t, 16))
                self.assertEqual(game.load(d.name).int('mem_alias'), 0x10000000)
        finally: shutil.rmtree(d.parent, ignore_errors=True); os.environ.pop('DM_GAMES_DIR', None)

    def test_mt_replay_option(self):   # --mt-replay: off unless given; the port's env rows then end with DM_VK_MT_REPLAY=1 (gfx_vk_dq.c)
        from porter import cli
        self.assertTrue(cli._parser().parse_args(['port', 'x', '--mt-replay']).mt_replay); self.assertFalse(cli._parser().parse_args(['port', 'x']).mt_replay)
        self.assertFalse(pipeline.Options(Path(".")).mt_replay)

    def test_vulkan_driver_option(self):   # --vulkan-driver FILE: parsed, and a file that is not a .so library fails before any stage
        from porter import cli
        self.assertEqual(cli._parser().parse_args(['port', 'x', '--vulkan-driver', 'libvulkan_freedreno.so']).vulkan_driver, 'libvulkan_freedreno.so')
        self.assertIsNone(cli._parser().parse_args(['port', 'x']).vulkan_driver)
        t = Path(tempfile.mkdtemp(prefix='nx2a_test_game_'))
        try:
            with self.assertRaises(pipeline.BuildError): pipeline.Port(pipeline.Options(t, vulkan_driver=t / 'driver.txt')).run()
        finally: shutil.rmtree(t, ignore_errors=True)

    def test_alias_headroom_option(self):   # --alias-headroom: parsed, checked before any stage, and kept as None (automatic) when not given
        from porter import cli
        self.assertEqual(cli._parser().parse_args(['port', 'x', '--alias-headroom', '16']).alias_headroom, 16)
        self.assertIsNone(cli._parser().parse_args(['port', 'x']).alias_headroom)
        for ok in (None, 0, pipeline.ALIAS_HEADROOM_MAX): self.assertEqual(pipeline.alias_headroom_gb(ok), ok)
        t = Path(tempfile.mkdtemp(prefix='nx2a_test_game_'))
        try:
            for bad in (-1, pipeline.ALIAS_HEADROOM_MAX + 1):
                with self.assertRaises(pipeline.BuildError) as e: pipeline.Port(pipeline.Options(folder=t, alias_headroom=bad)).run()
                self.assertIn('--alias-headroom', str(e.exception))
        finally: shutil.rmtree(t, ignore_errors=True)

    def test_mem_profile(self):   # --alias-headroom on a version the profile was brought up on: a copy in profile-mem/ in front, without patches.py; none without the option
        d = profile_dir([('status', 'playable')]); environ = dict(os.environ); t = Path(tempfile.mkdtemp(prefix='nx2a_test_build_'))
        try:
            (d / 'game.tsv').write_text((d / 'game.tsv').read_text(encoding='utf-8') + 'mem_reserve\t0x140000000\nmem_alias\t0x118000000\nmem_stack\t0x10000000\nsys_resource\t0x1000000\n',
                                        encoding='utf-8')
            (d / 'patches.py').write_text('CUTS = {}\n', encoding='utf-8')
            port = pipeline.Port(pipeline.Options(folder=t, alias_headroom=16)); port.env = {}; port.work = t; port.g = game.load(d.name)
            port.mem_profile()
            self.assertEqual(port.g.dir, (t / 'profile-mem' / d.name).resolve())
            self.assertEqual((port.g.int('mem_alias'), port.g.int('mem_reserve')), (0x518000000, 0x540000000))
            self.assertFalse((port.g.dir / 'patches.py').exists()); self.assertEqual(os.environ.get('DM_GAMES_FIRST'), '1')
            shutil.rmtree(t / 'profile-mem')   # as extract() does before it
            port.o.alias_headroom = None; port.g = game.load(d.name); port.mem_profile()
            self.assertFalse((t / 'profile-mem').exists()); self.assertEqual(port.g.dir, d.resolve())
        finally:
            os.environ.clear(); os.environ.update(environ); shutil.rmtree(d.parent, ignore_errors=True); shutil.rmtree(t, ignore_errors=True)

    def test_lost_patches(self):   # issue #26: a derived module of another build than the tested one, which has cuts, is named; one with cuts of its own, or no cuts lost, is not
        d = Path(tempfile.mkdtemp(prefix='nx2a_test_exefs_'))
        try:
            for m, bid in (('main', 'aa' * 16), ('sdk', 'bb' * 16), ('subsdk0', 'cc' * 16)): (d / m).write_bytes(b'NSO0' + bytes(0x3c) + bytes.fromhex(bid))
            builds = {'v65536': {'display': '1.1.0', 'main': '11' * 16, 'sdk': '22' * 16, 'subsdk0': 'cc' * 16}}
            table = {'22' * 16: ('dm/tools/sdk_patches.py', ['AddressRegionManager constructor: node area limit', 'AddressRegionManager constructor: length']), '33' * 16: ('x', ['y'])}
            lost = pipeline.lost_patches(builds, d, ['main', 'sdk', 'subsdk0'], table)
            self.assertEqual(len(lost), 1); self.assertTrue(lost[0].startswith('sdk build bbbbbbbb… is not the tested 22222222…'))
            self.assertIn('2 registered patches (dm/tools/sdk_patches.py: AddressRegionManager constructor)', lost[0])
            table['bb' * 16] = ('dm/tools/sdk_patches.py', ['z'])   # the derived build has cuts of its own
            self.assertEqual(pipeline.lost_patches(builds, d, ['main', 'sdk', 'subsdk0'], table), [])
            real = pipeline.registered_patches()   # the real files load (the SDK-wide rules, sdk_patches.py MATCH_CUTS, are matched by code, not by build: not in the table)
            self.assertTrue(any(f == 'dm/tools/sdk_patches.py' for f, _ in real.values()) and any(f.startswith('dm/games/') for f, _ in real.values()))
        finally: shutil.rmtree(d, ignore_errors=True)

    def test_app_stem(self):
        self.assertEqual(pipeline.app_stem(game.load(NEO), 'NEO'), 'NeoTheWorldEndsWithYou')

    def test_disk_space(self):   # twice the game files, less what the build folder holds, plus 1 GB
        from types import SimpleNamespace as NS
        d = Path(tempfile.mkdtemp(prefix='nx2a_test_game_')); (d / 'g.nsp').write_bytes(b'\0' * 2**20); work = d / 'build'
        p = NS(o=NS(folder=d), work=work); real = shutil.disk_usage
        try:
            shutil.disk_usage = lambda _: NS(free=2**30 + 2**20)
            with self.assertRaises(pipeline.BuildError): pipeline.Port.check_space(p)
            work.mkdir(); (work / 'x').write_bytes(b'\0' * 2**20); pipeline.Port.check_space(p)   # a build folder already there counts
        finally: shutil.disk_usage = real


def _hfs0(files):   # an HFS0 of (name, data): the XCI's partitions (hash fields left zero: nothing here reads them)
    strs = b''.join(n.encode() + b'\0' for n, _ in files); ent, off = b'', 0
    for i, (n, d) in enumerate(files):
        ent += struct.pack('<QQII8x32s', off, len(d), sum(len(m.encode()) + 1 for m, _ in files[:i]), 0, b''); off += len(d)
    return struct.pack('<4sII4x', b'HFS0', len(files), len(strs)) + ent + strs + b''.join(d for _, d in files)


@unittest.skipUnless(importlib.util.find_spec('Crypto'), 'no pycryptodome')
class Dumps(unittest.TestCase):   # porter/dumps.py: NSZ / XCZ → NSP; an NCZ made here as nsz makes it, from an NCA of a plain section and an AES-CTR one
    def setUp(self):
        from Crypto.Cipher import AES
        from porter import dumps; self.dumps = dumps
        rnd = random.Random(7); b = lambda n: bytes(rnd.getrandbits(8) for _ in range(n))
        self.key, self.ctr = b(16), b(8) + b'\0' * 8
        head, plain = b(0x4000), b(0x2000) + bytes(0x18000)   # the CTR part compresses (zeros), so zstd has something to do
        self.nca = head + plain[:0x2000] + AES.new(self.key, AES.MODE_CTR, nonce=self.ctr[:8], initial_value=0x6000 >> 4).encrypt(plain[0x2000:])
        self.name = hashlib.sha256(self.nca).hexdigest()[:32]
        secs = struct.pack('<8sQ', b'NCZSECTN', 2) + struct.pack('<QQQQ16s16s', 0x4000, 0x2000, 1, 0, bytes(16), bytes(16)) + struct.pack('<QQQQ16s16s', 0x6000, 0x18000, 3, 0, self.key, self.ctr)
        z = dumps.zstd_mod()
        self.solid = head + secs + z.compress(plain)
        bs = 1 << 14; blocks = [plain[i:i + bs] for i in range(0, len(plain), bs)]
        comp = [z.compress(k) if i else k for i, k in enumerate(blocks)]   # the first block stored as it is (random data: compressing does not shrink it)
        self.block = head + secs + struct.pack('<8sBBBBIQ', b'NCZBLOCK', 2, 1, 0, 14, len(blocks), len(plain)) + struct.pack(f'<{len(blocks)}I', *map(len, comp)) + b''.join(comp)
        self.d = Path(tempfile.mkdtemp(prefix='nx2a_test_dumps_'))

    def nsz(self, ncz, name='g.nsz'):
        files = [(self.name + '.ncz', ncz), ('0100000000aaa0000000000000000000.tik', b't' * 0x2c0)]
        hdr = self.dumps._pfs0_header([n for n, _ in files], [len(d) for _, d in files])
        (self.d / name).write_bytes(hdr + b''.join(d for _, d in files)); return self.d / name

    def check(self, src):
        self.assertEqual(self.dumps.size(src), len(self.dumps._pfs0_header([self.name + '.nca', 'x' * 36], [0, 0])) + len(self.nca) + 0x2c0)
        out = self.dumps.decompress(src, self.d / self.dumps.unpacked_name(src))
        self.assertEqual(out.stat().st_size, self.dumps.size(src)); self.assertEqual(out.stat().st_mtime_ns, src.stat().st_mtime_ns)
        with open(out, 'rb') as f: got = {n: (f.seek(o), f.read(s))[1] for n, o, s in self.dumps.entries(out)}
        self.assertEqual(got[self.name + '.nca'], self.nca); self.assertEqual(len(got), 2)

    def test_solid_and_block(self):
        self.check(self.nsz(self.solid)); self.check(self.nsz(self.block, 'b.nsz'))

    def test_first_section_in_the_header(self):   # nsz 5.0 starts the first section at 0xc00, inside the header it keeps as it is (a DLC NSZ it made, 2026-10-29)
        s = bytearray(self.solid); s[0x4000 + 16:0x4000 + 32] = struct.pack('<QQ', 0xc00, 0x2000 + 0x4000 - 0xc00)
        self.check(self.nsz(bytes(s)))

    def test_xcz_is_its_secure_partition(self):
        secure = _hfs0([(self.name + '.ncz', self.block), ('0100000000aaa0000000000000000000.tik', b't' * 0x2c0)])
        root = _hfs0([('update', _hfs0([])), ('secure', secure)])
        x = bytearray(0x200); x[0x100:0x104] = b'HEAD'; x[0x130:0x138] = struct.pack('<Q', 0x200)
        (self.d / 'g.xcz').write_bytes(bytes(x) + root); self.check(self.d / 'g.xcz')

    def test_damage_is_found(self):   # a changed byte: a zstd error, or an NCA whose SHA-256 is not its name; either way a BuildError and no NSP
        for ncz, at in ((self.solid, 0x4000 + 16 + 0x80 + 40), (self.block, -1), (self.block, 0x4000 + 16 + 0x80 + 24 + 100)):
            bad = bytearray(ncz); bad[at] ^= 1
            with self.assertRaises(pipeline.BuildError): self.dumps.decompress(self.nsz(bytes(bad)), self.d / 'out.nsp')
            self.assertFalse((self.d / 'out.nsp').exists())

    def test_scan_takes_compressed_dumps(self):   # upstream's scan_nsps with porter's kinds: the NSZ's ticket names its title ID
        self.nsz(self.solid, 'Game [v0].nsz')
        app, fam = pipeline.nsp.scan_nsps(self.d, self.dumps.EXTS)
        self.assertEqual(app, 0x0100000000AAA000); self.assertEqual(fam['base'].suffix, '.nsz')


@unittest.skipUnless(importlib.util.find_spec('Crypto'), 'no pycryptodome')
class UntaggedXci(unittest.TestCase):   # issue #28: a cartridge dump named by its release ("hr-bdlwa.xci") has no ticket and no [title ID]: the content meta NCA's header names it
    BASE = 0x01009D901BC56000

    def setUp(self):
        rnd = random.Random(28); self.hk = bytes(rnd.getrandbits(8) for _ in range(32))
        self.d = Path(tempfile.mkdtemp(prefix='nx2a_test_xci_'))
        self.keys = self.d / 'prod.keys'; self.keys.write_text(f'header_key = {self.hk.hex()}\n')

    def nca(self, tid, ctype):   # an NCA header (0xc00 bytes) whose second sector is encrypted as Nintendo does: AES-XTS, the sector number big-endian as the tweak
        from Crypto.Cipher import AES
        p = bytearray(0x200); p[:4] = b'NCA3'; p[5] = ctype; struct.pack_into('<Q', p, 0x10, tid)
        data, t = AES.new(self.hk[:16], AES.MODE_ECB), int.from_bytes(AES.new(self.hk[16:], AES.MODE_ECB).encrypt((1).to_bytes(16, 'big')), 'little')
        c = b''
        for i in range(0, 0x200, 16):
            tb = t.to_bytes(16, 'little'); c += bytes(a ^ b for a, b in zip(data.encrypt(bytes(a ^ b for a, b in zip(p[i:i + 16], tb))), tb))
            t = (t << 1 & (1 << 128) - 1) ^ (0x87 if t >> 127 else 0)
        return bytes(0x200) + c + bytes(0x800)

    def xci(self, name, ncas, key_area=False):
        root = _hfs0([('update', _hfs0([])), ('secure', _hfs0(ncas))])
        x = bytearray(0x200); x[0x100:0x104] = b'HEAD'; x[0x130:0x138] = struct.pack('<Q', 0x200)
        (self.d / name).write_bytes(bytes(0x1000 if key_area else 0) + bytes(x) + root)

    def test_scan_reads_the_meta_nca(self):   # an update's program NCA carries the base game's title ID: only the meta NCA's counts, and the smallest of them is the base game
        self.xci('hr-bdlwa.xci', [('a.nca', self.nca(self.BASE + 0x800, 0)), ('b.cnmt.nca', self.nca(self.BASE + 0x800, 1)), ('c.nca', self.nca(self.BASE, 0)), ('d.cnmt.nca', self.nca(self.BASE, 1))])
        app, fam = pipeline.nsp.scan_nsps(self.d, ('nsp', 'xci'), self.keys)
        self.assertEqual(app, self.BASE); self.assertEqual(fam['base'].name, 'hr-bdlwa.xci')
        self.xci('hr-bdlwa.xci', [('c.nca', self.nca(self.BASE + 0x800, 0)), ('d.cnmt.nca', self.nca(self.BASE + 0x800, 1))], key_area=True)
        self.assertEqual(pipeline.nsp.title_of(self.d / 'hr-bdlwa.xci', self.hk), self.BASE + 0x800)

    def test_without_keys_the_error_says_how(self):
        self.xci('hr-bdlwa.xci', [('d.cnmt.nca', self.nca(self.BASE, 1))])
        with self.assertRaisesRegex(pipeline.BuildError, r'hr-bdlwa\.xci: no ticket .*no prod\.keys.*\[01009D901BC56000\]\[v0\]'): pipeline.nsp.scan_nsps(self.d, ('nsp', 'xci'))
        self.keys.write_text(f'header_key = {bytes(32).hex()}\n')   # another key: the header does not decrypt to NCA3
        with self.assertRaisesRegex(pipeline.BuildError, 'no content meta NCA'): pipeline.nsp.scan_nsps(self.d, ('nsp', 'xci'), self.keys)


class FpsLocker(unittest.TestCase):   # porter/fpslocker.py: FPSLocker-Warehouse configs → the host's rows (hle_fpslock.c DM_FPS_LOCKS) and the rewriter's cuts
    BID = 'd49165c05ead8ae140f189aa97934f39'
    UNITY = ('ALL_FPS:\n  # Vsync\n  -\n    type: evaluate_write\n    address: [MAIN, 0x7A642D8]\n    value_type: int32\n    value: VSYNC_TARGET\n'
             '  -\n    type: evaluate_write\n    address: [MAIN, 0x7A642E4]\n    value_type: float\n    value: "1 / FPS_LOCK_TARGET"\n')   # the shape of Viractal 1.4.5's config

    def rows(self, text, image=b''):
        from porter import fpslocker
        tsv, cuts = fpslocker.convert(text, image, self.BID, 'test')
        return [l.split('\t') for l in tsv.splitlines() if not l.startswith('#')], cuts

    def test_unity_vsync_and_frame_time(self):   # 2 vsyncs / 1/30 s at 30; 1 vsync / 1/62 s at 60 (FPS_LOCK_TARGET is the target + 2 at the refresh rate)
        rows, cuts = self.rows(self.UNITY)
        self.assertEqual(rows[0], ['build', self.BID]); self.assertEqual(cuts, [])
        self.assertEqual(rows[1][1:], ['int32', '0x7a642d8', '2', '1', '-'])
        self.assertEqual(rows[2][1:3], ['float', '0x7a642e4']); self.assertAlmostEqual(float(rows[2][3]), 1 / 30); self.assertAlmostEqual(float(rows[2][4]), 1 / 62)

    def test_expressions(self):
        from porter.fpslocker import evaluate
        self.assertEqual(evaluate('TruncDec(FRAMETIME_TARGET, 2)', 30), 33.33); self.assertEqual(evaluate('2^3', 60), 8)
        self.assertEqual(evaluate('VSYNC_TARGET', 60), 1); self.assertEqual(evaluate('floor(FPS_TARGET / 7)', 60), 8)
        with self.assertRaises(Exception): evaluate('__import__("os")', 60)

    def test_pointers_lists_and_compares(self):
        rows, _ = self.rows('ALL_FPS:\n  -\n    type: compare\n    compare_address: [MAIN, 0x10, 0x8]\n    compare_type: "!="\n    compare_value_type: int8\n    compare_value: 0\n'
                            '    address: [MAIN, 0x20, 0x30, -4]\n    value_type: int32\n    value: [2, 2]\n  -\n    type: block\n    what: timing\n')
        self.assertEqual(rows[1][1:], ['int32', '0x20,0x30,-0x4', '2,2', '2,2', 'int8 0x10,0x8 != 0'])
        self.assertEqual(len(rows), 2)   # block timing is FPSLocker's own frame delay: dropped

    def test_refused_whole(self):   # entries work together: one the host cannot apply refuses the config
        from porter import fpslocker
        for text in (self.UNITY + 'DECLARATIONS:\n  -\n    type: variable\n    name: x\n    value_type: float\n    default_value: 0\n',
                     self.UNITY + 'MASTER_WRITE:\n  -\n    type: asm_a64\n    main_offset: 0x10\n    instructions: [[b.ne, :goto1], [nop], :goto1, [nop]]\n',   # labels parse
                     'ALL_FPS:\n  -\n    type: write\n    address: [HEAP, 0x10]\n    value_type: int32\n    value: 1\n',
                     'ALL_FPS:\n  -\n    type: compare\n    compare_address: [MAIN, 0x10]\n    compare_type: "=="\n    compare_value_type: int8\n    compare_value: 1\n    value_type: refresh_rate\n    value: 30\n',
                     'ALL_FPS: [\n'):
            with self.assertRaises(fpslocker.Refused): fpslocker.convert(text, b'', self.BID)

    def test_master_write_bytes_are_cuts(self):   # the original bytes come from main's image, so the rewriter can assert them
        image = bytes(range(256)) * 4
        _, cuts = self.rows(self.UNITY + 'MASTER_WRITE:\n  -\n    type: bytes\n    main_offset: 0x10\n    value_type: uint8\n    value: [0x1F, 0x20, 0x03]\n', image)
        self.assertEqual(cuts, [(0x10, '101112', '1f2003', 'FPSLocker-Warehouse MASTER_WRITE entry 1')])
        from porter import fpslocker
        ns = {}; exec(fpslocker.patches_py(self.BID, cuts), ns); self.assertEqual(ns['CUTS'][self.BID][0][:3], (0x10, '101112', '1f2003'))

    def test_url(self):
        from porter import fpslocker
        self.assertTrue(fpslocker.url('0100fd501f95a000', self.BID).endswith('/patches/0100FD501F95A000/D49165C05EAD8AE1.yaml'))


class Saves(unittest.TestCase):   # porter/saves.py: the save folders on the device, a .zip or folder to import (no device: the shell answers are made up)
    class FakeAdb:
        def __init__(self, sizes, times, packages=('org.nx2a.a',)): self.sizes, self.times, self.packages, self.cmds = sizes, times, packages, []
        def remote_sizes(self, rdir): return dict(self.sizes)
        def shell(self, script, **kw):
            self.cmds.append(script)
            if script.startswith('pm list packages'): return ''.join(f'package:{p}\n' for p in self.packages)
            if 'stat -c "%Y %n"' in script: return ''.join(f'{t} {n}\n' for n, t in self.times.items())
            return ''

    def test_listing(self):   # newest first; a device save counts with its own; hidden folders (an export still copying) and the request files left out
        from porter import saves
        adb = self.FakeAdb({'20261030-100000/a.dat': 10, '20261030-100000.device/d.dat': 5, '20261029-090000/a.dat': 7, 'result.txt': 40, '.x.part/a.dat': 9, 'console/sub/b.bin': 3},
                           {'20261030-100000': 200, '20261030-100000.device': 200, '20261029-090000': 100, 'console': 300})
        rows = saves.listing(adb, 'org.nx2a.a')
        self.assertEqual([r['name'] for r in rows], ['console', '20261030-100000', '20261029-090000'])
        self.assertEqual((rows[1]['files'], rows[1]['bytes'], rows[1]['device']), (2, 15, True))
        self.assertEqual((rows[0]['files'], rows[0]['bytes'], rows[0]['device']), (1, 3, False))

    def test_pick_app(self):
        from porter import saves
        from porter.upstream import BuildError
        self.assertEqual(saves.pick_app(self.FakeAdb({}, {}), ''), 'org.nx2a.a')
        two = self.FakeAdb({}, {}, ('org.nx2a.a', 'org.nx2a.b'))
        self.assertEqual(saves.pick_app(two, 'org.nx2a.b'), 'org.nx2a.b')
        with self.assertRaises(BuildError): saves.pick_app(two, '')
        with self.assertRaises(BuildError): saves.pick_app(two, 'org.nx2a.c')

    def test_zip(self):   # a .zip of one folder imports that folder; one of loose files imports them; a path out of the archive is refused
        import zipfile
        from porter import saves
        from porter.upstream import BuildError
        t = Path(tempfile.mkdtemp(prefix='nx2a_test_saves_'))
        with zipfile.ZipFile(t / 'one.zip', 'w') as z: z.writestr('MySave/main.dat', b'x'); z.writestr('MySave/sub/b.dat', b'y')
        with zipfile.ZipFile(t / 'loose.zip', 'w') as z: z.writestr('main.dat', b'x'); z.writestr('other.dat', b'y')
        with zipfile.ZipFile(t / 'bad.zip', 'w') as z: z.writestr('../evil.dat', b'x')
        u = Path(tempfile.mkdtemp(dir=t)); self.assertEqual(saves._unpacked(t / 'one.zip', u), u / 'MySave')
        u = Path(tempfile.mkdtemp(dir=t)); self.assertEqual(sorted(p.name for p in saves._unpacked(t / 'loose.zip', u).iterdir()), ['main.dat', 'other.dat'])
        with self.assertRaises(BuildError): saves._unpacked(t / 'bad.zip', Path(tempfile.mkdtemp(dir=t)))
        self.assertEqual(saves._unpacked(t, None), t)   # a folder: as it is


class ShaderBundleKey(unittest.TestCase):   # Port.shaders: the bundle's reuse key covers shader_vk.py's switches, and a profile's env DM_VK_VT builds it with --vt (split and shader_vk.py faked)
    def setUp(self): self.environ = dict(os.environ)
    def tearDown(self): os.environ.clear(); os.environ.update(self.environ)

    def port(self, rows, env=None):   # a Port at its shader stage, for a throwaway profile with these android.tsv rows, the tools' environment env
        d = profile_dir([('status', 'playable')] + rows); t = Path(tempfile.mkdtemp(prefix='nx2a_test_build_'))
        self.addCleanup(shutil.rmtree, d.parent, True); self.addCleanup(shutil.rmtree, t, True)
        p = pipeline.Port(pipeline.Options(folder=t, jobs=1)); g = game.load(d.name)
        p.g, p.ap, p.env, p.work, p.aw, p.logs, p.tools = g, android_profile.AndroidProfile(g), dict(env or {}), t, t / 'android', t / 'logs', {'dotnet': Path('dotnet')}
        p.versions, p.dirs, p.derived, p.provisional, p.oodle_why, p.aoc = [], {}, '', False, '', None
        p.translator = lambda: (t / 'dmsh_xlate.dll', 'translator key'); p.aw.mkdir()
        return p

    def shaders(self, p):   # p.shaders() → (shader_vk.py's command line, None when the bundle was reused; the key it is kept under)
        cmds = []
        def run(cmd, log, env=None, **kw):   # shader_vk.py: one shader into the bundle (cmd[3], its .partial)
            cmds.append([str(c) for c in cmd]); Path(cmd[3]).write_bytes(b'DMSV'); log.parent.mkdir(parents=True, exist_ok=True); log.write_text('1 of 1 shaders\n')
        with mock.patch.object(pipeline.nsp, 'split_shaders', lambda *a, **kw: None), mock.patch.object(pipeline.nsp, 'run', run), contextlib.redirect_stdout(io.StringIO()):
            p.shaders()   # (its stage lines kept off the test output)
        return (cmds[0] if cmds else None), (p.aw / 'shaders.dmsv.key').read_text().split('\n')[0]

    def test_switches(self):   # DM_SHADER_WIDE_WORDS=1 builds the bundle again under another key; unset and 0 are the same bundle (shader_vk.py: (value or '0') != '0'), reused
        p = self.port([])
        cmd, k0 = self.shaders(p); self.assertIsNotNone(cmd)
        p.env['DM_SHADER_WIDE_WORDS'] = '0'
        cmd, k = self.shaders(p); self.assertIsNone(cmd); self.assertEqual(k, k0)
        p.env['DM_SHADER_WIDE_WORDS'] = '1'
        cmd, k1 = self.shaders(p); self.assertIsNotNone(cmd); self.assertNotEqual(k1, k0)
        cmd, k = self.shaders(p); self.assertIsNone(cmd); self.assertEqual(k, k1)   # and that bundle is kept while the switch stays
        # one sorted text of the values shader_vk.py makes of them: SPILL_KEEP int(value or 0), a flag on for anything but '' and '0'; DM_VK_VT and other variables left out
        self.assertEqual(pipeline.shader_switches({'DM_SHADER_WIDE_WORDS': 'yes', 'DM_SHADER_HMMA_F16': '', 'DM_SHADER_SPILL_KEEP': '32768', 'DM_SHADER_NAN_MINMAX': '0',
                                                   'DM_VK_VT': '1', 'PATH': '/bin'}), 'DM_SHADER_SPILL_KEEP=32768 DM_SHADER_WIDE_WORDS=1')
        self.assertEqual(pipeline.shader_switches({'DM_SHADER_SPILL_KEEP': '0', 'DM_SHADER_SPIRV_OPT': '0'}), '')
        # DRAW_PARAMS is on by default: unset, '' and '1' are the default bundle, only '0' another
        self.assertEqual(pipeline.shader_switches({'DM_SHADER_DRAW_PARAMS': '1'}), '')
        self.assertEqual(pipeline.shader_switches({'DM_SHADER_DRAW_PARAMS': ''}), '')
        self.assertEqual(pipeline.shader_switches({'DM_SHADER_DRAW_PARAMS': '0'}), 'DM_SHADER_DRAW_PARAMS=0')

    def test_switches_are_shader_vks(self):   # the key's switches are every DM_SHADER_ variable shader_vk.py and the translator's scripts read (a new one fails here until it is in the key)
        src = [REPO / 'nxnative/dm/android/tools/shader_vk.py', *sorted((REPO / 'nxnative/dm/gfx/tools').glob('*.py'))]
        read = {n for f in src for n in re.findall(r"""(?:environ(?:\.get\(|\[)|getenv\()\s*['"](DM_SHADER_\w+)""", f.read_text(encoding='utf-8'))}
        self.assertEqual(read, set(pipeline.SHADER_SWITCHES))

    def test_profile_vt(self):   # a profile's env DM_VK_VT=1: the bundle built with --vt, under another key than the same profile's without it (vt=1 / vt=0)
        cmd, k_off = self.shaders(self.port([])); self.assertNotIn('--vt', cmd)
        cmd, k_on = self.shaders(self.port([('env', 'DM_VK_VT=1')])); self.assertIn('--vt', cmd); self.assertNotEqual(k_on, k_off)
        self.assertEqual(self.shaders(self.port([('env', 'DM_VK_VT=2')]))[1], k_on)   # any value but '' and 0: shader_vk.py has only --vt
        cmd, k = self.shaders(self.port([], env={'DM_VK_VT': '1'})); self.assertIn('--vt', cmd); self.assertEqual(k, k_on)   # the tools' own DM_VK_VT, as shader_vk.py's default

    def test_profile_without_vt(self):   # no env DM_VK_VT row, another env row, DM_VK_VT=0 or empty, or a last row of 0 (the app's env.txt keeps the last): no --vt, the key of none
        cmd, k_off = self.shaders(self.port([])); self.assertNotIn('--vt', cmd)
        for rows in ([('env', 'DM_VK_MT_REPLAY=1')], [('env', 'DM_VK_VT=0')], [('env', 'DM_VK_VT=')], [('env', 'DM_VK_VT=1'), ('env', 'DM_VK_VT=0')]):
            cmd, k = self.shaders(self.port(rows, env={'DM_VK_VT': '0'})); self.assertNotIn('--vt', cmd, rows); self.assertEqual(k, k_off, rows)


if __name__ == '__main__':
    unittest.main()
