"""Bounded archive reads and the shader split's image inputs.
nx2android: without upstream's tests of the iOS IPA packer (dm/ios), of nx_extract writing images (its own tests) and of the Metal-side gl_shaders.py / vk_shaders.py, which read
folders here (OpenGL and Vulkan games are not ported to Android)."""
import hashlib
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / 'dm/tools'), str(REPO / 'dm/gfx/tools')]
import romfs
import romfs_image


class ReadImage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src = self.root / 'files'; self.src.mkdir()
        (self.src / 'data/empty').mkdir(parents=True)
        (self.src / 'a').write_bytes(b'prefix!')
        self.data = b'\x03\x02\x23\x07' + b'ABC\0' * 20000
        (self.src / 'data/archive').write_bytes(self.data)
        (self.src / 'data/zero').touch()
        (self.src / 'data/test.vert').write_text('#version 320 es\nvoid main() {}')
        with zipfile.ZipFile(self.src / 'data/game.zip', 'w') as z: z.writestr('member', b'archive member')
        self.image = self.root / 'RomFS.img'; romfs_image.pack(self.src, self.image)
        self.virtual = romfs.Path(self.image)

    def test_metadata_paths_and_normal_output_paths(self):
        self.assertTrue(self.virtual.is_dir()); self.assertFalse(self.virtual.is_file())
        self.assertEqual((self.virtual / 'data/archive').read_bytes(), self.data)
        self.assertEqual((self.virtual / 'data/test.vert').read_text(), '#version 320 es\nvoid main() {}')
        actual = sorted((p.relative_to(self.src).as_posix(), p.is_dir(), p.stat().st_size if p.is_file() else 0) for p in self.src.rglob('*'))
        virtual = sorted((p.relative_to(self.virtual).as_posix(), p.is_dir(), p.stat().st_size if p.is_file() else 0) for p in self.virtual.rglob('*'))
        self.assertEqual(virtual, actual)
        self.assertFalse((self.virtual / 'missing').exists())
        with self.assertRaises(FileNotFoundError): (self.virtual / 'missing').read_bytes()
        with self.assertRaises(OSError): (self.virtual / 'data/archive').open('wb')
        with romfs.open(self.root / 'output', 'wb') as f: f.write(b'normal output')
        self.assertEqual((self.root / 'output').read_bytes(), b'normal output')

    def test_seek_read_and_mmap_cannot_cross_an_entry_boundary(self):
        with romfs.open(self.virtual / 'data/archive', 'rb') as f:
            f.seek(-5, 2); self.assertEqual(f.read(100), self.data[-5:])
            f.seek(len(self.data) + 10); self.assertEqual(f.read(), b'')
            with romfs.mmap.mmap(f.fileno(), 0, access=romfs.mmap.ACCESS_READ) as m:
                self.assertEqual(len(m), len(self.data)); self.assertEqual(m[:], self.data)
                self.assertEqual(struct.unpack_from('<I', m)[0], 0x07230203)
                found = re.search(b'ABC', m); assert found is not None; self.assertEqual(found.start(), 4)
                self.assertEqual(hashlib.sha256(m).digest(), hashlib.sha256(self.data).digest())
                self.assertEqual(m.find(b'ABC'), 4); self.assertEqual(m.rfind(b'ABC'), len(self.data) - 4)
                self.assertEqual(m.find(b'archive member'), -1)
        with romfs.open(self.virtual / 'data/zero', 'rb') as f: self.assertEqual(f.read(), b'')
        with romfs.open(self.virtual / 'data/game.zip', 'rb') as f, zipfile.ZipFile(f) as z:
            self.assertEqual(z.read('member'), b'archive member')

    def test_build_scanners_discover_the_same_inputs_from_image_and_folder(self):
        with patch.dict(os.environ, DM_GAME='010020D01AD24000'):
            import shader_split
        jobs, paths = shader_split.zip_jobs([(str(self.virtual / 'data/game.zip'), 'data/game.zip')])
        self.assertEqual(paths, {str(self.virtual / 'data/game.zip')})
        self.assertEqual(jobs[0][2][1], ('member',))
        self.assertEqual(shader_split.scan_file(jobs[0])[1], {})
        # Image replacement must invalidate the process's cached index.
        (self.src / 'new').write_text('new entry'); romfs_image.pack(self.src, self.image)
        self.assertEqual((self.virtual / 'new').read_text(), 'new entry')

    @unittest.skipUnless(shutil.which('zstd'), 'zstd command required')
    def test_zstd_file_inside_an_image(self):
        # Mario Wonder's .zs files: the zstd command cannot open a path inside the image, so unzs feeds it the bytes.
        with patch.dict(os.environ, DM_GAME='010020D01AD24000'):
            import shader_split
        (self.src / 'data/table.byml.zs').write_bytes(subprocess.run(['zstd', '-c', '-q'], input=self.data, capture_output=True, check=True).stdout)
        romfs_image.pack(self.src, self.image)
        self.assertEqual(shader_split.unzs(str(self.virtual / 'data/table.byml.zs')), self.data)

    def test_damaged_flag_is_read_on_files_only(self):
        with open(self.image, 'rb') as f: _, ioff, _, _ = romfs_image.HEAD.unpack(f.read(romfs_image.HEAD.size))
        loaded = romfs.Image(self.image); file_k, dir_k = loaded.find('a'), loaded.find('data')
        def flagged(k, flags):
            raw = bytearray(self.image.read_bytes()); raw[ioff + k * romfs.ENT.size + romfs.ENT.size - 1] = flags
            out = self.root / 'flagged.img'; out.write_bytes(raw); return out
        self.assertEqual(romfs.Image(flagged(file_k, 1)).entry(file_k)[5], 1)
        with self.assertRaises(ValueError): romfs.Image(flagged(file_k, 2))
        with self.assertRaises(ValueError): romfs.Image(flagged(dir_k, 1))
        self.assertEqual(romfs_image.read_index(flagged(file_k, 1)), romfs_image.read_index(self.image))


if __name__ == '__main__': unittest.main()
