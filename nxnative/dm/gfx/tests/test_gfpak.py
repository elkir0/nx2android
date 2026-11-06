"""GFPAK archives (Game Freak), without game data: each entry's compression, Oodle through the game's own decoder (the Mac host's, or entries the Android device decoded)."""
import contextlib
import io
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zlib

import lz4.block

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import shader_split


def gfpak(entries):
    """A GFPAK of (compression, raw length, stored bytes) entries: the 0x30-byte header, the file table, then the data"""
    table = 0x30
    data_at = table + 0x18 * len(entries)
    rows, data = b'', b''
    for ct, usz, stored in entries:
        rows += struct.pack('<HHII4xQ', 0, ct, usz, len(stored), data_at + len(data))
        data += stored
    return b'GFLXPACK' + struct.pack('<IIIIQQQ', 0x1000, 0, len(entries), 0, table, 0, 0) + rows + data


class GfpakTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.real_oodle = shader_split.oodle
        def fake_oodle(comp, n):   # stands in for the game's decoder in the Mac host (DM_OODLE_SERVE)
            self.calls.append((bytes(comp), n)); return b'K' * n
        shader_split.oodle = fake_oodle

    def tearDown(self):
        shader_split.oodle = self.real_oodle

    def test_entries_by_compression(self):
        # Sword's entries are LZ4; Pokémon Legends: Arceus's are Oodle Kraken (compression 3)
        raw = b'BNSH' + bytes(60)
        d = gfpak([(0, len(raw), raw), (1, len(raw), zlib.compress(raw)), (2, len(raw), lz4.block.compress(raw, store_size=False)),
                   (3, 0x40, b'kraken stream')])
        with mock.patch.dict(os.environ, {'DM_OODLE_HOST': 'host', 'DM_GFPAK_DECODED': ''}):   # the Mac host's decoder
            got = list(shader_split.gfpak_members(d, 'sample.gfpak'))
        self.assertEqual(got, [('#0', raw), ('#1', raw), ('#2', raw), ('#3', b'K' * 0x40)])
        self.assertEqual(self.calls, [(b'kraken stream', 0x40)])   # only the Oodle entry goes to the decoder

    def test_oodle_decoded_on_the_device(self):
        # nx2android: the porter has the game's decoder on the Android device decode the Oodle entries that hold a BNSH file (porter/pipeline.py device_oodle), into
        # DM_GFPAK_DECODED/<label with / as __>.dec ('DGFD', u64 archive size, u32 count, then u32 index, u32 length and the bytes of each); entries it left out hold no shader
        raw = b'BNSH' + bytes(60)
        d = gfpak([(3, len(raw), b'kraken 0'), (3, 0x40, b'kraken 1'), (0, len(raw), raw)])
        with tempfile.TemporaryDirectory() as t:
            Path(t, 'bin__sample.gfpak.dec').write_bytes(b'DGFD' + struct.pack('<QI', len(d), 1) + struct.pack('<II', 0, len(raw)) + raw)
            with mock.patch.dict(os.environ, {'DM_GFPAK_DECODED': t, 'DM_OODLE_HOST': ''}):
                got = list(shader_split.gfpak_members(d, '/bin/sample.gfpak'))
        self.assertEqual(got, [('#0', raw), ('#2', raw)])
        self.assertEqual(self.calls, [])   # the Mac host's decoder is not asked

    def test_oodle_without_a_decoder_is_left_out(self):
        raw = b'BNSH' + bytes(60)
        d = gfpak([(3, 0x40, b'kraken stream'), (0, len(raw), raw)])
        with mock.patch.dict(os.environ, {'DM_GFPAK_DECODED': '', 'DM_OODLE_HOST': ''}), contextlib.redirect_stdout(io.StringIO()) as log:
            got = list(shader_split.gfpak_members(d, 'sample.gfpak'))
        self.assertEqual(got, [('#1', raw)])
        self.assertIn('[RED] sample.gfpak: 1 of 2 entries are Oodle Kraken and nothing decodes them', log.getvalue())

    def test_unknown_compression_stops(self):
        d = gfpak([(4, 8, b'????????')])
        with self.assertRaisesRegex(SystemExit, 'entry 0 has compression 4'):
            list(shader_split.gfpak_members(d, 'sample.gfpak'))


if __name__ == '__main__':
    unittest.main()
