# The Android shader route (python -m unittest discover tests): shader_vk.py's NVN2 (Switch 2) jobs and reflection, the translator's Vulkan forms of what upstream does in MSL
# (compute shared memory past 32 KB in a buffer, task and mesh shaders) and gfx_vk_shader.c's per-primitive fragment variant. No game files: control sections and code are made here.
# The translator tests need DMSH_TEST_XLATE (the built dmsh_xlate.dll, or an apphost) and spirv-val, as upstream's translator tests do; the variant test a host C compiler.
import ast, hashlib, json, os, re, shutil, struct, subprocess, sys, tempfile, unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SHADER_VK = REPO / 'nxnative/dm/android/tools/shader_vk.py'
SHADER_C = REPO / 'nxnative/dm/android/gfx_vk_shader.c'

# shader_vk.py runs as a script (arguments, the profile, the translation): its functions and tables, without the run
_tree = ast.parse(SHADER_VK.read_text(encoding='utf-8'))
_keep = {'base_job', 'nvn2', 'sph_outputs', 'mesh_info', 'stage_memory', 'own_shared', 'nvn2_job', 'cs_facts', 'classify', 'refl_bin', 'captured', 'records', 'first_store'}
def _names(t): return [t.id] if isinstance(t, ast.Name) else [n for e in t.elts for n in _names(e)] if isinstance(t, ast.Tuple) else ['?']
_run = {'HERE', 'TOOLS', 'GAME', 'FACTS', 'MAP', 'XLATE_DLL'}   # the script's paths and its profile
_body = [n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name in _keep or isinstance(n, ast.Assign)
         and all(x.isupper() and x not in _run for t in n.targets for x in _names(t))]
SV = {'struct': struct, 're': re, 'os': os, 'hashlib': hashlib, 'collections': __import__('collections'), 'FACTS': {}}
exec(compile(ast.Module(body=_body, type_ignores=[]), str(SHADER_VK), 'exec'), SV)


def nvn2_control(stage=0, threads=32, payload=256, task=0, max_v=0, max_p=0, prim_at=0, inst=0x100, c1=(0x100, 0x100), area=0x100, size=0xd0):
    """An NVN2 control section: magic, version 0xc9, +0x38 the instructions' length, +0x40 / +0x44 the c1 segment's bounds (the end = the code area's length), +0x56 the stage,
    +0x58 the threads (a compute shader's thread group from +0x58 as three words), +0x5a the payload, +0x5e a mesh shader's task, +0x60 / +0x62 its vertices and primitives"""
    c = bytearray(size); struct.pack_into('<II', c, 0, 0x98761234, 0xc9); struct.pack_into('<I', c, 0x38, inst); struct.pack_into('<II', c, 0x40, c1[0], area)
    c[0x56] = stage; struct.pack_into('<2H', c, 0x58, threads, payload); c[0x5e] = task; struct.pack_into('<2H', c, 0x60, max_v, max_p)
    if prim_at: struct.pack_into('<I', c, 0x64, prim_at); c[0x68] = 0x80
    return bytes(c)


def sph(pos=0xf0, generics=b''):   # an SPH whose OMAP enables the position's components (byte 0x35, bits 4..7) and generic attributes (a nibble each from 0x36)
    s = bytearray(0x80); s[0x35] = pos; s[0x36:0x36 + len(generics)] = generics
    return bytes(s)


class Nvn2Jobs(unittest.TestCase):
    def job(self, stage, ctl, area):
        with tempfile.TemporaryDirectory() as d:
            j, c1 = SV['nvn2_job']('k', stage, ctl, area, d)
            return j, c1, Path(j['code']).read_bytes()

    def test_graphics_keeps_its_sph_and_takes_c1_from_the_code_area(self):
        area = sph() + bytes(range(0x80)) * 2 + b'\x5a' * 0x100   # 0x80 SPH + 0x100 instructions, then the c1 segment
        ctl = nvn2_control(inst=0x180, c1=(0x180, len(area)), area=len(area))
        j, c1, code = self.job('vs', ctl, area)
        self.assertEqual((j['arch'], j['stage'], len(code), c1), ('sm86', 'vs', 0x180, b'\x5a' * 0x100))
        self.assertNotIn('nvn2_cs', j)
        for k in ('metal_image_atomics', 'raw_gmem'): self.assertIs(j[k], False)   # the Metal chain's forms stay off for Vulkan (Program.cs)
        self.assertEqual((j['handle_arrays'], j['shared_max'], j['spill_slices']), (32, 0x8000, 512))

    def test_compute_layout_and_facts_from_the_control_section(self):
        ctl = bytearray(nvn2_control(stage=5, inst=0x40, area=0x40)); ctl[0xa9] = 4; ctl[0xb1:0xb6] = bytes([1, 2, 3, 4, 5]); struct.pack_into('<H', ctl, 0xba, 0x90)
        struct.pack_into('<3I', ctl, 0x58, 64, 2, 1); struct.pack_into('<I', ctl, 0x64, 0xc001)
        j, _, code = self.job('cs', bytes(ctl), bytes(0x40))
        self.assertEqual(j['nvn2_cs'], [1, 2, 3, 0x90, 16, 4, 5])   # uniform, storage, texture entries, size, push bytes, separate textures, separate samplers (InstEmitAmpere.cs AMapC0)
        self.assertEqual((j['local'], j['shared']), ([64, 2, 1], 0xc010))   # not in the answer table: as shader_split.py reads them (+0x58, +0x64 rounded to 16)
        self.assertEqual(len(code), 0x40)

    def test_task_memory(self):   # dmsh_meshstage.py memory: the payload after the input at 0x10
        ctl = nvn2_control(stage=6, payload=256, inst=0x100, area=0x100)
        j, _, code = self.job('ts', ctl, sph() + bytes(0x80))
        self.assertEqual((j['stage'], j['stage_memory'], j['shared'], j['local']), ('cs', [0, 0x10, 0x110, 0x210], 0x210, [32, 1, 1]))
        self.assertEqual(j['mesh'], {'kind': 'ts', 'threads': 32, 'payload': 256})
        self.assertEqual(len(code), 0x80)   # translated as compute: its SPH taken off

    def test_mesh_memory_and_outputs(self):   # the shape of Mario Kart World's largest: 256 vertices, 192 primitives, a per-primitive SPH after the instructions
        area = sph(0xf0, b'\x00\x00\xff') + bytes(0x80) + sph(0x00, b'\xff\xff')
        ctl = nvn2_control(stage=7, threads=32, payload=256, task=1, max_v=256, max_p=192, prim_at=0x100, inst=0x100, area=len(area))
        j, _, _ = self.job('ms', ctl, area)
        m = j['mesh']
        self.assertEqual(m['vertex'], [[-1, 0], [-1, 1], [-1, 2], [-1, 3], [4, 0], [4, 1], [4, 2], [4, 3], [5, 0], [5, 1], [5, 2], [5, 3]])
        self.assertEqual(m['primitive'], [[a, c] for a in range(4) for c in range(4)])
        out = 0x10 + 256; prim = out + 8 * 28 * 32 * 4   # 8 blocks of 32, 12 + 16 components
        self.assertEqual(j['stage_memory'], [0, 0x10, out, prim]); self.assertEqual(j['shared'], prim + 4 + 3 * 192)
        self.assertTrue(m['task']); self.assertNotIn('refused', j)

    def test_own_shared_memory_is_refused(self):
        sts = struct.pack('<QQ', 0x7988, 0)
        ctl = nvn2_control(stage=7, max_v=32, max_p=16, inst=0x90, area=0x90)
        j, _, _ = self.job('ms', ctl, sph() + sts)
        self.assertIn('shared memory of its own', j['refused'])

    def test_maxwell_jobs_keep_the_android_fields(self):
        self.assertFalse(SV['nvn2'](b'\x34\x12\x76\x98\x01\x00\x00\x00' + bytes(0x878)))
        j = SV['base_job']('k', 'ps', 'x.bin', b'')
        self.assertEqual((j['sb_align'], j['subgroup'], j['gather_bias'], j['gs_passthrough']), (256, 128, 8, False))


class Reflection(unittest.TestCase):
    def classify(self, stage, name, res=None, **b):
        return SV['classify'](stage, dict(dict(name=name, set=0, binding=1, type='ubo', count=1, dim=None, arrayed=0, ms=0, fmt=0), **b), res or {'sbufs': []})

    def test_push_constants_and_the_first_workgroup_are_driver_banks(self):   # gfx_vk.h RK_DRV: reg = the bank (DMSH_PUSH_BANK, DMSH_MESH_FIRST_SLOT)
        self.assertEqual(self.classify('cs', 'cp_c24')['bank'], 24); self.assertEqual(self.classify('cs', 'cp_c24')['kind'], 'drvcb')
        e = self.classify('ms', 'mesh_first'); self.assertEqual((e['kind'], e['bank']), ('drvcb', 29))

    def test_spill_binding_carries_words_and_slices(self):
        e = self.classify('cs', 'shared_spill', {'sbufs': [], 'spill_words': 12288, 'spill_slices': 512}, type='ssbo')
        self.assertEqual(e['kind'], 'spill')
        r = SV['refl_bin']([e], b'', [64, 1, 1])
        kind, desc, _, _, binding, count, reg, samp, fmt, off, sarg = struct.unpack_from('<BBBBHHhBBii', r, 20)
        self.assertEqual((kind, desc, off, sarg, samp), (SV['KIND']['spill'], SV['DESC']['ssbo'], 12288, 512, 0))

    def test_task_and_mesh_storage_tables(self):   # dmsh_abi.h DMSH_C0_SSBO_TS / _MS
        for st, base in (('ts', 0x670), ('ms', 0x770)):
            res = {'sbufs': [{'binding': 3, 'cb_slot': 0, 'cb_offset': (base + 0x20) // 4, 'flags': 1}]}
            e = self.classify(st, 'cp_s0', res, type='ssbo')
            self.assertEqual((e['kind'], e['reg'], e['rw']), ('sbuf', 2, True))   # BufferUsageFlags.Write is bit 0

    def test_bundle_stages(self):
        self.assertEqual((SV['STAGES']['ts'], SV['STAGES']['ms']), (6, 7))


class Captures(unittest.TestCase):
    def test_nvn2_capture_keys_as_the_host_does(self):   # hle_nvn.c keys_hash: the first 0x48 control bytes with +0x30 as zero, then the code area
        code = sph(0xf0) + bytes(0x80)
        c = bytearray(nvn2_control(inst=len(code), area=len(code))); struct.pack_into('<I', c, 0x30, 0xd8)   # a BNSH section's length at +0x30
        c[0x56] = 0; s = bytearray(code); struct.pack_into('<I', s, 0, 5 << 10)   # SPH stage 5: pixel
        key = hashlib.sha256(bytes(c[:0x30]) + bytes(4) + bytes(c[0x34:0x48]) + bytes(s)).hexdigest()[:16]
        with tempfile.TemporaryDirectory() as d:
            Path(d, key + '.bin').write_bytes(b'DMSC' + struct.pack('<2I', len(c), len(s)) + bytes(c) + bytes(s))
            Path(d, '0123456789abcdef.bin').write_bytes(b'DMSC' + struct.pack('<2I', len(c), len(s)) + bytes(c) + bytes(s))   # not its hash: left out
            got = SV['captured'](d)
        self.assertEqual(list(got), [key]); self.assertEqual(got[key][0], 'ps')
        rec = got[key][1]; self.assertEqual(rec[0x38:0x38 + len(s)], bytes(s)); self.assertEqual(rec[0x38 + len(s):], bytes(c))


# A compute shader's preamble in the translator's SPIR-V form (Mario Kart World's scenery lists, upstream e07cba1c): not-first = -local.x != workgroup.x << 6 (local size 64), a local
# that is 1 in the first invocation (b), and one guarded store of it to word 2 of storage slot 2 (c0 word 204 = 0x330: its address; bound from the 256-byte boundary below it)
FIRST = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %wg %lid
OpExecutionMode %main LocalSize LOCAL 1 1
OpName %c0 "cp_c0"
OpName %s0 "cp_s0"
OpDecorate %wg BuiltIn WorkgroupId
OpDecorate %lid BuiltIn LocalInvocationId
OpDecorate %ra ArrayStride 4
OpMemberDecorate %sb 0 Offset 0
OpDecorate %sb BufferBlock
OpDecorate %s0 DescriptorSet 0
OpDecorate %s0 Binding 2
OpDecorate %v4a ArrayStride 16
OpMemberDecorate %cb 0 Offset 0
OpDecorate %cb Block
OpDecorate %c0 DescriptorSet 0
OpDecorate %c0 Binding 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%bool = OpTypeBool
%v3u = OpTypeVector %uint 3
%v4f = OpTypeVector %float 4
%u4096 = OpConstant %uint 4096
%v4a = OpTypeArray %v4f %u4096
%cb = OpTypeStruct %v4a
%pcb = OpTypePointer Uniform %cb
%c0 = OpVariable %pcb Uniform
%ra = OpTypeRuntimeArray %uint
%sb = OpTypeStruct %ra
%psb = OpTypePointer Uniform %sb
%s0 = OpVariable %psb Uniform
%pin3 = OpTypePointer Input %v3u
%wg = OpVariable %pin3 Input
%lid = OpVariable %pin3 Input
%pinu = OpTypePointer Input %uint
%puf = OpTypePointer Uniform %float
%puu = OpTypePointer Uniform %uint
%pfi = OpTypePointer Function %int
%pff = OpTypePointer Function %float
%pfb = OpTypePointer Function %bool
%i0 = OpConstant %int 0
%i1 = OpConstant %int 1
%i2 = OpConstant %int 2
%i6 = OpConstant %int 6
%i8 = OpConstant %int 8
%i51 = OpConstant %int 51
%in256 = OpConstant %int -256
%main = OpFunction %void None %fn
%l0 = OpLabel
%vlid = OpVariable %pff Function
%vsh = OpVariable %pfi Function
%vneg = OpVariable %pfi Function
%vnf = OpVariable %pfb Function
%va = OpVariable %pfi Function
%vb = OpVariable %pfi Function
%1 = OpAccessChain %pinu %lid %i0
%2 = OpLoad %uint %1
%3 = OpBitcast %float %2
OpStore %vlid %3
%4 = OpAccessChain %pinu %wg %i0
%5 = OpLoad %uint %4
%6 = OpBitcast %int %5
%7 = OpShiftLeftLogical %int %6 %i6
OpStore %vsh %7
%8 = OpLoad %float %vlid
%9 = OpBitcast %int %8
%10 = OpNot %int %9
%11 = OpIAdd %int %10 %i1
OpStore %vneg %11
%12 = OpLoad %int %vneg
%13 = OpLoad %int %vsh
%14 = OpINotEqual %bool %12 %13
OpStore %vnf %14
%15 = OpLoad %int %vneg
OpStore %va %15
%16 = OpLoad %bool %vnf
%17 = OpLogicalNot %bool %16
OpSelectionMerge %m1 None
OpBranchConditional %17 %t1 %m1
%t1 = OpLabel
OpStore %va %i1
OpBranch %m1
%m1 = OpLabel
%18 = OpLoad %int %va
OpStore %vb %18
%19 = OpLoad %bool %vnf
%20 = OpLogicalNot %bool %19
OpSelectionMerge %m2 None
OpBranchConditional %20 %t2 %m2
%t2 = OpLabel
%21 = OpAccessChain %puf %c0 %i0 %i51 %i0
%22 = OpLoad %float %21
%23 = OpBitcast %int %22
%24 = OpIAdd %int %23 %i8
%25 = OpAccessChain %puf %c0 %i0 %i51 %i0
%26 = OpLoad %float %25
%27 = OpBitcast %int %26
%28 = OpBitwiseAnd %int %27 %in256
%29 = OpISub %int %24 %28
%30 = OpShiftRightLogical %int %29 %i2
%31 = OpAccessChain %puu %s0 %i0 %30
%32 = OpLoad %int %vb
%33 = OpBitcast %uint %32
OpStore %31 %33
OpBranch %m2
%m2 = OpLabel
LATER
OpReturn
OpFunctionEnd
"""
GUARD_AGAIN = """%40 = OpLoad %bool %vnf
OpSelectionMerge %m3 None
OpBranchConditional %40 %t3 %m3
%t3 = OpLabel
OpBranch %m3
%m3 = OpLabel"""


@unittest.skipUnless(shutil.which('spirv-as') and shutil.which('spirv-val'), 'needs spirv-as and spirv-val')
class FirstStore(unittest.TestCase):
    BINDS = [dict(kind='sbuf', binding=2, name='cp_s0', reg=2)]
    RES = {'sbufs': [{'binding': 2, 'cb_slot': 0, 'cb_offset': 204, 'flags': 1}]}

    def run_one(self, local=64, later='', src=None):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'f.spvasm').write_text((src or FIRST).replace('LOCAL', str(local)).replace('LATER', later))
            subprocess.run(['spirv-as', '--target-env', 'spv1.3', str(Path(d, 'f.spvasm')), '-o', str(Path(d, 'f.spv'))], check=True, capture_output=True)
            spv = Path(d, 'f.spv').read_bytes()
            out, first = SV['first_store'](spv, self.BINDS, [local, 1, 1], self.RES)
            Path(d, 'o.spv').write_bytes(out)
            v = subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', str(Path(d, 'o.spv'))], capture_output=True, text=True)
            self.assertEqual(v.returncode, 0, v.stderr)
            return spv, out, first

    def test_the_store_leaves_the_shader_for_the_replay(self):
        spv, out, first = self.run_one()
        self.assertEqual(first, [{'reg': 2, 'word': 2, 'value': 1}])   # word 2 of slot 2, the local's first-invocation value
        self.assertEqual(len(out), len(spv) - 12)   # the one OpStore (3 words) gone
        r = SV['refl_bin']([], b'', [64, 1, 1], first)
        self.assertEqual(r[20:], b'FSTR' + struct.pack('<IiII', 1, 2, 2, 1))

    def test_the_guard_read_again_leaves_the_shader_as_it_is(self):   # upstream's rule: a guarded store it does not understand
        spv, out, first = self.run_one(later=GUARD_AGAIN)
        self.assertEqual((first, out), ([], spv))

    def test_another_local_size_is_not_the_first_invocation(self):   # the shift must be log2 of the local size
        spv, out, first = self.run_one(local=128)
        self.assertEqual((first, out), ([], spv))

    def test_the_wide_words_index_form(self):   # shader_vk.py WIDE_WORDS (GlobalToStorage): ((address - base) >> 2) + words instead of (address + bytes - base) >> 2
        wide = """%21 = OpAccessChain %puf %c0 %i0 %i51 %i0
%22 = OpLoad %float %21
%23 = OpBitcast %int %22
%25 = OpAccessChain %puf %c0 %i0 %i51 %i0
%26 = OpLoad %float %25
%27 = OpBitcast %int %26
%28 = OpBitwiseAnd %int %27 %in256
%29 = OpISub %int %23 %28
%30a = OpShiftRightArithmetic %int %29 %i2
%30 = OpIAdd %int %30a %i2
%31 = OpAccessChain %puu %s0 %i0 %30"""
        old = FIRST[FIRST.index('%21 = OpAccessChain'):FIRST.index('%31 = OpAccessChain %puu %s0 %i0 %30') + len('%31 = OpAccessChain %puu %s0 %i0 %30')]
        spv, out, first = self.run_one(src=FIRST.replace(old, wide))
        self.assertEqual(first, [{'reg': 2, 'word': 2, 'value': 1}])


def xlate_cmd():
    x = os.environ.get('DMSH_TEST_XLATE')
    if not x: return None
    return [shutil.which('dotnet') or 'dotnet', x] if x.endswith('.dll') else [x]


AMPERE_EXIT = (0x000fea0003800000, 0x000000000000794d)   # hi, lo (upstream's test_dmsh_ampere_mem.py encodings)
def ainst(hi, lo): return struct.pack('<QQ', lo, hi)
def amov(rd, value): return ainst(0x000fc60000000f00, value << 32 | rd << 16 | 0x7802)                       # MOV Rd, imm32
def asts(offset, rb): return ainst(0x000fe80000000800, offset << 40 | rb << 32 | 0xff << 24 | 0x7388)       # STS.U32 [RZ + offset], Rb
ISBERD_WG = (0x000fe20000007800, 0x00000000ff007923)    # ISBERD R0, mode 0x78: the workgroup's number
STAGE_R0 = (0x0041e8000000b800, 0x00001000ff007927)     # 0x927 [RZ + 0x10], R0, mode 0xb8: a payload word
ATTRIBUTE = (0x0001e2000000f800, 0x0000800307007927)    # 0x927 [R7 + 0x80], R3, mode 0xf8: an attribute word
COUNT = (0x0001e20000008800, 0x00000307ff007927)        # 0x927 [RZ + 3], R7, mode 0x88: the primitive count
INDEX = (0x0001e20000008000, 0x0000040605057927)        # 0x927 [R5 + 4], R6, mode 0x80: an index byte


@unittest.skipUnless(xlate_cmd() and shutil.which('spirv-val') and shutil.which('spirv-dis'), 'set DMSH_TEST_XLATE to the built translator; needs spirv-val and spirv-dis')
class Translator(unittest.TestCase):
    def translate(self, code, **job):
        with tempfile.TemporaryDirectory(prefix='shader_vk_test_') as d:
            Path(d, 'k.bin').write_bytes(code)
            j = SV['base_job']('k', job.pop('stage', 'cs'), str(Path(d, 'k.bin')), b''); j.update(arch='sm86', **job)
            Path(d, 'jobs.json').write_text(json.dumps([j]))
            r = subprocess.run(xlate_cmd() + [str(Path(d, 'jobs.json')), d], capture_output=True, text=True)
            self.assertTrue(Path(d, 'k.spv').exists(), r.stdout + r.stderr)
            v = subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', str(Path(d, 'k.spv'))], capture_output=True, text=True)
            self.assertEqual(v.returncode, 0, v.stderr + v.stdout)
            dis = subprocess.run(['spirv-dis', str(Path(d, 'k.spv'))], capture_output=True, text=True).stdout
            return dis, json.loads(Path(d, 'k.res.json').read_text())

    def test_48_kb_of_shared_memory_go_to_a_buffer(self):
        dis, res = self.translate(amov(3, 7) + asts(0x10, 3) + ainst(*AMPERE_EXIT), nvn2_cs=[0, 0, 0, 0x10, 0], local=[64, 1, 1], shared=0xc000)
        self.assertIn('"shared_spill"', dis); self.assertIn('BuiltIn NumWorkgroups', dis); self.assertNotIn(' Workgroup ', dis)
        self.assertEqual((res['spill_words'], res['spill_slices']), (0x3000, 512))

    def test_48_kb_split_keeps_its_first_32_in_workgroup_memory(self):   # SharedMemorySpill.Split (job spill_keep, shader_vk.py SPILL_KEEP)
        dis, res = self.translate(amov(3, 7) + asts(0x10, 3) + ainst(*AMPERE_EXIT), nvn2_cs=[0, 0, 0, 0x10, 0], local=[64, 1, 1], shared=0xc000, spill_keep=0x8000)
        self.assertIn('"shared_spill"', dis); self.assertIn('OpTypeArray %uint %uint_8192', dis); self.assertIn('OpTypePointer Workgroup', dis); self.assertIn('OpULessThan', dis)
        self.assertEqual((res['spill_words'], res['spill_slices']), (0x1000, 512))

    def test_vote_takes_one_ballot(self):   # InstEmitWarp.EmitBallot: one ballot for a 32-lane warp in a subgroup of up to 128 lanes, its word picked by lane id >> 5
        vote = ainst(0x000fe20000000000 | 7 << 23 | 7 << 17 | 1 << 8, 2 << 16 | 0x7806)   # VOTE.ANY R2, PT, PT
        dis, _ = self.translate(vote + asts(0x10, 2) + ainst(*AMPERE_EXIT), nvn2_cs=[0, 0, 0, 0x10, 0], local=[64, 1, 1], shared=0x100)
        self.assertEqual(dis.count('OpGroupNonUniformBallot'), 1); self.assertIn('OpVectorExtractDynamic', dis)

    def test_nan_safe_min_max_and_saturation(self):   # job nan_minmax (shader_vk.py NAN_MINMAX): Maxwell's rule for a NaN operand
        tid = ainst(0x000e2e0000002100, 3 << 16 | 0x7919)                                  # S2R R3, SR_TID.X
        fmul_sat = ainst(0x000fe20000000000 | 4 << 20 | 1 << 13, 0x40000000 << 32 | 3 << 24 | 2 << 16 | 0x7820)   # FMUL.SAT R2, R3, 2.0 (scale 84..86: 4, none)
        code = tid + fmul_sat + asts(0x10, 2) + ainst(*AMPERE_EXIT)
        dis, _ = self.translate(code, nvn2_cs=[0, 0, 0, 0x10, 0], local=[64, 1, 1], shared=0x100)
        self.assertIn('FClamp', dis); self.assertNotIn('NClamp', dis)
        dis, _ = self.translate(code, nvn2_cs=[0, 0, 0, 0x10, 0], local=[64, 1, 1], shared=0x100, nan_minmax=True)
        self.assertIn('NClamp', dis); self.assertNotIn('FClamp', dis)

    def test_shared_memory_within_the_limit_stays(self):
        dis, res = self.translate(amov(3, 7) + asts(0x10, 3) + ainst(*AMPERE_EXIT), nvn2_cs=[0, 0, 0, 0x10, 0], local=[64, 1, 1], shared=0x1000)
        self.assertNotIn('shared_spill', dis); self.assertIn(' Workgroup ', dis); self.assertNotIn('spill_words', res)

    def test_task_shader(self):
        mesh = {'kind': 'ts', 'threads': 32, 'payload': 256}
        dis, _ = self.translate(ainst(*ISBERD_WG) + ainst(*STAGE_R0) + ainst(*AMPERE_EXIT), stage_memory=[0, 0x10, 0x110, 0x210], shared=0x210, local=[32, 1, 1], mesh=mesh)
        self.assertIn('OpEntryPoint TaskEXT', dis); self.assertIn('OpEmitMeshTasksEXT', dis); self.assertIn('"mesh_first"', dis)
        self.assertIn('Version: 1.4', dis)

    def test_mesh_shader(self):
        mesh = {'kind': 'ms', 'threads': 32, 'payload': 128, 'task': False, 'max_v': 64, 'max_p': 32,
                'vertex': [[-1, c] for c in range(4)] + [[1, 0], [1, 1]], 'primitive': [[0, 0]]}
        out = 0x10 + 128; prim = out + 2 * 7 * 32 * 4
        dis, _ = self.translate(ainst(*ATTRIBUTE) + ainst(*COUNT) + ainst(*INDEX) + ainst(*AMPERE_EXIT), stage_memory=[0, 0x10, out, prim], shared=prim + 4 + 3 * 32,
                                local=[32, 1, 1], mesh=mesh)
        for s in ('OpEntryPoint MeshEXT', 'OutputVertices 64', 'OutputPrimitivesEXT 32', 'OutputTrianglesEXT', 'OpSetMeshOutputsEXT', 'BuiltIn PrimitiveTriangleIndicesEXT',
                  'PerPrimitiveEXT', 'BuiltIn Position', '"mesh_first"'):
            self.assertIn(s, dis)
        # a triangle's indices stored whole (uvec3), never one component chosen at run time: the Adreno 840's compiler crashed on that form (MeshStage.cs)
        self.assertIn('_ptr_Output_v3uint', dis); self.assertNotIn('_ptr_Output_uint ', dis)


@unittest.skipUnless((shutil.which('cc') or shutil.which('clang')) and shutil.which('spirv-as') and shutil.which('spirv-val'), 'needs a host C compiler, spirv-as and spirv-val')
class PerPrimitiveVariant(unittest.TestCase):
    # A fragment shader in the translator's SPIR-V 1.3 form: inputs at locations 0 and 1, a storage buffer (Uniform + BufferBlock) it writes through an access chain, a uniform buffer
    FS = '''OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in0 %in1 %out0
OpExecutionMode %main OriginUpperLeft
OpName %in0 "in0"
OpName %in1 "in1"
OpName %out0 "out0"
OpName %sb "sb"
OpName %ssbo "ssbo"
OpName %ubo "ubo"
OpDecorate %in0 Location 0
OpDecorate %in0 Flat
OpDecorate %in1 Location 1
OpDecorate %in1 NoPerspective
OpDecorate %out0 Location 0
OpDecorate %sb BufferBlock
OpMemberDecorate %sb 0 Offset 0
OpDecorate %ra ArrayStride 4
OpDecorate %ssbo DescriptorSet 0
OpDecorate %ssbo Binding 1
OpDecorate %ub Block
OpMemberDecorate %ub 0 Offset 0
OpDecorate %ubo DescriptorSet 0
OpDecorate %ubo Binding 2
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%v4 = OpTypeVector %float 4
%ra = OpTypeRuntimeArray %uint
%sb = OpTypeStruct %ra
%psb = OpTypePointer Uniform %sb
%pu = OpTypePointer Uniform %uint
%ub = OpTypeStruct %v4
%pub = OpTypePointer Uniform %ub
%pv4u = OpTypePointer Uniform %v4
%pin = OpTypePointer Input %v4
%pout = OpTypePointer Output %v4
%ssbo = OpVariable %psb Uniform
%ubo = OpVariable %pub Uniform
%in0 = OpVariable %pin Input
%in1 = OpVariable %pin Input
%out0 = OpVariable %pout Output
%i0 = OpConstant %int 0
%u0 = OpConstant %uint 0
%main = OpFunction %void None %fn
%l = OpLabel
%a = OpLoad %v4 %in0
%b = OpLoad %v4 %in1
%c = OpAccessChain %pv4u %ubo %i0
%cv = OpLoad %v4 %c
%s = OpFAdd %v4 %a %b
%t = OpFAdd %v4 %s %cv
OpStore %out0 %t
%p = OpAccessChain %pu %ssbo %i0 %u0
OpStore %p %u0
OpReturn
OpFunctionEnd
'''

    def test_variant_is_valid_spirv_14(self):
        src = SHADER_C.read_text()
        def fn(sig):
            a = src.index(sig); return src[a:src.index('\n}\n', a) + 3]
        harness = '#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n' + fn('static uint32_t spv_prim_locs(') + fn('static uint32_t *spv_per_primitive(') + r'''
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); uint32_t *w = malloc(n); if (fread(w, 1, n, f) != (size_t)n) return 3; fclose(f);
    uint32_t no = 0, *o = spv_per_primitive(w, (uint32_t)(n / 4), (uint32_t)strtoul(argv[3], NULL, 16), &no); if (!o) return 2;
    f = fopen(argv[2], "wb"); fwrite(o, 4, no, f); fclose(f); return 0;
}
'''
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); (d / 'h.c').write_text(harness); (d / 'fs.spvasm').write_text(self.FS)
            cc = shutil.which('cc') or shutil.which('clang')
            subprocess.run([cc, '-O1', '-o', str(d / 'h'), str(d / 'h.c')], check=True, capture_output=True)
            subprocess.run(['spirv-as', '--target-env', 'spv1.3', str(d / 'fs.spvasm'), '-o', str(d / 'fs.spv')], check=True, capture_output=True)
            r = subprocess.run([str(d / 'h'), str(d / 'fs.spv'), str(d / 'pp.spv'), '2'], capture_output=True, text=True)   # location 1 per primitive
            self.assertEqual(r.returncode, 0, r.stderr)
            v = subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', str(d / 'pp.spv')], capture_output=True, text=True)
            self.assertEqual(v.returncode, 0, v.stderr + v.stdout)
            dis = subprocess.run(['spirv-dis', str(d / 'pp.spv')], capture_output=True, text=True).stdout
        self.assertIn('Version: 1.4', dis); self.assertIn('OpCapability MeshShadingEXT', dis)
        self.assertRegex(dis, r'OpDecorate %in1 PerPrimitiveEXT'); self.assertNotIn('%in0 PerPrimitiveEXT', dis); self.assertNotIn('%in1 NoPerspective', dis)
        self.assertIn('OpDecorate %sb Block', dis); self.assertIn('StorageBuffer %sb', dis)
        self.assertRegex(dis, r'OpEntryPoint Fragment %\w+ "main" %in0 %in1 %out0 %ssbo %ubo\n')   # every global in the interface (SPIR-V 1.4)


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless((shutil.which('cc') or shutil.which('clang')) and shutil.which('spirv-as') and shutil.which('spirv-val'), 'needs a host C compiler, spirv-as and spirv-val')
class LodShiftVariant(unittest.TestCase):
    # A fragment shader in the translator's form that takes level numbers of a 2D array at binding 3 (query LOD, explicit LOD, a fetch with a level, a size by level) and samples a 2D
    # texture at binding 4 with an explicit LOD too, which is not shifted
    FS = '''OpCapability Shader
OpCapability ImageQuery
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %uv %out0
OpExecutionMode %main OriginUpperLeft
OpName %c "c"
OpName %t "t"
OpName %q "q"
OpName %qx "qx"
OpName %im "im"
OpName %tc "tc"
OpName %f "f"
OpName %sz "sz"
OpDecorate %uv Location 0
OpDecorate %out0 Location 0
OpDecorate %arr DescriptorSet 1
OpDecorate %arr Binding 3
OpDecorate %flat DescriptorSet 1
OpDecorate %flat Binding 4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v2 = OpTypeVector %float 2
%v3 = OpTypeVector %float 3
%v4 = OpTypeVector %float 4
%iv2 = OpTypeVector %int 2
%iv3 = OpTypeVector %int 3
%img = OpTypeImage %float 2D 0 1 0 1 Unknown
%simg = OpTypeSampledImage %img
%img2 = OpTypeImage %float 2D 0 0 0 1 Unknown
%simg2 = OpTypeSampledImage %img2
%psi = OpTypePointer UniformConstant %simg
%psi2 = OpTypePointer UniformConstant %simg2
%pin = OpTypePointer Input %v2
%pout = OpTypePointer Output %v4
%arr = OpVariable %psi UniformConstant
%flat = OpVariable %psi2 UniformConstant
%uv = OpVariable %pin Input
%out0 = OpVariable %pout Output
%f0 = OpConstant %float 0
%i1 = OpConstant %int 1
%i5 = OpConstant %int 5
%main = OpFunction %void None %fn
%l = OpLabel
%c = OpLoad %v2 %uv
%t = OpLoad %simg %arr
%q = OpImageQueryLod %v2 %t %c
%qx = OpCompositeExtract %float %q 0
%c3 = OpCompositeConstruct %v3 %c %f0
%s = OpImageSampleExplicitLod %v4 %t %c3 Lod %qx
%im = OpImage %img %t
%tc = OpCompositeConstruct %iv3 %i5 %i5 %i1
%f = OpImageFetch %v4 %im %tc Lod %i1
%sz = OpImageQuerySizeLod %iv3 %im %i1
%t2 = OpLoad %simg2 %flat
%s2 = OpImageSampleExplicitLod %v4 %t2 %c Lod %qx
%a = OpFAdd %v4 %s %f
%b = OpFAdd %v4 %a %s2
OpStore %out0 %b
OpReturn
OpFunctionEnd
'''

    def run_variant(self, src_c, alpha_first):
        def fn(sig):
            a = src_c.index(sig); return src_c[a:src_c.index('\n}\n', a) + 3]
        a = src_c.index('typedef struct { const uint32_t *w; uint32_t fn, bound, *at, *nt, nnt;'); spvi = src_c[a:src_c.index('static uint32_t *spv_int_images(', a)]
        harness = ('#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\nenum { VK_COL0_NONE, VK_COL0_FLOAT, VK_COL0_INT, VK_COL0_OTHER };\n#define VK_ALPHA_PUSH_OFF 0\n'
                   + fn('static int spv_col0(') + spvi + fn('static uint32_t spvi_const(') + fn('static uint32_t *spv_alpha_test(') + fn('static uint32_t *spv_lod_shift(') + r'''
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); uint32_t *w = malloc(n); if (fread(w, 1, n, f) != (size_t)n) return 3; fclose(f);
    uint32_t nw = (uint32_t)(n / 4); const char *why = NULL;
    if (atoi(argv[3])) { uint32_t na = 0; uint32_t *a = spv_alpha_test(w, nw, &na, &why); if (!a) { fprintf(stderr, "alpha: %s\n", why); return 2; } w = a; nw = na; }
    uint16_t bind[1] = {3}; uint32_t no = 0, *o = spv_lod_shift(w, nw, bind, 1, 12, &no, &why); if (!o) { fprintf(stderr, "%s\n", why); return 2; }
    f = fopen(argv[2], "wb"); fwrite(o, 4, no, f); fclose(f); return 0;
}
''')
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); (d / 'h.c').write_text(harness); (d / 'fs.spvasm').write_text(self.FS)
            cc = shutil.which('cc') or shutil.which('clang')
            subprocess.run([cc, '-O1', '-o', str(d / 'h'), str(d / 'h.c')], check=True, capture_output=True)
            subprocess.run(['spirv-as', '--target-env', 'spv1.3', str(d / 'fs.spvasm'), '-o', str(d / 'fs.spv')], check=True, capture_output=True)
            r = subprocess.run([str(d / 'h'), str(d / 'fs.spv'), str(d / 'ls.spv'), '1' if alpha_first else '0'], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            v = subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', str(d / 'ls.spv')], capture_output=True, text=True)
            self.assertEqual(v.returncode, 0, v.stderr + v.stdout)
            return subprocess.run(['spirv-dis', str(d / 'ls.spv')], capture_output=True, text=True).stdout

    def test_level_numbers_take_the_shift(self):
        dis = self.run_variant(SHADER_C.read_text(), alpha_first=False)
        self.assertRegex(dis, r'%\d+ = OpImageQueryLod %v2float %t %c\n *%\d+ = OpCompositeConstruct %v2float (%\d+) \1\n *%q = OpFAdd %v2float')   # + B to both components
        self.assertEqual(dis.count('OpFSub %float %qx'), 1)                                        # binding 3's explicit LOD only, not binding 4's
        self.assertRegex(dis, r'OpShiftRightArithmetic %v3int %tc %\d+\n *%f = OpImageFetch %v4float %im %\d+ Lod %\d+')   # a fetch below level B: the coordinates shifted
        self.assertRegex(dis, r'%sz = OpShiftLeftLogical %v3int %\d+ %\d+')                       # a size below level B: level 0's scaled up
        self.assertIn('OpMemberDecorate %_struct_', dis); self.assertIn(' 0 Offset 12', dis); self.assertIn('PushConstant', dis)
        self.assertRegex(dis, r'OpBitwiseAnd %uint %\d+ %uint_15')

    def test_composes_with_the_alpha_tests_block(self):   # one push constant block an entry point: the alpha test's takes a third member
        dis = self.run_variant(SHADER_C.read_text(), alpha_first=True)
        self.assertEqual(dis.count('OpVariable %_ptr_PushConstant'), 1)
        self.assertIn(' 0 Offset 0', dis); self.assertIn(' 1 Offset 4', dis); self.assertIn(' 2 Offset 12', dis)
