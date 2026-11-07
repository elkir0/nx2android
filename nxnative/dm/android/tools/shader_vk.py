#!/usr/bin/env python3
# Android shader build: a game's Switch shaders (the package directory shader_split.py writes) → Vulkan SPIR-V + replay reflection, packed into one bundle.
# The same translator as the Apple chain (dm/gfx/tools/dmsh_xlate_bin, Ryujinx's translator with TargetApi.Vulkan) and the same jobs as dmsh_build.py
# (control section / code area / c1 split, the VertexA + VertexB merge, the profile's answer table), but no MSL: Vulkan takes the SPIR-V as it is.
# Geometry shaders are translated as Vulkan geometry shaders (the Apple chain emulates them in compute kernels). NVN2 (Switch 2) shaders go to the translator's Ampere frontend
# (nvn2_job); their task and mesh shaders become VK_EXT_mesh_shader stages, the bundle's stages 6 and 7.
#   DM_GAME=<id> shader_vk.py <package dir> <out bundle> [--jobs N] [--keys k1,k2] [--xlate <dmsh_xlate.dll>] [--capture <dir>]
# --capture: shaders whose code no game file holds, saved on the device by the Vulkan backend (gfx_vk_shader.c vk_shader_missing: <key>.bin = "DMSC", u32 control length, u32 code length,
#   control section, code area), translated beside the package's; a file whose content does not hash to its name, or whose key the package has, is left out
# Bundle (read by the Vulkan backend, dm/android/gfx_vk_shader.c): "DMSV" u32 version, u32 count, then count entries {u64 key, u32 stage, u32 spv_off, u32 spv_len,
#   u32 refl_off, u32 refl_len, u32 pad} sorted by key, then the blobs. refl is compact JSON: bindings, c1, local size, notes.
import argparse, collections, concurrent.futures as cf, hashlib, json, os, re, shutil, struct, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent.parent / 'gfx/tools'
sys.path.insert(0, str(TOOLS))
import dmsh_facts, dmsh_chain   # noqa: E402  (DM_GAME picks the profile)

ap = argparse.ArgumentParser()
ap.add_argument('pkg'); ap.add_argument('out')
ap.add_argument('--jobs', type=int, default=os.cpu_count() or 8)
ap.add_argument('--keys', default='')
ap.add_argument('--dotnet', default=os.environ.get('DOTNET', 'dotnet'))
ap.add_argument('--capture', help='a folder of shaders captured at run time on the device (<key>.bin)')
ap.add_argument('--xlate', help="the translator (dmsh_xlate.dll) to run (default: the iOS chain's dm/gfx/tools/dmsh_xlate_bin; the porter builds one from the in-repo source)")
ap.add_argument('--spv-dir', help='also write each module of the bundle as <dir>/<key>.spv (for spirv-val, spirv-dis)')
ap.add_argument('--vt', action='store_true', default=os.environ.get('DM_VK_VT', '0') not in ('', '0'),
                help="shader-side virtual texturing (spv_vt.py; the replay's gfx_vk_vt.c): fragment shaders sample their candidate 2D arrays through vt_sample / vt_qlod (also DM_VK_VT=1)")
a = ap.parse_args()

GAME = dmsh_chain.load_game()
FACTS = dmsh_facts.load()
MAP = str(GAME.table('shader_map.tsv'))
XLATE_DLL = a.xlate or str(TOOLS / 'dmsh_xlate_bin/dmsh_xlate.dll')
STAGES = {'vs': 0, 'ps': 1, 'gs': 2, 'hs': 3, 'ds': 4, 'cs': 5, 'ts': 6, 'ms': 7}   # NVN stage numbers (NVN2's task and mesh stages 6 and 7)
# Storage buffers are bound from a 256-byte boundary below their address (gfx_vk_exec.c SB_ALIGN, the same number): Vulkan's largest minStorageBufferOffsetAlignment, so a multiple of
# every device's; the translator offsets every storage access by the address's distance from that boundary (Program.cs QueryHostStorageBufferOffsetAlignment)
SB_ALIGN = 256
# Maxwell's warps are 32 lanes; Adreno runs 64 or 128 (a wave's size is the driver's choice per pipeline). Translated for a group above 32 (Program.cs QueryHostSubgroupSize),
# the translator emulates 32-lane warps inside it: lane id & 31, the ballot word of the lane's own 32, shuffles inside its 32, votes by ballot. Its 128 path picks the word by
# lane id >> 5, so it holds for 64 too. At 32 the warp code read only the first 32 lanes' ballot (Tears of the Kingdom: 32 shaders vote, ballot or shuffle).
# The choice holds for Ampere too: Vulkan cannot ask Adreno for 32 (VK_EXT_subgroup_size_control's smallest size there is 64), so NVN2's warps are emulated the same way. Its VOTE,
# VOTEU and SHFL take the same helpers; REDUX, MATCH, HMMA, LDSM and MOVM, which name the lanes of their warp, take this lane's warp in the subgroup (InstEmitAmpereTensor.cs
# AWarpShuffle). Mario Kart World 1.8.0: 157 shaders vote, 30 run the tensor instructions, 25 MATCH, 5 REDUX
SUBGROUP = 128
# Adreno rounds a gather coordinate that sits on a texel edge the other way from Maxwell, so a gather can pick the wrong 2x2 texels. Eden moves gather coordinates
# 1/512 texel on Qualcomm and Turnip (#4565, TotK's large dark squares); Ryujinx's TexturePass does the same with 8 bits (Program.cs QueryHostGatherBiasPrecision).
# Tears of the Kingdom on the Adreno 830 drew dark boxes of whole 16x16 tiles in its tiled lighting (607 shaders gather, 38 of them compute): in 2 of 12 camera
# steps at the Great Sky Island pond and every frame at the Temple of Time's stone circle; with the bias, 0 in 24
GATHER_BIAS = 8
# Ryujinx's arrays of handles (a texture whose handle traces to no slot indexes the pools; a computed index into c[2] its handle words) keep 32 elements, as before the Apple
# chain sized them to the whole pools: the Vulkan replay binds every element as a descriptor of its own (gfx_vk_exec.c stage_set) and has no pool binding (Program.cs QueryTextureArrayLength*)
HANDLE_ARRAYS = 32
# Compute shared memory past what a device gives a workgroup: Adreno's maxComputeSharedMemorySize is 32 KB (as Metal's limit, dmsh_abi.h DMSH_TG_MAX), Ampere gives up to 100 KB. Mario Kart
# World's DrivePassShaderDefault has 168 compute shaders of 48 KB; a pipeline that declares more than the limit is invalid. The translator gives such a shader its shared memory in a
# storage buffer instead, a slice per workgroup (Program.cs QueryHostSharedMemoryLimit; the binding kind 'spill' below, which the replay fills)
SHARED_MAX = 0x8000
# The storage buffer's slices: workgroup w of a dispatch takes slice w modulo this (dmsh_abi.h DMSH_TG_GROUPS, as the Apple chain's tg_spill), so the replay binds slices x the
# reflection's spill words x 4 bytes (gfx_vk_shader.c vk_spill_buffer): 24 MB for 48 KB shaders. Workgroups this far apart in a dispatch must not run together
SPILL_SLICES = 512
# The first bytes of such a shared memory can stay in workgroup memory, the buffer's slice holding the rest; each access then tests its word (Program.cs
# QueryHostSharedMemorySpillKeep, SharedMemorySpill.Split). Mario Kart World's 168 keep 48-byte records at the invocation's global index (times a count), so with 32 KB
# kept a dispatch's first 682 records (10 workgroups at one an invocation) never touch the buffer, and the buffer is 8 MB instead of 24. DM_SHADER_SPILL_KEEP=<bytes> (32768: the device's whole
# limit); 0, the default, keeps all of it in the buffer, as before: not yet run on a device (no run so far cut a dispatch into parts, so the buffer costs only its
# own loads and stores, and compute pipelines are built at a shader's first dispatch, where a driver that refused the form would lose that pass)
SPILL_KEEP = int(os.environ.get('DM_SHADER_SPILL_KEEP') or 0)
# A global memory access traced to no storage buffer (an address the shader loaded from memory) goes, on the Mac, through the memory pool table and write log the host binds
# (GlobalToStorage TryGenerateRawStorageOp, physical storage buffer pointers: the gmem_map / gmem_log kinds of dmsh_abi.h). The Vulkan replay binds neither and does not enable
# buffer device addresses for its pools, so such an access is dropped, as before (Mario Kart World 1.8.0: 2 shaders, gsys's common vertex shader and agl's common compute)
RAW_GMEM = False
# Float min / max / clamp: FMin / FMax / FClamp leave the result of a NaN operand to the device; Maxwell's FMNMX gives the other operand and .SAT gives 0, as NMin / NMax / NClamp do
# (Program.cs QueryHostNanSafeMinMax). The Apple chains take that rule since Super Mario Galaxy 2's half-white Mario on the A20 Pro (56704a0b: a NaN spot term in
# max(0, x) that the Switch and the M4 Pro make 0). DM_SHADER_NAN_MINMAX=1: NMin / NMax / NClamp; the default 0 keeps FMin / FMax / FClamp until a device run shows
# what the Adreno compiler makes of them (the same instruction if its min / max already return the other operand, more if not: Mario Kart World 1.8.0 has 232,000
# of them in its pixel shaders)
NAN_MINMAX = (os.environ.get('DM_SHADER_NAN_MINMAX') or '0') != '0'
# spirv-opt over each module (DM_SHADER_SPIRV_OPT=1: -O keeping every binding and interface variable; a module it fails on, or whose result does not validate, stays as
# translated). Off: over Mario Kart World 1.8.0's 11,603 modules -O validates everywhere and takes 52 % of the vertex and pixel shaders' instructions out, but 89 % of
# what it takes out (a 200-shader sample) is loads, stores, variables, access chains and bitcasts of the translator's register variables, which a driver's own SSA
# construction removes as well; it leaves 7.5 % fewer arithmetic, extended and sampling instructions, and compute shaders grow 21 % (inlined helpers). Not measured
# on a device: a bundle made with it is the A/B (gfx_vk_shader.c DM_VK_SHADERS)
SPIRV_OPT = (os.environ.get('DM_SHADER_SPIRV_OPT') or '0') != '0'
# The words of a wide storage load or store (Ampere's LDG / STG .64 / .128, which the frontend splits into 32-bit words at address + 4j) take one word index plus j
# (Program.cs QueryHostWideStorageWords, GlobalToStorage) instead of (address + 4j - base) >> 2 each, so a driver can see that they are neighbours and read them whole;
# upstream 1b9fbea4 does it in the Metal source (Splatoon Raiders' lit pass 70 → 16 ms on an M1). Mario Kart World 1.8.0: 16,449 vec4 and 7,542 vec2 runs of such words in
# 2,382 shaders (PARITY's census). DM_SHADER_WIDE_WORDS=1: that form; the default 0 keeps each word's own index, as before, until a device run shows Adreno's compiler merging them
WIDE_WORDS = (os.environ.get('DM_SHADER_WIDE_WORDS') or '0') != '0'
# agl's NDSR upscaler (30 compute shaders of Mario Kart World 1.8.0, 144-288 HMMA.16816 with an f16 accumulator a thread): the translator's helper converts each half to float and makes
# 64 FP32 FMAs a lane an HMMA (HelperFunctionManager dmsh_hmma16816_f16). DM_SHADER_HMMA_F16=1: the helper in packed half arithmetic instead, 32 two-wide half FMAs and no conversion of
# A and B, its sums in half precision as the console's f16 accumulator allows (CodeGen/Spirv TensorF16; the device needs shaderFloat16). The default 0 keeps the float sum, the same
# results as before: the half sum is not bit-exact, and no run has shown the game engaging the network on the phone yet (gfx_vk_shader.c says it when one of them loads)
HMMA_F16 = (os.environ.get('DM_SHADER_HMMA_F16') or '0') != '0'
# A vertex shader's base vertex, base instance and draw index (c0 + 0x640, 0x644, 0x648, which NVN's driver writes for every draw) from Vulkan's BaseVertex, BaseInstance and
# DrawIndex (Program.cs QueryHasConstantBufferDrawParameters). The replay can put a direct draw's base instance in c0 (gfx_vk_exec.c g_draw_binst) but not an indirect draw's,
# whose record the GPU writes: it put 0, so Mario Kart World's GPU-culled vegetation (DrawElementsIndirect records whose first instance the culling writes: 15, 23...) read its
# instances from the start of the list, whose order the culling's atomics change from frame to frame: at its first race's start line, trees drawn black came and went in 25-50 %
# of the frames, 1-2 % with this, and most Free Roam places are steady (Galaxy Z Fold, 2026-11-06; some trees there still flicker, from another cause).
# DM_SHADER_DRAW_PARAMS=0: from c0, as before
DRAW_PARAMS = (os.environ.get('DM_SHADER_DRAW_PARAMS') or '1') != '0'

def records(d):
    r = [m.start() for m in re.finditer(b'BTCDBTCD', d)]
    return [d[x:(r[k + 1] if k + 1 < len(r) else len(d))] for k, x in enumerate(r)]

SPH_STAGE = {1: 'vs', 2: 'hs', 3: 'ds', 4: 'gs', 5: 'ps'}   # SPH word 0 bits 10..13 (shader_split.py sph_stage)
def captured(d):
    """The run-time captures in folder d (--capture): {key: (stage, record in the package's form)}"""
    out = {}
    for nm in sorted(os.listdir(d)) if d and os.path.isdir(d) else []:
        if not re.fullmatch(r'[0-9a-f]{16}\.bin', nm): continue
        with open(os.path.join(d, nm), 'rb') as f: b = f.read()
        if len(b) < 12 or b[:4] != b'DMSC': print(f'[RED] capture {nm}: not a captured shader', flush=True); continue
        nctl, ncode = struct.unpack_from('<2I', b, 4); ctl, code = b[12:12 + nctl], b[12 + nctl:12 + nctl + ncode]
        if len(code) != ncode or len(ctl) != nctl or ctl[:4] != bytes.fromhex('34127698'): print(f'[RED] capture {nm}: truncated', flush=True); continue
        # the key's control bytes (hle_nvn.c keys_hash): an NVN2 section's first 0x48 with +0x30 as zero (gfx_vk_shader.c keeps 0xd0 of it, for nvn2_job), else the whole section saved
        kc = ctl[:0x30] + bytes(4) + ctl[0x34:0x48] if nvn2(ctl) else ctl
        if hashlib.sha256(kc + code).hexdigest()[:16] != nm[:16]: print(f'[RED] capture {nm}: its content does not hash to its name', flush=True); continue
        if nvn2(ctl):   # NVN2: compute, task and mesh by the control section's stage byte (+0x56: 5, 6, 7), else the 0x80-byte SPH at the code area's start (no code header)
            st = {5: 'cs', 6: 'ts', 7: 'ms'}.get(ctl[0x56]) if len(ctl) > 0x56 else None
            if not st and len(code) >= 0x80: st = SPH_STAGE.get(struct.unpack_from('<I', code, 0)[0] >> 10 & 0xf)
        else: st = 'cs' if code[:4] == bytes.fromhex('79563412') else SPH_STAGE.get(struct.unpack_from('<I', code, 0x30)[0] >> 10 & 0xf) if code[:4] == bytes.fromhex('78563412') and len(code) > 0x80 else None
        if not st: print(f'[RED] capture {nm}: no stage in its code area header', flush=True); continue
        rec = bytearray(b'BTCDBTCD' + bytes(0x30)); struct.pack_into('<2I', rec, 0x28, len(ctl), len(code)); rec += code + ctl
        if rec.find(bytes.fromhex('34127698')) != 0x38 + len(code): print(f'[RED] capture {nm}: the control magic appears in its code area', flush=True); continue   # job_for splits there
        out[nm[:16]] = (st, bytes(rec))
    return out

def vertex_ab(key, code, ctl):   # dmsh_build.py vertex_ab: a VertexA + VertexB program merged as Ryujinx does
    if len(code) < 0x80 or len(ctl) < 0x760 or code[:4] != bytes.fromhex('78563412'): return None
    w0 = struct.unpack_from('<I', code, 0x30)[0]
    if w0 & 0x1f != 1 or (w0 >> 10) & 0xf != 0: return None
    a_sph, a_len, _, b_sph, b_len, _ = struct.unpack_from('<6I', ctl, 0x748)
    if not any((a_sph, a_len, b_sph, b_len)): return None
    bw0 = struct.unpack_from('<I', code, b_sph)[0] if 0x80 <= b_sph <= len(code) - 0x50 else 0
    if a_sph != 0x30 or b_sph != struct.unpack_from('<I', ctl, 0x700)[0] + 0x30 or bw0 & 0x1f != 1 or (bw0 >> 10) & 0xf != 1 or (a_len - 0x50) % 32 or (b_len - 0x50) % 32 or b_sph + b_len > len(code):
        return None
    sa, sb = list(struct.unpack_from('<20I', code, a_sph)), list(struct.unpack_from('<20I', code, b_sph))
    sph = [sb[0] | sa[0] & ~(0xf << 10)] + [max(sa[i] & 0xffffff, sb[i] & 0xffffff) | sb[i] & ~0xffffff for i in (1, 2, 3)] + [sb[4]] + [sa[i] | sb[i] for i in range(5, 20)]
    av = bytearray(code[a_sph + 0x50:a_sph + a_len]); b = code[b_sph + 0x50:b_sph + b_len]
    for o in range(0, len(av), 8):
        if o % 32 == 0: continue
        ins = struct.unpack_from('<Q', av, o)[0]
        if ins >> 52 == 0xe30 and ins & 0x1f == 0xf: struct.pack_into('<Q', av, o, 0xe24 << 52 | ((len(av) - (o + 8)) & 0xffffff) << 20 | ins & 0xfffff)
    return code[:0x30] + struct.pack('<20I', *sph) + bytes(av) + b, code[b_sph + b_len:]

def xfb_layout(key, ctl):
    """A vertex shader's transform feedback as its control section declares it (where the NVN driver reads it; dmsh_build.py tf_layout reads the one-output form): +0x6dc u16 stride of
    each buffer, +0x6e8 u8 number of outputs it captures, +0xdc + 0x80 b those outputs (0x20 + n = generic output n), +0x4dc + 0x80 b their component counts (u8 each); the outputs
    follow each other in the buffer (up to 9 outputs interleaved in one buffer, stride 144, in the game it was found with). → the translator's job "xfb": per buffer None or
    {stride, words}, words = the output word at each 4-byte position (Maxwell's numbering, 0x20 + 4 n + component); None without feedback or in a form not known"""
    strides = struct.unpack_from('<4H', ctl, 0x6dc)
    if not any(strides): return None
    out = []
    for b in range(4):
        if not strides[b]: out.append(None); continue
        n = ctl[0x6e8 + b]; outs, comps = ctl[0xdc + 0x80 * b:0xdc + 0x80 * b + n], ctl[0x4dc + 0x80 * b:0x4dc + 0x80 * b + n]
        words = [0x20 + 4 * (o - 0x20) + c for o, k in zip(outs, comps) for c in range(k)]
        if not n or any(not 0x20 <= o < 0x40 for o in outs) or any(not 1 <= k <= 4 for k in comps) or 4 * len(words) != strides[b]:
            print(f'[RED] {key}: transform feedback buffer {b} in a form not known (stride {strides[b]}, outputs {list(outs)}, components {list(comps)}): translated without capture', flush=True)
            return None
        out.append({'stride': strides[b], 'words': words})
    return out

def base_job(key, stage, cp, c1):   # the fields every Android job has (Program.cs reads each of them; a job without one gets the Apple chain's answer)
    return {'key': key, 'stage': stage, 'code': cp, 'tex': {}, 'img': {}, 'attr': {}, 'fout': {}, 'c1': c1.hex(), 'debug': False, 'reduced': True, 'sb_align': SB_ALIGN, 'subgroup': SUBGROUP,
            'gather_bias': GATHER_BIAS, 'gs_passthrough': False, 'metal_image_atomics': False, 'handle_arrays': HANDLE_ARRAYS, 'shared_max': SHARED_MAX, 'spill_slices': SPILL_SLICES,
            'spill_keep': SPILL_KEEP, 'raw_gmem': RAW_GMEM, 'nan_minmax': NAN_MINMAX, 'wide_words': WIDE_WORDS, 'hmma_f16': HMMA_F16,
            'draw_params': DRAW_PARAMS}

# ---- NVN2 (Switch 2): Ampere code, translated by the translator's Ampere frontend (job arch sm86), as dmsh_build.py job_for and dmsh_ampere.py read it ----
def nvn2(ctl):   # an NVN2 control section: magic, version 0xc9 (NVN's have version 1 and are 0x880 bytes)
    return len(ctl) >= 0x48 and struct.unpack_from('<II', ctl) == (0x98761234, 0xc9)

def sph_outputs(sph):
    """The components an SPH's OMAP enables, in the order of the attribute area (dmsh_meshstage.py sph_outputs): [(None, c)] for the position's (byte 0x35, bits 4..7), then [(n, c)]
    for generic attribute n's (bytes 0x36 on, a nibble an attribute, bit c its component)"""
    out = [(None, c) for c in range(4) if sph[0x35] >> (4 + c) & 1]
    for n in range(32): out += [(n, c) for c in range(4) if sph[0x36 + n // 2] >> (4 * (n % 2) + c) & 1]
    return out

def mesh_info(ctl, area):
    """A task or mesh shader's facts (dmsh_meshstage.py info): control +0x56 its stage (6 task, 7 mesh), +0x58 its workgroup's threads, +0x5a its payload's bytes, +0x5e whether a mesh
    shader has a task, +0x60 / +0x62 its most vertices and primitives, +0x64 the offset of its per-primitive SPH where +0x68 bit 7 is set; the outputs as [[attribute or -1 for the
    position, component]] (vertex: the SPH's, primitive: the second SPH's generic ones)"""
    kind = {6: 'ts', 7: 'ms'}.get(ctl[0x56]) if len(ctl) > 0x56 else None
    threads, payload = struct.unpack_from('<2H', ctl, 0x58)
    out = {'kind': kind, 'threads': threads, 'payload': payload}
    if kind == 'ms':
        max_v, max_p = struct.unpack_from('<2H', ctl, 0x60)
        at = struct.unpack_from('<I', ctl, 0x64)[0] if ctl[0x68] & 0x80 else 0
        prim = sph_outputs(area[at:at + 0x80]) if at else []
        out.update(task=bool(ctl[0x5e]), max_v=max_v, max_p=max_p, vertex=[[-1 if n is None else n, c] for n, c in sph_outputs(area[:0x80])],
                   primitive=[[n, c] for n, c in prim if n is not None])
    return out

MESH_INPUT, MESH_BLOCK = 0x10, 32   # dmsh_meshstage.py INPUT, BLOCK: the stage input's offset in the stage memory; vertices and primitives of an attribute block
def stage_memory(i):
    """The stage memory's offsets in shared memory (the translator's job stage_memory: the workgroup's number, the input, the outputs, a mesh shader's primitive bytes) and its bytes,
    laid out as dmsh_meshstage.py memory lays it out: a task's payload after the input; a mesh shader's attribute blocks of 32 vertices and 32 primitives, each component 0x80 bytes
    (a word a vertex or primitive), the vertex components first, then the primitive bytes: the count at +3, three index bytes a triangle from +4"""
    out = MESH_INPUT + (i['payload'] + 15 & ~15)
    if i['kind'] == 'ts': return [0, MESH_INPUT, out, out + i['payload']], out + (i['payload'] + 3 & ~3)
    blocks = (max(i['max_v'], i['max_p'], 1) + MESH_BLOCK - 1) // MESH_BLOCK
    prim = out + blocks * max(len(i['vertex']) + len(i['primitive']), 1) * MESH_BLOCK * 4
    return [0, MESH_INPUT, out, prim], prim + (4 + 3 * max(i['max_p'], 1) + 3 & ~3)

def own_shared(code):   # instructions with shared memory of their own (LDS, STS, ATOMS: dmsh_meshstage.py own_shared), which would lie over the stage memory
    return any(struct.unpack_from('<H', code, o)[0] & 0xfff in (0x984, 0x388, 0x988, 0x38c, 0x98c, 0xf8c) for o in range(0, len(code) - 15, 16))

def nvn2_job(key, stage, ctl, area, w):
    """An NVN2 shader's job (dmsh_build.py job_for): the instructions are the code area's first control +0x38 bytes (the 0x80-byte SPH of the graphics stages included), its c1
    segment lies between control +0x40 and +0x44. NVN's control fields the Maxwell jobs read (VertexA / B, transform feedback, early fragment tests) are not in an NVN2 section"""
    a0, e0 = struct.unpack_from('<II', ctl, 0x40)
    c1 = area[a0:e0] if a0 <= e0 <= len(area) else b''
    code = area[:struct.unpack_from('<I', ctl, 0x38)[0]]
    extra, jstage = {'arch': 'sm86'}, stage
    # compute lays out its driver constants in c[0] per shader: its push constants' words (+0xa9), its uniform buffer, storage buffer, texture, separate texture and separate
    # sampler entries (+0xb1, +0xb2, +0xb3, +0xb4, +0xb5; images come last) and the size (+0xba); the translator maps them to c[24] and NVN's fixed tables (InstEmitAmpere.cs AMapC0)
    if stage == 'cs': extra['nvn2_cs'] = [ctl[0xb1], ctl[0xb2], ctl[0xb3], struct.unpack_from('<H', ctl, 0xba)[0], ctl[0xa9] * 4, ctl[0xb4], ctl[0xb5]]
    if stage in ('ts', 'ms'):
        # translated as compute, its SPH taken off and its stage memory in shared memory (InstEmitAmpereMem.cs AIsberd, AStageOut); the translator then writes it as a Vulkan task or
        # mesh shader around that body (job "mesh": Program.cs QueryHostMeshStage). Graphics stages' c[0] and c[2] tables stay: the translator reads their addresses out of the code
        mesh = mesh_info(ctl, area)
        lay, size = stage_memory(mesh)
        if mesh['kind'] != stage: extra['refused'] = f'control +0x56 says {mesh["kind"]}, the split {stage}'
        elif own_shared(code[0x80:]): extra['refused'] = 'shared memory of its own, which would lie over its stage memory'
        extra.update(stage_memory=lay, local=[mesh['threads'], 1, 1], shared=size, mesh=mesh); code = code[0x80:]; jstage = 'cs'
    cp = os.path.join(w, key + '.bin')
    with open(cp, 'wb') as f: f.write(code)
    j = base_job(key, jstage, cp, c1); j.update(extra)
    if stage in ('ts', 'ms'): return j, c1   # its threads and memory come from its control section (above)
    if key in FACTS: j.update(FACTS[key])
    else: j.update(cs_facts(ctl, 0x58, 0x64) if stage == 'cs' else {}, no_facts=True)
    return j, c1

def cs_facts(ctl, at, sh):
    """A compute shader's thread group and shared memory from its control section, as shader_split.py writes them into the answer table (NVN2: +0x58, +0x64; NVN: +0x748,
    +0x754): for a shader the table does not have, captured at run time (--capture)"""
    if len(ctl) < sh + 4: return {}
    x, y, z = struct.unpack_from('<3I', ctl, at)
    return {'local': [x, y, z], 'shared': (struct.unpack_from('<I', ctl, sh)[0] + 15) // 16 * 16} if x * y * z else {}

def job_for(key, stage, rec, w):   # dmsh_build.py job_for; transform feedback is captured natively (xfb_layout)
    ci = rec.find(bytes.fromhex('34127698')); code = rec[0x38:ci]
    if nvn2(rec[ci:]): return nvn2_job(key, stage, rec[ci:], code, w)
    c1o, c1t = struct.unpack_from('<2I', rec, ci + 0x700)
    c1 = rec[0x38 + c1o:0x38 + c1t]
    ab = vertex_ab(key, code, rec[ci:]) if stage == 'vs' else None
    if ab: code, c1 = ab
    if stage != 'cs': code = code[0x30:]
    elif code[:4] == bytes.fromhex('79563412'): code = code[0x100:]
    cp = os.path.join(w, key + '.bin'); open(cp, 'wb').write(code)
    xfb = xfb_layout(key, rec[ci:]) if stage == 'vs' else None
    j = base_job(key, stage, cp, c1)
    if xfb: j['xfb'] = xfb
    # GLSLC marks a fragment shader declared layout(early_fragment_tests) at control +0x718 (bit 0); NVN then sets the 3D engine's FORCE_EARLY_FRAGMENT_TESTS, which
    # the code area does not carry. Monster Hunter Rise's occlusion culling draws boxes over a 128x64 depth and stores a flag per visible box: tested after the shader,
    # every box's fragments ran and flagged it (15 ms a frame in Kamura village on Adreno 650)
    if stage == 'ps' and len(rec) >= ci + 0x71c and struct.unpack_from('<I', rec, ci + 0x718)[0] & 1: j['early_z'] = True
    if key in FACTS: j.update(FACTS[key])
    else: j.update(cs_facts(rec[ci:], 0x748, 0x754) if stage == 'cs' else {}, no_facts=True)
    return j, c1

# ---- SPIR-V reflection: descriptor bindings with the translator's names, decoded as dmsh_build.py reflect decodes the MSL names ----
OP_NAME, OP_DECORATE, OP_VARIABLE, OP_TYPE_POINTER, OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY, OP_TYPE_IMAGE, OP_TYPE_SAMPLED_IMAGE, OP_TYPE_SAMPLER, OP_TYPE_STRUCT, OP_CONSTANT, OP_EXECUTION_MODE = 5, 71, 59, 32, 28, 29, 25, 27, 26, 30, 43, 16
DEC_BINDING, DEC_SET, DEC_BLOCK, DEC_BUFFER_BLOCK, DEC_NONWRITABLE = 33, 34, 2, 3, 24
SC_UNIFORM_CONSTANT, SC_UNIFORM, SC_STORAGE_BUFFER = 0, 2, 12
def spirv_bindings(spv):
    w = struct.unpack(f'<{len(spv) // 4}I', spv)
    names, deco, types, consts, vars_ = {}, collections.defaultdict(dict), {}, {}, []
    i = 5
    while i < len(w):
        n, op = w[i] >> 16, w[i] & 0xffff; ins = w[i:i + n]
        if op == OP_NAME: names[ins[1]] = bytes(struct.pack(f'<{n - 2}I', *ins[2:])).split(b'\0')[0].decode()
        elif op == OP_DECORATE: deco[ins[1]][ins[2]] = ins[3] if n > 3 else 1
        elif op in (OP_TYPE_POINTER,): types[ins[1]] = ('ptr', ins[2], ins[3])
        elif op == OP_TYPE_ARRAY: types[ins[1]] = ('array', ins[2], ins[3])
        elif op == OP_TYPE_RUNTIME_ARRAY: types[ins[1]] = ('rarray', ins[2])
        elif op == OP_TYPE_IMAGE: types[ins[1]] = ('image', ins[3], ins[7], ins[5], ins[6], ins[8])   # dim, sampled (1 sampled, 2 storage), arrayed, multisampled, image format
        elif op == OP_TYPE_SAMPLED_IMAGE: types[ins[1]] = ('sampled', ins[2])
        elif op == OP_TYPE_SAMPLER: types[ins[1]] = ('sampler',)
        elif op == OP_TYPE_STRUCT: types[ins[1]] = ('struct',)
        elif op == OP_CONSTANT and n == 4: consts[ins[2]] = ins[3]
        elif op == OP_VARIABLE: vars_.append((ins[2], ins[1], ins[3]))   # id, pointer type, storage class
        i += n
    out = []
    for vid, pt, sc in vars_:
        d = deco.get(vid, {})
        if DEC_BINDING not in d: continue
        t = types.get(types[pt][2]) if pt in types else None; tid = types[pt][2] if pt in types else None
        count = 1
        while t and t[0] in ('array', 'rarray'):
            count = consts.get(t[2], 1) if t[0] == 'array' else 0; tid = t[1]; t = types.get(tid)
        if sc == SC_UNIFORM: dt = 'ssbo' if deco.get(tid, {}).get(DEC_BUFFER_BLOCK) else 'ubo'
        elif sc == SC_STORAGE_BUFFER: dt = 'ssbo'
        elif t and t[0] == 'sampled': dt = 'tbuf' if types.get(t[1], ('',))[1:2] == (5,) else 'sampled'   # Dim 5 = Buffer: a uniform texel buffer
        elif t and t[0] == 'image': dt = 'storage_tbuf' if t[1] == 5 and t[2] == 2 else 'storage' if t[2] == 2 else 'tbuf' if t[1] == 5 else 'image'
        elif t and t[0] == 'sampler': dt = 'sampler'
        else: dt = '?'
        img = t if t and t[0] == 'image' else types.get(t[1]) if t and t[0] == 'sampled' else None
        out.append({'name': names.get(vid, ''), 'set': d.get(DEC_SET, 0), 'binding': d[DEC_BINDING], 'type': dt, 'count': count,
                    'dim': img[1] if img else None, 'arrayed': img[3] if img else 0, 'ms': img[4] if img else 0, 'fmt': img[5] if img else 0})
    local = None
    i = 5
    scalars = {}   # type id → ('f' | 'i' | 'u')
    while i < len(w):
        n, op = w[i] >> 16, w[i] & 0xffff
        if op == OP_EXECUTION_MODE and w[i + 2] == 17: local = list(w[i + 3:i + 6])   # LocalSize
        elif op == 21: scalars[w[i + 1]] = 'i' if w[i + 3] else 'u'                  # OpTypeInt width signedness
        elif op == 22: scalars[w[i + 1]] = 'f'                                        # OpTypeFloat
        elif op == 23: scalars[w[i + 1]] = scalars.get(w[i + 2], 'f')                 # OpTypeVector: its component's kind
        i += n
    # the stage's interface: Input (1) / Output (3) variables with a Location → masks of all, signed integer and unsigned integer locations
    io = {1: [0, 0, 0], 3: [0, 0, 0]}
    for vid, pt, sc in vars_:
        if sc not in io or 30 not in deco.get(vid, {}): continue   # 30 = Location
        loc = deco[vid][30]
        if loc >= 32: continue
        t = types.get(pt); tid = t[2] if t else None
        while tid in types and types[tid][0] in ('array', 'rarray'): tid = types[tid][1]
        k = scalars.get(tid, 'f')
        io[sc][0] |= 1 << loc
        if k == 'i': io[sc][1] |= 1 << loc
        if k == 'u': io[sc][2] |= 1 << loc
    return out, local, io

# Handle decoding: as dmsh_build.py decode_tex / decode_ubo_tex / reflect (names: <stage>_t<type>_cb<slot>_<handle>, <stage>_c<n>, <stage>_s<n>, <stage>_i…)
TEX_BASE, IMG_BASE, IMG_SLOTS, SEPTEX_BASE, SEPTEX_SLOT0, SAMP_BASE = 0x20, 0x120, 8, 0x168, 128, 0x568
MESH_FIRST_SLOT = 29   # dmsh_abi.h DMSH_MESH_FIRST_SLOT
SSBO = {'vs': 0x110, 'cs': 0x310, 'gs': 0x410, 'ps': 0x510, 'ts': 0x670, 'ms': 0x770}   # a stage's storage buffer table in c[0] (dmsh_abi.h DMSH_C0_SSBO_*; NVN2's task and mesh stages' at 0x670 and 0x770)
def classify(stage, b, res):
    nm = b['name']; e = dict(b)
    # shader-side virtual texturing (spv_vt.py): the bindings its transform added. vt_pool_<k>'s reg (its candidate's binding) is main's, from the transform's info; off is k
    if nm in ('vt_ctl', 'vt_pt', 'vt_fb'): e.update(kind=nm); return e
    if re.fullmatch(r'vt_pool_\d+', nm): e.update(kind='vt_pool', off=int(nm[8:])); return e
    m = re.fullmatch(r'[a-z]+_c(\d+)', nm)
    if m:
        n = int(m.group(1))
        # a driver bank: c[0] and c[2], and NVN2's push constants c[24] (nvnCommandBufferPush*Constants, dmsh_abi.h DMSH_PUSH_BANK; NVN2 compute's c[0] push region is mapped
        # there, InstEmitAmpere.cs AMapC0), which the replay fills from the pushed block (gfx_vk.h RK_DRV)
        e.update(kind='cbuf', reg=n - 3) if 3 <= n <= 16 else e.update(kind='c1' if n == 1 else 'drvcb', bank=n); return e
    # a task shader's, or a draw-launched mesh shader's, first workgroup (Program.cs QueryHostMeshStage: word 0 = the draw's first): the bank the replay fills it as (dmsh_abi.h DMSH_MESH_FIRST_SLOT)
    if nm == 'mesh_first': e.update(kind='drvcb', bank=MESH_FIRST_SLOT); return e
    # a compute shader's shared memory in a storage buffer (Program.cs QueryHostSharedMemoryLimit): a slice of the reflection's spill words a workgroup, SPILL_SLICES slices
    if nm == 'shared_spill': e.update(kind='spill', off=res.get('spill_words'), samp_arg=res.get('spill_slices')); return e
    if nm == 'support_buffer': e.update(kind='support'); return e
    m = re.fullmatch(r'[a-z]+_s(\d+)', nm)
    if m:
        idx = int(m.group(1)); sb = sorted(res.get('sbufs', []), key=lambda x: x['binding'])
        s = sb[idx] if idx < len(sb) else None; off = s['cb_offset'] * 4 if s and s['cb_slot'] == 0 else None; base = SSBO.get(stage)
        e.update(kind='sbuf', rw=bool(s and s['flags'] & 1))   # BufferUsageFlags.Write is bit 0 (dmsh_build.py reflect reads the same; bit 1 was never set)
        if stage == 'cs' and off is not None and 0x210 <= off < 0x310 and (off - 0x210) % 16 == 0: e.update(kind='ubo_ptr', reg=(off - 0x210) // 16)
        elif s and s['cb_slot'] >= 3: e.update(kind='sbuf_ubo', reg=s['cb_slot'] - 3, off=s['cb_offset'] * 4)
        else: e['reg'] = (off - base) // 16 if off is not None and base is not None and 0 <= off - base < 0x100 and (off - base) % 16 == 0 else None
        return e
    # Textures and images, named by the translator <stage>_<t|i><type>_<cb<slots>|tcb>_<handle> (dmsh_build.py decode_tex / decode_ubo_tex): tcb = the driver's handle table c2;
    # cb<slots> = texture buffer slot in the low 16 bits, sampler buffer slot + 1 in the high 16 bits. Handle: word offsets o0 (bits 0-13), o1 (14-27), type (28-31):
    # 0 one combined handle (the sampler is the texture handle's high bits), 1 the sampler handle in another word, 3 a constant sampler ID in o1.
    # Vulkan binds each as a combined image sampler: 'tex' / 'tbuf' slots name the texture, samp_* where the sampler comes from.
    m = re.match(r'[a-z]+_([ti])\w*?_(?:cb(\d+)|tcb)_([0-9A-F]+)', nm)
    if m:
        kind = m.group(1); slots = int(m.group(2)) if m.group(2) else -1; h = int(m.group(3), 16)
        tb = (slots & 0xffff) if slots >= 0 else 2; sb = ((slots >> 16) - 1) if slots >= 0 and (slots >> 16) else tb
        o0, o1, ty = h & 0x3fff, (h >> 14) & 0x3fff, h >> 28
        if tb == 2:   # the driver's handle table c2
            def slot(word, base, cb_): return (word * 4 - base) // 8 if cb_ == 2 and word * 4 >= base and (word * 4 - base) % 8 == 0 else None
            words = [o0] + ([o1] if ty else [])
            if kind == 'i':
                e.update(kind='img', reg=next((slot(x, IMG_BASE, tb) for x in words if IMG_BASE <= x * 4 < IMG_BASE + 8 * IMG_SLOTS and slot(x, IMG_BASE, tb) is not None), None))
                if b['type'] == 'storage_tbuf': e['kind'] = 'img_tbuf'
                return e
            if b['count'] > 1 or b['count'] == 0:   # an array of c2 handles indexed at run time: element k = the handle at off + 8k
                e.update(kind='tex_c2a', off=o0 * 4 if ty == 0 else None); return e
            tw = [o0] if ty == 3 else words
            t = next((slot(x, TEX_BASE, tb) for x in tw if slot(x, TEX_BASE, tb) is not None and x * 4 < SEPTEX_BASE), None)
            if t is None: t = next((SEPTEX_SLOT0 + slot(x, SEPTEX_BASE, tb) for x in tw if slot(x, SEPTEX_BASE, tb) is not None and x * 4 < SAMP_BASE), None)
            e.update(kind='tbuf' if b['type'] == 'tbuf' else 'tex', reg=t)
            if ty == 3: e['samp_id'] = o1
            elif ty: e['samp'] = next((slot(x, SAMP_BASE, sb) for x in words if x * 4 >= SAMP_BASE and slot(x, SAMP_BASE, sb) is not None), None)
            else: e['samp_tex'] = True   # the sampler ID is in the same handle as the texture
            if t is not None and e['kind'] == 'tex' and t in set(res.get('fetch1d', ())): e['fetch1d'] = True
            return e
        if 3 <= tb <= 16 and ty in (0, 1, 3) and not (ty == 1 and not 3 <= sb <= 16):   # bindless: the handle is in the game's own uniform buffer
            e.update(kind=('img_ubo' if kind == 'i' else 'tbuf_ubo' if b['type'] == 'tbuf' else 'tex_ubo'), reg=tb - 3, off=o0 * 4)
            if ty == 3: e['samp_id'] = o1
            elif ty == 1: e['samp_ubo'] = [sb - 3, o1 * 4]
            return e
    e.update(kind='?'); return e

# Binary reflection (dm/android/gfx_vk.h dmsv_refl / dmsv_bind): header {u32 nbind, u32 c1_len, u32 local[3]}, nbind × 20-byte bindings, then c1, then "FSTR" (first_store) and "ORDR"
# (ordered_alloc) sections when the shader has them. binding: u8 kind, u8 desc (descriptor type), u8 dim (SPIR-V Dim, 0xff none), u8 flags (1 arrayed, 2 multisampled, 4 written), u16 binding, u16 count,
#          i16 reg, u8 samp (sampler source), u8 fmt (SPIR-V image format), i32 off, i32 samp_arg (a spill binding: off = the words a workgroup takes, samp_arg = the slices)
KIND = {'support': 0, 'cbuf': 1, 'c1': 2, 'drvcb': 3, 'sbuf': 4, 'ubo_ptr': 5, 'sbuf_ubo': 6, 'tex': 7, 'tbuf': 8, 'img': 9, 'img_tbuf': 10, 'tex_c2a': 11,
        'tex_ubo': 12, 'tbuf_ubo': 13, 'img_ubo': 14, 'spill': 15, 'vt_pool': 16, 'vt_ctl': 17, 'vt_pt': 18, 'vt_fb': 19, '?': 255}   # (vt_*: gfx_vk_vt.h RK_VT_*)
DESC = {'ubo': 0, 'ssbo': 1, 'sampled': 2, 'storage': 3, 'tbuf': 4, 'storage_tbuf': 5, 'image': 6, 'sampler': 7, '?': 255}
SAMP_NONE, SAMP_TEX, SAMP_SLOT, SAMP_ID, SAMP_UBO = 0, 1, 2, 3, 4
kinds_all = collections.Counter(); vt_stats = collections.Counter()
if a.vt: import spv_vt   # noqa: E402  (beside this script)
# ---- stores only a dispatch's first invocation makes (upstream e07cba1c, dmsh_build.py first_store; nvn_mtl.m makes them before the dispatch) ----
# A compute shader's preamble `if (-local.x == workgroup.x << log2(local size)) buffer[k] = value`: Mario Kart World's scenery lists (2ce9071c2f22d74f, 5fea9b7ddb30e8a8: one
# guarded block a store, a value in a local set to a constant under the same guard; 0823b08c2027a38d: a local size with rows, the test on y and x, workgroup z zero) reset a list's
# counter in the first invocation while the others append to it. The console starts that workgroup first; Adreno, as Apple's GPUs, starts workgroups in no order, so a late reset wiped
# entries already appended and groups of cacti came and went. The stores leave the shader and the replay makes them, in order, before the dispatch (gfx_vk_exec.c first_stores).
# Read on the translator's SPIR-V, where the bundle's reflection is decoded too (storage buffer binding → slot, as classify), so the Apple chain's translator is unchanged: its Function
# variables (the AST's locals) are followed by value, the guard and the stores' addresses evaluated as linear forms of the invocation and workgroup ids and c0's words. Only this exact
# condition counts (it holds for one invocation of a dispatch with a flat local size, or rows with z 1); a guarded block that holds anything else, another use of the guard after the
# stores, or more than 4 stores leave the whole shader as it is (649e36162cad9e79 of the base game: its first invocation also loops over a uniform's count)
# Also read (2026-11-06, found offline over 1.8.0's 729 compute shaders): an index made with OpIMul by a constant (workgroup.x * 32 + local.x), a guard the branch tests itself
# with no local holding it, and addresses that add a word of the shader's own c1. That takes in 9d65b36d2c14177f and a52144763f3f4d33 (the visible terrain cells the vegetation
# placement reads: their counter, slot 1 word 2, reset by the first invocation while every workgroup appends) and 4e584bc1f755b0ed; and two clear-only shaders, where moving the
# store ahead changes nothing. Still not read: per-layer resets whose address depends on workgroup z (811ae21f81975fcb, c5e9da99764f162e)
# instructions that are not values: stores, calls, atomics, image writes, barriers, control flow (anything else has no effect but its result)
STMT = {62, 63, 57, 99, 224, 225, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 4416, *range(227, 243)}
def first_store(spv, binds, local, res, c1=b''):   # c1: the job's own c1 bytes, whose words are constants here
    w = list(struct.unpack(f'<{len(spv) // 4}I', spv)); ins, i = [], 5
    while i < len(w): n = w[i] >> 16; ins.append((i, w[i] & 0xffff, w[i:i + n])); i += n or 1
    if not local or len(local) < 3 or local[2] != 1: return spv, []
    names = {x[2][1]: bytes(struct.pack(f'<{len(x[2]) - 2}I', *x[2][2:])).split(b'\0')[0].decode() for x in ins if x[1] == 5}
    built = {x[2][1]: x[2][3] for x in ins if x[1] == 71 and len(x[2]) >= 4 and x[2][2] == 11}   # BuiltIn: 26 WorkgroupId, 27 LocalInvocationId
    consts = {x[2][2]: x[2][3] - (1 << 32) * (x[2][3] >> 31) for x in ins if x[1] == 43 and len(x[2]) == 4}
    defs = {x[2][2]: x[2] for x in ins if x[1] in (61, 65, 124, 128, 130, 132, 194, 195, 196, 199, 200, 166, 167, 168, 170, 171, 126)}
    fvars = {x[2][2] for x in ins if x[1] == 59 and x[2][3] == 7}
    c0 = next((v for v, nm in names.items() if re.fullmatch(r'[a-z]+_c0', nm)), None)
    c1v = next((v for v, nm in names.items() if re.fullmatch(r'[a-z]+_c1', nm)), None)   # the shader's own c1 (the job's c1 bytes): constants here
    sbuf = {b['binding']: b for b in binds if b['kind'] == 'sbuf'}
    var_bind = {x[2][1]: x[2][3] for x in ins if x[1] == 71 and len(x[2]) == 4 and x[2][2] == 33}   # Binding
    sb_by_index = sorted(res.get('sbufs', []), key=lambda x: x['binding'])
    def lin(*terms, k=0):
        d = {'k': k}
        for t, c in terms: d[t] = d.get(t, 0) + c
        return d
    def add(a, b, s=1):
        if not isinstance(a, dict) or not isinstance(b, dict): return None
        d = dict(a)
        for t, c in b.items(): d[t] = d.get(t, 0) + s * c
        return {t: c for t, c in d.items() if c or t == 'k'}
    def ev(x, vals):   # a value: a linear form {symbol: coefficient, 'k': constant}, a condition ('cond', atoms, holds when they are all zero), ('shr2', form), or None
        if x in consts: return lin(k=consts[x])
        d = defs.get(x)
        if d is None: return None
        o = d[0] & 0xffff
        if o == 124: return ev(d[3], vals)   # OpBitcast
        if o == 61:   # OpLoad: a local's value, an invocation or workgroup id, a word of c0
            p = d[3]
            if p in fvars: return vals.get(p)
            a = defs.get(p)
            if a is not None and a[0] & 0xffff == 65:
                ix = [consts.get(v) for v in a[4:]]
                if None in ix: return None
                if built.get(a[3]) in (26, 27) and len(ix) == 1: return lin((('wg' if built[a[3]] == 26 else 'lid', ix[0]), 1))
                if a[3] == c0 and len(ix) == 3 and ix[0] == 0: return lin((('c0', 4 * ix[1] + ix[2]), 1))
                if a[3] == c1v and c1v is not None and len(ix) == 3 and ix[0] == 0 and 16 * ix[1] + 4 * ix[2] + 4 <= len(c1):   # a word of c1: its value
                    return lin(k=struct.unpack_from('<i', c1, 16 * ix[1] + 4 * ix[2])[0])
            return None
        if o == 200:   # OpNot: -x - 1
            v = ev(d[3], vals); return add(lin(k=-1), v, -1) if isinstance(v, dict) else None
        if o in (128, 130):   # OpIAdd, OpISub; a word index plus words (WIDE_WORDS: (offset >> 2) + j) is the index of the offset plus their bytes
            a, b = ev(d[3], vals), ev(d[4], vals)
            if o == 128 and isinstance(a, tuple) and a[0] == 'shr2' and isinstance(b, dict) and set(b) == {'k'}: return ('shr2', add(a[1], lin(k=4 * b['k'])))
            if o == 128 and isinstance(b, tuple) and b[0] == 'shr2' and isinstance(a, dict) and set(a) == {'k'}: return ('shr2', add(b[1], lin(k=4 * a['k'])))
            return add(a, b, 1 if o == 128 else -1)
        if o == 132:   # OpIMul, one side a constant (9d65b36d2c14177f's global index: workgroup.x * 32 + local.x)
            a_, b_ = ev(d[3], vals), ev(d[4], vals)
            if isinstance(a_, dict) and isinstance(b_, dict) and set(b_) == {'k'}: return {t: c * b_['k'] for t, c in a_.items()}
            if isinstance(a_, dict) and isinstance(b_, dict) and set(a_) == {'k'}: return {t: c * a_['k'] for t, c in b_.items()}
            return None
        if o == 196:   # OpShiftLeftLogical by a constant
            v, n = ev(d[3], vals), consts.get(d[4])
            return {t: c << n for t, c in v.items()} if isinstance(v, dict) and n is not None else None
        if o == 199:   # OpBitwiseAnd of one word of c0 with -256: the 256-byte boundary below it (SB_ALIGN)
            v, m = ev(d[3], vals), consts.get(d[4])
            syms = [t for t, c in v.items() if t != 'k' and c] if isinstance(v, dict) else None
            return lin((('align', syms[0][1]), 1)) if m == -256 and syms and len(syms) == 1 and v[syms[0]] == 1 and syms[0][0] == 'c0' and not v['k'] else None
        if o in (194, 195):   # OpShiftRightLogical (or Arithmetic: WIDE_WORDS) by 2: a word index (of a byte offset in 0 .. 255 + 4 words here, where the two agree)
            v = ev(d[3], vals); return ('shr2', v) if isinstance(v, dict) and consts.get(d[4]) == 2 else None
        if o in (170, 171):   # OpIEqual / OpINotEqual: lid.d + local[d] wg.d == 0 holds for lid.d == wg.d == 0 alone; wg.d == 0; lid.d == 0
            v = add(ev(d[3], vals), ev(d[4], vals), -1)
            if not isinstance(v, dict) or v['k']: return None
            t = {s: c for s, c in v.items() if s != 'k' and c}
            if any(c < 0 for c in t.values()): t = {s: -c for s, c in t.items()}
            dims = {s[1] for s in t}
            if len(dims) != 1: return None
            dd = dims.pop()
            if t not in ({('lid', dd): 1, ('wg', dd): local[dd]}, {('wg', dd): 1}, {('lid', dd): 1}): return None
            return ('cond', frozenset(t), o == 170)
        if o == 168:   # OpLogicalNot
            v = ev(d[3], vals); return ('cond', v[1], not v[2]) if isinstance(v, tuple) and v[0] == 'cond' else None
        if o == 167:   # OpLogicalAnd of two that hold
            a, b = ev(d[3], vals), ev(d[4], vals)
            return ('cond', a[1] | b[1], True) if all(isinstance(v, tuple) and v[0] == 'cond' and v[2] for v in (a, b)) else None
        return None
    want = {('lid', 0), ('wg', 0)} | ({('lid', 1), ('wg', 1), ('wg', 2)} if local[1] != 1 else set())
    if local[0] & (local[0] - 1) or local[1] & (local[1] - 1): return spv, []
    ep = next((x for x in ins if x[1] == 15), None)
    if ep is None or ep[2][1] != 5: return spv, []
    k = next(j for j, x in enumerate(ins) if x[1] == 54 and x[2][2] == ep[2][2])
    k += 2
    while k < len(ins) and ins[k][1] == 59: k += 1
    vals, fvals, guard, stores = {}, {}, None, []
    while k < len(ins):   # the entry function's top level, up to the guard, then its stores
        at, o, x = ins[k]
        if o == 62 and x[1] in fvars:   # a local: its value, and its value in the first invocation
            v, fv = ev(x[2], vals), ev(x[2], fvals)
            if guard is None and isinstance(v, tuple) and v[0] == 'cond' and v[1] == want: guard = x[1]
            vals[x[1]], fvals[x[1]] = v, fv; k += 1; continue
        if o not in STMT: k += 1; continue   # a value: evaluated where used
        if o == 247 and k + 1 < len(ins) and ins[k + 1][1] == 250:   # OpSelectionMerge, OpBranchConditional
            br = ins[k + 1][2]; c = ev(br[1], fvals)
            if not (isinstance(c, tuple) and c[0] == 'cond' and c[1] == want): break
            if guard is None: guard = -1   # the branch tests the condition itself, no local holds it (9d65b36d2c14177f, a52144763f3f4d33, 4e584bc1f755b0ed)
            if not c[2] or br[3] != x[1] or k + 2 >= len(ins) or ins[k + 2][1] != 248 or ins[k + 2][2][1] != br[2]: return spv, []
            j = k + 3
            while j < len(ins) and ins[j][1] != 249:   # the block: values, locals set, storage buffer stores; anything else leaves the shader as it is
                jat, jo, jx = ins[j]
                if jo == 62 and jx[1] in fvars: fvals[jx[1]] = ev(jx[2], fvals)
                elif jo == 62:
                    a = defs.get(jx[1])
                    if a is None or a[0] & 0xffff != 65 or a[3] not in var_bind or var_bind[a[3]] not in sbuf or len(a) != 6 or consts.get(a[4]) != 0: return spv, []
                    ix, val = ev(a[5], fvals), ev(jx[2], fvals)
                    b = sbuf[var_bind[a[3]]]; m = re.fullmatch(r'[a-z]+_s(\d+)', b['name']); sb = sb_by_index[int(m.group(1))] if m and int(m.group(1)) < len(sb_by_index) else None
                    if not (isinstance(ix, tuple) and ix[0] == 'shr2' and isinstance(val, dict) and set(val) == {'k'}) or sb is None or sb['cb_slot'] != 0 or b.get('reg') is None: return spv, []
                    f = ix[1]; base = [t for t, c in f.items() if t != 'k' and c]
                    if sorted(base) != sorted([('c0', sb['cb_offset']), ('align', sb['cb_offset'])]) or f[('c0', sb['cb_offset'])] != 1 or f[('align', sb['cb_offset'])] != -1 or f['k'] % 4 or f['k'] < 0: return spv, []
                    stores.append((jat, {'reg': b['reg'], 'word': f['k'] // 4, 'value': val['k'] & 0xffffffff}))
                elif jo in STMT: return spv, []
                j += 1
            if j + 1 >= len(ins) or ins[j][2][1] != x[1] or ins[j + 1][1] != 248 or ins[j + 1][2][1] != x[1]: return spv, []
            k = j + 2; continue
        break
    if guard is None or not stores or len(stores) > 4: return spv, []
    if any(x[1] == 61 and x[2][3] == guard for x in ins[k:]): return spv, []   # the guard read again after the stores: an if this does not understand
    drop = {at for at, _ in stores}
    out = w[:5]
    for at, o, x in ins:
        if at not in drop: out += x
    return struct.pack(f'<{len(out)}I', *out), [e for _, e in stores]

# ---- a compute shader that allocates through an atomic (dispatch runs its workgroups in order: gfx_vk_exec.c dispatch, DM_VK_ORDERED) ----
# A small dispatch's workgroups start in index order on the Switch's NVIDIA GPU, so atomics they make on one global counter (base = atomicAdd(counter, my_total)) resolve in that order in
# practice. Adreno starts workgroups in no order, so a game relying on that (found: Mario Kart World's vegetation instance allocation) hands its items different offsets every frame. Read on
# the translated module: any atomic read-modify-write on a storage buffer whose result is used. Workgroup-memory atomics do not count (they order invocations inside one workgroup), nor an
# atomic whose result feeds nothing (OpAtomicStore returns nothing; OpAtomicLoad is no read-modify-write).
# OpAtomicExchange, CompareExchange(Weak), IIncrement, IDecrement, IAdd, ISub, SMin, UMin, SMax, UMax, And, Or, Xor — not OpAtomicLoad (227) nor OpAtomicStore (228)
OP_ATOMICS = set(range(229, 243))   # OpAtomicExchange (229) .. OpAtomicXor (242)
OP_MEMBER_NAME, OP_LOAD, OP_STORE, OP_MEMBER_DECORATE, OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN = 6, 61, 62, 72, 65, 66   # (the other opcodes and decorations are named above)
SC_FUNCTION = 7
def ordered_alloc(spv):
    w = list(struct.unpack(f'<{len(spv) // 4}I', spv)); ins, i = [], 5
    while i < len(w): n = w[i] >> 16; ins.append((w[i] & 0xffff, w[i:i + n])); i += n or 1
    deco = {}   # target id → {decoration: operand} (OpDecorate and OpMemberDecorate both, the latter keyed past the former's numbers)
    for o, x in ins:
        if o == OP_DECORATE and len(x) >= 3: deco.setdefault(x[1], {})[x[2]] = x[3] if len(x) > 3 else 1
        elif o == OP_MEMBER_DECORATE and len(x) >= 3: deco.setdefault(x[1], {})[x[2] + 100000] = x[3] if len(x) > 3 else 1
    ptrs = {x[1]: (x[2], x[3]) for o, x in ins if o == OP_TYPE_POINTER}   # id: (storage class, pointee type)
    def_of = {x[2]: x for o, x in ins if o in (OP_VARIABLE, OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and len(x) >= 3}   # a pointer value → the instruction that made it
    def buf_ptr(v, d=0):
        x = def_of.get(v)   # a chain's own result type can be _ptr_Uniform_uint inside the block: resolve to the base variable, whose pointee is the struct
        if x is None: return False
        if x[0] & 0xffff != OP_VARIABLE: return buf_ptr(x[3], d + 1) if d < 16 else False   # the chain's base pointer
        t = x[1]   # OpVariable [result type, result, storage class]
        if t not in ptrs: return False
        sc, pt = ptrs[t]; dd = deco.get(pt, {})
        return sc in (12, 5349) or (sc == 2 and (dd.get(DEC_BLOCK) or dd.get(DEC_BUFFER_BLOCK)))   # StorageBuffer, PhysicalStorageBuffer, Uniform on a Block / BufferBlock struct
    loads = {x[3] for o, x in ins if o == OP_LOAD and len(x) >= 4}   # the pointers some OpLoad reads
    dead_vars = {x[2] for o, x in ins if o == OP_VARIABLE and len(x) >= 4 and x[3] == SC_FUNCTION and x[2] not in loads}   # the translator writes every register into one
    for o, x in ins:
        if o not in OP_ATOMICS or len(x) < 6 or not buf_ptr(x[3]): continue   # x[1] result type, x[2] result, x[3] pointer
        # "used": the result id is an operand of some instruction other than OpName / OpMemberName / OpDecorate / OpMemberDecorate, and not an OpStore's value into a dead Function variable
        for o2, y2 in ins:
            if y2 is x or o2 in (OP_NAME, OP_MEMBER_NAME, OP_DECORATE, OP_MEMBER_DECORATE) or len(y2) < 3 or x[2] not in y2[2:]: continue
            if o2 == OP_STORE and y2[1] in dead_vars: continue   # OpStore <pointer y2[1]> <value y2[2]>
            return True
    return False

def refl_bin(binds, c1, local, first=(), ordered=False):
    """... then, when the shader has first-invocation stores (first_store), "FSTR", u32 count, count x {i32 storage slot, u32 word, u32 value} after c1, and when it allocates through a
    storage-buffer atomic whose result it uses (ordered_alloc), "ORDR", u32 1 (gfx_vk_shader.c reads both)"""
    out = [struct.pack('<5I', len(binds), len(c1), *(list(local) + [1, 1, 1])[:3])]
    for e in binds:
        kind = KIND.get(e['kind'], 255)
        reg = e.get('reg'); reg = -1 if reg is None else reg
        if e['kind'] == 'drvcb': reg = e.get('bank', 0)
        off = e.get('off') if e.get('off') is not None else -1
        samp, sarg = SAMP_NONE, 0
        if e['kind'] == 'spill': sarg = e.get('samp_arg') or 0   # spill: off = the words a workgroup takes, samp_arg = the slices
        elif e.get('samp_tex'): samp = SAMP_TEX
        elif e.get('samp') is not None: samp, sarg = SAMP_SLOT, e['samp']
        elif e.get('samp_id') is not None: samp, sarg = SAMP_ID, e['samp_id']
        elif e.get('samp_ubo'): samp, sarg = SAMP_UBO, e['samp_ubo'][0] | e['samp_ubo'][1] << 8
        flags = (1 if e.get('arrayed') else 0) | (2 if e.get('ms') else 0) | (4 if e.get('rw') else 0) | (8 if e.get('fetch1d') else 0)
        out.append(struct.pack('<BBBBHHhBBii', kind, DESC.get(e['type'], 255), 0xff if e.get('dim') is None else e['dim'], flags, e['binding'], e['count'],
                               reg, samp, e.get('fmt') or 0, off, sarg))
    return (b''.join(out) + bytes(c1)
            + (b'FSTR' + struct.pack('<I', len(first)) + b''.join(struct.pack('<iII', e['reg'], e['word'], e['value']) for e in first) if first else b'')
            + (b'ORDR' + struct.pack('<I', 1) if ordered else b''))

def main():
    pc, loc = {}, {}
    for ln in open(MAP, encoding='utf-8'):
        c = ln.rstrip('\n').split('\t')
        if len(c) > 4 and c[0].endswith('.bin'): loc.setdefault(c[3], (c[0], int(c[1])))
    cap = {k: v for k, v in captured(a.capture).items() if k not in loc}
    if cap: print(f'{len(cap)} shaders captured at run time: {" ".join(f"{k} ({st})" for k, (st, _) in sorted(cap.items()))}', flush=True)
    keys = sorted(k for k in (a.keys.split(',') if a.keys else [*loc, *cap]) if k in loc or k in cap)
    packs = {}
    w = tempfile.mkdtemp(prefix='shvk_', dir=os.environ.get('WORK'))
    jobs, c1s, stages = {}, {}, {}
    for k in keys:
        if k in cap: stages[k], rec = cap[k]
        else:
            name, idx = loc[k]
            if name not in packs: packs[name] = records(open(os.path.join(a.pkg, name), 'rb').read())
            stages[k], rec = name[:2], packs[name][idx]
        jobs[k], c1s[k] = job_for(k, stages[k], rec, w)
    refused = {k: jobs[k]['refused'] for k in keys if jobs[k].get('refused')}   # not translated (a task or mesh shader whose form is not known: nvn2_job)
    todo = [k for k in keys if k not in refused]
    chunks = [todo[i::a.jobs] for i in range(a.jobs) if todo[i::a.jobs]]
    def xl(ch):
        jp = os.path.join(w, f'job_{abs(hash(ch[0]))}.json'); json.dump([jobs[k] for k in ch], open(jp, 'w'))
        return subprocess.run([a.dotnet, XLATE_DLL, jp, w], capture_output=True, text=True, errors='replace')
    print(f'translating {len(todo)} shaders in {len(chunks)} batches ...', flush=True)
    with cf.ThreadPoolExecutor(max(len(chunks), 1)) as ex: rs = list(ex.map(xl, chunks))
    errs, why = collections.Counter(), dict(refused)
    for r in rs:
        for ln in r.stderr.splitlines():
            if ln.strip(): errs[ln.strip()[:120]] += 1
            m = re.match(r'\[RED\] ([0-9a-f]{16}): (.*)', ln.strip())
            if m: why[m.group(1)] = m.group(2)[:160]
    entries, fails, per_stage, firsts, ordereds = [], collections.Counter(), collections.defaultdict(collections.Counter), {}, {}
    if a.spv_dir: os.makedirs(a.spv_dir, exist_ok=True)
    for k in keys:
        spv_p, res_p = os.path.join(w, k + '.spv'), os.path.join(w, k + '.res.json')
        if not os.path.exists(spv_p):
            fails['refused' if k in refused else 'no output'] += 1; per_stage[stages[k]]['failed'] += 1
            print(f'[RED] {k} ({stages[k]}): {why.get(k, "no output")}', flush=True); continue
        per_stage[stages[k]]['translated'] += 1
        spv = open(spv_p, 'rb').read(); res = json.load(open(res_p)) if os.path.exists(res_p) else {}
        st = stages[k]; vt = None
        if a.vt and st == 'ps':   # (before the reflection, which files the bindings the transform adds)
            try: spv, vt = spv_vt.transform(spv)
            except Exception as ex: print(f'[RED] {k}: spv_vt.py: {ex}; left as it is', flush=True)
            if vt: vt_stats['shaders'] += 1; vt_stats['candidates'] += len(vt['cands']); vt_stats['without requests'] += vt['fb'] is None
        binds, local, io = spirv_bindings(spv)
        cl = [classify(st, b, res) for b in binds]
        for e in cl:
            if e['kind'] == 'vt_pool': e['reg'] = dict((kk, bb) for bb, kk in vt['cands']).get(e['off']) if vt else None   # the candidate's texture binding
        first = []; ordered = False
        if st == 'cs':   # the first invocation's stores, for the replay (first_store), and the allocating atomics that want its workgroups in order (ordered_alloc)
            spv, first = first_store(spv, cl, local or jobs[k].get('local'), res, c1=bytes.fromhex(jobs[k].get('c1') or ''))
            if first: firsts[k] = first
            ordered = ordered_alloc(spv)
            if ordered: ordereds[k] = True
        for e in cl: kinds_all[e['kind']] += 1
        # header words 2-4: a compute shader's local size; a vertex shader's input locations (all, signed int, unsigned int); a pixel shader's output locations
        hdr3 = (local or jobs[k].get('local') or [1, 1, 1]) if st in ('cs', 'ts', 'ms') else io[1] if st == 'vs' else io[3] if st == 'ps' else [0, 0, 0]
        entries.append((int(k, 16), STAGES.get(st, 0), spv, refl_bin(cl, c1s[k], hdr3, first, ordered)))
    if SPIRV_OPT:   # after the reflection and first_store, which read the translator's form
        def opt(e):
            p = os.path.join(w, f'{e[0]:016x}.opt.spv'); open(p, 'wb').write(e[2])
            r = subprocess.run(['spirv-opt', '-O', '--preserve-bindings', '--preserve-interface', '--target-env=vulkan1.3', p, '-o', p + '.o'], capture_output=True)
            ok = not r.returncode and not subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', p + '.o'], capture_output=True).returncode
            return (e[0], e[1], open(p + '.o', 'rb').read() if ok else e[2], e[3]), ok
        with cf.ThreadPoolExecutor(a.jobs) as ex: done = list(ex.map(opt, entries))
        entries = [d[0] for d in done]
        print(f'spirv-opt (DM_SHADER_SPIRV_OPT): {sum(d[1] for d in done)} of {len(done)} modules optimized, the others as translated', flush=True)
    if a.spv_dir:
        for key, st, spv, refl in entries: open(os.path.join(a.spv_dir, f'{key:016x}.spv'), 'wb').write(spv)
    entries.sort(key=lambda e: e[0])
    hdr = struct.pack('<4sII', b'DMSV', 1, len(entries)); ent_sz = 32
    off = len(hdr) + ent_sz * len(entries); table, blobs = [], []
    for key, st, spv, refl in entries:
        table.append(struct.pack('<QIIIIII', key, st, off, len(spv), off + len(spv), len(refl), 0)); blobs += [spv, refl]; off += len(spv) + len(refl)
    with open(a.out, 'wb') as f: f.write(hdr + b''.join(table) + b''.join(blobs))
    print(f'{len(entries)} of {len(keys)} shaders → {a.out} ({off / 2**20:.1f} MB); failed: {dict(fails)}; binding kinds: {dict(kinds_all)}')
    print('per stage: ' + ', '.join(f'{st} {c["translated"]} translated, {c["failed"]} failed' for st, c in sorted(per_stage.items(), key=lambda kv: STAGES.get(kv[0], 9))))
    if a.vt: print(f'virtual texturing (spv_vt.py): {vt_stats["shaders"]} fragment shaders, {vt_stats["candidates"]} candidate 2D arrays, {vt_stats["without requests"]} shaders without requests (a discard or a depth write)')
    print(f'first-invocation stores made by the replay (first_store): {len(firsts)} compute shaders' + ''.join(f'\n  {k}: ' + ', '.join(f'slot {e["reg"]} word {e["word"]} = {e["value"]}' for e in v) for k, v in sorted(firsts.items())))
    print(f'ordered allocations (ordered_alloc): {len(ordereds)} compute shaders' + ''.join(f'\n  {k}' for k in sorted(ordereds)))
    for e, n in errs.most_common(8): print(f'  translator: {n:5d}  {e}')
    shutil.rmtree(w, ignore_errors=True)

main()
