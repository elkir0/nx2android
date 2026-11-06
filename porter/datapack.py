# The data pack: everything the app reads from its files folder on the device (/sdcard/Android/data/<package>/files, front_android.c), as a folder beside the APK:
#   mods/          the rewritten code modules (patch_nso.py's outputs)
#   RomFS/         the game's files (the packed version's merged RomFS), or
#   RomFS.img      the same as one image (dm/tools/romfs_image.py: a NUP's version is unpacked into one), which the app takes for DM_ROMFS when it is there
#   aoc/<index>/   the DLC the player has (aoc/<index>.img: a NUP's)
#   shaders.dmsv   the Vulkan shaders (dm/android/tools/shader_vk.py)
#   nintendo_ext.ttf  the button-icon font, the stand-in for the Nintendo extension shared font (dm/tools/make_ext_font.py; env.txt's DM_SHARED_FONT_EXT)
#   env.txt        host switches: the game language, the DLC folder, the profile's env rows, then the port's (--fps60, --mt-replay)
#   settings.txt   the pause menu's first-launch defaults (the profile's setting rows, then the port's: --touch); the installer never replaces the player's own
#   nx2a-data.json what the pack is: the game, the package, the version, the device features the profile requires (android.tsv requires), which the installer checks
# Big trees and images are hard links into the extraction when the volume allows, so the pack takes no extra space; the pack is rebuilt whole when anything in it changes.
import json, os, shutil, time
from pathlib import Path

from porter.upstream import pack_nup

MANIFEST = 'nx2a-data.json'
KEEP_ON_DEVICE = ('settings.txt',)          # the player's choices win over the pack's defaults
OWNED = ('mods', 'RomFS', 'RomFS.img', 'aoc')   # what the pack owns on the device: files there that the pack lacks are stale (an older version's, a RomFS tree an image replaced) and removed
GAME_FILES = ('RomFS', 'RomFS.img', 'aoc')  # the game's own files, which change only with its version: the install compares them by size alone (never an MD5 of a 23 GB image)
STAMP = '.inputs.json'                      # beside an image: romfs_image.py's reuse stamp, not a game file


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try: os.link(src, dst)
    except OSError: shutil.copy2(src, dst)


def link_tree(src, dst):   # (files, bytes)
    n = b = 0
    for root, _, files in os.walk(src):
        for f in files:
            s = Path(root) / f
            link_or_copy(s, dst / s.relative_to(src)); n += 1; b += s.stat().st_size
    return n, b


def renew_stamp(img):
    """The reuse stamp of a RomFS image (<image>.inputs.json: romfs_image.py publish, the source's digest and the image's device, inode, size, mtime and ctime) made to hold again
    when only the ctime differs: the data pack's hard link to the image, and its removal with an older pack, change the ctime and nothing else, and the next port would write the
    image again (23 GB for Mario Kart World). A file rewritten in place has another mtime, a new one another inode: those stamps stay as they are"""
    stamp, ident = Path(f'{img}{STAMP}'), pack_nup.romfs_image.file_identity
    try: st = json.loads(stamp.read_text()); now = ident(img)
    except (OSError, ValueError): return
    if isinstance(st, dict) and isinstance(st.get('image'), list) and st['image'][:4] == now[:4] and st['image'] != now:
        part = stamp.with_name(stamp.name + '.part'); part.write_text(json.dumps({**st, 'image': now})); os.replace(part, stamp)


def link_image(src, dst, linked):   # a RomFS image (RomFS.img, aoc/<index>.img) into the pack, as link_or_copy does; linked: the images, whose stamps build() renews at its end. Returns the bytes
    link_or_copy(src, dst); linked.append(src)
    return Path(src).stat().st_size


def link_game(src, dst, linked):   # a RomFS (an image or a folder) or aoc/ (folders and images, their stamps left out) into the pack: (files, bytes)
    if src.is_file(): return 1, link_image(src, dst, linked)
    n = b = 0
    for p in sorted(src.iterdir()) if dst.name == 'aoc' else [src]:
        if p.name.endswith((STAMP, '.dump.key')) or p.name.startswith('.'): continue   # the extraction's reuse stamps (romfs_image.py, porter.dumpdir)
        if p.is_file(): n += 1; b += link_image(p, dst / p.name, linked)
        else: dn, db = link_tree(p, dst if p == src else dst / p.name); n += dn; b += db
    return n, b


def files_folder(package): return f'/sdcard/Android/data/{package}/files'


def build(dst, *, profile, version, display, patched, romfs, shaders, aoc=None, language='en-US', package=None, env=(), settings=None, files=None, progress=None):
    """Write the pack to dst (replacing it). profile: the AndroidProfile; package: the app's, when not the profile's; env: more env.txt rows, after the profile's; settings: settings.txt values over the profile's;
    files: {name in the pack: source} (the 60 fps memory writes, fpslock.tsv; the button-icon font). Returns the manifest"""
    package = package or profile.package
    tmp = dst.with_name(dst.name + '.partial')
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    n = b = 0; linked = []
    for what, src, sub in (('the rewritten modules', patched, 'mods'), ('the RomFS', romfs, 'RomFS.img' if romfs and romfs.is_file() else 'RomFS'), ('the DLC', aoc, 'aoc')):
        if not src: continue
        if progress: progress(what)
        dn, db = link_tree(src, tmp / sub) if sub == 'mods' else link_game(src, tmp / sub, linked); n += dn; b += db
    link_or_copy(shaders, tmp / 'shaders.dmsv'); n += 1; b += shaders.stat().st_size
    for name, src in (files or {}).items(): shutil.copyfile(src, tmp / name); n += 1; b += Path(src).stat().st_size
    rows = [f'# Written by nx2android\'s porter for {profile.name} ({profile.id}); KEY=VALUE per line, read by the app at launch',
            f'DM_LANG={language}']
    if aoc: rows.append(f'DM_AOC_DIR={files_folder(package)}/aoc')
    rows += profile.env + list(env)
    (tmp / 'env.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8', newline='\n')
    if sets := {**profile.settings, **(settings or {})}: (tmp / 'settings.txt').write_text(''.join(f'{k}={v}\n' for k, v in sets.items()), encoding='utf-8', newline='\n')
    man = dict(title_id=profile.id, name=profile.name, package=package, version=version, display=display, language=language, requires=profile.requires,
               files=n, bytes=b, romfs='image' if romfs and romfs.is_file() else 'folder',
               dlc=sorted(int(p.name.removesuffix('.img')) for p in aoc.iterdir() if (p.is_dir() or p.suffix == '.img') and p.name.removesuffix('.img').isdigit()) if aoc else [],
               made=time.strftime('%Y-%m-%d %H:%M:%S'))
    (tmp / MANIFEST).write_text(json.dumps(man, indent=2), encoding='utf-8')
    if dst.exists(): shutil.rmtree(dst)
    for i in range(20):   # Windows: a folder just removed can stay delete-pending while another process (an indexer, a virus scan) holds it, and the rename is refused meanwhile
        try: tmp.rename(dst); break
        except PermissionError:
            if i == 19: raise
            time.sleep(0.5)
    for img in linked: renew_stamp(img)   # after the older pack's links went too
    return man


def read_manifest(d):
    try: return json.loads((Path(d) / MANIFEST).read_text(encoding='utf-8'))
    except (OSError, ValueError): return None
