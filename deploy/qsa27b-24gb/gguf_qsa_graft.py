"""Add QSA indexers and metadata to a Qwen3.8-27B GGUF, leaving every existing tensor byte-identical. No dependencies.
Follows the QSA converter: indexers stay F32, norms are stored as 1 + weight, ratios are 4 on full-attention layers.
usage: gguf_qsa_graft.py <in.gguf> <qsa-indexers.safetensors> <out.gguf>    Refuses to overwrite <out.gguf>."""
import json, os, struct, sys
src, idx, out = sys.argv[1:4]
if os.path.exists(out): raise SystemExit(f"refusing to overwrite {out}")
f = open(src, "rb"); assert f.read(4) == b"GGUF"
ver, nt, nkv = struct.unpack("<IQQ", f.read(20))
def rs(): return f.read(struct.unpack("<Q", f.read(8))[0])
SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}; FMT = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
kv0 = f.tell(); align = 32; keys = {}
for _ in range(nkv):
    k = rs().decode(); t = struct.unpack("<I", f.read(4))[0]
    if t == 8: keys[k] = rs()
    elif t == 9:
        et, n = struct.unpack("<IQ", f.read(12))
        if et == 8:
            for _ in range(n): rs()
        else: f.seek(SZ[et]*n, 1)
        keys[k] = None
    else:
        keys[k] = struct.unpack("<"+FMT[t], f.read(SZ[t]))[0]
align = keys.get("general.alignment", 32); kv1 = f.tell()
assert keys["general.architecture"] == b"qwen35" and keys["qwen35.block_count"] == 65, "expected the -mtp qwen35 GGUF"
for k in keys:
    assert "indexer" not in k and "compress_ratios" not in k, f"already has QSA metadata: {k}"
T = []
for _ in range(nt):
    n = rs(); nd = struct.unpack("<I", f.read(4))[0]; d = struct.unpack(f"<{nd}Q", f.read(8*nd)); ty, off = struct.unpack("<IQ", f.read(12)); T.append([n, d, ty, off])
base = (f.tell()+align-1)//align*align; end = os.path.getsize(src)
order = sorted(range(nt), key=lambda i: T[i][3])
for a, b in zip(order, order[1:]+[None]): T[a].append((T[b][3] if b is not None else end-base) - T[a][3])
s = open(idx, "rb"); l = struct.unpack("<Q", s.read(8))[0]; hdr = json.loads(s.read(l)); s0 = 8+l
MAP = {"q_proj.weight": "q_proj.weight", "k_proj.weight": "k_proj.weight", "q_norm": "q_norm.weight", "k_norm": "k_norm.pre_weight", "k_block_norm": "k_norm.weight"}
NEW = []
for name, v in hdr.items():
    if name == "__metadata__": continue
    assert v["dtype"] == "F32", name
    pre, suf = name.split("._qsa_indexer."); layer = int(pre.split(".")[3]); assert (layer+1) % 4 == 0
    a, b = v["data_offsets"]; s.seek(s0+a); raw = s.read(b-a)
    if suf.endswith("norm"):
        cnt = len(raw)//4; raw = struct.pack(f"<{cnt}f", *[x+1 for x in struct.unpack(f"<{cnt}f", raw)])  # double then round is exact for one add
    NEW.append((f"blk.{layer}.indexer.{MAP[suf]}".encode(), tuple(reversed(v["shape"])), 0, raw))
assert len(NEW) == 80, len(NEW)
def kv_u32(k, v): b = k.encode(); return struct.pack("<Q", len(b))+b+struct.pack("<II", 4, v)
ratios = [4 if (i+1) % 4 == 0 and i < 64 else 0 for i in range(65)]
kr = b"qwen35.attention.compress_ratios"
extra = kv_u32("qwen35.attention.indexer.head_count", 4)+kv_u32("qwen35.attention.indexer.key_length", 128)+kv_u32("qwen35.attention.indexer.top_k", 2048)
extra += struct.pack("<Q", len(kr))+kr+struct.pack("<IIQ", 9, 5, 65)+struct.pack("<65i", *ratios)
pad = lambda n: (-n) % align
rows, o = [], 0
for n, d, ty, off, sz in T: rows.append((n, d, ty, o)); o += sz
newoff = []
for n, d, ty, raw in NEW: newoff.append(o); rows.append((n, d, ty, o)); o += len(raw)+pad(len(raw))
with open(out, "wb") as w:
    w.write(b"GGUF"+struct.pack("<IQQ", ver, nt+len(NEW), nkv+4))
    f.seek(kv0); w.write(f.read(kv1-kv0)); w.write(extra)
    for n, d, ty, a in rows: w.write(struct.pack("<Q", len(n))+n+struct.pack("<I", len(d))+struct.pack(f"<{len(d)}Q", *d)+struct.pack("<IQ", ty, a))
    w.write(b"\0"*pad(w.tell())); data0 = w.tell()
    for n, d, ty, off, sz in T:
        f.seek(base+off); left = sz
        while left:
            b = f.read(min(left, 1 << 24)); w.write(b); left -= len(b)
    for n, d, ty, raw in NEW: w.write(raw); w.write(b"\0"*pad(len(raw)))
print(out, os.path.getsize(out)/1e9, "GB;", nt+len(NEW), "tensors, data at", data0)
