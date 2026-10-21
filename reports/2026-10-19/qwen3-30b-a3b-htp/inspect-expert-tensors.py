#!/usr/bin/env python3
"""Read GGUF metadata and MoE tensor headers without paging in tensor data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct

TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
    40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
}
SUPPORTED_HTP_MUL_MAT_ID = {"Q4_0", "Q4_1", "Q8_0", "IQ4_NL", "MXFP4",
                            "Q4_K", "Q5_K", "Q6_K"}
SCALARS = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
           4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
           10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}


class Reader:
    def __init__(self, path):
        self.f = Path(path).open("rb")

    def unpack(self, fmt):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise ValueError("unexpected end of GGUF header")
        return struct.unpack(fmt, data)[0]

    def string(self):
        size = self.unpack("<Q")
        if size > 64 * 1024 * 1024:
            raise ValueError(f"unreasonable GGUF string length: {size}")
        data = self.f.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF string")
        return data.decode("utf-8", errors="replace")

    def skip_value(self, kind, capture=False):
        if kind == 8:
            value = self.string()
            return value if capture else None
        if kind == 9:
            element_kind = self.unpack("<I")
            count = self.unpack("<Q")
            if count > 100_000_000:
                raise ValueError("unreasonable GGUF array length")
            if element_kind in SCALARS:
                self.f.seek(SCALARS[element_kind][1] * count, 1)
            elif element_kind == 8:
                for _ in range(count):
                    self.string()
            else:
                raise ValueError(f"unsupported GGUF array type {element_kind}")
            return None
        if kind not in SCALARS:
            raise ValueError(f"unsupported GGUF metadata type {kind}")
        fmt, size = SCALARS[kind]
        data = self.f.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF metadata")
        return struct.unpack(fmt, data)[0] if capture else None


def inspect(path):
    r = Reader(path)
    if r.f.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = r.unpack("<I")
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    tensor_count = r.unpack("<Q")
    metadata_count = r.unpack("<Q")
    metadata = {}
    important = {"general.architecture", "general.file_type"}
    for _ in range(metadata_count):
        key = r.string()
        kind = r.unpack("<I")
        value = r.skip_value(kind, key in important or key.endswith(".expert_count")
                             or key.endswith(".expert_used_count"))
        if value is not None:
            metadata[key] = value
    expert_types = Counter()
    expert_tensors = []
    for _ in range(tensor_count):
        name = r.string()
        n_dims = r.unpack("<I")
        if n_dims > 8:
            raise ValueError(f"invalid tensor dimension count for {name}: {n_dims}")
        shape = [r.unpack("<Q") for _ in range(n_dims)]
        tensor_type = r.unpack("<I")
        r.unpack("<Q")
        if "expert" in name.lower() or "_exps" in name.lower():
            name_type = TYPE_NAMES.get(tensor_type, f"TYPE_{tensor_type}")
            expert_types[name_type] += 1
            expert_tensors.append({"name": name, "shape": shape, "type": name_type})
    unknown = sorted({entry["type"] for entry in expert_tensors
                      if entry["type"] not in SUPPORTED_HTP_MUL_MAT_ID})
    return {
        "file": str(Path(path)), "bytes": Path(path).stat().st_size,
        "gguf_version": version, "tensor_count": tensor_count,
        "architecture": metadata.get("general.architecture"),
        "file_type": metadata.get("general.file_type"),
        "expert_count": metadata.get(f"{metadata.get('general.architecture')}.expert_count"),
        "expert_used_count": metadata.get(f"{metadata.get('general.architecture')}.expert_used_count"),
        "expert_tensor_count": len(expert_tensors),
        "expert_tensor_types": dict(sorted(expert_types.items())),
        "unsupported_expert_types": unknown,
        "unsupported_examples": [entry for entry in expert_tensors
                                 if entry["type"] in unknown][:32],
        "htp_expert_type_gate": "pass" if expert_tensors and not unknown else "fail",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = inspect(args.model)
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
