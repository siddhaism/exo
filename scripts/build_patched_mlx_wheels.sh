#!/bin/zsh
set -euo pipefail

if (( $# != 3 )); then
  print -u2 "usage: $0 MLX_CHECKOUT OUTPUT_DIR DEVELOPER_DIR"
  exit 2
fi

mlx_checkout=$1
output_dir=$2
developer_dir=$3

if [[ ! -f "$mlx_checkout/setup.py" || ! -d "$mlx_checkout/.git" ]]; then
  print -u2 "MLX_CHECKOUT must be a clean MLX git checkout"
  exit 2
fi
if [[ -n "$(git -C "$mlx_checkout" status --porcelain)" ]]; then
  print -u2 "MLX checkout is dirty; refusing a non-reproducible build"
  exit 2
fi
if [[ ! -x "$developer_dir/usr/bin/xcodebuild" ]]; then
  print -u2 "DEVELOPER_DIR does not contain an Xcode toolchain"
  exit 2
fi

mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
mlx_commit=$(git -C "$mlx_checkout" rev-parse HEAD)
xcode_version=$("$developer_dir/usr/bin/xcodebuild" -version | tr '\n' ' ')

export DEVELOPER_DIR="$developer_dir"
export PYPI_RELEASE=1
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-8}

cd "$mlx_checkout"
python setup.py clean --all
MLX_BUILD_STAGE=2 python setup.py bdist_wheel --dist-dir "$output_dir"
python setup.py clean --all
MLX_BUILD_STAGE=1 python setup.py bdist_wheel --dist-dir "$output_dir"

python - "$output_dir" "$mlx_commit" "$xcode_version" <<'PY'
import hashlib
import json
import pathlib
import sys
import zipfile

output = pathlib.Path(sys.argv[1])
mlx_commit = sys.argv[2]
xcode_version = sys.argv[3]
wheels = sorted(output.glob("mlx*.whl"))
if len(wheels) != 2:
    raise SystemExit(f"expected exactly two MLX wheels, found {len(wheels)}")

manifest = {
    "mlxCommit": mlx_commit,
    "xcodeVersion": xcode_version,
    "pythonVersion": sys.version,
    "wheels": [],
}
for wheel in wheels:
    names = zipfile.ZipFile(wheel).namelist()
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    manifest["wheels"].append(
        {
            "filename": wheel.name,
            "sha256": digest,
            "hasCore": any(name.endswith("core.cpython-313-darwin.so") for name in names),
            "hasLibmlx": any(name.endswith("libmlx.dylib") for name in names),
            "hasLibjaccl": any(name.endswith("libjaccl.dylib") for name in names),
            "hasMetallib": any(name.endswith("mlx.metallib") for name in names),
        }
    )

(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY
