import argparse
import json
import numpy as np
import parpy
from parpy.types import F32, I32
import torch
import sys

torch.manual_seed(1234)

nx = parpy.types.shape_var()
ny = parpy.types.shape_var()
nz = parpy.types.shape_var()

@parpy.jit
def next(
    c0: F32,
    c1: F32,
    A0: parpy.types.buffer(F32, [nx, ny, nz]),
    i: I32,
    j: I32,
    k: I32
) -> F32:
    if i == 0 or i == nx-1 or j == 0 or j == ny-1 or k == 0 or k == nz-1:
        return A0[i,j,k]
    else:
        return (A0[i,j,k+1] + A0[i,j,k-1] + A0[i,j-1,k] + A0[i,j+1,k] + A0[i-1,j,k] + A0[i+1,j,k]) * c1 + A0[i,j,k] * c0

@parpy.jit
def stencil(
    iterations: I32,
    A0: parpy.types.buffer(F32, [nx, ny, nz]),
):
    c0 = parpy.builtin.convert(1.0 / 6.0, F32)
    c1 = parpy.builtin.convert(1.0 / 6.0 / 6.0, F32)
    for _iter in range(iterations):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    A0[i,j,k] = next(c0, c1, A0, i, j, k)

def matches_compute_kernels(entry, iterations):
    entry_name = entry["Name"]
    if entry_name.find("parpy_stencil") >= 0:
        return True
    if entry_name.find("segmap") >= 0 and entry["Instances"] == iterations:
        return True
    return False

def parse_performance(json_file, iterations):
    with open(json_file, "r") as f:
        time = 0.0
        for entry in json.loads(f.read()):
            if matches_compute_kernels(entry, iterations):
                time += entry["Med (ns)"]
        return time

command = sys.argv[1]

if command == "generate-input":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-file", type=str, required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args(sys.argv[2:])

    buffer = torch.randn((args.ny, args.nz), dtype=torch.float32)

    np.set_printoptions(threshold=sys.maxsize, linewidth=1<<30)
    with open(args.output_file, "w+") as f:
        f.write(f"{args.iterations}\n{args.nx}\n")
        f.write(np.array2string(np.asarray(buffer), separator=', '))
elif command == "run":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args(sys.argv[2:])

    buffer = torch.randn((args.ny, args.nz), dtype=torch.float32)

    # Expand by repeating each [ny, nz] matrix nx times
    expanded = buffer.unsqueeze(0).expand(args.nx, -1, -1).cuda()

    p = {
        'nx': parpy.threads(args.nx),
        'ny': parpy.threads(args.ny),
        'nz': parpy.threads(args.nz),
    }
    stencil(args.iterations, expanded, opts=parpy.par(p))

    np.set_printoptions(threshold=sys.maxsize, linewidth=1<<30)
    with open(args.output_file, "w+") as f:
        f.write(np.array2string(np.asarray(expanded.cpu()), separator=" "))
elif command == "print-results":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args(sys.argv[2:4])

    print("\\begin{tabular}{l|cc}")
    print("Problem set & \\frameworkname & Futhark\\\\")
    print("\\hline")
    for nx in [int(x) for x in sys.argv[4:]]:
        parpy_results = int(parse_performance(f"data/parpy-stencil-{nx}.json", args.iterations))
        futhark_results = int(parse_performance(f"data/futhark-stencil-{nx}.json", args.iterations))
        label = "Default" if nx == 512 else "Small"
        print(f"{label} & {parpy_results} ns & {futhark_results} ns\\\\")
    print("\\end{tabular}")
else:
    sys.stderr.write(f"Unsupported command {command}")
    sys.exit(1)
