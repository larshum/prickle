import argparse
import json
import numpy as np
import torch
import parpy
import sys

torch.manual_seed(1234)

T = parpy.types.type_var()
M = parpy.types.shape_var()
N = parpy.types.shape_var()

@parpy.jit
def softmax(
    x: parpy.types.buffer(T, [M, N]),
    y: parpy.types.buffer(T, [M, N]),
):
    for row in range(M):
        m = parpy.reduce.max(x[row,:N])
        y[row,:N] = parpy.math.exp(x[row,:N] - m)
        s = parpy.reduce.sum(y[row,:N])
        y[row,:N] /= s

def matches_compute_kernel_names(entry):
    entry_name = entry["Name"]
    if entry_name.find("softmax") >= 0:
        return True
    if entry_name.find("transpose") >= 0:
        return True
    return False

def parse_performance(json_file):
    with open(json_file, "r") as f:
        time = 0.0
        for entry in json.loads(f.read()):
            if matches_compute_kernel_names(entry):
                time += entry["Med (ns)"]
        return time

command = sys.argv[1]

if command == "generate-input":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-file", type=str, required=True)
    parser.add_argument("-n", type=int, required=True)
    args = parser.parse_args(sys.argv[2:])

    M = 100
    N = args.n
    x = torch.randn((M, N), dtype=torch.float32)
    y = torch.empty_like(x)

    np.set_printoptions(threshold=sys.maxsize, linewidth=1<<30)
    with open(args.output_file, "w+") as f:
        f.write(np.array2string(np.asarray(x), separator=", "))
elif command == "run":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-file", type=str, required=True)
    parser.add_argument("-n", type=int, required=True)
    parser.add_argument("-r", "--repeats", type=int, default=100)
    args = parser.parse_args(sys.argv[2:])

    M = 100
    N = args.n
    x = torch.randn((M, N), dtype=torch.float32, device='cuda')
    y = torch.empty_like(x)

    p = {
        'M': parpy.threads(M),
        'N': parpy.threads(512),
    }
    opts = parpy.par(p)

    for i in range(args.repeats):
        softmax(x, y, opts=opts)

    np.set_printoptions(threshold=sys.maxsize, linewidth=1<<30)
    with open(args.output_file, "w+") as f:
        f.write(np.array2string(np.asarray(y.cpu()), separator=" "))
elif command == "print-results":
    print("\\begin{tabular}{l|cc}")
    print("Columns & \\frameworkname & Futhark\\\\")
    print("\\hline")
    for size in [int(x) for x in sys.argv[2:]]:
        parpy_results = int(parse_performance(f"data/parpy-softmax-{size}.json"))
        futhark_results = int(parse_performance(f"data/futhark-softmax-{size}.json"))
        print(f"{size} & {parpy_results} ns & {futhark_results} ns\\\\")
    print("\\end{tabular}")
else:
    sys.stderr.write(f"Unsupported command {command}")
    sys.exit(1)
