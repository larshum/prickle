run: build
	podman run --rm -it --volume ../..:/root/custom --device "nvidia.com/gpu=all" futhark-container "bash"

build:
	podman build --format docker --rm -t futhark-container .

clean:
	rm -f *.sqlite *.nsys-rep *.txt *.json *.h softmax.c stencil.c softmax stencil
	rm -rf data/
