## Full docker

```bash
docker build --tag tmponnxleak .
docker run -it --rm tmponnxleak
```

## Run locally

### prepare onnx deps

```bash
docker build . --target=downloader --tag=onnx-downloader
docker run -it --rm -v "$(pwd)"/:/hostlib onnx-downloader cp -r /onnxruntime /hostlib/.
```

### run service local

```bash
OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$(pwd)"/onnxruntime/lib go run .
```
