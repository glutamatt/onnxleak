## Run locally

### prepare onnx deps

```bash
docker build . --target=downloader --tag=onnx-downloader
docker run -it --rm -v "$(pwd)"/:/hostlib onnx-downloader cp -r /onnxruntime /hostlib/.
```

### run service

```bash
LD_LIBRARY_PATH="$(pwd)"/onnxruntime/lib go run main.go
