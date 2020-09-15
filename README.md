## Run with docker

```bash
docker build --tag tmponnxleak .
docker run -it --rm tmponnxleak
```

## Run local go

### get onnx dependencies

```bash
docker build . --target=downloader --tag=onnx-downloader
docker run -it --rm -v "$(pwd)"/:/hostlib onnx-downloader cp -r /onnxruntime /hostlib/.
```

### run service local

```bash
OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$(pwd)"/onnxruntime/lib go run .
```


# Github Issue

## Possible Memory leak over sessions released

I **can't achieve to cap memory consumption** in my work flow.

I create and release sessions over time, from a single **long running process** (an HTTP real-time serving application), in order to infer on **daily** retrained models.

I create **one new session per new trained model**

**Outdated model sessions are closed** when replaced by newer ones, but **memory usage keep growing**.

I tried to replicate my work flow in **[a demo app](https://github.com/glutamatt/onnxleak)** : we can see memory increasing (despite go runtime memory is constant)

```
#1 New Session (6 threads, 10 MB resident, 71 MB go sys)
#2 New Session (7 threads, 46 MB resident, 73 MB go sys)
#3 New Session (22 threads, 55 MB resident, 73 MB go sys)
#4 New Session (22 threads, 62 MB resident, 73 MB go sys)
#5 New Session (25 threads, 65 MB resident, 73 MB go sys)
...
#20 New Session (28 threads, 136 MB resident, 73 MB go sys)
...
#50 New Session (30 threads, 238 MB resident, 74 MB go sys)
...
#100 New Session (35 threads, 292 MB resident, 74 MB go sys)
...
#200 New Session (43 threads, 301 MB resident, 74 MB go sys)
...
#500 New Session (50 threads, 321 MB resident, 74 MB go sys)
...
#750 New Session (50 threads, 326 MB resident, 74 MB go sys)
...
#1000 New Session (55 threads, 336 MB resident, 74 MB go sys)
```

> here is running one session after the other (open, infer, close, open, infer, close, ...) with 10000 concurrent prediction for each session
>
> the model is always the same (a [3MB onnx file](https://github.com/glutamatt/onnxleak/blob/fd18756faf292bea257561e6b0a83d50296a98b6/model.onnx), with 3 hidden dense layers, 1 input (shape [1, 1415]) , 1 output (shape [1, 100]))
>
> the input is always the [same "zero filled"](https://github.com/glutamatt/onnxleak/blob/fd18756faf292bea257561e6b0a83d50296a98b6/main.go#L37) vector

