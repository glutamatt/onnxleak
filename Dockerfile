
# Microsoft Nuget package extract
FROM ubuntu:bionic as downloader
ARG ORT_VERSION=1.4.0
RUN apt-get update && apt-get install -y wget atool
RUN mkdir /onnxruntime
WORKDIR /onnxruntime
RUN wget -nv -O microsoft.ml.onnxruntime.nupkg https://globalcdn.nuget.org/packages/microsoft.ml.onnxruntime.${ORT_VERSION}.nupkg
RUN aunpack microsoft.ml.onnxruntime.nupkg
RUN mkdir -p /onnxruntime/include
RUN mkdir -p /onnxruntime/lib
RUN cp microsoft.ml.onnxruntime/runtimes/linux-x64/native/libonnxruntime.so /onnxruntime/lib/.
RUN cp microsoft.ml.onnxruntime/build/native/include/*.h /onnxruntime/include/.
RUN cd /onnxruntime/lib/ && ln -s libonnxruntime.so libonnxruntime.so.${ORT_VERSION}
RUN rm -fr microsoft.ml.onnxruntime microsoft.ml.onnxruntime.nupkg

FROM golang:1.15-buster as builder
WORKDIR /app
COPY --from=downloader /onnxruntime onnxruntime
COPY model.onnx .
ADD onnxruntime_go_api.* ./
ADD ./ /app

ENV OMP_WAIT_POLICY PASSIVE
ENV OMP_NUM_THREADS 1
ENV LD_LIBRARY_PATH=/app/onnxruntime/lib

CMD go run .
