
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

#WORKDIR /app
#RUN mkdir -p onnx/lib/onnxruntime
#COPY --from=downloader /onnxruntime /onnxruntime
#ADD go.* ./
#RUN go mod download
#RUN go mod verify
#ADD ./ /app
#RUN cp -r /onnxruntime onnx/lib/.
##RUN LD_LIBRARY_PATH=/onnxruntime/lib/ go test -race -count=1 -p 1 ./...
#RUN go build -o recorabanne cmd/main.go
#
#FROM busybox:1.31-glibc
#
#COPY --from=downloader /etc/ssl/certs /etc/ssl/certs
#
#WORKDIR /lib
#COPY --from=builder /lib/x86_64-linux-gnu/libdl.so.2 .
#COPY --from=builder /lib/x86_64-linux-gnu/librt.so.1 .
#COPY --from=builder /lib/x86_64-linux-gnu/libgcc_s.so.1 .
#COPY --from=builder /usr/lib/x86_64-linux-gnu/libstdc++.so.6 .
#COPY --from=builder /usr/lib/x86_64-linux-gnu/libgomp.so.1 .
#
#WORKDIR /app
#COPY --from=builder /onnxruntime/lib/* ./onnx/lib/
#COPY --from=builder /app/recorabanne .
#
#EXPOSE 8080
#ENV LD_LIBRARY_PATH /app/onnx/lib/
#ENV OMP_WAIT_POLICY PASSIVE
#ENV OMP_NUM_THREADS 1
#ENV GIN_MODE release
#CMD ["/app/recorabanne"]
