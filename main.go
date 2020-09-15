package main

// #cgo CFLAGS: -I${SRCDIR}/onnxruntime/include
// #cgo LDFLAGS: -L${SRCDIR}/onnxruntime/lib -lonnxruntime
// #include <onnxruntime/include/onnxruntime_c_api.h>
// #include <onnxruntime_go_api.h>
import "C"
import (
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"unsafe"
)

//single shared references for the whole process
var ortAPI *C.OrtApi
var ortEnv *C.OrtEnv

func main() {
	ortAPI = C.GetOrtApi()
	checkStatus(C.CreateEnv(ortAPI, C.ORT_LOGGING_LEVEL_ERROR, C.CString("main_env"), &ortEnv))

	var sessionOptions *C.OrtSessionOptions
	checkStatus(C.CreateSessionOptions(ortAPI, &sessionOptions))
	checkStatus(C.SetSessionGraphOptimizationLevel(ortAPI, sessionOptions, 0))
	checkStatus(C.DisableCpuMemArena(ortAPI, sessionOptions))
	checkStatus(C.DisableMemPattern(ortAPI, sessionOptions))

	modelPath := C.CString("model.onnx")
	inputNames := []*C.char{C.CString("input.1")}
	outputNames := []*C.char{C.CString("46")}
	inputShape := []int64{1, 1415}
	inputSlice := make([]float32, inputShape[1])
	inputTensorLen := C.ulong(inputShape[1] * 4) //4 bytes per float
	outputLength := 100
	predPerSession := 10000
	sessionCounter := 0

	for { // create a new session, run {predPerSession} concurrent predictions and close the session
		sessionCounter++
		threadCount, memory, goMem := stats()
		fmt.Printf("#%d New Session (%d threads, %d MB resident, %d MB go sys)\n", sessionCounter, threadCount, memory, goMem)

		var clonedSessionOptions *C.OrtSessionOptions
		checkStatus(C.CloneSessionOptions(ortAPI, sessionOptions, &clonedSessionOptions))
		var session *C.OrtSession
		checkStatus(C.CreateSession(ortAPI, ortEnv, modelPath, clonedSessionOptions, &session))

		waitGroup := sync.WaitGroup{}
		waitGroup.Add(predPerSession)

		for predCounter := 0; predCounter < predPerSession; predCounter++ {
			go func(mustPrintPredictionSamples bool) {
				var inputTensor *C.OrtValue
				var outputTensor *C.OrtValue
				var memoryInfo *C.OrtMemoryInfo
				checkStatus(C.CreateCpuMemoryInfo(ortAPI, C.OrtDeviceAllocator, C.OrtMemTypeDefault, &memoryInfo))
				checkStatus(C.CreateTensorWithDataAsOrtValue(
					ortAPI,
					memoryInfo,
					unsafe.Pointer(&inputSlice[0]),
					inputTensorLen,
					(*C.long)(unsafe.Pointer(&inputShape[0])),
					2, //number of dimensions
					C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
					&inputTensor,
				))
				checkStatus(C.Run(
					ortAPI,
					session,
					nil,
					(**C.char)(unsafe.Pointer(&inputNames[0])),
					(**C.OrtValue)(unsafe.Pointer(&inputTensor)),
					1, //one input
					(**C.char)(unsafe.Pointer(&outputNames[0])),
					1, //one output
					&outputTensor,
				))
				resPointer := unsafe.Pointer(nil)
				checkStatus(C.GetTensorMutableData(ortAPI, outputTensor, &resPointer))
				if mustPrintPredictionSamples {
					predictionsNoCopy := floatSlice(resPointer, outputLength)
					fmt.Printf("sample predictions from nocopy: %v ... %v\n", predictionsNoCopy[:3], predictionsNoCopy[outputLength-3:])
				}
				C.ReleaseMemoryInfo(ortAPI, memoryInfo)
				C.ReleaseValue(ortAPI, inputTensor)
				C.ReleaseValue(ortAPI, outputTensor)
				waitGroup.Done()
			}(sessionCounter%50 == 0 && (predCounter%(predPerSession/10)) == 0)
		}

		waitGroup.Wait()
		C.ReleaseSession(ortAPI, session)
		C.ReleaseSessionOptions(ortAPI, clonedSessionOptions)
	}
}

func floatSlice(dataPointer unsafe.Pointer, size int) (slice []float32) {
	predHeader := (*reflect.SliceHeader)(unsafe.Pointer(&slice))
	predHeader.Data = uintptr(dataPointer)
	predHeader.Len = size
	predHeader.Cap = size
	return
}

func checkStatus(status *C.OrtStatus) {
	if status != nil {
		ortError := C.GoString(C.GetErrorMessage(ortAPI, status))
		C.ReleaseStatus(ortAPI, status)
		ortErrorLocation := ""
		if _, fileName, fileLine, ok := runtime.Caller(1); ok {
			ortErrorLocation = fmt.Sprintf("at %s:%d ", fileName, fileLine)
		}
		panic(fmt.Errorf("onnx runtime error %s: '%s'", ortErrorLocation, ortError))
	}
}

func stats() (threads, mem int, goMem int) {
	stat, _ := ioutil.ReadFile(fmt.Sprintf("/proc/%d/stat", os.Getpid()))
	threadsPos := 19
	memPos := 23
	parts := strings.SplitN(string(stat), " ", memPos+1)
	fmt.Sscan(parts[threadsPos], &threads)
	fmt.Sscan(parts[memPos], &mem)
	mem *= os.Getpagesize()
	mem /= 1000000 // MB

	goMemStats := new(runtime.MemStats)
	runtime.ReadMemStats(goMemStats)
	goMem = int(goMemStats.Sys) / 1000000 // MB
	return
}
