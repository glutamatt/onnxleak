package main

// #cgo CFLAGS: -I${SRCDIR}/onnxruntime/include
// #cgo LDFLAGS: -L${SRCDIR}/onnxruntime/lib -lonnxruntime
// #include <onnxruntime/include/onnxruntime_c_api.h>
// #include <onnxruntime_go_api.h>
import "C"
import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"time"
	"unsafe"
)

var ortAPI *C.OrtApi
var ortEnv *C.OrtEnv

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	ortAPI = C.GetOrtApi()
	checkStatus(C.CreateEnv(ortAPI, C.ORT_LOGGING_LEVEL_ERROR, C.CString("main_env"), &ortEnv))
	sessionOptions := getSessionOptions()
	modelPath := C.CString("model.onnx")
	inputNames := []*C.char{C.CString("input.1")}
	outputNames := []*C.char{C.CString("46")}
	var outputLength = 100
	inputShape := []int64{1, 1415}
	predPerSession := 1000

	sessionCounter := 0
	for {
		sessionCounter++
		time.Sleep(500 * time.Millisecond)
		threadCount, memory := stats()
		fmt.Printf("#%d New Session (%d threads, %d MB)\n", sessionCounter, threadCount, memory)
		//Let's GO
		var clonedSessionOptions *C.OrtSessionOptions
		var session *C.OrtSession

		checkStatus(C.CloneSessionOptions(ortAPI, sessionOptions, &clonedSessionOptions))
		checkStatus(C.CreateSession(ortAPI, ortEnv, modelPath, clonedSessionOptions, &session))

		wg := sync.WaitGroup{}
		wg.Add(predPerSession)

		for predCounter := 0; predCounter < predPerSession; predCounter++ {
			go func(printPred bool) {
				var inputTensor *C.OrtValue
				var outputTensor *C.OrtValue
				var memoryInfo *C.OrtMemoryInfo
				//checkStatus(C.CreateCpuMemoryInfo(ortAPI, C.OrtDeviceAllocator, C.OrtMemTypeDefault, &memoryInfo))
				checkStatus(C.CreateCpuMemoryInfo(ortAPI, C.OrtArenaAllocator, C.OrtMemTypeDefault, &memoryInfo))
				inputTensorLen := C.ulong(inputShape[1] * 4) //4 bytes per float
				inputSlice := make([]float32, inputShape[1])
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
				predictionsNoCopy := floatSlice(resPointer, outputLength)
				if printPred {
					fmt.Printf("predictions nocopy: %v ... %v\n", predictionsNoCopy[:3], predictionsNoCopy[outputLength-3:])
				}
				C.ReleaseMemoryInfo(ortAPI, memoryInfo)
				C.ReleaseValue(ortAPI, inputTensor)
				C.ReleaseValue(ortAPI, outputTensor)
				wg.Done()
			}(sessionCounter % 20 == 0)
		}

		wg.Wait()
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

func getSessionOptions() (ortSessionOptions *C.OrtSessionOptions) {
	checkStatus(C.CreateSessionOptions(ortAPI, &ortSessionOptions))
	checkStatus(C.SetSessionGraphOptimizationLevel(ortAPI, ortSessionOptions, 0))
	//checkStatus(C.DisableCpuMemArena(ortAPI, ortSessionOptions))
	//checkStatus(C.DisableMemPattern(ortAPI, ortSessionOptions))
	return ortSessionOptions
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

func stats() (threads, mem int) {
	stat, _ := ioutil.ReadFile(fmt.Sprintf("/proc/%d/stat", os.Getpid()))
	threadsPos := 19
	memPos := 23
	parts := strings.SplitN(string(stat), " ", memPos+1)
	fmt.Sscan(parts[threadsPos], &threads)
	fmt.Sscan(parts[memPos], &mem)
	mem *= os.Getpagesize()
	mem /= 1000000 // MB
	return
}
