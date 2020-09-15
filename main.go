package main
// #cgo CFLAGS: -I${SRCDIR}/onnxruntime/include
// #cgo LDFLAGS: -L${SRCDIR}/onnxruntime/lib -lonnxruntime
// #include <onnxruntime/include/onnxruntime_c_api.h>
// #include <onnxruntime_go_api.h>
import "C"
import (
	"fmt"
	"log"
	"runtime"
)


var ortAPI *C.OrtApi
func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	ortAPI = C.GetOrtApi()
	var ortEnv *C.OrtEnv
	checkStatus(C.CreateEnv(ortAPI, C.ORT_LOGGING_LEVEL_ERROR, C.CString("main_env"), &ortEnv))
	log.Printf("ortEnv %#v",ortEnv)









	//checkStatus(C.CreateSession(ortAPI, ortEnv, cFilePath, sessionOptions, &ortSession))
}


func checkStatus(status *C.OrtStatus) {
	if status == nil {
		return
	}
	ortError := C.GoString(C.GetErrorMessage(ortAPI, status))
	C.ReleaseStatus(ortAPI, status)
	ortErrorLocation := ""
	if _, fileName, fileLine, ok := runtime.Caller(1); ok {
		ortErrorLocation = fmt.Sprintf("at %s:%d ", fileName, fileLine)
	}
	panic(fmt.Errorf("onnx runtime error %s: '%s'", ortErrorLocation, ortError))
}
