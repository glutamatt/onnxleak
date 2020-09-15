#include <onnxruntime/include/onnxruntime_c_api.h>

const OrtApi *GetOrtApi();
OrtStatus *CreateEnv(OrtApi *api, OrtLoggingLevel l, const char *logid, OrtEnv **out);
void ReleaseStatus(OrtApi *api, OrtStatus *s);
void ReleaseSession(OrtApi *api, OrtSession *s);
void ReleaseSessionOptions(OrtApi *api, OrtSessionOptions *options);
void ReleaseMemoryInfo(OrtApi *api, OrtMemoryInfo *s);
void ReleaseValue(OrtApi *api, OrtValue *v);
void ReleaseTypeInfo(OrtApi *api, OrtTypeInfo *i);
void ReleaseTensorTypeAndShapeInfo(OrtApi *api, OrtTensorTypeAndShapeInfo *v);
const char *GetErrorMessage(OrtApi *api, const OrtStatus *s);
OrtStatus *CreateSessionOptions(OrtApi *api, OrtSessionOptions **options);
OrtStatus *CloneSessionOptions(OrtApi *api, OrtSessionOptions *options, OrtSessionOptions **clonedOptions);
OrtStatus *SetSessionGraphOptimizationLevel(OrtApi *api, OrtSessionOptions *options, GraphOptimizationLevel graph_optimization_level);
OrtStatus *DisableCpuMemArena(OrtApi *api, OrtSessionOptions *options);
////  ORT_API2_STATUS(DisableMemPattern, _Inout_ OrtSessionOptions* options);
OrtStatus *DisableMemPattern(OrtApi *api, OrtSessionOptions *options);



OrtStatus *CreateCpuMemoryInfo(OrtApi *api, enum OrtAllocatorType type, enum OrtMemType mem_type, OrtMemoryInfo **out);
OrtStatus *CreateTensorWithDataAsOrtValue(OrtApi *api, const OrtMemoryInfo *info,
                                          void *p_data, size_t p_data_len, const int64_t *shape, size_t shape_len,
                                          ONNXTensorElementDataType type, OrtValue **out);
OrtStatus *Run(OrtApi *api, OrtSession *sess,
               const OrtRunOptions *run_options,
               const char *const *input_names, const OrtValue *const *input, size_t input_len,
               const char *const *output_names, size_t output_names_len, OrtValue **output);
OrtStatus *GetTensorMutableData(OrtApi *api, OrtValue *value, void **out);
OrtStatus *SessionGetInputCount(OrtApi *api, const OrtSession *sess, size_t *out);
OrtStatus *SessionGetInputName(OrtApi *api, const OrtSession *sess, size_t index, OrtAllocator *allocator, char **value);
OrtStatus *SessionGetInputTypeInfo(OrtApi *api, const OrtSession *sess, size_t index, OrtTypeInfo **type_info);
OrtStatus *GetTensorTypeAndShape(OrtApi *api, const OrtValue *value, OrtTensorTypeAndShapeInfo **out);
OrtStatus *GetDimensionsCount(OrtApi *api, const OrtTensorTypeAndShapeInfo *info, size_t *out);
OrtStatus *GetDimensions(OrtApi *api, const OrtTensorTypeAndShapeInfo *info, int64_t *dim_values, size_t dim_values_length);
OrtStatus *GetTensorElementType(OrtApi *api, const OrtTensorTypeAndShapeInfo *info, enum ONNXTensorElementDataType *out);
OrtStatus *SessionGetOutputCount(OrtApi *api, _In_ const OrtSession* sess, _Out_ size_t* out);
OrtStatus *SessionGetOutputName(OrtApi *api, const OrtSession *sess, size_t index, OrtAllocator *allocator, char **value);
OrtStatus *SessionGetOutputTypeInfo(OrtApi *api, const OrtSession *sess, size_t index,
                                   OrtTypeInfo **type_info);
OrtStatus *CastTypeInfoToTensorInfo(OrtApi *api, const OrtTypeInfo* info, const OrtTensorTypeAndShapeInfo** out);

OrtStatus *CreateSession(OrtApi *api,  const OrtEnv* env,  const ORTCHAR_T* model_path,
                                           const OrtSessionOptions* options,  OrtSession** out);

OrtStatus *GetAllocatorWithDefaultOptions(OrtApi *api,  OrtAllocator** out);

