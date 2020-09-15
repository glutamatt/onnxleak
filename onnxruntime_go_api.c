#include <onnxruntime/include/onnxruntime_c_api.h>

const OrtApi *GetOrtApi()
{
    return OrtGetApiBase()->GetApi(1);
}
OrtStatus *CreateEnv(OrtApi *api, OrtLoggingLevel l, const char *logid, OrtEnv **out)
{
    return api->CreateEnv(l, logid, out);
}
void ReleaseStatus(OrtApi *api, OrtStatus *s)
{
    api->ReleaseStatus(s);
}
void ReleaseSession(OrtApi *api, OrtSession *s)
{
    api->ReleaseSession(s);
}
void ReleaseSessionOptions(OrtApi *api, OrtSessionOptions *options)
{
    api->ReleaseSessionOptions(options);
}
void ReleaseMemoryInfo(OrtApi *api, OrtMemoryInfo *s)
{
    api->ReleaseMemoryInfo(s);
}
void ReleaseValue(OrtApi *api, OrtValue *v)
{
    api->ReleaseValue(v);
}
void ReleaseTypeInfo(OrtApi *api, OrtTypeInfo *i)
{
    api->ReleaseTypeInfo(i);
}
void ReleaseTensorTypeAndShapeInfo(OrtApi *api, OrtTensorTypeAndShapeInfo *v)
{
    api->ReleaseTensorTypeAndShapeInfo(v);
}
const char *GetErrorMessage(OrtApi *api, const OrtStatus *s)
{
    return api->GetErrorMessage(s);
}
OrtStatus *CreateSessionOptions(OrtApi *api, OrtSessionOptions **options)
{
    return api->CreateSessionOptions(options);
}
OrtStatus *CloneSessionOptions(OrtApi *api, OrtSessionOptions *options, OrtSessionOptions **clonedOptions)
{
    return api->CloneSessionOptions(options, clonedOptions);
}
OrtStatus *SetSessionGraphOptimizationLevel(OrtApi *api, OrtSessionOptions *options, GraphOptimizationLevel graph_optimization_level)
{
    return api->SetSessionGraphOptimizationLevel(options, graph_optimization_level);
}
OrtStatus *DisableCpuMemArena(OrtApi *api, OrtSessionOptions *options)
{
    return api->DisableCpuMemArena(options);
}

//OrtStatus *DisableMemPattern(OrtApi *api, OrtSessionOptions *options);
OrtStatus *DisableMemPattern(OrtApi *api, OrtSessionOptions *options)
{
    return api->DisableMemPattern(options);
}
OrtStatus *CreateCpuMemoryInfo(OrtApi *api, enum OrtAllocatorType type, enum OrtMemType mem_type, OrtMemoryInfo **out)
{
    return api->CreateCpuMemoryInfo(type, mem_type, out);
}
OrtStatus *CreateTensorWithDataAsOrtValue(OrtApi *api, const OrtMemoryInfo *info,
                                          void *p_data, size_t p_data_len, const int64_t *shape, size_t shape_len,
                                          ONNXTensorElementDataType type, OrtValue **out)
{
    return api->CreateTensorWithDataAsOrtValue(info, p_data, p_data_len, shape, shape_len, type, out);
}

OrtStatus *Run(OrtApi *api, OrtSession *sess,
               const OrtRunOptions *run_options,
               const char *const *input_names, const OrtValue *const *input, size_t input_len,
               const char *const *output_names, size_t output_names_len, OrtValue **output)
{
    return api->Run(sess, run_options, input_names, input, input_len, output_names, output_names_len, output);
}

OrtStatus *GetTensorMutableData(OrtApi *api, OrtValue *value, void **out)
{
    return api->GetTensorMutableData(value, out);
}
OrtStatus *SessionGetInputCount(OrtApi *api, const OrtSession *sess, size_t *out)
{
    return api->SessionGetInputCount(sess, out);
}

OrtStatus *SessionGetInputName(OrtApi *api, const OrtSession *sess, size_t index,
                               OrtAllocator *allocator, char **value)
{
    return api->SessionGetInputName(sess, index, allocator, value);
}

OrtStatus *SessionGetInputTypeInfo(OrtApi *api, const OrtSession *sess, size_t index,
                                   OrtTypeInfo **type_info)
{
    return api->SessionGetInputTypeInfo(sess, index, type_info);
}

OrtStatus *GetTensorTypeAndShape(OrtApi *api, const OrtValue *value, OrtTensorTypeAndShapeInfo **out)
{
    return api->GetTensorTypeAndShape(value, out);
}

OrtStatus *GetDimensionsCount(OrtApi *api, const OrtTensorTypeAndShapeInfo *info, size_t *out)
{
    return api->GetDimensionsCount(info, out);
}

OrtStatus *GetDimensions(OrtApi *api, const OrtTensorTypeAndShapeInfo *info, int64_t *dim_values, size_t dim_values_length)
{
    return api->GetDimensions(info, dim_values, dim_values_length);
}

OrtStatus *GetTensorElementType(OrtApi *api, const OrtTensorTypeAndShapeInfo *info, enum ONNXTensorElementDataType *out)
{
    return api->GetTensorElementType(info, out);
}
OrtStatus *SessionGetOutputCount(OrtApi *api, _In_ const OrtSession *sess, _Out_ size_t *out)
{
    return api->SessionGetOutputCount(sess, out);
}

OrtStatus *SessionGetOutputName(OrtApi *api, const OrtSession *sess, size_t index, OrtAllocator *allocator, char **value)
{
    return api->SessionGetOutputName(sess, index, allocator, value);
}

OrtStatus *SessionGetOutputTypeInfo(OrtApi *api, const OrtSession *sess, size_t index,
                                   OrtTypeInfo **type_info)
{
    return api->SessionGetOutputTypeInfo(sess, index, type_info);
}

OrtStatus *CastTypeInfoToTensorInfo(OrtApi *api, const OrtTypeInfo* info, const OrtTensorTypeAndShapeInfo** out)
{
    return api->CastTypeInfoToTensorInfo(info, out);
}

OrtStatus *CreateSession(OrtApi *api,  const OrtEnv* env,  const ORTCHAR_T* model_path,
                                           const OrtSessionOptions* options,  OrtSession** out)
{
    return api->CreateSession(env,model_path,options, out);
}

OrtStatus *GetAllocatorWithDefaultOptions(OrtApi *api,  OrtAllocator** out)
{
    return api->GetAllocatorWithDefaultOptions(out);
}
