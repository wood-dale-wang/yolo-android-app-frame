#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>

#include <mutex>
#include <string>
#include <vector>

#include "ncnn/net.h"
#include "ncnn/gpu.h"

namespace {

constexpr const char* kTag = "YoloNcnnJni";
constexpr int kInputWidth = 640;
constexpr int kInputHeight = 640;

std::mutex g_mutex;
ncnn::Net g_net;
bool g_initialized = false;
bool g_gpu_created = false;
int g_out_shape[3] = {1, 0, 0};

void log_error(const char* message) {
    __android_log_print(ANDROID_LOG_ERROR, kTag, "%s", message);
}

std::string jstring_to_std(JNIEnv* env, jstring value) {
    const char* chars = env->GetStringUTFChars(value, nullptr);
    std::string out(chars ? chars : "");
    if (chars) {
        env->ReleaseStringUTFChars(value, chars);
    }
    return out;
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_org_lzuxc_ml_YoloDetector_nativeInit(
    JNIEnv* env,
    jobject /*thiz*/,
    jobject asset_manager,
    jstring param_path,
    jstring bin_path,
    jboolean use_vulkan
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_initialized) {
        return JNI_TRUE;
    }

    if (asset_manager == nullptr || param_path == nullptr || bin_path == nullptr) {
        log_error("nativeInit received null arguments.");
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
    if (!mgr) {
        log_error("Failed to get AAssetManager.");
        return JNI_FALSE;
    }

    const bool want_vulkan = use_vulkan == JNI_TRUE;
    if (want_vulkan && ncnn::get_gpu_count() > 0) {
        ncnn::create_gpu_instance();
        g_gpu_created = true;
        g_net.opt.use_vulkan_compute = true;
    } else {
        g_net.opt.use_vulkan_compute = false;
    }
    g_net.opt.num_threads = 4;

    const std::string param = jstring_to_std(env, param_path);
    const std::string bin = jstring_to_std(env, bin_path);

    if (g_net.load_param(mgr, param.c_str()) != 0) {
        log_error("Failed to load ncnn param from assets.");
        if (g_gpu_created) {
            ncnn::destroy_gpu_instance();
            g_gpu_created = false;
        }
        return JNI_FALSE;
    }

    if (g_net.load_model(mgr, bin.c_str()) != 0) {
        log_error("Failed to load ncnn bin from assets.");
        g_net.clear();
        if (g_gpu_created) {
            ncnn::destroy_gpu_instance();
            g_gpu_created = false;
        }
        return JNI_FALSE;
    }

    g_initialized = true;
    return JNI_TRUE;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_org_lzuxc_ml_YoloDetector_nativeDetect(
    JNIEnv* env,
    jobject /*thiz*/,
    jobject bitmap
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_initialized || bitmap == nullptr) {
        g_out_shape[0] = 1;
        g_out_shape[1] = 0;
        g_out_shape[2] = 0;
        return env->NewFloatArray(0);
    }

    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        log_error("AndroidBitmap_getInfo failed.");
        return env->NewFloatArray(0);
    }

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        log_error("Bitmap format must be RGBA_8888.");
        return env->NewFloatArray(0);
    }

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        log_error("AndroidBitmap_lockPixels failed.");
        return env->NewFloatArray(0);
    }

    ncnn::Mat input = ncnn::Mat::from_pixels_resize(
        static_cast<const unsigned char*>(pixels),
        ncnn::Mat::PIXEL_RGBA2RGB,
        static_cast<int>(info.width),
        static_cast<int>(info.height),
        kInputWidth,
        kInputHeight
    );

    AndroidBitmap_unlockPixels(env, bitmap);

    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    input.substract_mean_normalize(nullptr, norm_vals);

    ncnn::Extractor ex = g_net.create_extractor();
    ex.input("in0", input);

    ncnn::Mat out;
    if (ex.extract("out0", out) != 0) {
        log_error("ncnn extractor failed on blob out0.");
        g_out_shape[0] = 1;
        g_out_shape[1] = 0;
        g_out_shape[2] = 0;
        return env->NewFloatArray(0);
    }

    ncnn::Mat out2d;
    if (out.dims == 1) {
        out2d = out.reshape(out.w, 1);
    } else if (out.dims == 2) {
        out2d = out;
    } else if (out.dims == 3) {
        out2d = out.reshape(out.w, out.h * out.c);
    } else {
        log_error("Unsupported ncnn output dims.");
        g_out_shape[0] = 1;
        g_out_shape[1] = 0;
        g_out_shape[2] = 0;
        return env->NewFloatArray(0);
    }

    g_out_shape[0] = 1;
    g_out_shape[1] = out2d.h;
    g_out_shape[2] = out2d.w;

    std::vector<float> values(static_cast<size_t>(out2d.w) * static_cast<size_t>(out2d.h));
    for (int y = 0; y < out2d.h; y++) {
        const float* row = out2d.row(y);
        for (int x = 0; x < out2d.w; x++) {
            values[static_cast<size_t>(y) * out2d.w + x] = row[x];
        }
    }

    jfloatArray result = env->NewFloatArray(static_cast<jsize>(values.size()));
    if (result == nullptr) {
        return env->NewFloatArray(0);
    }

    env->SetFloatArrayRegion(result, 0, static_cast<jsize>(values.size()), values.data());
    return result;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_org_lzuxc_ml_YoloDetector_nativeGetOutputShape(
    JNIEnv* env,
    jobject /*thiz*/
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    jintArray shape = env->NewIntArray(3);
    if (shape == nullptr) {
        return env->NewIntArray(0);
    }

    env->SetIntArrayRegion(shape, 0, 3, g_out_shape);
    return shape;
}

extern "C" JNIEXPORT void JNICALL
Java_org_lzuxc_ml_YoloDetector_nativeRelease(
    JNIEnv* /*env*/,
    jobject /*thiz*/
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_initialized) {
        g_net.clear();
        g_initialized = false;
    }

    if (g_gpu_created) {
        ncnn::destroy_gpu_instance();
        g_gpu_created = false;
    }

    g_out_shape[0] = 1;
    g_out_shape[1] = 0;
    g_out_shape[2] = 0;
}
