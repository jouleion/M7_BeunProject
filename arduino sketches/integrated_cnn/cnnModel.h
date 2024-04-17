#ifndef CNN_MODEL_H
#define CNN_MODEL_H

#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// change this to model_name.h
#include "dumb_CNN.h"
// change this to model_name
#define MODEL_NAME dumb_CNN


char buffer[500];

class cnnModel {
public:
    cnnModel(int width, int height);
    float* predict(float** image);

private:
    void setup_cnn();

    int width, height;

    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    int inference_count = 0;


    // we set the memory size to be used by our inputs, intermediate data and output variables here
    // these are the reader instructions
    /*
          Note: If the allocation of tensors fails, your TensorArenaSize is proba-
          bly too small. Try to increase its size and re-upload your code. You can
          check how much bytes of the TensorArena are actually used by calling
          size_t arena_used_size = interpreter->arena_used_bytes(); after success-
          fully allocating all tensors.
    */
    static constexpr int kTensorArenaSize = 10000;
    uint8_t tensor_arena[kTensorArenaSize];
};

#endif // CNN_MODEL_H
