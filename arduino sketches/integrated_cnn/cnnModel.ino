#include "cnnModel.h"

cnnModel::cnnModel(int width, int height) : width(width), height(height) {
    // Setup the neural network
    setup_cnn();
}


void cnnModel::setup_cnn() {
    // put model in useable data structure

    // get the model from the model header file
    model = tflite::GetModel(MODEL_NAME);

    // Check if the model is compatible with the version of TensorFlow Lite Micro
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        sprintf(buffer, "--- Model provided is schema version %d not equal to supported "
                        "version %d.",
                model->version(), TFLITE_SCHEMA_VERSION);
        Serial.println(buffer);
        return;
    } else {
        Serial.println("--- Model loaded");
    }

    // This pulls in all the operation implementations we need.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::AllOpsResolver resolver;

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("--- AllocateTensors() failed");
        return;
    } else {
        Serial.println("--- Tensors allocated");
    }

    // to check the actual bytes that are have to be allocated, run this code snippet
    // size_t arena_used_size = interpreter->arena_used_bytes();

    size_t used_size = interpreter->arena_used_bytes();
    Serial.print("--- Arena used bytes: ");
    Serial.println(used_size);

    // Obtain pointers to the model's input and output tensors.
    // this is where we write the inputs later
    input = interpreter->input(0);
    output = interpreter->output(0);
}


float* cnnModel::predict(float** spectrogram) {
  
    // place the pixel data in the input tensor of the model
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            input->data.f[i * width + j] = spectrogram[i][j];
        }
    }

    // run the model to calculate the outcome
    TfLiteStatus invoke_status = interpreter->Invoke();

    // check for errors
    if (invoke_status != kTfLiteOk) {
        sprintf(buffer, "Invoke failed on input data\n");
        Serial.println(buffer);
        //return [-1];
    }

    // get the output from the model's output tensor and store it in an array
    float y[NUM_KEYWORDS];
    for (int i = 0; i < NUM_KEYWORDS; i++) {
        y[i] = output->data.f[i];
    }

    // return the prediction
    return(y);
}
