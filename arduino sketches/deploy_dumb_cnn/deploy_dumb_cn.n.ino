/*
    Boilerplate code for a neural network on the esp32
    Setup is a bit involved, but the loop has some nice comments explaining the process of predictions
*/

#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// change this to model_name.h
#include "dumb_CCC.h"
// change this to model_name
#define MODEL_NAME dumb_CNN

// Globals, used for compatibility with Arduino-style sketches.
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  int inference_count = 0;

  // we set the memory size to be used by our inputs, intermediate data and output variables here
  constexpr int kTensorArenaSize = 2000;            // see reader for instructions
  uint8_t tensor_arena[kTensorArenaSize];
}

char buffer[500];

void setup() {
  Serial.begin(9600);
  delay(250);

  setup_neural_network();
}

void setup_neural_network(){
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.

  // this should get the model from the model header file
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

  size_t used_size = interpreter->arena_used_bytes();
  Serial.print("--- Arena used bytes: ");
  Serial.println(used_size);

  // Obtain pointers to the model's input and output tensors.
  // this is where we write the inputs later
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Place the input in the model's input tensor
  // single float:
  // input->data.f[0] = x;

  // image should have scaled data, range is a float from [0, 1]

  // this should be like this:
  // for(each row){
  //   for(each pixel){
  //     input->data.f[rows*128 + i] = pixel_value;
  //   }
  // }

  // Do the prediction with the given input
  TfLiteStatus invoke_status = interpreter->Invoke();

  // check for errors
  if (invoke_status != kTfLiteOk) {
    sprintf(buffer, "Invoke failed on x: %f\n",
            static_cast<double>(x));
    Serial.println(buffer);
    return;
  }

  // get the output from model's output tensor. store in an array.
  //   float y[] = {};
  //   for(number of keywords){
  //     y[i] = output->data.f[0];
  //   }

  // show the prediction somehow
  // output_prediction(y);

  // short delay
  delay(25);
}
