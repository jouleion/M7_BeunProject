// look at the boilerplate file to understand, not this file
// code from the programming tutotorial

#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// TODO: change this line according to your model filename
#include "model1.h"


// This constant represents the range of x values our model was trained on,
// which is from 0 to (2 * Pi). We approximate Pi to avoid requiring additional
// libraries.
const float kXrange = 2.f * 3.14159265359f;
const int kInferencesPerCycle = 100;

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 2000;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

char buffer[500];

// Handles the output and prints two helper lines for the Serial Plotter
void HandleOutput(float x, float y) {
  Serial.print("lower:");
  Serial.print(-2.);
  Serial.print(",");
  Serial.print("upper:");
  Serial.print(2.);
  Serial.print(",");
  Serial.print("x:");
  Serial.print(x);
  Serial.print(",");
  Serial.print("y:");
  Serial.println(y);
}

void setup() {
  Serial.begin(9600);
  delay(250);
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.

  // TODO: change the call to the model according to your model name
  model = tflite::GetModel(model1);

//   Check if the model is compatible with the version of TensorFlow Lite Micro
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
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

void loop() {
  // Calculate an x value to feed into the model. We compare the current
  // inference_count to the number of inferences per cycle to determine
  // our position within the range of possible x values the model was
  // trained on, and use this to calculate a value.
  float position = static_cast<float>(inference_count) / static_cast<float>(kInferencesPerCycle);
  float x = position * kXrange;

  // Place the input in the model's input tensor
  input->data.f[0] = x;

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    sprintf(buffer, "Invoke failed on x: %f\n",
            static_cast<double>(x));
    Serial.println(buffer);
    return;
  }

  // Obtain the output from model's output tensor
  float y = output->data.f[0];

  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  HandleOutput(x, y);

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;

  delay(25);
}
