import os


def convert_tflite_to_c(tflite_path: str = 'model.tflite',  model_name: str = 'model', save_path: str = 'models') -> str:
    """
    Converts TFLite models into C-compatible header files for Arduino etc.

    Parameters
    ----------
    tflite_path : str
        Path to the TFLite model file
        Default path is 'model.tflite'
    model_name : str
        Output model name
        Default value is 'model'
    save_path : str
        Path where c array will be stored
        Default is "models"

    Returns
    -------
    model_path : str
        Path to the converted C-compatible model

    Raises
    ------
    ValueError
        If the provided model is not a TFLite model
    """

    # Check if file is a tflite model. If not, raise ValueError.
    if not tflite_path.endswith('.tflite'):
        raise ValueError("The provided file is not a TFLite model.")

    # Open the TFLite model file in binary mode and read its content into 'tflite_content'.
    with open(tflite_path, 'rb') as tflite_file:
        tflite_content = tflite_file.read()

    # Calculate the length of 'tflite_content' (i.e., the size of the TFLite model in bytes).
    array_length = len(tflite_content)

    # Split 'tflite_content' into chunks of 12 bytes each and convert each chunk to a hexadecimal string.
    # This is done so that the TFLite model can be represented as an array in C-compatible format.
    hex_lines = [', '.join([f'0x{byte:02x}' for byte in tflite_content[i:i + 12]]) for i in
                 range(0, len(tflite_content), 12)]

    # Join the chunks of hexadecimal strings with newlines to format them neatly.
    hex_array = ',\n     '.join(hex_lines)

    # prepare the file name in correct path
    new_file_name = os.path.join(save_path, model_name + '.h')

    # Open a header file in write mode and write out the TFLite model as an array.
    with open(new_file_name, 'w') as header_file:
        # Write the length of the TFLite model to the header file.
        header_file.write(f'unsigned int {model_name}_len = {array_length};\n\n')

        # Write out the TFLite model as an array in C-compatible format.
        header_file.write(f'alignas(8) const unsigned char {model_name}[] = {{\n     ')
        header_file.write(f'{hex_array}\n')
        header_file.write(f'}};\n')

    # Return the name of the generated header file.
    return model_name + '.h'


if __name__ == "__main__":
    convert_tflite_to_c('model.tflite', 'model')

