tflite_convert \
    --output_file=./"$1".tflite \
    --graph_def_file=./"$1".pb \
    --input_arrays=input_1 \
    --output_arrays=softmax/Softmax
