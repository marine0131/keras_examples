tflite_convert \
    --output_file=./"$2" \
    --graph_def_file=./"$1" \
    --input_arrays=input_1_1 \
    --output_arrays=softmax_1/Softmax
