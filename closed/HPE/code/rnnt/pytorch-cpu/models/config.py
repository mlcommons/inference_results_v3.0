class RNNTParam:
    # Transcription
    trans_input_size = 240  # 80*3
    trans_hidden_size = 1024
    pre_num_layers = 2
    post_num_layers = 3
    stack_time_factor = 2
    # Prediction
    pred_hidden_size = 320
    pred_num_layers = 2
    # Joint
    joint_hidden_size = 512
    num_labels = 29
    # [SOS, SPACE, a~z, ', BLANK]
    # [-1, 0, 1~26, 27, 28]
    SOS = -1
    BLANK = 28
    max_symbols_per_step = 30
    sample_rate = 16000
