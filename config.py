class DefaultConfig (object) :

    train_dataset_root = 'train_dataset path'
    test_dataset_root = 'test_dataset path'

    cubic_sequence_length = 30
    cube_side_length = 29

    batch_size = 8
    max_epoch = 200
    transformer_encoders_number = 12
    learning_rate = 1e-6
    num_indexes = 4

    num_cross_fold = 10
    use_gpu = False

    in_channels = 1
    num_levels = 4
    f_maps = 16
    dim_hidden = 3456
    num_heads = 3
    dim_head = 18
    num_encoders = 8
    num_linear = 2
    num_class = 2

opt  = DefaultConfig()