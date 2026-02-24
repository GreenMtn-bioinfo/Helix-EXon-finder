from tf_quiet import import_tf_quietly
tf = import_tf_quietly()
import keras
from tcn import TCN


### Keras definitions of the model architectures that were trained and evaluated for https://doi.org/10.64898/2025.12.19.694709


## 1) A Temporal Convolutional Network (TCNs) classifier by Bai et al.
## https://github.com/philipperemy/keras-tcn; https://arxiv.org/abs/1803.01271
def TCN_classifier(input_shape: tuple, 
                   n_classes: int, 
                   n_filters: int = 64, 
                   kernel_size: int = 3, 
                   dilations: tuple = (1, 2, 4, 8, 16, 32), 
                   n_stacks: int = 1, 
                   dense_layer_dims: list = None,
                   print_summary: bool = False):
    
    i = keras.Input(shape=input_shape)
    
    o = TCN(return_sequences=False,
        nb_filters=n_filters,
        kernel_size=kernel_size,
        nb_stacks=n_stacks,
        dilations=dilations,
        padding='same', # We do not care about causality or future leakage in these sequences, as we are not predicting the future and they are not true time series!
        use_skip_connections=True,
        dropout_rate=0.0,
        activation='relu',
        kernel_initializer='he_normal',
        use_batch_norm=False,
        use_layer_norm=False,
        go_backwards=False,
        return_state=False)(i)
    
    if not dense_layer_dims is None: # In case we want additional dense layers before the final softmax/class returning layer
        for layer_dim in dense_layer_dims:
            o = keras.layers.Dense(units=layer_dim, activation='relu')(o)
    
    o = keras.layers.Dense(n_classes, activation='softmax')(o)
    
    model = keras.Model(inputs=[i], outputs=[o])
    
    if print_summary:
        model.summary()
        
    return model


## 2)  Based on MBDA-Net from a paper by Gao et al.: 
## https://link.springer.com/article/10.1007/s11042-022-12809-z
def MBDA_Net(input_shape: tuple, 
             n_classes: int, 
             filters: int = 96, 
             kernel_sizes: list = [3, 5, 7, 11], 
             strides: int = 3, 
             mem_units: int = 96, 
             final_dense: int =  96,
             bidirectional: bool = True,
             print_summary: bool = False):
    
    
    input = keras.Input(shape=input_shape)
    
    ### MULTI-HEAD ATTENTION: INTAKE
    
    # A)
    conv1 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_sizes[3], strides=strides, padding="same")(input)
    a = keras.layers.BatchNormalization()(conv1)
    a = keras.layers.ReLU()(a)

    # B)
    conv2 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_sizes[2], strides=strides, padding="same")(input)
    b = keras.layers.BatchNormalization()(conv2)
    b = keras.layers.ReLU()(b)
    
    # C)
    conv3 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_sizes[1], strides=strides, padding="same")(input)
    c = keras.layers.BatchNormalization()(conv3)
    c = keras.layers.ReLU()(c)
    
    # D)
    conv4 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_sizes[0], strides=strides, padding="same")(input)
    d = keras.layers.BatchNormalization()(conv4)
    d = keras.layers.ReLU()(d)
    
    ## MULTI-HEAD ATTENTION: Combination
    
    # AB)
    ab = keras.layers.Concatenate(axis=2)([a, b])
    ab = keras.layers.Attention()([ab, ab])
    ab = keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding="same")(ab)
    
    # AB_C)
    ab_c = keras.layers.Concatenate(axis=2)([ab, c])
    ab_c = keras.layers.Attention()([ab_c, ab_c])
    ab_c = keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding="same")(ab_c)
    
    # AB_C_D)
    ab_c_d = keras.layers.Concatenate(axis=2)([ab_c, d])
    ab_c_d = keras.layers.Attention()([ab_c_d, ab_c_d])
    ab_c_d = keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding="same")(ab_c_d)
    
    ## BiLSTMs
    if bidirectional:
        bi_a = keras.layers.Bidirectional(keras.layers.LSTM(mem_units, return_sequences=True))(a)
        bi_ab = keras.layers.Bidirectional(keras.layers.LSTM(mem_units, return_sequences=True))(ab)
        bi_b = keras.layers.Bidirectional(keras.layers.LSTM(mem_units, return_sequences=True))(b)
        bi_ab_c = keras.layers.Bidirectional(keras.layers.LSTM(mem_units, return_sequences=True))(ab_c)
        bi_ab_c_d = keras.layers.Bidirectional(keras.layers.LSTM(mem_units, return_sequences=True))(ab_c_d)
    else:
        bi_a = keras.layers.LSTM(mem_units, return_sequences=True)(a)
        bi_ab = keras.layers.LSTM(mem_units, return_sequences=True)(ab)
        bi_b = keras.layers.LSTM(mem_units, return_sequences=True)(b)
        bi_ab_c = keras.layers.LSTM(mem_units, return_sequences=True)(ab_c)
        bi_ab_c_d = keras.layers.LSTM(mem_units, return_sequences=True)(ab_c_d)
    
    ## BiLSTM fusion
    lstm_fusion = keras.layers.add([bi_a, bi_b, bi_ab, bi_ab_c, bi_ab_c_d])
    lstm_fusion = keras.layers.Conv1D(filters=filters, kernel_size=kernel_sizes[0], strides=1, padding="same")(lstm_fusion)
    lstm_fusion = keras.layers.BatchNormalization()(lstm_fusion)
    lstm_fusion = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(lstm_fusion)
    lstm_fusion = keras.layers.Dense(units=final_dense, activation='relu')(lstm_fusion) # This layer was in the paper (omit due to adjustment in architecture?)
    
    ## FINAL output (combination of multi-head attention scores and BiLSTM fusion)
    output = keras.layers.Concatenate(axis=2)([lstm_fusion, ab_c_d])
    output = keras.layers.Dense(units=final_dense*2, activation='relu')(output) # These layers were in the paper (replace with a final average pooling layer?)
    output = keras.layers.Dense(units=final_dense*2, activation='relu')(output) # These layers were in the paper (replace with a final average pooling layer?)
    
    ## This was NOT in the paper! Used to aggregate the final feature map for whole-seq classification
    output = keras.layers.GlobalAveragePooling1D()(output) 
    output = keras.layers.Dense(units=n_classes, activation='softmax')(output)
    model = keras.Model(input, output)
    
    if print_summary:
        model.summary()
    
    return model


## A basic LSTM-based classifier that can be bi-directional if desired
def LSTM_classifier(input_shape: tuple, 
                    n_classes: int, 
                    memory_units: int = 64, 
                    bidirectional: bool = False,
                    dense_layer_dims: list = None, 
                    print_summary: bool = False):
    
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    
    forward_layer = keras.layers.LSTM(memory_units, dropout=0,
                                        activation = 'tanh', return_sequences=False,
                                        recurrent_activation = 'sigmoid',
                                        recurrent_dropout = 0,
                                        unroll = False,
                                        use_bias = True)
    
    if bidirectional:
        backward_layer = keras.layers.LSTM(memory_units, dropout=0, go_backwards=True,
                                            activation = 'tanh', return_sequences=False,
                                            recurrent_activation = 'sigmoid',
                                            recurrent_dropout = 0,
                                            unroll = False,
                                            use_bias = True)
        model.add(keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer))
        
    else:
        model.add(forward_layer)
    
    if not dense_layer_dims is None: # In case we want additional dense layers before the final softmax/class returning layer
        for layer_dim in dense_layer_dims:
            model.add(keras.layers.Dense(units=layer_dim, activation='relu'))

    model.add(keras.layers.Dense(n_classes, activation='softmax'))
    
    if print_summary:
        model.summary()
    
    return model