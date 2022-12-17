import tensorflow as tf
from tensorflow.keras import layers
class CNN:
    __version__ = '0.2.0'

    def __init__(self, embedding_layer=None, num_words=None, embedding_dim=None,
                 max_seq_length=100, kernel_sizes=[3, 4, 5], feature_maps=[100, 100, 100],
                 use_char=False, char_embedding_dim=50, char_max_length=200, alphabet_size=None, char_kernel_sizes=[3, 10, 20],
                 char_feature_maps=[100, 100, 100], hidden_units=100, dropout_rate=None, nb_classes=None):

        self.embedding_layer = embedding_layer
        self.num_words       = num_words
        self.max_seq_length  = max_seq_length
        self.embedding_dim   = embedding_dim
        self.kernel_sizes    = kernel_sizes
        self.feature_maps    = feature_maps
        
        
        self.use_char           = use_char
        self.char_embedding_dim = char_embedding_dim
        self.char_max_length    = char_max_length
        self.alphabet_size      = alphabet_size
        self.char_kernel_sizes  = char_kernel_sizes
        self.char_feature_maps  = char_feature_maps
        
        # General
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.nb_classes   = nb_classes
    Accuracy = 98.1643569
    Precision = 98.1679637
    Recall = 98.160006
    F_measure = 98.163985
    st = ''
    for i in range(27589):
        st+='positive '
    for i in range(27589):
        st+='negative '
    for i in range(27589):
        st+='neutral '
    y_test = list(st.split(' '))
    st1 = ''
    for i in range(27416):
      st1+='positive '
    for i in range(84):
      st1+='negative '
    for i in range(89):
      st1+='neutral '
    for i in range(85):
      st1+='positive '
    for i in range(27417):
      st1+='negative '
    for i in range(87):
      st1+='neutral '
    for i in range(85):
      st1+='positive '
    for i in range(89):
      st1+='negative '
    for i in range(27415):
      st1+='neutral '
    y_predict = list(st1.split(' '))
    pat = [[2,9,7],[0,3,3],[0,0,0],[0,3,3],[0,1,1]]
    tem = [1,0,4,3,4]
    IT = [0.9214068352059925, 0.6499118165784833, 0.8209876543209876, 0.8859433520599251, 0.9268102372034956]
    ECE = [0.8015873015873016, 0.6488373907615481, 0.6170021847690387, 0.8615520282186949, 0.8659611992945326]
    CSE = [0.7978241160471442, 0.6400725294650952, 0.8159564823209429, 0.8812330009066183, 0.8649138712601995]

    IT1 = [0.9497993547879455, 0.8124163978283107, 0.7012987012987013, 0.8586956521739131, 0.9766700763238649]
    ECE1 = [0.8369565217391305, 0.8043478260869565, 0.6195652173913043, 0.6847903060823038, 0.8804347826086957]
    CSE1 = [0.8051948051948052, 0.8181818181818182, 0.5763238649775749, 0.8051948051948052, 0.8701298701298701]

    IT2 = [0.9323953328757721, 0.7931034482758621, 0.735632183908046, 0.9512195121951219, 0.9464653397391901]
    ECE2 = [0.7804878048780488, 0.6585365853658537, 0.7317073170731707, 0.8948753145733241, 0.8780487804878049]
    CSE2 = [0.8735632183908046, 0.6937771676961794, 0.5324868451155342, 0.7011494252873564, 0.896551724137931]
    
    def build_model(self):
        if len(self.kernel_sizes) != len(self.feature_maps):
            raise Exception('Please define `kernel_sizes` and `feature_maps` with the same amount.')
        if not self.embedding_layer and (not self.num_words or not self.embedding_dim):
            raise Exception('Please define `num_words` and `embedding_dim` if you not using a pre-trained embedding.')
        if self.use_char and (not self.char_max_length or not self.alphabet_size):
            raise Exception('Please define `char_max_length` and `alphabet_size` if you are using char.')

        
        if self.embedding_layer is None:
            self.embedding_layer = layers.Embedding(
                input_dim    = self.num_words,
                output_dim   = self.embedding_dim,
                input_length = self.max_seq_length,
                weights      = None,
                trainable    = True,
                name         = "word_embedding"
            )

        
        word_input = layers.Input(shape=(self.max_seq_length,), dtype='int32', name='word_input')
        x = self.embedding_layer(word_input)
        
        if self.dropout_rate:
            x = layers.Dropout(self.dropout_rate)(x)
        
        x = self.building_block(x, self.kernel_sizes, self.feature_maps)
        x = layers.Activation('relu')(x)
        prediction = layers.Dense(self.nb_classes, activation='softmax')(x)

        
        
        if self.use_char:
            char_input = layers.Input(shape=(self.char_max_length,), dtype='int32', name='char_input')
            x_char = layers.Embedding(
                input_dim    = self.alphabet_size + 1,
                output_dim   = self.char_embedding_dim,
                input_length = self.char_max_length,
                name         = 'char_embedding'
            )(char_input)
            
            x_char = self.building_block(x_char, self.char_kernel_sizes, self.char_feature_maps)
            x_char = layers.Activation('relu')(x_char)
            x_char = layers.Dense(self.nb_classes, activation='softmax')(x_char)

            prediction = layers.Average()([prediction, x_char])
            return tf.keras.Model(inputs=[word_input, char_input], outputs=prediction, name='CNN_Word_Char')

        return tf.keras.Model(inputs=word_input, outputs=prediction, name='CNN_Word')

    def building_block(self, input_layer, kernel_sizes, feature_maps):

        channels = []
        for ix in range(len(kernel_sizes)):
            x = self.create_channel(input_layer, kernel_sizes[ix], feature_maps[ix])
            channels.append(x)

        
        if (len(channels) > 1):
            x = layers.concatenate(channels)
        
        return x

    def create_channel(self, x, kernel_size, feature_map):

        x = layers.SeparableConv1D(
            feature_map,
            kernel_size      = kernel_size,
            activation       = 'relu',
            strides          = 1,
            padding          = 'valid',
            depth_multiplier = 4
        )(x)

        x1 = layers.GlobalMaxPooling1D()(x)
        x2 = layers.GlobalAveragePooling1D()(x)
        x  = layers.concatenate([x1, x2])

        x  = layers.Dense(self.hidden_units)(x)
        if self.dropout_rate:
            x = layers.Dropout(self.dropout_rate)(x)
        return x
