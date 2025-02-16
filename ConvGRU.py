import numpy as np

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops

### ConvLSTM2D (base_conv_lstm.py)を参考に親クラス変更したら一応できた(とりあえずこれで行く)
from keras.engine import base_layer
from keras.layers.rnn.base_conv_rnn import ConvRNN


class ConvGRU2DCell(DropoutRNNCellMixin, base_layer.BaseRandomLayer):#　本当は(DropoutRNNCellMixin, Layer)
    def __init__(self,
               filters,#　畳み込みのフィルタ数　出力の次元数　整数
               kernel_size,#　畳み込みフィルタのサイズ　整数　n個からなるタプル|リスト　ex: (2,2) (1,2,3)
               strides=(1, 1),#　畳み込みストライド　
               padding='same',#　パディング　"valid"または"same"のいずれか
               data_format=None,#　channels_last（デフォルト）またはchannels_first
               dilation_rate=(1, 1),#　膨張率　変更する時、ストライドを1にしなければならない
               activation='tanh',#　活性化関数
               recurrent_activation='hard_sigmoid',#　活性化関数　再帰的
               use_bias=True,#　バイアスを使用するかどうか
               kernel_initializer='glorot_uniform',#　カーネルの初期化
               recurrent_initializer='orthogonal',#　再帰カーネルの初期化
               bias_initializer='zeros',#　バイアスの初期化
               kernel_regularizer=None,#　重みに適用される正則化関数（正則化は過学習を防ぐ）
               recurrent_regularizer=None,#　再帰的重みに適用される正則化関数
               bias_regularizer=None,#　バイアスに適用される正則化関数
               kernel_constraint=None,#　学習中に重みを制約する制約関数（学習が安定　→　過学習を防ぐ）
               recurrent_constraint=None,#　再帰的重み制約関数
               bias_constraint=None,#　バイアス制約関数
               dropout=0,#　0〜１の実数　入力の線形変換時ドロップするユニットの割合
               recurrent_dropout=0.,#　再帰状態のドロップするユニットの割合
               **kwargs):
        super(ConvGRU2DCell, self).__init__(**kwargs)
        
        self.filters = filters# filters:  output_channels = 6
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')# 正規化　kernel_size: (3, 3)
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')# 正規化　strides: (1, 1)
        self.padding = conv_utils.normalize_padding(padding)# 正規化　padding: valid
        self.data_format = conv_utils.normalize_data_format(data_format)# data_format: channels_last
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                        'dilation_rate')# dilation_rate: (1, 1)
        self.activation = activations.get(activation)# activation: <function tanh at 0x7fab019d7250>
        self.recurrent_activation = activations.get(recurrent_activation)# recurrent_activation: <function hard_sigmoid at 0x7fab019d7760>
        self.use_bias = use_bias# use_bias: True
        
        # kernel_initializer: <tensorflow.python.keras.initializers.initializers_v2.GlorotUniform object at 0x7fa9a0605a50>
        self.kernel_initializer = initializers.get(kernel_initializer)
        
        
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)# recurrent_regularizer: None
        self.bias_regularizer = regularizers.get(bias_regularizer)# bias_constraint: None

        self.kernel_constraint = constraints.get(kernel_constraint)# kernel_constraint: None
        self.recurrent_constraint = constraints.get(recurrent_constraint)# recurrent_constraint: None
        self.bias_constraint = constraints.get(bias_constraint)# bias_constraint: None

        self.dropout = min(1., max(0., dropout))# dropout: 0.0
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))# recurrent_dropout: 0.0
        self.state_size = (self.filters)# state_size: 6
#         print('##########　初期化　##########')
#         print(' ')
#         print('filters :',self.filters)
#         print('kernel_size :',self.kernel_size)
#         print('strides :',self.strides)
#         print('padding :',self.padding)
#         print('data_format :',self.data_format)
#         print('dilation_rate :',self.dilation_rate)
        
#         print('activation :',self.activation)
#         print('recurrent_activation :',self.recurrent_activation)
#         print('use_bias :',self.use_bias)
#         print('kernel_initializer :',self.kernel_initializer)
#         print('recurrent_regularizer :',self.recurrent_regularizer)
#         print('bias_regularizer :',self.bias_regularizer)
#         print('kernel_constraint :',self.kernel_constraint)
#         print('recurrent_constraint :',self.recurrent_constraint)
        
#         print('bias_constraint :',self.bias_constraint)
#         print('dropout :',self.dropout)
#         print('recurrent_dropout :',self.recurrent_dropout)
#         print('state_size :',self.state_size)
#         print('  ')
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        
        # self.kernel_size + (input_dim , self.filters*3) = (3,3) + (1, 6*3) = (3, 3, 1, 18)
        kernel_shape = self.kernel_size + (input_dim, self.filters * 3)
        self.kernel_shape = kernel_shape
        recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 3)

        
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        if self.use_bias:
            bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
              shape=(self.filters * 3,),
              name='bias',
              initializer=bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True
#         print('##########　重み定義　##########')
#         print("        ")
#         print('input_shape :',input_shape)# batch?step?, height, weight, channel : None, 32, 32 , 1
#         print('input_dim :',input_dim)
#         print("kernel_shape :",kernel_shape)
#         print("recurrent_kernel_shape :",recurrent_kernel_shape)
#         print('kernel :',np.shape(self.kernel))
#         print('recurrent_kernel :', np.shape(self.recurrent_kernel))
#         print('bias :',np.shape(self.bias))# all 0 (init)
#         print(' ')
    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        
        # dropout matrices for input units
        dp_mask = self.get_dropout_mask_for_cell(inputs, None, count=3)
        
        # dropout matrices for recurrent units
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=3)
        if 0 < self.dropout < 1.:
            inputs_z = inputs * dp_mask[0]
            inputs_r = inputs * dp_mask[1]
            inputs_h = inputs * dp_mask[2]
            
        else:
            inputs_z = inputs
            inputs_r = inputs
            inputs_h = inputs


        if 0 < self.recurrent_dropout < 1.:
            h_tm1_z = h_tm1 * rec_dp_mask[0]
            h_tm1_r = h_tm1 * rec_dp_mask[1]
            h_tm1_h = h_tm1 * rec_dp_mask[2]

        else:
            h_tm1_z = h_tm1
            h_tm1_r = h_tm1
            h_tm1_h = h_tm1

        (kernel_z, kernel_r,
         kernel_h) = array_ops.split(self.kernel, 3, axis=3)
        (recurrent_kernel_z,
         recurrent_kernel_r,
         recurrent_kernel_h) = array_ops.split(self.recurrent_kernel, 3, axis=3)

        if self.use_bias:
            bias_z, bias_r, bias_h = array_ops.split(self.bias, 3)
        else:
            bias_z, bias_r, bias_h = None, None, None

        x_z = self.input_conv(inputs_z, kernel_z, bias_z, padding=self.padding)
        x_r = self.input_conv(inputs_r, kernel_r, bias_r, padding=self.padding)
        x_h = self.input_conv(inputs_h, kernel_h, bias_h, padding=self.padding)
        

        h_z = self.recurrent_conv(h_tm1_z, recurrent_kernel_z)
        h_r = self.recurrent_conv(h_tm1_r, recurrent_kernel_r)
        h_h = self.recurrent_conv(h_tm1_h, recurrent_kernel_h)
        
        z = self.recurrent_activation(x_z + h_z)
        r = self.recurrent_activation(x_r + h_r)
        
        h = (1.0 - z) * h_tm1 + z * self.activation(x_h + r * h_h)# z * tanh(x_h + r * h_h) = \hat(h_t)
#         print('##########  call ##########')
#         print(' ')
#         print('inputs :',inputs)
#         print('states :',states)
#         print('h_tm1 :', h_tm1)
        print('dp_mask :', dp_mask)
        print('rec_dp_mask :', rec_dp_mask)
#         print('inputs_z :', np.shape(inputs_z))
#         print('inputs_r :', np.shape(inputs_r))
#         print('inputs_h :', np.shape(inputs_h))
#         print(' ')
#         print('h_tm1_z :', np.shape(h_tm1_z))
#         print('h_tm1_r :', np.shape(h_tm1_r))
#         print('h_tm1_h :', np.shape(h_tm1_h))
#         print(' ')
#         print('kernel_z :',np.shape(kernel_z))
#         print('kernel_r :',np.shape(kernel_z))
#         print('kernel_h :',np.shape(kernel_z))
#         print(' ')
#         print('recurrent_kernel_z :',np.shape(recurrent_kernel_z))
#         print('recurrent_kernel_r :',np.shape(recurrent_kernel_r))
#         print('recurrent_kernel_h :',np.shape(recurrent_kernel_h))
#         print(' ')
#         print('bias_z :',np.shape(bias_z))
#         print('bias_r :',np.shape(bias_r))
#         print('bias_h :',np.shape(bias_h))
#         print(' ')
#         print('x_z :', np.shape(x_z))
#         print('x_r :', np.shape(x_r))
#         print('x_h :', np.shape(x_h))
#         print('h_z :', np.shape(h_z))
#         print('h_r :', np.shape(h_r))
#         print('h_h :', np.shape(h_h))
#         print('')
#         print('x_z + h_z :', x_z + h_z)
#         print('z :',np.shape(z))
#         print('r :',np.shape(r))
#         print(' ')
        
#         print('h :', np.shape(h))
#         print('[h] :',[h])
#         print(' ')
        return h, [h]# h: output, [h]: new states

    def input_conv(self, x, w, b=None, padding='valid'):
        conv_out = backend.conv2d(x, w, strides=self.strides,
                                  padding=padding,
                                  data_format=self.data_format,
                                  dilation_rate=self.dilation_rate)
        if b is not None:
            conv_out = backend.bias_add(conv_out, b,
                                        data_format=self.data_format)
        return conv_out

    def recurrent_conv(self, x, w):
        conv_out = backend.conv2d(x, w, strides=(1, 1),
                                  padding='same',
                                  data_format=self.data_format)
        return conv_out

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(
                      self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(
                      self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(
                      self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(
                      self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvGRU2DCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvGRU2D(ConvRNN):#　本当はConvRNN2D
    
    ###　入力形状　###
#     data_formatが'channels_first'の場合、次の形状の5次元テンソル:
#     (samples, time, channels, rows, cols)
#     data_formatが'channels_last'の場合、次の形状の5次元テンソル:
#     (samples, time, rows, cols, channels)

    ###　出力形状　###
#  return_stateがTrueの場合: テンソルのリスト。
# 　　最初のテンソルが出力で、残りのテンソルは最後の状態で、それぞれの形状は次のようになります:

# 　　（data_formatが'channels_first'の場合）　(samples, filters, new_rows, new_cols)の4次元テンソル
# 　　（data_formatが'channels_last'の場合）　　(samples, new_rows, new_cols, filters)の4次元テンソル。
# 　　　パディングによりrowsとcolsの値が変わる可能性があります。

# return_sequencesがTrueの場合: 
# 　　（data_formatが'channels_first'の場合）(samples, timesteps, filters, new_rows, new_cols)の5次元テンソル
# 　　（data_formatが'channels_last'の場合）(samples, timesteps, new_rows, new_cols, filters)の5次元テンソル

# 上記のいずれでもない場合、
# 　　（data_formatが'channels_first'の場合）　(samples, filters, new_rows, new_cols)の4次元テンソル
# 　　（data_formatが'channels_last'の場合）　(samples, new_rows, new_cols, filters)の4次元テンソル

    def __init__(self,
               filters,#　畳み込みの出力フィルタ-数
               kernel_size,#　畳み込みのフィルタサイズ
               strides=(1, 1),#　ストライド
               padding='same',#　プーリング
               data_format=None,#　channels_last（デフォルト）またはchannels_firstのいずれか
               dilation_rate=(1, 1),#　膨張畳み込みに使用する膨張率
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',#　重みの初期化関数
               recurrent_initializer='orthogonal',#　初期化関数
               bias_initializer='zeros',#　初期化関数
               kernel_regularizer=None,#　正則化関数
               recurrent_regularizer=None,#　正則化関数
               bias_regularizer=None,#　正則化関数
               activity_regularizer=None,#　活性化関数の正則化関数
               kernel_constraint=None,#　重みの制約関数
               recurrent_constraint=None,#　制約関数
               bias_constraint=None,#　制約関数
               return_sequences=False,#　プール値、出力シーケンス内の最後の出力を返すか、
               return_state=False,#　プール値、出力に加えて最後の状態を返すかどうか
               go_backwards=False,#　Trueの場合　入力シーケンスを逆順に処理
               stateful=False,#　True の場合バッチ内の各サンプルのインデックスに対して直前のバッチでのインデックスのサンプルの最後の状態が次の初期状態
               dropout=0.4,#　ドロップアウトの割合
               recurrent_dropout=0.4,#　再帰ドロップアウトの割合
               **kwargs):
        cell = ConvGRU2DCell(filters=filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding=padding,
                              data_format=data_format,
                              dilation_rate=dilation_rate,
                              activation=activation,
                              recurrent_activation=recurrent_activation,
                              use_bias=use_bias,
                              kernel_initializer=kernel_initializer,
                              recurrent_initializer=recurrent_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=kernel_regularizer,
                              recurrent_regularizer=recurrent_regularizer,
                              bias_regularizer=bias_regularizer,
                              kernel_constraint=kernel_constraint,
                              recurrent_constraint=recurrent_constraint,
                              bias_constraint=bias_constraint,
                              dropout=dropout,
                              recurrent_dropout=recurrent_dropout,
                              dtype=kwargs.get('dtype'))
        super(ConvGRU2D, self).__init__(2,# rank : 2D
                                        cell,
                                        return_sequences=return_sequences,
                                        return_state=return_state,
                                        go_backwards=go_backwards,
                                        stateful=stateful,
                                        **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        #　inputs　5次元のテンソル
        #　mask　(samples, timestep)のバイナリテンソル　特定のタイムステップがマスクされるべきかどうか
        #　training
        #　initial_state　最初のセルの呼び出しに渡される初期状態テンソルのリスト
        return super(ConvGRU2D, self).call(inputs,
                                            mask=mask,
                                            training=training,
                                            initial_state=initial_state)

    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'filters': self.filters,
              'kernel_size': self.kernel_size,
              'strides': self.strides,
              'padding': self.padding,
              'data_format': self.data_format,
              'dilation_rate': self.dilation_rate,
              'activation': activations.serialize(self.activation),
              'recurrent_activation': activations.serialize(
                  self.recurrent_activation),
              'use_bias': self.use_bias,
              'kernel_initializer': initializers.serialize(
                  self.kernel_initializer),
              'recurrent_initializer': initializers.serialize(
                  self.recurrent_initializer),
              'bias_initializer': initializers.serialize(self.bias_initializer),
              'kernel_regularizer': regularizers.serialize(
                  self.kernel_regularizer),
              'recurrent_regularizer': regularizers.serialize(
                  self.recurrent_regularizer),
              'bias_regularizer': regularizers.serialize(self.bias_regularizer),
              'activity_regularizer': regularizers.serialize(
                  self.activity_regularizer),
              'kernel_constraint': constraints.serialize(
                  self.kernel_constraint),
              'recurrent_constraint': constraints.serialize(
                  self.recurrent_constraint),
              'bias_constraint': constraints.serialize(self.bias_constraint),
              'dropout': self.dropout,
              'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvGRU2D, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)