import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tensorflow.python.keras.regularizers import l2

from collections import OrderedDict

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat

def build_input_layers(feature_columns, prefix=''):
    """
    Input 是 TensorFlow/Keras 中用于定义模型输入层的类。
    input_layer = Input(shape=shape, name=None, dtype=None, **kwargs)
    """
    # # input_features = OrderedDict() 这行代码创建了一个有序字典来存储输入特征。
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(
                shape=(1,), name=prefix + fc.name, dtype=fc.dtype) # 批量大小 × 1（单个ID）
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(
                shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name,
                                            dtype=fc.dtype)
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=prefix + fc.weight_name,
                                                       dtype="float32")
            if fc.length_name is not None:
                input_features[fc.length_name] = Input((1,), name=prefix + fc.length_name, dtype='int32')

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features

def build_embedding_layers(feature_columns):
    """
    Embedding 层的作用是将高维稀疏的类别ID转换为低维稠密的向量表示。
    Embedding(
    input_dim,      # 词汇表大小（最大整数索引 + 1）
    output_dim,     # 嵌入向量的维度
    input_length=None,  # 输入序列长度（对于序列输入）
    embeddings_initializer='uniform',
    mask_zero=False,     # 是否忽略0值（用于变长序列）
    name=None)
    """
    embedding_layer_dict = {}
    
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            embedding_layer_dict[fc.name] = Embedding(fc.vocabulary_size, fc.embedding_dim, name='emb_' + fc.name)
        elif isinstance(fc, VarLenSparseFeat):
            # 这行代码中的 fc.vocabulary_size + 1 和 mask_zero=True 是处理稀疏特征时的两个重要技巧。
            # 这里加1是因为mask有个默认的embedding，占用了一个位置，比如mask为0， 而0这个位置的embedding是mask专用了
            """
            # 假设用户ID特征
            fc.vocabulary_size = 1000  # 训练集中有1000个不同的用户
            embedding = Embedding(1000 + 1, 16)  # 实际创建1001个嵌入向量

            # 索引分配：
            # 0: [MASK] 或 [PAD]  (mask_zero=True时用于填充)
            # 1: 用户1
            # 2: 用户2
            # ...
            # 1000: 用户1000
            # 如果有新用户，可以映射到索引0或其他预留位置
            """
            embedding_layer_dict[fc.name] = Embedding(fc.vocabulary_size + 1, fc.embedding_dim, name='emb_' + fc.name, mask_zero=True)

    return embedding_layer_dict

def embedding_lookup(feature_columns, input_layer_dict, embedding_layer_dict):
    # 这个函数是Input层与embedding层给接起来
    embedding_list = []
    
    for fc in feature_columns:
        _input = input_layer_dict[fc]
        _embed = embedding_layer_dict[fc]
        embed = _embed(_input)
        embedding_list.append(embed)

    return embedding_list

def concat_input_list(input_list):
    feature_nums = len(input_list)
    if feature_nums > 1:
        return Concatenate(axis=1)(input_list)
    elif feature_nums == 1:
        return input_list[0]
    else:
        return None

# 将所有的sparse特征embedding拼接
def concat_embedding_list(feature_columns, input_layer_dict, embedding_layer_dict, flatten=False):
    """
    离散特征经过embedding之后得到各自的embedding向量，这里存储在一个列表中
    :feature_columns:A list. 里面的每个元素是namedtuple(元组的一种扩展类型，同时支持序号和属性名访问组件)类型，表示的是数据的特征封装版
    :input_layer_dict:A dict. 这是离散特征对应的层字典 {'sparse_name': Input(shap, name, dtype)}形式
    :embedding_layer_dict: A dict. 离散特征构建的embedding层字典，形式{'sparse_name': Embedding(vocabulary_size, embedding_dim, name)}
    """
    embedding_list = [] # 创建一个空列表，用于存储所有特征的嵌入向量
    for fc in feature_columns: # fc 是特征列对象，包含特征的所有配置信息
        _input = input_layer_dict[fc.name]  # 获取输入层 
        _embed = embedding_layer_dict[fc.name]  # B x 1 x dim  获取对应的embedding层
        embed = _embed(_input)    # B x dim  将input层输入到embedding层中
        # 这是函数式API的核心：将输入层传递给嵌入层。 输入形状：(batch_size, 1)  输出形状：(batch_size, 1, embedding_dim)
        
        # 是否需要flatten, 如果embedding列表最终是直接输入到Dense层中，需要进行Flatten，否则不需要(特征拼接后输入全连接网络则需要)
        if flatten:
            # 需要Flatten的情况：(batch_size, 1, embedding_dim) → (batch_size, embedding_dim)
            # from tensorflow.keras.layers import Flatten
            # Flatten() 是 Keras 中的一个层，用于将多维输入展平为一维。让我详细解释它的用法和作用：
            # 连接CNN/Embedding与Dense层, 保持batch维度不变
            embed = Flatten()(embed)
            # 这段代码等价于：flatten_layer = Flatten() embed = flatten_layer(embed)
            
        embedding_list.append(embed)
    
    return embedding_list


# NoMask层的作用是移除输入张量的掩码信息
# 当我们需要连接多个具有不同掩码的张量时，Keras的Concatenate层会检查掩码一致性。如果掩码不一致，就会报错。NoMask层可以解决这个问题。
class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)
    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)
    # 去掉了mask的属性 比如历史行为序列中
    def call(self, x, mask=None, **kwargs):
        return x
    def compute_mask(self, inputs, mask):
        return None

# 这是一个增强版的拼接函数，支持可选的掩码处理。
# inputs: 要拼接的张量列表 axis: 拼接的维度（默认-1，即最后一个维度） mask: 是否保留掩码信息
def concat_func(inputs, axis=-1, mask=False):
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)
    
def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise NotImplementedError("dnn_feature_columns can not be empty list")


class DNN(Layer):
    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed
        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]
        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]
        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)
        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):
        deep_input = inputs
        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            fc = self.activation_layers[i](fc)
            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc
        return deep_input


def activation_layer(activation):
    if activation in ("dice", "Dice"):
        act_layer = Dice()
    elif isinstance(activation, (str, str)):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer


class PredictionLayer(Layer):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
         - **use_bias**: bool.Whether add bias term or not.
    """
    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")
        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)

        output = tf.reshape(x, (-1, 1))
        return output

def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)

def softmax(logits, dim=-1, name=None):
    try:
        return tf.nn.softmax(logits, dim=dim, name=name)
    except TypeError:
        return tf.nn.softmax(logits, axis=dim, name=name)

"""MMOE模型"""
def MMOE(dnn_feature_columns, num_experts=3, expert_dnn_hidden_units=(256, 128), tower_dnn_hidden_units=(64,),
        gate_dnn_hidden_units=(), l2_reg_embedding=0.00001, l2_reg_dnn=0, dnn_dropout=0, dnn_activation='relu',
        dnn_use_bn=False, task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr')):
    
    num_tasks = len(task_names)
    
    # 异常判断
    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))
    
    # 构建Input层并将Input层转成列表作为模型的输入
    input_layer_dict = build_input_layers(dnn_feature_columns)
    input_layers = list(input_layer_dict.values())
    
    # 筛选出特征中的sparse和Dense特征， 后面要单独处理
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns))
    
    # 获取Dense Input
    dnn_dense_input = []
    for fc in dense_feature_columns:
        dnn_dense_input.append(input_layer_dict[fc.name])
    
    # 构建embedding字典
    embedding_layer_dict = build_embedding_layers(dnn_feature_columns)

    # 离散的这些特特征embedding之后，然后拼接，然后直接作为全连接层Dense的输入，所以需要进行Flatten
    dnn_sparse_embed_input = concat_embedding_list(sparse_feature_columns, input_layer_dict, embedding_layer_dict, flatten=False)
    
    # 把连续特征和离散特征合并起来
    dnn_input = combined_dnn_input(dnn_sparse_embed_input, dnn_dense_input)
    
    # 建立专家层
    expert_outputs = []
    for i in range(num_experts):
        expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=2022, name='expert_'+str(i))(dnn_input)
        expert_outputs.append(expert_network)
    
    # 将多个专家网络的输出堆叠成一个三维张量
    # # 语法：Lambda(函数)(输入)
    # # Lambda(lambda x: 对x进行某些操作)(input_tensor)
    expert_concat = Lambda(lambda x: tf.stack(x, axis=1))(expert_outputs)
    
    # 建立多门控机制层
    mmoe_outputs = []
    for i in range(num_tasks):  # num_tasks=num_gates
        # 建立门控层
        gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=2022, name='gate_'+task_names[i])(dnn_input)
        gate_out = Dense(num_experts, use_bias=False, activation='softmax', name='gate_softmax_'+task_names[i])(gate_input)
        gate_out = Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)
        
        # gate multiply the expert
        """
        tf.reduce_sum(
        input_tensor,
        axis=None,
        keepdims=None,
        name=None)
        input_tensor：这是待处理的张量。
        axis：这个参数指定了沿着哪个维度（轴）进行求和。如果不指定，则默认对所有元素进行求和。
        keepdims：这个参数决定了求和后是否保持原张量的维度。如果设置为True，则输出的张量将保持输入张量的形状；如果设置为False或不设置（默认为False），则结果会降低维度。
        name：这个参数允许用户为这个操作指定一个名字，这在TensorFlow的计算图中有助于识别
        """
        gate_mul_expert = Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False), name='gate_mul_expert_'+task_names[i])([expert_concat, gate_out])
        
        mmoe_outputs.append(gate_mul_expert)
    
    # 每个任务独立的tower
    task_outputs = []
    for task_type, task_name, mmoe_out in zip(task_types, task_names, mmoe_outputs):
        # 建立tower
        tower_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=2022, name='tower_'+task_name)(mmoe_out)
        logit = Dense(1, use_bias=False, activation=None)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit)
        task_outputs.append(output)
    
    model = Model(inputs=input_layers, outputs=task_outputs)
    return model