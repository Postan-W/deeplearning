#keras的tokenizer可以方便的将输入语料向量化。它可以帮你清洗数据，最后得到向量表示
#pad_sequneces用来对数据进行截断或者补充操作,返回张量
"""
maxlen：None或整数，为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0.在命名实体识别任务中，主要是指句子的最大长度

dtype：返回的numpy array的数据类型

padding：‘pre’或‘post’，确定当需要补0时，在序列的起始还是结尾补

truncating：‘pre’或‘post’，确定当需要截断序列时，从起始还是结尾截断

value：浮点数，此值将在填充时代替默认的填充值0

tokenizer
参数
num_words: 需要保留的最大词数，基于词频。只有最常出现的num_words-1词会被保留。
filters: 一个字符串，其中每个元素是一个将从文本中过滤掉的字符。默认值是所有标点符号，加上制表符和换行符，减去 ' 字符。
lower: 布尔值。是否将文本转换为小写。
split: 字符串。按该字符串切割文本。
char_level: 如果为 True，则每个字符都将被视为标记。
oov_token: 如果给出，它将被添加到word_index中，并用于在 text_to_sequence调用期间替换词汇表外的单词。

num_words的作用是在转换成序列时，只有词频最高的那些词才会被考虑，句子中其他的词直接丢弃，所以出来的样本序列，里面的词汇表只有num_words-1个
"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
samples = ['today is saturday today today and and and andsunny','it is said is that tomorrow would rain']
#为了不使输入向量空间太大，有时我们限制一下只保留前N个最常出现的单词
tokenizer = Tokenizer(num_words=20)
tokenizer.fit_on_texts(samples)
print(tokenizer.word_index)
sequences = tokenizer.texts_to_sequences(samples)
"""
tokenization的操作中因为限制了num_words也就是每个句子在num_words最高频出现的词之外的词就被舍弃了，况且每个句子本身长度也通常不一样，所以一般每个句子的向量维度就基本不会一样。而pad_sequnces操作，maxlen给出了，那么不足的补0，过长的截断，就保证了长度一样
"""
result = pad_sequences(sequences,maxlen=6,padding='post',truncating='post')
#[[1, 2, 4, 1, 1, 3, 3, 3, 5], [6, 2, 7, 2, 8, 9, 10, 11]]
print(result)
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
