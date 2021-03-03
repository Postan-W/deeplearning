#词干提取与词形还原
import nltk
from nltk.stem import PorterStemmer as stemmer
from nltk.stem import WordNetLemmatizer as wnl
from nltk import word_tokenize
from nltk import sent_tokenize
stems = stemmer()

words = ['annoys','annoying','annoyed','studies']

words_stem = [stems.stem(word=word) for word in words]

#上面的studies就无法还原成study。用词形还原工具则可以
#词形还原器的能力更强，但是对于大型语料库可以感受到它的速度更慢


lemmatizer = wnl()

words_lemmatizer = [lemmatizer.lemmatize(word=word,pos='v') for word in words]
print(words_lemmatizer)
"""
注意，这里是需要wordnet语料的，但是由于中国防火墙的问题程序是无法下载的。在网上很容易搜索及下载到wordnet，
然后放到相应的文件夹下。到底放在哪个文件夹？程序报错的信息会直接引导你找到正确答案
"""

#标记化单词
s = 'hi! my name is Trump'
word_tokenization = word_tokenize(s)
print(word_tokenization)

sentence_tokenization = sent_tokenize(s)
print(sentence_tokenization)

#删除停止词比如and、or、be等虚词
from nltk.corpus import stopwords

s = "the weather is hot and i want to go for a swim"
stop_words = set(stopwords.words('english'))

tokens = word_tokenize(s)

tokens = [word for word in tokens if not word in stop_words]
print(tokens)
