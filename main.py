import argparse
import numpy as np
import os
import re
import csv
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from nltk.tokenize import word_tokenize, WordPunctTokenizer, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk import FreqDist
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
from tensorflow.keras.utils import to_categorical

# 파일 경로 설정
BOOK_LIST = ["book1.txt","book2.txt","book3.txt","book4.txt","book5.txt","book6.txt","book7.txt"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOK_PATH = os.path.join(BASE_DIR, "books", BOOK_LIST[0])

def save_file(data):
  save_path = os.path.join(os.getcwd(), "data", f"data.csv")
  with open(save_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

def load_data():
  """파일을 읽어오는 공통 함수"""
  try:
    with open(BOOK_PATH, 'r', encoding='utf-8') as f:
      return f.read()
  except FileNotFoundError:
    print(f"에러: {BOOK_PATH} 파일을 찾을 수 없습니다.")
    return ""
  
def step1_file_sentence():
  """
  [실습 1] 파일 전체를 문장 단위로 분리하기
  데이터셋 구축의 첫걸음입니다.
  """
  raw_text = load_data()
  # 텍스트가 너무 길면 앞부분 1000자만 샘플로 확인
  sentences = sent_tokenize(raw_text)

  content = []

  pattern = r"Page \|.*?- J\.K\. Rowling"

  print(f"전체 문장 개수: {len(sentences)}")
  print("-" * 30)

  for sent in sentences:
      # 줄바꿈(\n) 제거 후 출력
      clean_text = re.sub(pattern, " ", sent)
      replace_text = clean_text.replace('\n', ' ').replace('  ',' ').replace('“','').replace('”','').replace('’',"'").replace('—',"")
      content.append(replace_text)

  return content

def word_tokenization():
  sentences = step1_file_sentence()
  wordList = []
  for sentence in sentences:
    wordList.append(text_to_word_sequence(sentence))

  return wordList


def lemmatization():
  lemmatizer = WordNetLemmatizer()
  words_list = word_tokenization()
  lemmatizer_words = []
  for words in words_list:
    lemmatizer_words.append([lemmatizer.lemmatize(word, pos='v') for word in words])
  return lemmatizer_words


def stopWord():
  stop_words = set(stopwords.words('english')) 
  word_list = lemmatization()
  result = []
  for words in word_list:
    list = []
    for word in words:
      if word not in stop_words:
        list.append(word.lower())
    result.append(list)
  # save_file(result)
  return result


def integerEncoding():
  # global vocab, preprocessed_sentences
  print("[전처리 단계]")
  sentences = stopWord()
  vocab = {} 
  preprocessed_sentences = [] 

  for words in sentences:
    result = []
    for word in words: 
      if len(word) > 2: # 3) 길이가 짧은 단어 제거 (단어의 의미가 적은 경우가 많음)
        result.append(word)
        if word not in vocab:
          vocab[word] = 0 
        vocab[word] += 1
    preprocessed_sentences.append(result) 
  print(f"전처리된 문장: {preprocessed_sentences}")
  print("="*200)  

  # 단어 집합(Vocab) 출력 및 특정 단어 빈도 확인
  print('단어 집합 :', vocab)
  print("="*200)

# def getFrequency(vocab, vocab_size: int = 5):
#   # most_common()은 빈도수가 높은 순으로 리스트를 반환
#   vocab = vocab.most_common(vocab_size)
#   print(f"빈도수가 높은 상위 5개의 단어: {vocab}")
#   return vocab

# def step3():
#   print("[방법 3: NLTK FreqDist 활용]")
#   # np.hstack을 사용하여 2차원 리스트를 1차원으로 풀어서 전달
#   vocab = FreqDist(np.hstack(preprocessed_sentences))
#   print(f"barber라는 단어의 빈도수 : {vocab['barber']}")

#   # 상위 5개 추출 및 딕셔너리 컴프리헨션으로 인덱싱 (enumerate 활용)
#   vocab = getFrequency(vocab, 5)
#   word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab)}
#   print(f"상위 5개 인덱스 부여: {word_to_index}")


def padding():
  preprocessed_sentences = stopWord()
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(preprocessed_sentences)
  encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
  max_len = max(len(item) for item in encoded)
  padded = pad_sequences(encoded, padding='post', maxlen=max_len)
  save_file(padded)
  
  return(padded)



def one_hot_encoding():
  word_token = padding()
  max_list = []
  for word_list in word_token: 
    temp = int(max(item for item in word_list))
    max_list.append(temp)

  max_size = (max(item for item in max_list))
  print(max_size)
  encoding_list = []

  for i, word_list in enumerate(word_token):
    temp_list = [0] * (max_size + 1)
    for word_index in word_list:
      temp_list[int(word_index)] = 1
    encoding_list.append(temp_list)

  # save_file(encoding_list)
  return encoding_list


one_hot_encoding()