import argparse
import os
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from nltk.tokenize import word_tokenize, WordPunctTokenizer, sent_tokenize
from nltk.tag import pos_tag

# 파일 경로 설정
BOOK_LIST = ["book1.txt","book2.txt","book3.txt","book4.txt","book5.txt","book6.txt","book7.txt"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOK_PATH = os.path.join(BASE_DIR, "books", BOOK_LIST[0])

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
  
  print(f"전체 문장 개수: {len(sentences)}")
  print("-" * 30)
  print("문장 미리보기:")
  for i, sent in enumerate(sentences[:12]):
      # 줄바꿈(\n) 제거 후 출력
      print(f"{i+1}: {sent.replace('\n', ' ')}")


def step0():
  """
  영어 단어 토큰화(Word Tokenization) 비교
  - 다양한 라이브러리가 구두점(., !), 줄임표(Don't)를 어떻게 처리하는지 확인
  """
  txt = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
  
  # 1. word_tokenize: NLTK의 기본 토큰화. "Don't"를 "Do"와 "n't"로 분리
  print('단어 토큰화1 :', word_tokenize(txt))
  
  # 2. WordPunctTokenizer: 구두점을 별도로 분리하는 특징이 있음. "Don't"를 "Don", "'", "t"로 분리
  print('단어 토큰화2 :', WordPunctTokenizer().tokenize(txt))
  
  # 3. text_to_word_sequence: 케라스(TensorFlow) 제공. 모든 문자를 소문자로 바꾸고 구두점(마침표, 컴마 등)을 제거
  print('단어 토큰화3 :', text_to_word_sequence(txt))

def step1():
  """
  영어 문장 토큰화(Sentence Tokenization)
  - 마침표(.)가 단순히 약어(Ph.D.)에 쓰인 것인지 문장의 끝인지 구분하는 능력을 확인
  """
  txt1 = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
  txt2 = "I am actively looking for Ph.D. students. and you are a Ph.D student."
  
  # sent_tokenize: NLTK에서 제공하는 영어 문장 분리 도구
  print('문장 토큰화1 :', sent_tokenize(txt1))

  # Ph.D. 같은 약어를 인식하여 문장 중간에서 끊기지 않도록 처리
  print('문장 토큰화2 :', sent_tokenize(txt2))

def step3():
  """
  영어 품사 태깅(Part-of-Speech Tagging)
  - 단어의 역할(명사, 동사, 형용사 등)을 정의된 태그로 표시
  """
  txt = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
  
  # 먼저 단어 토큰화를 수행
  tokenized_sentence = word_tokenize(txt)
  print('단어 토큰화 :', tokenized_sentence)

  # pos_tag: NLTK 제공. PRP(인칭 대명사), VBP(동사), RB(부사) 등의 태그를 붙임
  print('품사 태깅 :', pos_tag(tokenized_sentence))

if __name__ == '__main__':
  # 터미널(CLI)에서 -s 인자를 받아 실행할 단계를 결정
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', type=int, help="실행할 단계 선택 (step1~step5, 미입력 시 step0)")
  try:
    args = parser.parse_args()
    # 입력된 step 값에 따라 해당 함수 실행
    if args.s == 1:
      step1()
    elif args.s == 3:
      step3() 
    else:
      # 기본값 또는 step0 입력 시 실행
      # step0()
      step1_file_sentence()
  except SystemExit:
    # argparse 오류 시(잘못된 인자 입력 등) 프로그램 종료 처리
    print("잘못된 인자 입력이 발생했습니다.")
    exit()
