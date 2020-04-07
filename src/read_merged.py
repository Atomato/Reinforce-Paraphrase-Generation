from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
import os
import numpy as np
import pandas as pd
import re
import konlpy
from konlpy.tag import Kkma
from konlpy.utils import pprint
import sys
from sentence_decode import BeamSearch
import copy
import kss
import pickle as pkl

def isnan(data):
    if type(data)==type(0.0):
        if np.isnan(data):
            return True
    return False

def pageExtractor(data, col):
    if isnan(data['Page']):
        return False
    else:
        pages = data['Page']
        if '-' in pages:
        #일부만 표출되는 경우, 여러 페이지에 걸쳐서 표출되는 경우
            pages = pages.split('\n')
            for p in pages:
                p_tmp = p.split('-')
                if p_tmp[0] in col:
                    page = p_tmp[1].replace('P','').split(',')
                    if len(page)==1:
                        return page[0]
                    else:
                        return page
            return 'X'

        else:
            return pages.replace('P','')


def df_to_list(dataframe, columns):
    resultList = []
    n_data = len(dataframe)
    page_prev = False
    for i in range(n_data):
        data = dataframe.iloc[i]
        word_list = []
        if isnan(data[columns[0]]):  # nan인지 아닌지 확인
            # nan인 경우:
            continue
        else:
            # nan 이 아닌 경우
            word_list.append(data[columns[0]])
            for col in columns[1:-1]:
                if isnan(data[col]):
                    continue
                else:
                    if not page_prev:
                        # 이전 페이지가 저장이 안돼있으면
                        page = pageExtractor(data, col)
                        page_prev = page
                    page = pageExtractor(data, col)
                    if not page:
                        # page가 False(==nan)이면
                        page = page_prev
                        word_list.append([[data[col]], (col, page)])
                    else:
                        word_list.append([[data[col]], (col, page)])
                        page_prev = page

        resultList.append(word_list)
    return resultList

def split_sentences(sentence):
    #sentence= ['매우 긴 문장 여러개 묶음']
    sentenceList_tmp = []
    sentenceList = []
    sentenceList_fin = []
    equationList = []
    sentence = sentence[0]
    sentence = sentence.replace('\ne', '\\KE')
    sentenceList_tmp += sentence.split('\n')
    for sentence in sentenceList_tmp:
        sentenceList+=kss.split_sentences(sentence.strip().replace('\\KE', '\ne'))
    for i in range(len(sentenceList)):
        tmp  = equation_replacement(sentenceList[i])
        if tmp!='' and tmp!=' ' and tmp!='  ':
            sentenceList_fin.append(tmp[0])
            equationList.append(tmp[1])
    return sentenceList_fin, equationList

def equation_replacement(sentence):
    ################Find '$'####################
    eq_index = []
    equations = []
    sentence = sentence.replace("\frac", "\\frac")
    # \f가 \x0c로 바뀌는 문제 발생
    for i in range(len(sentence)):
        if sentence[i]=='$':
            eq_index+=[i]
    if len(eq_index)%2 != 0 or len(eq_index)== 0 :
        return sentence, equations
    ################Find '$'####################
    for i in range(0, len(eq_index),2):
        equations.append(sentence[eq_index[i]: eq_index[i+1]+1])
    for eq in equations:
        if eq == '$=$' or eq == '$ = $':
            sentence = sentence.replace(eq, "(등호)")
        elif len(eq)==3:
            sentence = sentence.replace(eq, "(미지수)")
        elif eq == "$\rightarrow$" or eq == "$\Rightarrow$" or eq == "$rightarrow$" or eq == "$Rightarrow$"or eq == "$\\rightarrow$" or eq == "$\\Rightarrow$":
            sentence = sentence.replace(eq, "(화살표)")
        else:
            sentence = sentence.replace(eq, "(수식)")
    return sentence, equations

def sentence_paraphrase(BS_object, sentence):
    ################Remove the order and restore##############################

    tip = re.compile("참고[0-9]+[.]\ ")
    numbering = re.compile("\([0-9]+\)\ ")
    match_tmp = tip.match(sentence)
    if match_tmp:
        beginning = match_tmp.group()
    else:
        match_tmp = numbering.match(sentence)
        if match_tmp:
            beginning = match_tmp.group()
        else:
            if sentence[:2]=='- ':
                beginning = '- '
            else:
                beginning = ''
    if len(beginning):
        sentence = sentence.replace(beginning, '')

    tmp = BS_object.decode(sentence)
    while (not tmp):
        tmp = BS_object.decode(sentence)
        print("Decoding error, trying again")
    sentence = beginning+tmp
    return sentence

def data_list_paraphrase(model_filename, data_list):
    sentenceList = []
    indexList = []
    BS = BeamSearch(model_filename, data_class='test')
    for i in range(len(data_list)):
        for j in range(len(data_list[i])-1):
            for k in range(len(data_list[i][j+1][0])):
                sentence = data_list[i][j+1][0][k]
                sentence = sentence_paraphrase(BS, sentence)
                sentenceList.append(sentence)
                indexList.append([i,j,k])
    return sentenceList, indexList


def replace_sentence_token(sentence):
    #############<expr>에 대한 작업###################
    split_tmp = sentence.split("<EXPR>")
    n_expr = len(split_tmp) - 1
    sentence = split_tmp[0]
    i = 0
    while (i < n_expr):
        sentence = sentence + "<EXPR" + str(i) + ">" + split_tmp[i + 1]
        i += 1
    #############<expr>에 대한 작업###################

    #############<arrw>에 대한 작업###################
    split_tmp = sentence.split("<ARRW>")
    n_arrw = len(split_tmp) - 1
    sentence = split_tmp[0]
    i = 0
    while (i < n_arrw):
        sentence = sentence + "<ARRW" + str(i) + ">" + split_tmp[i + 1]
        i += 1
    #############<arrw>에 대한 작업###################

    #############<unvar>에 대한 작업###################
    split_tmp = sentence.split("<UNVAR>")
    n_unvar = len(split_tmp) - 1
    sentence = split_tmp[0]
    i = 0
    while (i < n_unvar):
        sentence = sentence + "<UNVAR" + str(i) + ">" + split_tmp[i + 1]
        i += 1
    #############<unvar>에 대한 작업###################

    #############<equl>에 대한 작업###################
    split_tmp = sentence.split("<EQUL>")
    n_equl = len(split_tmp) - 1
    sentence = split_tmp[0]
    i = 0
    while (i < n_equl):
        sentence = sentence + "<EQUL" + str(i) + ">" + split_tmp[i + 1]
        i += 1
    #############<equl>에 대한 작업###################
    return sentence, n_expr, n_arrw, n_unvar, n_equl

def recover_equation(sentence, equationList):
    if len(equationList)==0:
        return sentence
    sentence_tuple = replace_sentence_token(sentence)
    sentence = sentence_tuple[0]
    n_expr = sentence_tuple[1]
    n_arrw = sentence_tuple[2]
    n_unvar = sentence_tuple[3]
    n_equl = sentence_tuple[4]
    n_expr_eq = n_arrw_eq = n_unvar_eq = n_equl_eq = -1

    for eq in equationList:
        if eq == '$=$' or eq == '$ = $':
            n_equl_eq+=1
            token = "<EQUL"+str(n_equl_eq)+">"
            sentence = sentence.replace(token, eq)
        elif len(eq)==3:
            n_unvar_eq+=1
            token = "<UNVAR"+str(n_unvar_eq)+">"
            sentence = sentence.replace(token, eq)
        elif eq == "$\rightarrow$" or eq == "$\Rightarrow$" or eq == "$rightarrow$" or eq == "$Rightarrow$"or eq == "$\\rightarrow$" or eq == "$\\Rightarrow$":
            n_arrw_eq +=1
            token = "<ARRW"+str(n_arrw_eq)+">"
            sentence = sentence.replace(token, eq)
        else:
            n_expr_eq+=1
            token = "<EXPR"+str(n_expr_eq)+">"
            sentence = sentence.replace(token, eq)
    return sentence

def decode_merge_file(model_filename = '../log/MLE/best_model/model_best_2800', data_filename = '../data/kor/excel/중1_문자와식_개념완성_Merge-1_pipeline제작2.xlsx'):

    df = pd.read_excel(data_filename, sheet_name = "Merge")
    cols = ['하위개념어', '정의M', '과정M', '성질M', '예', '참고M', 'Page']
    data_list = df_to_list(df, cols)

    for i in range(len(data_list)):
        for j in range(len(data_list[i]) - 1):
            sentenceList, equationList = split_sentences(data_list[i][j + 1][0])
            data_list[i][j + 1].append(data_list[i][j + 1][1])
            data_list[i][j + 1][0] = sentenceList
            data_list[i][j + 1][1] = equationList
        if (data_list[i][1][2][0] == '정의M'):
            data_list[i][1][0][0] = data_list[i][0] + ': ' + data_list[i][1][0][0]

    sL, iL = data_list_paraphrase(model_filename, data_list)

    transformedSentence = copy.deepcopy(data_list)
    recoveredSentence = copy.deepcopy(transformedSentence)
    x = 0
    for i, j, k in iL:
        transformedSentence[i][j + 1][0][k] = sL[x]
        recoveredSentence[i][j + 1][0][k] = recover_equation(sL[x],transformedSentence[i][j + 1][1][k])
        x += 1
    return recoveredSentence


if __name__ == '__main__':
    try:
        model_filenmae = sys.argv[1]
    except:
        model_filename = '../log/MLE/best_model/model_best_2800'

    try:
        data_filename = sys.argv[2]
    except:
        data_filename = '../data/kor/excel/중1_문자와식_개념완성_Merge-1_pipeline제작2.xlsx'

    df = pd.read_excel(data_filename, sheet_name = "Merge")
    cols = ['하위개념어', '정의M', '과정M', '성질M', '예', '참고M', 'Page']
    data_list = df_to_list(df, cols)

    for i in range(len(data_list)):
        for j in range(len(data_list[i]) - 1):
            sentenceList, equationList = split_sentences(data_list[i][j + 1][0])
            data_list[i][j + 1].append(data_list[i][j + 1][1])
            data_list[i][j + 1][0] = sentenceList
            data_list[i][j + 1][1] = equationList
        if (data_list[i][1][2][0] == '정의M'):
            data_list[i][1][0][0] = data_list[i][0] + ': ' + data_list[i][1][0][0]

    sL, iL = data_list_paraphrase(model_filename, data_list)

    transformedSentence = copy.deepcopy(data_list)
    recoveredSentence = copy.deepcopy(transformedSentence)
    x = 0
    for i, j, k in iL:
        transformedSentence[i][j + 1][0][k] = sL[x]
        recoveredSentence[i][j + 1][0][k] = recover_equation(sL[x],transformedSentence[i][j + 1][1][k])
        x += 1
    print(recoveredSentence)
    with open('../log/MLE/eq_recovered_list.pkl', 'wb') as f:
        pkl.dump(recoveredSentence, f)

