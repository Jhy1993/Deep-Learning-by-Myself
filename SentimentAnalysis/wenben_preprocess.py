# -*- coding: utf-8 -*-
import sys
import os
import jieba
reload(sys)
sys.setdefaultencoding('utf-8')
corpus_path="C:\Users\Jhy\Desktop\corpus"
#seg_path=""
#get ls
dir_list=os.listdir(corpus_path)
for mydir in dir_list:
    file_name=os.path.join(corpus_path,mydir)
    file_read=open(file_name,'rb')
    raw_corpus=file_read.read()
    seg_corpus=jieba.cut(raw_corpus)
    print seg_corpus