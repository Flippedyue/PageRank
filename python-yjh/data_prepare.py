import numpy as np
import os
from os import write

'''
    对原始数据进行简单预处理
    生成了文件:

    index_list.txt :  7115个网页节点的编码
    new_data.txt   :  原始文件中的节点序号修改为编码后的编号
    Mat.txt        :  按目的节点升序排序，每行只一个目的节点。每行内容依次：src degree dest
    Mat2.txt       :  每行内容：src degree dest  每行的dest存入src的所有目的节点
    old_rank.txt   :  初始化rank值为 1/N

'''

if __name__ == "__main__":
    
    '''
    Step 1:
        读出所有节点，去重排序并编码

    '''
    datax=[]
    datay=[]
    index_list=set()    
    with open("WikiData.txt", "r", encoding="utf8") as f:
        for line in f:
            _from = int(line.split()[0])
            _to = int(line.split()[1])
            datax.append(_from)
            datay.append(_to)
            index_list.add(_to)
            index_list.add(_from)
    pageNum=len(index_list)
    print("pageNum="+str(pageNum)+"\n")

    with open("index_list.txt","w",encoding="utf8") as f:
        for i in index_list:
            f.write(str(i)+"\n")

    encoder=[]
    with open("index_list.txt","r",encoding="utf8") as f:
        for line in f:
            encoder.append(int(line))
    
    

    '''
    Step 2:
        用新的序号代替原节点，生成 new_data

    '''
    for i in range(len(datax)):
        datax[i]=encoder.index(datax[i])
        datay[i]=encoder.index(datay[i])
    data=np.vstack((datax,datay)).T
    with open("new_data.txt","w",encoding="utf8") as f:
        for i in range(len(datax)):
            f.write(str(data[i][0])+" "+str(data[i][1])+"\n")



    '''
    Step 3:
        初始化稀疏矩阵需要的信息
        Mat2.txt :  src degree dest1_dest2_ 
        Mat.txt  :  src degree dest

    '''
    M_line=[]
    m_dict=dict()
    for i in range(pageNum):
        m_dict={"src":i,"degree":0,"dest":[]}
        M_line.append(m_dict)
        #m_dict.clear()       #clear会直接删掉导致出错

    for i in range(len(datax)):
        #data[i]的取值范围：0-7114
        #这里保存了，所有出现在datax中的节点的 degree & dest,没出现的分别初始化为0和[]
        M_line[datax[i]]['degree']+=1
        M_line[datax[i]]["dest"].append(datay[i])
    with open("Mat2.txt", 'w', encoding='utf-8') as f:
        for i in range(pageNum):
            str_dest=""
            for j in range(len(M_line[i]['dest'])):
                str_dest+=str(M_line[i]['dest'][j])+"_"
            f.write(str(M_line[i]['src'])+' '+str(M_line[i]['degree'])+' '+str_dest+"\n")
    
    M2_line=np.zeros((len(datax),3),dtype=int)
    k=0
    for i in range(pageNum):  # M_lines长度就是7115
        if M_line[i]['degree']!=0:
            for j in range(len(M_line[i]['dest'])):
                M2_line[k][0]=M_line[i]['src']
                M2_line[k][1]=M_line[i]['degree']
                M2_line[k][2]=M_line[i]['dest'][j]
                k+=1    
    M2_line= M2_line[np.argsort(M2_line[:,2])]
    with open("Mat.txt", 'w', encoding='utf-8') as f:
        for i in range(len(M2_line[:,0])):
            f.write(str(M2_line[i][0])+" "+str(M2_line[i][1])+" "+str(M2_line[i][2])+"\n")



    '''
    Step 4:
        初始化rank值

    '''
    with open("old_rank.txt", "w", encoding="utf8") as f:
        for i in range(pageNum):
            f.writelines(str(1 /pageNum) + "\n")