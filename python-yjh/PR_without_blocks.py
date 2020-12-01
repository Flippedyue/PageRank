import numpy as np
import os
from scipy.sparse import coo_matrix
from scipy import sparse
import datetime


'''
本程序中使用了幂迭代的方法计算，采用了稀疏矩阵，
但未将数据分块，实现pagerank的计算。
构建稀疏矩阵时，使用scipy中的coo_matrix，
转成bsr_matrix后使用scipy.sparse内置的.dot()方法进行矩阵乘法。
程序本地运行时间 7s
内存占用约800M

'''



start = datetime.datetime.now()
index_list=[]
old_rank=[]
new_rank=[]
Beta=0.85

#函数用于结合memory_profiler计算程序占用内存
#在循环中调用，显示在最后一次迭代结束时，程序当前占用内存
@profile
def show_mem():
    pass


'''
Step 1:
    获取节点总数和编码

'''
with open("index_list.txt","r",encoding="utf-8") as f:
    for line in f:
        index_list.append(int(line))
pageNum=len(index_list)



'''
Step 2:
    读入old_rank.txt,初始化old_rank，new_rank

'''
old_rank_list = open("old_rank.txt", "r", encoding="utf-8").readlines()
old_rank = np.array(old_rank_list, dtype=np.float32)
new_rank = np.zeros(pageNum)



'''
Step 3:
    构建稀疏矩阵：
    1、从Mat.txt中读取需要的信息 src degree dest
    2、

'''
def list_str2int(str_list)->list:
    '''
        在Mat.txt中，dest保存形式为 字符串：dest1_dest2_dest3_
        需要处理字符串，形成整形列表

    '''
    int_list=[]
    if str_list==None:
        return []
    else:
        str_list=str_list.split("_")
        for i in range(len(str_list)):
            if str_list[i]!="":
                int_list.append(int(str_list[i]))
        return int_list

#                                       #
#初始化字典列表，保存从Mat.txt中提取的信息#
#                                       #
M_line=[]
m_dict=dict()
for i in range(pageNum):
    m_dict={"src":i,"degree":0,"dest":[]}
    M_line.append(m_dict)

#        #
#读入内存#
#        #
with open("Mat2.txt","r",encoding="utf-8") as f:
    i=0
    for line in f:
        temp_src=int(line.split()[0])
        temp_degree=int(line.split()[1])
        if len(line.split())==2:
            temp_dest=[]
        else:
            temp_dest=list_str2int(str(line.split()[2]))
        M_line[i]['src']=temp_src
        M_line[i]['degree']=temp_degree
        M_line[i]['dest']=temp_dest
        i+=1
       
#             #
#初始化稀疏矩阵#
#             #
row=[]
col=[]
data=[]
for i in range(pageNum):
    if M_line[i]["degree"]!=0:
        for j in range(len(M_line[i]["dest"])):
            col.append(M_line[i]['src'])
            row.append(M_line[i]['dest'][j])
            data.append(1/M_line[i]['degree'])
    if M_line[i]["degree"]==0:
        for j in range(pageNum):
            col.append(M_line[i]['src'])
            row.append(j)
            data.append(1/pageNum)

sp_co_matrix=coo_matrix((data,(row,col)),shape=(pageNum,pageNum))
M=sp_co_matrix.tobsr()
print(type(M))



'''
Step 4:
    计算pagerank
    使用scipy.sparse定义的稀疏矩阵内置的.dot()实现矩阵乘法

'''
with open("Sum_without_block.txt","w",encoding="utf-8") as f:
    while True:
        sum_=0
        old_rank=old_rank.reshape(-1,1)
        new_rank=(Beta*M).dot(old_rank)
        new_rank+=(1-Beta)/pageNum
        flag=1
        for i in range(pageNum):
            if abs(new_rank[i]-old_rank[i])>1e-8:
                flag=0
            old_rank[i]=new_rank[i]
            sum_+=new_rank[i]
        f.write(str(float(sum_))+"\n")
        show_mem()
        if flag==1 :
            break
        
        


'''
Step 5:
    输出前100rank值得  节点和其rank值

'''
with open("Result_without_block.txt","w",encoding="utf-8") as f:
    for i in range(0,100):
        idx=int(np.argwhere(new_rank==max(new_rank))[0][0])
        f.write(str(index_list[idx])+"\t"+str(float(max(new_rank)))+"\n")
        print(str(index_list[idx])+"\t"+str(float(max(new_rank)))+"\n")
        new_rank[idx][0]=0

end = datetime.datetime.now()
run_time=(end-start).seconds
print("running time:  ",int(run_time/60),"min",run_time%60,"s")