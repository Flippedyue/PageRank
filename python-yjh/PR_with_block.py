import numpy as np
import os
import datetime

'''
本程序实现数据分块， block_stripe
对每个block
将目的节点为block中节点的对应的信息读入内存(src degree dest)
计算rank值，释放空间，
读取计算下一个block中节点rank值需要的信息。
本地运行时间约 12min 33s
程序内存占用约 48M

'''

start = datetime.datetime.now()
index_list=[]
old_rank=[]
new_rank=[]
mat_list=list()  
Beta=0.85 


#函数用于结合memory_profiler计算程序占用内存
#在循环中调用，显示在最后一次迭代结束时，程序当前占用内存
@profile
def show_mem():
    pass

'''
Step 1:
    获取节点总数和编码,定义block的数量和大小

'''
with open("index_list.txt","r",encoding="utf8") as f:
    for line in f:
        index_list.append(int(line))
pageNum=len(index_list)

block_size=1500
block_num=int(pageNum/block_size)+1


'''
Step 2:
    读入old_rank.txt,初始化old_rank，new_rank
    在内存中维护这两个数组

'''
old_rank_list = open("old_rank.txt", "r", encoding="utf8").readlines()
old_rank = np.array(old_rank_list, dtype=np.float32)
old_rank_list=[]
new_rank = np.zeros(pageNum)



'''
Step 3:
    计算pagerank

'''
def implement_rank(idx,mat_list,old_rank,new_rank):
    '''
    计算rank值的第一部分：B*ri/di
    idx      :  需要计算rank值的节点的编号
    mat_list :  保存从Mat.txt中逐行读取的信息： src degree dest
    
    '''
    for i in range(len(mat_list)):
        if int(mat_list[i][2])==idx:
                new_rank[idx]+=Beta*old_rank[int(mat_list[i][0])]/int(mat_list[i][1])
    return



'''
    进入迭代

'''
with open("Sum_with_block.txt","w",encoding="utf8") as f:
    while True:
        sum_=0
        new_rank = np.zeros(pageNum)
        for count in range(block_num):
            with open("Mat.txt","r",encoding="utf8") as f:   
                while(1):
                    line=f.readline()
                    if line=="":
                        break
                    if  int(line.split()[2])<(count+1)*block_size and int(line.split()[2])<pageNum:
                        if int(line.split()[2])>=(count)*block_size:
                            mat_list.append([line.split()[0],line.split()[1],line.split()[2]])
                    else:
                        f.close()
                        break
                if count!=block_num-1:
                    for i in range(count*block_size,(count+1)*block_size):
                        implement_rank(i,mat_list,old_rank,new_rank)
                else:
                    for i in range(count*block_size,pageNum):
                        implement_rank(i,mat_list,old_rank,new_rank)    
            mat_list=[]
        S = np.ones(pageNum) * (1 - sum(new_rank)) / pageNum
        new_rank += S
        flag=1
        for i in range(pageNum):
            if abs(new_rank[i]-old_rank[i])>1e-8:
                flag=0
            old_rank[i]=new_rank[i]
            sum_+=new_rank[i]
        print((str(float(sum_))))
        show_mem()
        if flag==1:
            break
       

with open("Result_with_block.txt","w",encoding="utf8") as f:
    for i in range(0,100):
        idx=int(np.argwhere(new_rank==max(new_rank))[0][0])
        f.write(str(index_list[idx])+"\t"+str(float(max(new_rank)))+"\n")
        print(str(index_list[idx])+"\t"+str(float(max(new_rank)))+"\n")
        new_rank[idx]=0

end = datetime.datetime.now()
run_time=(end-start).seconds
print("running time:  ",int(run_time/60),"min",run_time%60,"s")