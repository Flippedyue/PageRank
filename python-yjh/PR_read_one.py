import numpy as np
import os
import datetime

'''
本程序实现数据分块，每次从磁盘中读取目的节点为i的一组信息，
即指向节点i的其他节点序号和出度。
在此基础上计算节点i的rank值，每完成一个节点的rank值计算，释放内存。
因为I/O较为频繁，程序运行时间较长。
程序本地运行时间 28min 30s
内存占用约48M

'''


start = datetime.datetime.now()
file_pt=0      #保存上一次读取Mat.txt时的位置
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
with open("index_list.txt","r",encoding="utf8") as f:
    for line in f:
        index_list.append(int(line))
pageNum=len(index_list)

block_size=1500
block_num=int(pageNum/block_size)+1 #5



'''
Step 2:
    读入old_rank.txt,初始化old_rank，new_rank

'''
old_rank_list = open("old_rank.txt", "r", encoding="utf8").readlines()
old_rank = np.array(old_rank_list, dtype=np.float32)
new_rank = np.zeros(pageNum)



'''
Step 4:
    计算pagerank

'''
def get_rank(idx,file_pt,old_rank,filename="Mat.txt"):
    '''
        对每个节点i都打开一次文件，只读dest为i的信息。
        实际上，block的设置并不需要……
        在PR_with_block.py中实现block_stripe

    '''
    rank=0
    with open(filename,"r",encoding="utf8") as f:
        temp=0
        for line in f.readlines()[file_pt:]:
            if int(line.split()[2])==idx:
                temp+=1
                rank+=Beta*old_rank[int(line.split()[0])]/int(line.split()[1])
            else:
                break
        file_pt+=temp
    f.close()
    return file_pt,rank
        

'''
    进入迭代
    因为函数get_rank中使用中间变量rank将累加后计算结果直接赋给new_rank,
    所以迭代中不需要再将new_rank初始化为0

'''
with open("Sum_read_one.txt","w",encoding="utf8") as f:
    while True:
        print("ok1")
        file_pt=0
        sum_=0
        for count in range(block_num):
            if count!=block_num-1:
                for i in range(count*block_size,(count+1)*block_size):
                    file_pt,new_rank[i]=get_rank(i,file_pt,old_rank)
            else:
                for i in range(count*block_size,pageNum):
                    file_pt,new_rank[i]=get_rank(i,file_pt,old_rank)
        S = np.ones(pageNum) * (1 - sum(new_rank)) / pageNum
        new_rank += S
        flag=1
        for i in range(pageNum):
            if abs(new_rank[i]-old_rank[i])>1e-8:
                flag=0
            old_rank[i]=new_rank[i]
            sum_+=new_rank[i]
        f.write(str(float(sum_))+"\n")
        show_mem()
        print("ok222222222222")
        if flag==1:
            break
    


'''
Step 5:
    输出前100rank值的节点和其rank值

'''
with open("Result_read_one.txt","w",encoding="utf8") as f:
    for i in range(0,100):
        idx=int(np.argwhere(new_rank==max(new_rank))[0][0])
        f.write(str(index_list[idx])+"\t"+str(float(max(new_rank)))+"\n")
        print(str(index_list[idx])+"\t"+str(float(max(new_rank)))+"\n")
        new_rank[idx]=0
        

end = datetime.datetime.now()
run_time=(end-start).seconds
print("running time:  ",int(run_time/60),"min",run_time%60,"s")