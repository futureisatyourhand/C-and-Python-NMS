#---*coding:utf-8*---
import numpy as np
##检测里面的非极大值抑制代码
def nms(boxes):
    aeras=(boxes[:,2]-boxes[:,0]+1)*(boxes[:,3]-boxes[:,1]+1)
    print()
    inds=np.argsort(boxes[:,4])[::-1]
    res=[]
    eps=1e-8
    threshold=0.55
    while inds.size>0:
        i=inds[0]
        res.append(i)
        xmin=np.maximum(boxes[i,0],boxes[inds[1:],0])
        xmax=np.minimum(boxes[i,2],boxes[inds[1:],2])
        ymin=np.maximum(boxes[i,1],boxes[inds[1:],1])
        ymax=np.minimum(boxes[i,3],boxes[inds[1:],3])
        iw=np.maximum(xmax-xmin+1,0)
        ih=np.maximum(ymax-ymin+1,0)
        over=iw*ih
        union=aeras[i]+aeras[inds[1:]]-over+eps
        ious=over/union
        ind=np.where(ious<threshold)[0]
        inds=inds[ind+1]
    return res

##检测里面的优化版本的非极大值抑制版本：soft-nms
def softer_nms(boxes,method=2,threshold=0.5,sigma=0.1,eps=1e-8,thr=0.1):
    n=len(boxes)
    for i in range(n):
        max_pos=i
        max_score=boxes[i,4]
        flag=i
        xmin,ymin,xmax,ymax,ts=boxes[i,0],boxes[i,1],boxes[i,2],boxes[i,3],boxes[i,4]
        while flag<n:
            if max_score<boxes[flag,4]:
                max_score=boxes[flag,4]
                max_pos=flag
            flag+=1
        if max_pos!=i:
            #boxes[i,:],boxes[max_pos,:]=boxes[max_pos,:],boxes[i,:]
            boxes[i,:]=boxes[max_pos,:]
            boxes[max_pos,:]=xmin,ymin,xmax,ymax,ts
        pos=i+1
        while pos<n:
            xmin=max(boxes[i,0],boxes[pos,0])
            xmax=min(boxes[i,2],boxes[pos,2])
            ymin=max(boxes[i,1],boxes[pos,1])
            ymax=min(boxes[i,3],boxes[pos,3])
            iw,ih=max(xmax-xmin+1,0),max(ymax-ymin+1,0)
            over=iw*ih
            union=(boxes[i,2]-boxes[i,0]+1)*(boxes[i,3]-boxes[i,1]+1)-over+eps
            iou=over/union
            if method==1:
                if iou>threshold:
                    weight=0
                else:
                    weight=1
            elif method==2:
                weight=np.exp(-iou**2/sigma)
            else:
                weight=1
            boxes[pos,4]=weight*boxes[pos,4]
            if boxes[pos,4]<thr:
                boxes[pos,:],boxes[n-1,:]=boxes[n-1,:],boxes[pos,:]
                n-=1
                pos-=1
            pos+=1
    return [i for i in range(n)]

boxes=np.array([[10,20,100,200,0.1],
                [10,10,100,200,0.5],
                [100,25,125,321,0.7],
                [211,207,620,990,0.4],
                [200,30,300,310,0.8],
                [190,21,110,210,0.9]])
rtn=softer_nms(boxes)
print(rtn,boxes[rtn])
##传统的原始聚类方法:k-means
def kmeans(dataset,k):
    dataset=np.mat(dataset)
    n,m=dataset.shape
    centers=np.zeros((k,m))
    distance=np.mat(np.zeros((n,2)))
    for i in range(k):
        index=int(np.random.uniform(0,n))
        centers[i,:]=dataset[index,:]
    changed=True
    while changed:
        changed=False
        for i in range(n):
            minIndex=-1
            minDistance=float('inf')
            for j in range(k):
                dist=np.sqrt(np.sum(np.power(centers[j,:]-dataset[i,:],2)))
                if dist<minDistance:
                    minDistance=dist
                    minIndex=j
            if minIndex!=distance[i,0]:
                changed=True
                distance[i,:]=minIndex,minDistance**2
        for j in range(k):
            data=dataset[np.nonzero(distance[:,0].A==j)[0]]
            centers[j,:]=np.mean(data,0)
    return centers
dataset=np.random.randint(0,40,size=(20,5))
print("k-means:",kmeans(dataset,3))
##传统的优化类别中心的聚类方法：k-means++
def kmeans2(dataset,k):
    n,m=dataset.shape
    i=0
    centers=np.zeros((k,m))
    index=int(np.random.uniform(0,n))
    centers[0,:]=dataset[index,:]
    for i in range(1,k):
        dist=np.sqrt(np.sum(np.power(dataset[:,np.newaxis,:]-centers[:(i+1),:],2),2))
        ##
        print(dist.shape)
        d=list(np.argmin(dist,1))
        s=np.zeros((n,1))
        for j in range(n):
            s[j]=dist[j,d[j]]
        s=s/sum(s)
        centers[i,:]=dataset[np.argmax(s),:]
    
    changed=True
    distance=np.mat(np.zeros((n,2)))
    while changed:
        changed=False
        for i in range(n):
            d=np.sqrt(np.sum(np.power(centers-dataset[i,:][np.newaxis,:],2),1))
            j=np.argmin(d)
            if distance[i,0]!=j:
                distance[i,:]=j,d[j]
                changed=True
        for j in range(k):
            centers[j,:]=np.mean(dataset[np.nonzero(distance[:,0].A==j)[0]],axis=0)
    return centers
print("k-means++:",kmeans2(dataset,3))
##算法题：求N！中为0的个数
def ling_number(n):
    s=0
    while n>0:
        s+=n/5
        n/=5
    return s
print(ling_number(100))