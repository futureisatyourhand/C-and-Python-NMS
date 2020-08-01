# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:42:11 2020

@author: hlj0812.duapp.com
"""
import numpy as np

def nms(dets,overthreshold=0.7):
    xmin=dets[:,0]
    ymin=dets[:,1]
    xmax=dets[:,2]
    ymax=dets[:,3]
    scores=dets[:,0]
    inds=scores.argsort()[::-1]
    aeras=(xmax-xmin+1)*(ymax-ymin+1)
    rtn=[]
    while inds.size>0:
        i=inds[0]
        rtn.append(i)
        x1=np.maximum(xmin[i],xmin[inds[1:]])
        x2=np.minimum(xmax[i],xmax[inds[1:]])
        y1=np.maximum(ymin[i],ymin[inds[1:]])
        y2=np.minimum(ymax[i],ymax[inds[1:]])
        aera=np.maximum(x2-x1+1,0)*np.maximum(y2-y1+1,0)
        union=aeras[i]+aeras[inds[1:]]-aera
        over=aera/union
        ind=np.where(over<overthreshold)[0]
        inds=inds[ind+1]
    return rtn

def soft_nms(dets,sigma=0.1,method=2,Nt=0.5,threshold=0.1):
    box_lens=dets.shape[0]
    for i in range(box_lens):
        tmpx1,tmpy1,tmpx2,tmpy2=dets[i,0],dets[i,1],dets[i,2],dets[i,3]
        max_pos=i
        max_scores=dets[i,4]
        
        ##找到以i为基线，大于该box的最大的分数的box
        pos=i+1
        while pos<box_lens:
            if max_scores<dets[pos,4]:
                max_scores=dets[pos,4]
                max_pos=pos
            pos+=1
        # 选取置信度最高的框与当前的框进行交换，即为使用置信度最高的框与其余的框进行比对
        dets[i,:],dets[max_pos,:]=dets[max_pos,:],dets[i,:]
        
        ## 将置信度最高的 box 赋给临时变量方便后续计算该box与后面box的ious
        tmpx1,tmpy1,tmpx2,tmpy2=dets[i,0],dets[i,1],dets[i,2],dets[i,3]
        
        #根据最高的box与后面box的ious来修正检测得分
        #该过程是将box_lens进行压缩
        pos=i+1
        while pos<box_lens:
            x1,y1,x2,y2=dets[pos,0],dets[pos,1],dets[pos,2],dets[pos,3]
            iw=min(x2,tmpx2)-max(x1,tmpx1)+1
            ih=min(y2,tmpy2)-max(y1,tmpy1)+1
            if iw>0 and ih>0:
                overlap=iw*ih
                ious=overlap/((x2-x1+1)*(y2-y1+1)+(tmpx2-tmpx1+1)*(tmpy2-tmpy1+1)-overlap+1e-8)
                if method==1:
                    if ious>Nt:
                        weight=1-ious
                    else:
                        weight=1
                elif method==2:
                    weight=np.exp(-ious**2/sigma)
                else:
                    if ious>Nt:
                        weight=0
                    else:
                        weight=1
                ## 赋予该box新的置信度
                dets[pos,4]=weight*dets[pos,4]
                
                # 如果box得分低于阈值threshold，则通过与最后一个框交换来丢弃该框
                if dets[pos,4]<threshold:
                    dets[pos,:]=dets[box_lens-1,:]
                    box_lens-=1
                    pos-=1
            pos+=1
    return [i for i in range(box_lens)]
                
    
dets=np.array([
        [30,20,300,100,0.99],
        [25,20,200,100,0.8],
        [30,30,250,100,0.91],
        [10,10,100,100,0.6],
        [60,60,200,100,0.56],
        [30,25,290,190,0.5]])
print(dets)
rtn=soft_nms(dets)
print(dets[rtn])
