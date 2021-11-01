import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splprep,splev
from sklearn.cluster import DBSCAN
from scipy import interpolate
from numpy import linalg 
"""
Find path connecting data point
"""
class FindPath(object):
    def __init__(self, n = 20,traj_w=10,sf=1):
        self.n = n
        self.t = np.linspace(0,1,n)
        self.error = 1e100
        self.error_log=[]
        self.data_rc=[]
        self.tangent_v=[]
        self.w=traj_w
        self.sf=sf
        
    def fit(self, p1, p2,data):
        self.p1 = np.expand_dims(p1,axis=0)
        self.p2 = np.expand_dims(p2,axis=0)
        self.data = data
        self.cluster_labels=[]

        self.spline = p1 + np.multiply((p2 - p1) , np.expand_dims(self.t,axis=1))
        plt.plot(self.spline[:,0], self.spline[:,1]) 
        plt.show()
        
        self.best_spline=self.spline.copy()
        
        update=True
        while update:
            update=self.__move()
#             if len(self.error_log)>=2:
#                 if abs(self.error_log[-1]-self.error_log[-2])/self.error_log[-1]<0.005:
#                     update=False
#                 else:
#                     update=True
#             else:
#                 update=True
            print(update)


        return self.best_spline,self.data_rc
    
    def __move(self):
        

#         vor_pts=[[] for i in range(self.spline.shape[0])]
#         vor_traj=[[] for i in range(self.spline.shape[0])]

#         pts_sum=np.zeros_like(self.spline)
#         density = np.zeros(self.n)
#         error_arr=np.zeros(self.n)
# #         self.data=shuffle(self.data)
#         for ind in range(len(self.data)):
#             traj_t_span = self.data[ind].shape[0]
#             spline_inds=[]
#             spline_dist=[]
#             for i in range(traj_t_span):
#                 p = self.data[ind][i,:]

#                 dist = linalg.norm(self.spline - p[None,:], axis = 1)
#                 spline_inds.append(np.argmin(dist))
#                 spline_dist.append(np.amin(dist))
#             spline_inds=np.array(spline_inds)
#             spline_dist=np.array(spline_dist)
# #             print(spline_dist)
#             for j in range(self.n):
#                 vor_split_traj=np.where(spline_inds==j)[0]
#                 if len(vor_split_traj)>0:
#                     density[j]+=1
#                     pts_sum[j]+=np.mean(self.data[ind][vor_split_traj],axis=0)
#                     error_arr[j]+=np.mean(spline_dist[vor_split_traj])
#         print(density,np.std(density)) 
             

#                 vor_split_traj=consecutive_arrs(np.where(spline_inds==j)[0], stepsize=1)
#                 for u in range(len(vor_split_traj)):
#                     if len(vor_split_traj[u])>0:
#                         density[j]+=1
#                         pts_sum[j]+=np.mean(self.data[ind][vor_split_traj[u]],axis=0)
#                         error_arr[j]+=np.mean(spline_dist[vor_split_traj[u]])
#             print(error_arr)

#         density[[0,-1]] = 0
#         updateIndexs = np.nonzero(density)
        w=self.w
        traj_error_arr1,traj_sum1,pts_error_arr1,pts_sum1,density1,data_rc1=self.__error() 
        centers = self.spline.copy()
#         print(pts_sum1) 
        
        updateIndexs = np.where(density1>0)[0]
        centers[updateIndexs,:]=(np.divide(pts_sum1[updateIndexs,:],np.expand_dims(density1[updateIndexs],axis=1))+\
                                w*traj_sum1[updateIndexs,:]/len(self.data))/(1+w)



        self.spline,tangent_v=self.__discretize(centers)
        
        dot_color=np.arange(self.spline.shape[0])
        cm=plt.cm.get_cmap('jet')
        plt.figure(figsize=(7.5,5))
        plt.scatter(self.spline[:,0],self.spline[:,1],c=dot_color,cmap=cm)
        plt.xlabel('Morphology PC1',fontsize=16)
        plt.ylabel('Vimentin Haralick PC1',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        #plt.savefig(str(self.error)+'.png',dpi=300)
        plt.show()
#         print(tangent_v)
        traj_error_arr2,traj_sum2,pts_error_arr2,pts_sum2,density2,data_rc2=self.__error() 

#         print(w*np.sum(traj_error_arr2),np.sum(pts_error_arr2))
        error=w*np.sum(traj_error_arr2)+np.sum(pts_error_arr2)
#         print(self.error_log)


        self.error_log.append(error)

        if error<self.error:
            self.error=error
            self.best_spline=self.spline.copy()
            self.data_rc=data_rc2
            self.tangent_v=tangent_v
            return True
        else:
            return False

        

    def __discretize(self,centers,sample_n=1,st_move=0.02):

        
        spline=centers.copy()
        centers[0,:]=self.p1#+st_move*(centers[0,:]-self.p1)#self.spline[0,:]+st_move*(centers[0,:]-self.spline[0,:])
        centers[-1,:]=self.p2#+st_move*(centers[-1,:]-self.p2)#self.spline[-1,:]+st_move*(centers[-1,:]-self.spline[-1,:])
        t = np.linspace(0,1, sample_n* self.n)

        



        center_list=[centers[:,i] for i in range(centers.shape[1])]
        tck, u = interpolate.splprep(center_list,s=self.sf)
        spline_list=splev(t,tck,ext=3)
        for k in range(len(spline_list)):
            spline[:,k]=spline_list[k][::sample_n]

#         self.spline[1:-1,0]=v1[sample_n:-sample_n:sample_n]
#         self.spline[1:-1,1]=v2[sample_n:-sample_n:sample_n]
#         self.spline[1:-1,2]=v3[sample_n:-sample_n:sample_n]
#         self.spline[1:-1,3]=v4[sample_n:-sample_n:sample_n]

#         self.spline[[0,-1],0]=self.spline[[0,-1],0]+st_move*(v1[[0,-1]]-self.spline[[0,-1],0])
#         self.spline[[0,-1],1]=self.spline[[0,-1],1]+st_move*(v2[[0,-1]]-self.spline[[0,-1],1]) 
        
#         tck, u = interpolate.splprep([self.spline[:,0], self.spline[:,1],self.spline[:,2],self.spline[:,3]],s=s)   
#         l, r = [(1, tuple(self.p1))], [(2, tuple(self.p2))]
#         clamped_spline = interpolate.make_interp_spline(u, self.spline, bc_type=(l, r))
#         v1,v2,v3,v4 = clamped_spline(t).T([x_old, y_old, z_old]).
        
        
        # spline[:,0]=v1[::sample_n]
        # spline[:,1]=v2[::sample_n]
        # spline[:,2]=v3[::sample_n]
        # spline[:,3]=v4[::sample_n]
#         spline[0,:]=self.spline[0,:]+st_move*(np.array([v1[0],v2[0],v3[0],v4[0]])-self.spline[0,:])
#         spline[-1,:]=self.spline[-1,:]+st_move*(np.array([v1[-1],v2[-1],v3[-1],v4[-1]])-self.spline[-1,:])


        
        
#         plt.title('smooth')
#         plt.plot(self.spline[:,0], self.spline[:,1]) 
#         plt.show()
#         plt.plot(self.spline[:,2], self.spline[:,3]) 
#         plt.show()
        
        tangent_v=interpolate.splev(t,tck,der=1)
        
        return spline,tangent_v
        
    def __error(self):
        
        
        pts_error_arr=np.zeros(self.n)
        density = np.zeros(self.n)
        pts_sum=np.zeros_like(self.spline)
        
        traj_error_arr=np.zeros(self.n)
        traj_sum=np.zeros_like(self.spline)
        data_rc=[]
        for ind in range(len(self.data)):
            traj_t_span = self.data[ind].shape[0]
            spline_inds=[]
            spline_dist=[]
            for j in range(self.n):
                p=self.spline[j]
                
                dist=linalg.norm(self.data[ind]-p[None,:],axis=1)
#                 print(dist.shape)
                traj_error_arr[j]+=np.amin(dist)
                traj_sum[j]+=self.data[ind][np.argmin(dist),:]
                

            traj_rc=np.zeros((self.data[ind].shape[0],))
            for i in range(traj_t_span):
                p = self.data[ind][i,:]
                dist = linalg.norm(self.spline - p[None,:], axis = 1)
                pts_error_arr[np.argmin(dist)]+=np.amin(dist)
                pts_sum[np.argmin(dist)]+=self.data[ind][i,:]
                density[np.argmin(dist)]+=1
                traj_rc[i]=np.argmin(dist)
            data_rc.append(traj_rc)
                
                
#                 spline_inds.append(np.argmin(dist))
#                 spline_dist.append(np.amin(dist))
#             spline_inds=np.array(spline_inds)
#             spline_dist=np.array(spline_dist)
#             for j in range(self.n):
#                 vor_split_traj=np.where(spline_inds==j)[0]
#                 if len(vor_split_traj)>0:
#                     density[j]+=1
#                     pts_sum[j]+=np.mean(self.data[ind][vor_split_traj],axis=0)
#                     error_arr[j]+=np.mean(spline_dist[vor_split_traj])
#         print(density,np.std(density))
        return traj_error_arr,traj_sum,pts_error_arr,pts_sum,density,data_rc