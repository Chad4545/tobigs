import numpy as np
import numpy.linalg as lin
from sklearn.preprocessing import StandardScaler


class mypca(object):
    '''
    k : component 수
    n : 원래 차원
    components : 고유벡터 저장소 shape (k,n)
    explain_values : 고유값 shape (k,)
    '''
    
    k = None
    components = None
    explain_values= None
    
    def __init__(self, k=None, X_train=None):
        '''
        k의 값이 initial에 없으면 None으로 유지
        '''
        if k is not None :
            self.k = k       
        if X_train is not None:
            self.fit(X_train)
            
    def fit(self,X_train=None):
        if X_train is None:
            print('Input is nothing!')
            return
        if self.k is None:
            self.k = X_train.shape[1]
            
        #############################################
        # TO DO                                     #
        # 인풋 데이터의 공분산행렬을 이용해                 #
        # components와 explain_values 완성           # 
        #############################################
        cov_mat = np.cov(X_train.T)
        eigen_values = lin.eig(cov_mat)[0]
        eigen_vectors = lin.eig(cov_mat)[1].T
        
        self.explain_values  = eigen_values[:self.k]  # k,
        self.components = eigen_vectors[:self.k,:]    # k,n
        
        
        #############################################
        # END CODE                                  #
        #############################################
        
        return self.explain_values, self.components
    
    def transform(self,X=None):
        if X is None:
            print('Input is nothing!')
            return
        
        result = None
        '''
        N : X의 행 수
        result의 shape : (N, k)
        '''
        #############################################
        # TO DO                                     #
        # components를 이용해 변환결과인                 #
        # result 계산                               #
        #############################################
        result = self.components.dot(X.T).T
     
        
        #############################################
        # END CODE                                  #
        #############################################       
        return result
    
    def fit_transform(self,X=None):
        if X is None:
            print('Input is nothing!')
            return
        self.fit(X)
        return self.transform(X)