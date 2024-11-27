from utils import *
import os
import numpy as np
import cvxopt
import matplotlib.pyplot as plt
import time

class SoftMarginSVMQP:
    
    def __init__(self, C, kernel = 'linear'):

        '''
        Additional hyperparams allowed
        '''
        self.kernel = kernel
        self.C = C
        self.gamma = 0.1

        if kernel == 'linear':
            self.kernel_function = self.linear
        elif kernel == 'rbf':
            self.kernel_function = self.RBF
        pass
    
    def linear(self,x,y):
        return np.dot(x,y)
    
    def RBF(self,x,y,gamma=0.1):
        dist_sq = np.sum((x - y) ** 2)
        return np.exp(-gamma*dist_sq)
    
    def rbf_kernel(self, X1, X2):
        sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * X1.dot(X2.T)
        return np.exp(-self.gamma * sq_dists)
    
    def linear_kernel(self, X1, X2):
        return X1.dot(X2.T)
    
    def compute_kernel_matrix(self, X):
        if self.kernel == 'rbf':
            return self.rbf_kernel(X, X)
        else:
            return self.linear_kernel(X, X)
        

    def fit(self, X, y):
        '''
        TODO: Implement both lienar and RBF Kernels
        Function signature SHOULD NOT BE CHANGED. Please maintain the output format and output sizes

        Args:
            X : input data. Shape : (no. of examples , flattened dimension)
            y : output data. Shape : (no, of examples, )

        Ouput:
            None
        '''        
        n_examples,n_features = X.shape
        
        #Making a Kernel Mattrix where K[i,j] = Kernel(x[i],x[j])
        self.kernel_mattrix = self.compute_kernel_matrix(X)
        kernel_mattrix = self.kernel_mattrix
                        
        #Making the mattrix form of the dual to the SVM minimisation problem which can be put into the Quadratic solver to get the required values of Alpha
        # Prepare p and q
        temp_p = np.outer(y, y) * kernel_mattrix  # Shape (n_examples, n_examples)
        p = cvxopt.matrix(temp_p)                # Convert to cvxopt matrix
        q = cvxopt.matrix(-1 * np.ones(n_examples))  # Shape (n_examples, 1)
        
        # Prepare A and b
        A = cvxopt.matrix(y.reshape(1, -1), tc='d') 
        b = cvxopt.matrix(0.0)                    
        
        # Prepare G and h
        lhs1 = -np.identity(n_examples)           
        lhs2 = np.identity(n_examples)           
        G = cvxopt.matrix(np.vstack((lhs1, lhs2)))
        
        rhs1 = np.zeros(n_examples)             
        rhs2 = np.ones(n_examples) * self.C      
        h = cvxopt.matrix(np.hstack((rhs1, rhs2)))  
        
        solution = cvxopt.solvers.qp(p,q,G,h,A,b)
        lamda = np.ravel(solution['x'])

        sv_locs = lamda > 0
        
        self.sv = X[sv_locs]
        self.sv_values = y[sv_locs]
        self.lamda = lamda[sv_locs]

        if self.kernel == 'linear':
            self.W = np.sum((self.lamda * self.sv_values).reshape(-1, 1) * self.sv, axis=0)
        else: 
            self.W = None
        
        b_values = []
        for i in np.where(sv_locs)[0]:
            b_i = self.sv_values[i] - np.sum(self.lamda * self.sv_values * self.kernel_mattrix[i,sv_locs])
            b_values.append(b_i)

        self.b = np.mean(b_values)
        

    def predict(self, X):
        '''
        Function signature SHOULD NOT BE CHANGED. Please maintain the output format and output sizes

        Args:
            X : input data. Shape : (no. of examples , flattened dimension)
        Ouput:
            predictions : Shape : (no. of examples, )
        '''
        if self.kernel == 'linear':
            return np.where(X @ self.W + self.b >= 0, 1, -1)
        else:
            kernel_matrix_x_sv = self.rbf_kernel(X, self.sv) 
            decision_values = np.dot(kernel_matrix_x_sv, self.lamda * self.sv_values) + self.b
            predictions = np.where(decision_values >= 0, 1, -1)
            return predictions  
        
    def get_top_support_vectors_image(self, top_n=6, img_shape=(28, 28), output_filename="top_support_vectors.png"):          
            top_indices = np.argsort(-self.lamda)[:top_n]
            top_sv = self.sv[top_indices]

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_filename_with_time = f"{output_filename.split('.')[0]}_{timestamp}.png"

            fig, axs = plt.subplots(2, 3, figsize=(8, 5))
            for i, ax in enumerate(axs.flatten()):
                if i < top_n:
                    img = top_sv[i].reshape(img_shape)
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                    ax.set_title(f"λ={self.lamda[top_indices[i]]:.2f}")
                else:
                    ax.axis('off')  

            plt.tight_layout()
            plt.savefig(output_filename_with_time)
            plt.close()  