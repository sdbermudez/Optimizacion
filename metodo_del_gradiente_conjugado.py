import numpy as np
#Funcion no necesariamente cuadratica
def f(x):
    
    return  (100*(x[1]-x[0]**2)**2)+(1-x[0])**2 

def grad_f(x):
    
    return np.array([(-(400*(x[1]-x[0]**2)*x[0])-2*(1-x[0])), 200*(x[1]-x[0]**2)]) 

def linea_search(f, grad_f, xk, dk, alpha=1, tau=0.5, c1=0.1):
    
    while f(xk + alpha * dk) > f(xk) + c1 * alpha * np.dot(grad_f(xk), dk):
        alpha *= tau  
    return alpha

def gradiente_conjugado(f, grad_f, x0, tol=1e-6, max_iter=100, xaux=None):
    
    xk = np.array(x0, dtype=float)
    xk_new = np.array([None,None], dtype=float)
    rk = -grad_f(xk)
    pk = rk
    cont=0
    for _ in range(max_iter):
        cont+=1
        print(f"Iteración {cont}: xk = {xk}, grad={grad_f(xk)}, rk={rk}, pk={pk}")
        print("||xk-x*||=", np.linalg.norm(xk-xaux))
        #print(np.linalg.norm(xk-xaux))
        print()
        
        alpha_k = linea_search(f, grad_f, xk, pk)
        xk_new = xk + (alpha_k* pk)
        
        
        if np.linalg.norm(xk_new-xk) < tol:
            break
        rk_new = -grad_f(xk_new)
        beta_k = np.dot(np.transpose(rk_new), rk_new) / np.dot(np.transpose(rk), rk) 
        pk_new = rk_new + beta_k * pk
        
        xk, rk, pk = xk_new, rk_new, pk_new
    
    return xk  


x0 = np.array([1.2 , 1.2]) 
xaux = np.array([1.0, 1.0])
solucion = gradiente_conjugado(f, grad_f, x0,xaux=xaux)
print("Solución encontrada:", solucion)
