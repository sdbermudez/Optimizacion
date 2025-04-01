import numpy as np

def f(x):
    return  (100*(x[1]-x[0]**2)**2)+(1-x[0])**2

def grad_f(x):

    return np.array([-((400*x[0]*(x[1]-x[0]**2))+2*(1-x[0])), 200*(x[1]-x[0]**2)], dtype=float)  

def linea_search(f, grad_f, xk, dk, alpha=1, tau=0.5, c1=0.1):
    
    while f(xk + alpha * dk) > f(xk) + c1 * alpha * np.dot(grad_f(xk), dk):
        alpha *= tau  
    return alpha

def metodo_gradiente(f, grad_f, x0, xaux=None,tol=1e-6, max_iter=4000):
    xk = np.array(x0, dtype=float)  
    cont=0
    for _ in range(max_iter):
        cont+=1
        grad_k = grad_f(xk)
        print(f"iteracion ({cont}: xk={xk}, gradk={grad_k})")
        print("||xk-x*||=", np.linalg.norm(xk-xaux))
        print()
        temp=xk.copy()
        
        
        pk = -grad_k  
        alpha_k = linea_search(f, grad_f, xk, pk)
        
        xk = xk - (alpha_k * grad_f(xk))   
        if np.linalg.norm(xk-temp) < tol:
            break
    
    return xk  


x0 = np.array([1.2, 1.2])  # Punto inicial
xaux = np.array([1.0, 1.0])  # Punto de referencia (opcional)
solucion = metodo_gradiente(f, grad_f, x0, xaux=xaux)
print("Solución encontrada:", solucion)
print("Valor de la función objetivo:", f(solucion))
prueba = np.array([1.0, 1.0])
print("Gradiente en la solución:", grad_f(prueba))