import numpy as np
import matplotlib.pyplot as plt
import control as crt
from control import tf, feedback, step_response
from geneticalgorithm import geneticalgorithm as ga

def f(X):
    
    sis = tf([2], [4, 1]) #tf do sistema 2/(4s + 1)

    sisp = tf([X[0]],[1])   # tf do controlador proporcional Kp
    sisi = tf([X[0]*X[1]],[1, 0]) # tf do controlador integral Kp*Ki/s
    sisd = tf([X[0]*X[2], 0],[1]) # tf do controlador diferencial Kd*s

    sis2 = crt.parallel(sisp, sisi, sisd) # tf do PID
    sis3 = crt.series(sis, sis2)          # tf do sistema em série com o PID 

    mf = feedback(sis3, 1)  # malha fechada do sistema

    time = np.linspace(0, 20, 100) # tempo de simulação
    _, y1 = step_response(mf, time) # resposta do sistema em malha fechada com PID

    erro = sum(abs(np.array((y1 - 1)*time))) # integral do erro 

    return erro

varbound=np.array([[0,20]]*3)

model=ga(function=f,dimension=3,variable_type='real',variable_boundaries=varbound)

model.run()

X = model.best_variable
erro = model.best_function

print(f'Kp = {X[0]}')
print(f'Ki = {X[1]}')
print(f'Kd = {X[2]}')
print(f'erro = {erro}')

sis = tf([2], [4, 1]) #tf do sistema 2/(4s + 1)
 
sisp = tf([X[0]],[1])   # tf do controlador proporcional Kp
sisi = tf([X[0]*X[1]],[1, 0]) # tf do controlador integral Kp*Ki/s
sisd = tf([X[0]*X[2], 0],[1]) # tf do controlador diferencial Kd*s
 
sis2 = crt.parallel(sisp, sisi, sisd) # tf do PID
sis3 = crt.series(sis, sis2)          # tf do sistema em série com o PID 

mf = feedback(sis3, 1)  # malha fechada do sistema
mf_sem_PID = feedback(sis, 1)  # malha fechada do sistema sem PID

time = np.linspace(0, 20, 100) # tempo de simulação
_, y1 = step_response(mf, time) # resposta do sistema em malha fechada com PID
_, y2 = step_response(mf_sem_PID, time) # resposta do sistema em malha fechada sem PID
_, y3 = step_response(sis, time) # resposta do sistema em malha aberta
 
plt.figure(1)
plt.plot(time, y1, label='$y_1(t)$') 
plt.ylim([-0.1, 3])
plt.xlim([0, 20])
plt.xlabel('tempo [s]')
plt.ylabel('sinal [1]')
plt.legend()
plt.title('Resposta do Sistema com PID')
 
plt.figure(2)
plt.plot(time, y2, label='$y_1(t)$') 
plt.ylim([-0.1, 3])
plt.xlim([0, 20])
plt.xlabel('tempo [s]')
plt.ylabel('sinal [1]')
plt.legend()
plt.title('Resposta do Sistema sem PID')
 
plt.figure(3)
plt.plot(time, y3, label='$y_1(t)$')
plt.ylim([-0.1, 3])
plt.xlim([0, 20])
plt.xlabel('tempo [s]')
plt.ylabel('sinal [1]')
plt.legend()
plt.title('Resposta do Sistema em Malha Aberta')

