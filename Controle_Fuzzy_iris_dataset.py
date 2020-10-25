
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.datasets import load_iris

# 0 --> SETOSA
# 1 --> VERSICOLOR
# 2 --> VIRGINICA

iris = load_iris()

#Separando dados para treinamento

# Dados do comprimento da sépala
sp_comp_0_min = np.min(iris.data[0:25,:1])
sp_comp_0_max = np.max(iris.data[0:25,:1])
sp_comp_1_min = np.min(iris.data[50:75,:1])
sp_comp_1_med = np.mean(iris.data[50:75,:1])
sp_comp_2_min = np.min(iris.data[100:125,:1])
sp_comp_2_max = np.max(iris.data[100:125,:1])
sp_comp_intersecao_0_1 = np.min(np.intersect1d(iris.data[0:25,:1],iris.data[50:75,:1]))
sp_comp_intersecao_1_2 = np.max(np.intersect1d(iris.data[50:75,:1],iris.data[100:125,:1]))

# Dados da largura da sépala
sp_larg_0_min = np.min(iris.data[0:25,1:2])
sp_larg_0_max = np.max(iris.data[0:25,1:2])
sp_larg_1_min = np.min(iris.data[50:75,1:2])
sp_larg_1_max = np.mean(iris.data[50:75,1:2])
sp_larg_2_min = np.min(iris.data[100:125,1:2])
sp_larg_2_med = np.max(iris.data[100:125,1:2])
sp_larg_intersecao_1_2 = np.min(np.intersect1d(iris.data[50:75,1:2],iris.data[100:125,1:2]))
sp_larg_intersecao_2_0 = np.max(np.intersect1d(iris.data[100:125,1:2],iris.data[0:25,1:2]))

# Dados do comprimento da pétala
pt_comp_0_min = np.min(iris.data[0:25,2:3])
pt_comp_0_max = np.max(iris.data[0:25,2:3])
pt_comp_1_min = np.min(iris.data[50:75,2:3])
pt_comp_1_med = np.mean(iris.data[50:75,2:3])
pt_comp_2_min = np.min(iris.data[100:125,2:3])
pt_comp_2_max = np.max(iris.data[100:125,2:3])
pt_comp_intersecao_1_2 = np.max(np.intersect1d(iris.data[50:75,2:3],iris.data[100:125,2:3]))

# Dados da largura da pétala
pt_larg_0_min = np.min(iris.data[0:25,3:4])
pt_larg_0_max = np.max(iris.data[0:25,3:4])
pt_larg_1_min = np.min(iris.data[50:75,3:4])
pt_larg_1_med = np.mean(iris.data[50:75,3:4])
pt_larg_2_min = np.min(iris.data[100:125,3:4])
pt_larg_2_max = np.max(iris.data[100:125,3:4])
pt_larg_intersecao_1_2 = np.max(np.intersect1d(iris.data[50:75,3:4],iris.data[100:125,3:4]))



sp_comp = ctrl.Antecedent(np.arange(sp_comp_0_min, sp_comp_2_max, 0.1),'sp_comp')
sp_larg = ctrl.Antecedent(np.arange(sp_larg_1_min, sp_larg_0_max, 0.1),'sp_larg')
pt_comp = ctrl.Antecedent(np.arange(pt_comp_0_min, pt_comp_2_max, 0.1),'pt_comp')
pt_larg = ctrl.Antecedent(np.arange(pt_larg_0_min, pt_larg_2_max, 0.1),'pt_larg')
flor_iris = ctrl.Consequent(np.arange(0,3,0.1), 'flor_iris')

# Funções de pertinência do comprimento da sépala
sp_comp['0'] = fuzz.trapmf(sp_comp.universe,[sp_comp_0_min,sp_comp_0_min,sp_comp_intersecao_0_1,sp_comp_0_max])
sp_comp['1'] = fuzz.trimf(sp_comp.universe,[sp_comp_intersecao_0_1,sp_comp_1_med,sp_comp_intersecao_1_2])
sp_comp['2'] = fuzz.trapmf(sp_comp.universe,[sp_comp_2_min,sp_comp_intersecao_1_2,sp_comp_2_max,sp_comp_2_max])

# Funções de pertinência da largura da sépala
sp_larg['1'] = fuzz.trapmf(sp_larg.universe,[sp_larg_1_min,sp_larg_1_min,sp_larg_intersecao_1_2,sp_larg_1_max])
sp_larg['2'] = fuzz.trimf(sp_larg.universe,[sp_larg_2_min,sp_larg_2_med,sp_larg_intersecao_2_0])
sp_larg['0'] = fuzz.trapmf(sp_larg.universe,[sp_larg_0_min,sp_larg_intersecao_2_0,sp_larg_0_max,sp_larg_0_max])

# Funções de pertinência do comprimento da pétala
pt_comp['0'] = fuzz.trapmf(pt_comp.universe,[pt_comp_0_min,pt_comp_0_min,pt_comp_0_max,pt_comp_0_max])
pt_comp['1'] = fuzz.trimf(pt_comp.universe,[pt_comp_0_max,pt_comp_1_med,pt_comp_intersecao_1_2])
pt_comp['2'] = fuzz.trapmf(pt_comp.universe,[pt_comp_2_min,pt_comp_intersecao_1_2,pt_comp_2_max,pt_comp_2_max])

# Funções de pertinência da largura da pétala
pt_larg['0'] = fuzz.trapmf(pt_larg.universe,[pt_larg_0_min,pt_larg_0_min,pt_larg_0_max,pt_larg_0_max])
pt_larg['1'] = fuzz.trimf(pt_larg.universe,[pt_larg_0_max,pt_larg_1_med,pt_larg_intersecao_1_2])
pt_larg['2'] = fuzz.trapmf(pt_larg.universe,[pt_larg_2_min,pt_larg_intersecao_1_2,pt_larg_2_max,pt_larg_2_max])


# Funções de pertinência da classificação da Íris
flor_iris['0'] = fuzz.trapmf(flor_iris.universe, [0, 0, 1, 1])
flor_iris['1'] = fuzz.trapmf(flor_iris.universe, [1, 1, 2, 2])
flor_iris['2'] = fuzz.trapmf(flor_iris.universe, [2, 2, 3, 3])

# Regras de inferência
rule1 = ctrl.Rule(sp_comp['0'] & sp_larg['0'] & pt_comp['0'] & pt_larg['0'], flor_iris['0'])
rule2 = ctrl.Rule(sp_comp['1'] & sp_larg['1'] & pt_comp['1'] & pt_larg['1'], flor_iris['1'])
rule3 = ctrl.Rule(sp_comp['2'] & sp_larg['2'] & pt_comp['2'] & pt_larg['2'], flor_iris['2'])
rule4 = ctrl.Rule(sp_comp['0'] & sp_larg['0'] & pt_comp['0'] & pt_larg['1'], flor_iris['0'])
rule5 = ctrl.Rule(sp_comp['0'] & sp_larg['0'] & pt_comp['1'] & pt_larg['0'], flor_iris['0'])
rule6 = ctrl.Rule(sp_comp['0'] & sp_larg['1'] & pt_comp['0'] & pt_larg['0'], flor_iris['0'])
rule7 = ctrl.Rule(sp_comp['1'] & sp_larg['0'] & pt_comp['0'] & pt_larg['0'], flor_iris['0'])
rule8 = ctrl.Rule(sp_comp['0'] & sp_larg['0'] & pt_comp['0'] & pt_larg['2'], flor_iris['0'])
rule9 = ctrl.Rule(sp_comp['0'] & sp_larg['0'] & pt_comp['2'] & pt_larg['0'], flor_iris['0'])
rule10 = ctrl.Rule(sp_comp['0'] & sp_larg['2'] & pt_comp['0'] & pt_larg['0'], flor_iris['0'])
rule11 = ctrl.Rule(sp_comp['2'] & sp_larg['0'] & pt_comp['0'] & pt_larg['0'], flor_iris['0'])
rule12 = ctrl.Rule(sp_comp['1'] & sp_larg['1'] & pt_comp['1'] & pt_larg['0'], flor_iris['1'])
rule13 = ctrl.Rule(sp_comp['1'] & sp_larg['1'] & pt_comp['0'] & pt_larg['1'], flor_iris['1'])
rule14 = ctrl.Rule(sp_comp['1'] & sp_larg['0'] & pt_comp['1'] & pt_larg['1'], flor_iris['1'])
rule15 = ctrl.Rule(sp_comp['0'] & sp_larg['1'] & pt_comp['1'] & pt_larg['1'], flor_iris['1'])
rule16 = ctrl.Rule(sp_comp['1'] & sp_larg['1'] & pt_comp['1'] & pt_larg['2'], flor_iris['1'])
rule17 = ctrl.Rule(sp_comp['1'] & sp_larg['1'] & pt_comp['2'] & pt_larg['1'], flor_iris['1'])
rule18 = ctrl.Rule(sp_comp['1'] & sp_larg['2'] & pt_comp['1'] & pt_larg['1'], flor_iris['1'])
rule19 = ctrl.Rule(sp_comp['2'] & sp_larg['1'] & pt_comp['1'] & pt_larg['1'], flor_iris['1'])
rule20 = ctrl.Rule(sp_comp['2'] & sp_larg['2'] & pt_comp['2'] & pt_larg['0'], flor_iris['2'])
rule21 = ctrl.Rule(sp_comp['2'] & sp_larg['2'] & pt_comp['0'] & pt_larg['2'], flor_iris['2'])
rule22 = ctrl.Rule(sp_comp['2'] & sp_larg['0'] & pt_comp['2'] & pt_larg['2'], flor_iris['2'])
rule23 = ctrl.Rule(sp_comp['0'] & sp_larg['2'] & pt_comp['2'] & pt_larg['2'], flor_iris['2'])
rule24 = ctrl.Rule(sp_comp['2'] & sp_larg['2'] & pt_comp['2'] & pt_larg['1'], flor_iris['2'])
rule25 = ctrl.Rule(sp_comp['2'] & sp_larg['2'] & pt_comp['1'] & pt_larg['2'], flor_iris['2'])
rule26 = ctrl.Rule(sp_comp['2'] & sp_larg['1'] & pt_comp['2'] & pt_larg['2'], flor_iris['2'])
rule27 = ctrl.Rule(sp_comp['1'] & sp_larg['2'] & pt_comp['2'] & pt_larg['2'], flor_iris['2'])


iris_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7,\
                                rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16,\
                                    rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24,\
                                        rule25,rule26,rule27])

iris_controle = ctrl.ControlSystemSimulation(iris_ctrl)

cont = 0 

# Método para classificação da Íris
# se o valor da função de pertinência da classificação estiver entre 0 e 1 a íris é uma setosa
# se o valor da função de pertinência da classificação estiver entre 1 e 2 a íris é uma versicolor
# se o valor da função de pertinência da classificação estiver entre 2 e 3 a íris é uma virginica

for i in range(25,50):
    iris_controle.input['sp_comp'] = iris.data[i,:1]
    iris_controle.input['sp_larg'] = iris.data[i,1:2]
    iris_controle.input['pt_comp'] = iris.data[i,2:3]
    iris_controle.input['pt_larg'] = iris.data[i,3:4]

    iris_controle.compute()
       
    result = float(iris_controle.output['flor_iris'])
    
    
    if (result <= 1):
        cont = cont + 1

for i in range(75,100):
    iris_controle.input['sp_comp'] = iris.data[i,:1]
    iris_controle.input['sp_larg'] = iris.data[i,1:2]
    iris_controle.input['pt_comp'] = iris.data[i,2:3]
    iris_controle.input['pt_larg'] = iris.data[i,3:4]

    iris_controle.compute()
       
    result = float(iris_controle.output['flor_iris'])
    
    
    if (result <= 2 and result > 1):
        cont = cont + 1 
        
for i in range(125,150):
    iris_controle.input['sp_comp'] = iris.data[i,:1]
    iris_controle.input['sp_larg'] = iris.data[i,1:2]
    iris_controle.input['pt_comp'] = iris.data[i,2:3]
    iris_controle.input['pt_larg'] = iris.data[i,3:4]

    iris_controle.compute()
    
    result = float(iris_controle.output['flor_iris'])
    
    
    if (result <= 3 and result > 2):
        cont = cont + 1

# índice de desempenho        
ind_desp = (cont/75)*100
print(ind_desp) 