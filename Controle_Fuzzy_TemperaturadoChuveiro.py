
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


potencia = ctrl.Antecedent(np.arange(0, 101, 1), 'potência')
vazao = ctrl.Antecedent(np.arange(0, 101, 1), 'vazão')
temperatura = ctrl.Consequent(np.arange(15, 46, 1), 'temperatura')

# Funções de Pertinência para a potência
potencia['baixa'] = fuzz.trapmf(potencia.universe, [0, 0, 20, 50])
potencia['média'] = fuzz.trimf(potencia.universe, [20, 50, 80])
potencia['alta'] = fuzz.trapmf(potencia.universe, [50, 80, 100, 100])

# Funções de Pertinência para a vazão
vazao['baixa'] = fuzz.trapmf(vazao.universe, [0, 0, 20, 50])
vazao['média'] = fuzz.trimf(vazao.universe, [20, 50, 80])
vazao['alta'] = fuzz.trapmf(vazao.universe, [50, 80, 100, 100])

# Funções de Pertinência para a temperatura
temperatura['baixa'] = fuzz.trapmf(temperatura.universe, [15,15, 20, 30])
temperatura['agradável'] = fuzz.trimf(temperatura.universe, [20, 30, 40])
temperatura['alta'] = fuzz.trapmf(temperatura.universe, [30,40, 45, 45])



potencia.view()

vazao.view()

temperatura.view()

# Regras de Inferência
rule1 = ctrl.Rule(potencia['baixa'] & vazao['baixa'], temperatura['agradável'])
rule2 = ctrl.Rule(potencia['baixa'] & vazao['média'], temperatura['baixa'])
rule3 = ctrl.Rule(potencia['baixa'] & vazao['alta'], temperatura['baixa'])
rule4 = ctrl.Rule(potencia['média'] & vazao['baixa'], temperatura['agradável'])
rule5 = ctrl.Rule(potencia['média'] & vazao['média'], temperatura['agradável'])
rule6 = ctrl.Rule(potencia['média'] & vazao['alta'], temperatura['baixa'])
rule7 = ctrl.Rule(potencia['alta'] & vazao['baixa'], temperatura['alta'])
rule8 = ctrl.Rule(potencia['alta'] & vazao['média'], temperatura['alta'])
rule9 = ctrl.Rule(potencia['alta'] & vazao['alta'], temperatura['agradável'])



temperatura_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])

temperatura_controle = ctrl.ControlSystemSimulation(temperatura_ctrl)

# Entrada de potência e vazão
temperatura_controle.input['potência'] = 70
temperatura_controle.input['vazão'] = 50


temperatura_controle.compute()

# Saída da temperatura
print(temperatura_controle.output['temperatura'])
temperatura.view(sim=temperatura_controle)
