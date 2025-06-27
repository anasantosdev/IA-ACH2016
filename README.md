# IA-ACH2016
Projeto criado para a disciplina de Inteligência Artificial no curso de SI da EACH USP. 


Estrutura do projeto:
A pasta "dataset" contém todos os dados usados para treinar a Inteligência Artificial.

Arquivo principal: diabetes_classifier.py tem o papel de classificar os dados e prever o resultado correto.

Observações sobre as transformações dos dados no dataset de 2023: 

As variáveis foram renomeadas para fins de simplificação.

Todas as variáveis que representavam a ausência de algo foram substituídas pelo número 0 e os números conseguintes foram alterados para seguir a nova ordenação númerica. Como exemplo, a variável _RFHYPE6 foi substituída da seguinte forma:

1 -> 0  (não tem pressão alta)
2 -> 1  (tem pressão alta)

Outras substituições feitas para fins de simplificicação do projeto:
Substituições
3 (Diabetes somente na gravidez) -> 0 (Ausência de diabetes)
"6" ou "66" ("Não sei")                    -> REMOVIDA
"7" ou "77" ("Recusou responder")          -> REMOVIDA

Para a variável discreta cardinal Sex, foi decidida uma numeração aleatória de 0 para mulher e 1 para homem.