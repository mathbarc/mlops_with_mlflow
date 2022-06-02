# MLOps

O presente repositório contém o material da palestra sobre MLOps apresentada na TheDeveloper'sConference Innovation 2022.

- O que é MLOps;
- Os diferentes componentes de um processo MLOps;
- Os níveis de maturidade dos processos:
	- Manual;
	- Com automação de pipeline de ML;
	- Com automação CI/CD;
- E na sequência apresentarei uma pipeline construída utilizando a ferramenta MLFlow e uma api REST que consome modelos gerenciados pelo MLFlow.

## O que é MLOps?

MLOps é um conjunto de práticas que visa reduzir a distância entre o modelo de inteligência artificial e utilização deste modelo pelo usuário final.
No MLOps os ciclos do DevOps são mantidos, porém são adicionadas as etapas necessárias para o desenvolvimento de modelos de machine learning:
	- A captação e organização da base de dados, principal insumo para a criação de modelos;
	- A seleção e treinamento de modelos compatíveis com os padrões que se deseja extrair do conjunto de dados;
	- E a validação dos modelos com um conjunto de dados diferente do utilizado para treinar o modelo.

Estas três etapas são incluídas no processo como uma extensão da etapa de desenvolvimento mas na maioria dos casos elas ocorrem em paralelo. Vale ressaltar porém que o tempo de conclusão necessário para as etapas de ML e DEV são diferentes, o que muitas vezes gera uma disparidade entre o código desenvolvido e os modelos em criação.

## Componentes de um processo MLOps

Para reduzir essas disparidades no MLOps adotamos as seguintes fases de desenvolvimento:
	- A análise exploratória dos dados: Nesta etapa os cientistas de dados analisam o dataset com o objetivo de compreender o problema e analisar a viabilidade da solução;
	- Preparação dos dados: Neste ponto é realizada limpeza, organização e catalogação dos dados;
	- Treinamento de modelos: Etapa na qual os dados limpos e catalogados são utilizados para criar modelos que reconheçam os padrões desejados, assim como é assegurado o seu funcionamento correto junto a plataforma;
	- Quando o modelo está apto para uso, ele é disponibilizado em uma plataforma para processo de inferência utilizando dados novos vindos do ambiente de produção;
	- E portando precisa de um acompanhamento para averiguar se ele mantém sua integridade, sendo necessário o monitoramento dos seus resultados, principalmente dos indices de confiança nos resultado do modelo;
	- Dados com baixos índices de segurança no resultado são captados e destinados então para compor uma nova base de dados e o processo retorna para a etapa de treinamento.

Dado todo este processo iterativo, o que devemos esperar de um pipeline de MLOps? Bem, isso depende da complexidade e escala do seu produto.

## Processo Manual

Como primeiro pipeline temos aqui o Processo Manual. Nele a análise, preparação, treinamento e validação são todos realizados de forma manual, muitas vezes em scritps separados ou células de um notebook que são chamados manualmente nesta sequência para realizar o pipeline de treinamento de modelos.
Neste tipo de processo existe uma desconexão entre o time de data science que implementa o modelo e os engenheiros de software que desenvolvem o serviço que utilizará o modelo.
Apesar de seus problemas, esse processo não necessita de um ambiente estruturado para ser implantado, como é o caso de PoCs ou projetos cuja a viabilidade ainda não foi confirmada.

## Processo com automação de Pipeline ML

Quando movemos a criação de um processo automatizado de treinamento chegamos ao segundo estágio. Aqui temos uma pipeline automatizada para o processo o que permite execuções mais rápidas de experimentos. Outro grande ganho desta etapa é a criação de uma simetria entre o que é utilizado em todos os ambientes, com o pipeline automatizado é possível replicar no ambiente de produção o mesmo ambiente utilizado para desenvolvimento dos modelos de ML. Essa possibilidade trás uma oportunidade: a entrega contínua de modelos. Novas melhorias e aprimoramentos assim que validados podem ser movidos para produção com maior facilidade.
Este estágio é recomendado para produtos que irão começar a serem utilizados por uma user base restrita.

## Processo com automação de Pipeline CI/CD

Nesse ultimo estágio, o pipeline automatizado de machine learning possui testes e empacotamento automatizados via CI e entregas nos diversos ambientes do produto automatizados com praticas de CD, diminuindo ainda mais a distância entre o modelo e o contexto no qual ele será utilizado. Nessa fase do produto estamos lidando com equipes maiores, maior complexidade e uma user base maior também.
