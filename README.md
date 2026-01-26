<div align="center">
<h3> Clusterização da Base de Clientes </h3>
<h4> Um Projeto Estruturado de Engenharia de Dados, Machine Learning e Business Intelligence</h4> 
<div display ="inline">
  <img width="112" height="50" src= "https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
  <img width="112" height="50" src= "https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" />
  <img width="155" height="50" src= "https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img width="79" height="50" src= "https://img.shields.io/badge/Sql-003B57?style=for-the-badge&logo=sqlite&logoColor=white" />
  <img width="140" height="50" src= "https://img.shields.io/badge/Google Cloud-F80000?style=for-the-badge&logo=oracle&logoColor=black" />
  <img width="150" height="50" src= "https://img.shields.io/badge/Looker Studio-0078D4?style=for-the-badge&logo=microsoft-outlook&logoColor=white" />
  
  </div>
  </div>
    
<div align="center">

<p>Este README detalha o processo completo de análise de um dataset, desde a limpeza e engenharia de dados em SQL no Google Big Query, passando pela criação de um modelo Machine Learning em Python, até a construção de um dashboard no Looker Studio.</p>

</div>

## 📊 Dashboard Final

<div align="center">
<p>Abaixo, a versão final do dashboard, que consolida todos os insights gerados neste projeto.</p>
<img width="500" height="500" src= "https://i.ibb.co/ZpYFJz0X/clusterizacao.png" />
<br>
<br><a href="https://lookerstudio.google.com/u/0/reporting/8c8d4297-fb57-450a-b883-61889dea33d1/page/p_he0q3qw2ld">Acessar Dashboard → </a> <br />
</div>

## 📖 Sobre o Projeto

<p>Este projeto simula um caso de uso real do dia a dia de um profissional de dados, com o objetivo de analisar os clientes e realizar a segmentação em 3 níveis para identificação. O processo abrange três grandes áreas:</p>
<ol>
  <li><strong>Engenharia de Dados:</strong> Tratamento, limpeza e enriquecimento dos dados brutos utilizando SQL e Google Cloud Plataform (GCP) para prepará-los para análise e modelagem.</li>
  <li><strong>Machine Learning:</strong> Desenvolvimento de um modelo de clusterização utilizando Python e Scikit-learn.</li>
  <li><strong>Business Intelligence:</strong> Criação de um dashboard dinâmico no Looker Studio para visualizar os insights e permitir a exploração interativa dos dados.</li>
</ol>

## 🔗 Etapas de Construção do Projeto

<p>O projeto foi estruturado em camadas sequenciais listadas abaixo.</p>

### 1. Regras de Negócio
A primeira fase do projeto consistiu em definir com clareza qual problema seria resolvido e qual resultado era esperado. Essa etapa é fundamental e, sempre que possível, deve ser feita em conjunto com o usuário final, pois é ele quem conhece o contexto real do negócio.

Neste caso, o objetivo era ajudar o dono da loja a entender quais estratégias de venda aplicar para cada tipo de cliente. Para isso, ele disponibilizou três informações da base:
- Idade
- Renda anual
- Pontuação de gastos na loja (variando de 0 a 100)

A necessidade principal era segmentar os clientes em três perfis: Bronze, Prata e Ouro, de forma que cada grupo pudesse receber abordagens diferentes.

As regras de negócio foram definidas da seguinte forma:
- Bronze: clientes com menor pontuação de gastos considerando a renda anual e idade.
- Prata: clientes com pontuação baixa para intermediária considerando a renda anual e idade.
- Ouro: clientes com maior pontuação de gastos considerando a renda anual e idade.

Essas definições serviram como base para validar os resultados do modelo de clusterização, garantindo que os grupos gerados fizessem sentido do ponto de vista do negócio.

### 2. Leitura, Preparação e Segmentação dos Dados
Após a definição das regras de negócio, iniciei a etapa técnica com a leitura da base de clientes em formato CSV utilizando Python.

Em seguida, foi realizada uma análise estatística básica das três variáveis utilizadas no modelo:
- idade
- renda_anual
- pontuacao_gastos

Isso permitiu entender a distribuição dos dados, possíveis outliers e diferenças de escala entre as variáveis.

### 3. Padronização dos Dados
Como as variáveis possuem escalas muito diferentes (por exemplo, renda anual vs. pontuação de gastos), foi necessário aplicar padronização (z-score) antes da clusterização.

Essa etapa garante que nenhuma variável tenha mais peso que outra apenas por estar em uma escala maior.
 
Também foi feita uma comparação visual entre os valores originais e os padronizados para validar o processo.

### 4. Clusterização com K-Means
Com os dados padronizados, foi aplicado o algoritmo K-Means, definindo k = 3 clusters, de acordo com a regra de negócio (Bronze, Prata e Ouro).

Cada cliente passou a receber um identificador numérico de cluster (0, 1 ou 2).
O botão abaixo te redicionará ao código em Python.
<div align="center">
<br><a href="https://github.com/lucasratola137/Clusterizacao_Python/blob/main/Clusterizacao.py">Acessar Código no GitHub → </a> <br />
</div>

### 5. Exportação e Análises no Google Cloud
Após a segmentação, o dataset foi salvo em um novo arquivo. Esse arquivo foi enviado para o Google Cloud Platform, onde realizei análises adicionais e bem simples em SQL, utilizando:
- Funções de agregação (AVG, COUNT, etc.).
- Query com CASE para transformar os clusters numéricos em categorias de negócio (esta query foi a utilizada como fonte de dados do Dashboard).
<div align="center">
<p>Abaixo, a tabela criada no Google Cloud com a query utilizada para conexão com o Looker Studio.</p>
<img width="500" height="500" src= "https://i.ibb.co/MkjVY5wQ/GCP.png" />
  
</div>
 Essa etapa permitiu validar os clusters sob a ótica do negócio e gerar insights para estratégias de venda por perfil de cliente.
 O botão abaixo te redicionará aos códigos em SQL.
<div align="center">
<br><a href="https://github.com/lucasratola137/Clusterizacao_Python/blob/main/Query.sql">Acessar Código no GitHub → </a> <br />
</div>

### 6. Conexão Fonte de Dados e Criação do Dashboard
Por fim, nesta etapa foi realizada a conexão entre o Google Cloud e Looker Studio para criação do Dashboard com os gráficos para análise do usuário final.
