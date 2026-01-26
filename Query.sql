



#agregação média apenas para validação
SELECT a.cluster,avg(a.renda_anual) as media_renda ,avg(a.pontuacao_gasto) as media_gasto
from projetocluster.clientes_segmentados a
group by a.cluster;

#query utilizada como fonte de dado para conexão com o Looker Studio
SELECT 
a.*, 
case 
when a.cluster = 2 then 'Ouro'
when a.cluster = 1 then 'Prata'
when a.cluster = 0 then 'Bronze'
else 'not found'
end as tipo_cliente
from projetocluster.clientes_segmentados a
order by a.idCliente asc;
