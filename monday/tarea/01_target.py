"""Construye competencia_01 (crudo + clase_ternaria) con DuckDB.

Regla del target, para cada (cliente, foto_mes) en que el cliente ESTA:
  - BAJA+1   : no esta en el mes siguiente
  - BAJA+2   : esta en el mes siguiente pero no en el subsiguiente
  - CONTINUA : esta en los dos meses siguientes
  - NULL     : no se puede saber (ultimos periodos, se ven las ventanas)
"""
import duckdb, pathlib

BASE = pathlib.Path(__file__).resolve().parent.parent.parent
CRUDO = BASE / "monday" / "competencia_01_crudo.csv"
OUT_CSV = BASE / "monday" / "competencia_01.csv"
OUT_PQ = BASE / "monday" / "competencia_01.parquet"

con = duckdb.connect()
con.execute("pragma threads=4")

con.execute(f"""
create or replace table crudo as
select * from read_csv_auto('{CRUDO}')
""")

con.execute("""
create or replace table competencia_01 as
with periodos as (
    select distinct foto_mes from crudo
), clientes as (
    select distinct numero_de_cliente from crudo
), todo as (
    select numero_de_cliente, foto_mes from clientes cross join periodos
), flags as (
    select
        t.numero_de_cliente
      , t.foto_mes
      , c.* exclude (numero_de_cliente, foto_mes)
      , case when c.numero_de_cliente is null then 0 else 1 end as mes_0
      , lead(case when c.numero_de_cliente is null then 0 else 1 end, 1)
            over (partition by t.numero_de_cliente order by t.foto_mes) as mes_1
      , lead(case when c.numero_de_cliente is null then 0 else 1 end, 2)
            over (partition by t.numero_de_cliente order by t.foto_mes) as mes_2
    from todo t
    left join crudo c using (numero_de_cliente, foto_mes)
)
select
    * exclude (mes_0, mes_1, mes_2)
  , case
        when mes_1 is null then null          -- no hay mes+1 en la ventana
        when mes_1 = 0    then 'BAJA+1'
        when mes_2 is null then null          -- hay mes+1 pero no mes+2
        when mes_2 = 0    then 'BAJA+2'
        else 'CONTINUA'
    end as clase_ternaria
from flags
where mes_0 = 1
""")

print("filas:", con.sql("select count(*) from competencia_01").fetchone()[0])
print(con.sql("""
    pivot competencia_01 on clase_ternaria using count(numero_de_cliente) group by foto_mes order by foto_mes
""").df().to_string(index=False))

con.execute(f"copy competencia_01 to '{OUT_PQ}' (format parquet)")
con.execute(f"copy competencia_01 to '{OUT_CSV}' (format csv, header)")
print("\nescritos:", OUT_PQ.name, "y", OUT_CSV.name)
