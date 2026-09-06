# `z301_Sobre_la_incertidumbre` — pendiente

**Consigna original:** `monday/z301_Sobre_la_incertidumbre.ipynb` (Alejandro Bolaños)

> ## Tarea:
>
> * Ejecute todo de nuevo para su semilla y los hiperparámetros que considere.
> * Ya tiene la búsqueda de Optuna. Duda existencial: Dado que no hay tanta distancia
>   entre el primer modelo y los subsiguientes en optuna y que a su vez hicimos apenas
>   5 muestreos, no será posible que el mejor modelo en SSS no sea el mejor en el
>   privado de **Mayo**?
>     * Quédese con los top 5 de optuna. Use más muestras para cada modelo (20 o más)
>     * Calcule la ganancia y su distribución en los conjuntos de validación
>     * Cómo son las medias? ayudan a elegir el mejor modelo en el privado? con que confianza?
>     * Sólo mirando la distribución del muestreo de validación, tiene herramientas para
>       elegir el mejor modelo en el privado?
>     * Puede pasar que comparando 2 modelos uno le gane al otro en el público, pero no
>       así en el privado? En que porcentaje pasa?
>
> Colaboración:
> * Recuerde compartir con tus compañeros los nuevos scripts que hayas generado y las
>   configuraciones que hayas probado por Zulip

## Material ya disponible

`mis_tareas/z201_arboles/06_features.py` tiene una medición out-of-time hecha por
adelantado (entrena en 202104, mide en 202105 y 202106). Salió algo que es materia
prima para esta tarea: **entre los dos meses de test hay 54% de diferencia de ganancia**
($185M contra $286M) con el mismo modelo. Esa varianza es exactamente el fenómeno que
esta clase quiere que midas.
