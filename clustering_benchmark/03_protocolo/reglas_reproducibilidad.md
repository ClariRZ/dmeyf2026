# Reglas de reproducibilidad

1. Cada experimento tiene ID único.
2. Nunca sobrescribir resultados.
3. Guardar parámetros exactos.
4. Guardar semilla.
5. Guardar versión del dataset.
6. Guardar lista exacta de variables.
7. Guardar versiones de librerías cuando sea posible.
8. Registrar warnings y errores.
9. No eliminar resultados negativos.
10. No cambiar parámetros después de ver el resultado sin crear un nuevo ID.
11. Visualizaciones no sustituyen métricas.
12. UMAP/t-SNE se usan para visualizar, no para probar existencia de clusters.
13. No nombrar clusters antes del perfilado.
14. Separar decisiones previas de interpretaciones posteriores.
15. Toda comparación entre métodos debe usar el mismo dataset congelado.
