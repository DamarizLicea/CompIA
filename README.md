# Clasificación de Titulares de Noticias usando Redes Convolucionales y EDA: Un Enfoque Basado en CNN y Synonym Replacement


Damariz Licea

## Abstract
Este trabajo presenta un enfoque para la clasificación de titulares de noticias mediante una red neuronal convolucional (CNN), complementado con técnicas de aumento de datos basadas en sinónimos a través de WordNet. Utilizando el "News Category Dataset", el proyecto abarca desde la limpieza de texto hasta la evaluación del modelo, implementando procesos de tokenización, embeddings y convoluciones, con métricas transparentes para el investigador. Como mejora, el modelo incorpora early stopping y checkpoints para optimizar su rendimiento. La evaluación del desempeño se realizó mediante una matriz de confusión y métricas como precisión, recall y f1-score por clase, junto con la accuracy global, utilizando la función classification_report de sklearn. Esta investigación se fundamenta en trabajos previos sobre CNN para texto (Kim, 2014) y técnicas de aumento de datos, específicamente aplicando la estrategia EDA (Easy Data Augmentation) de Wei y Zou (2019) para el reemplazo de palabras por sinónimos, demostrando que modelos relativamente simples pueden lograr resultados efectivos con estrategias de procesamiento adecuadas.

## Introducción

### Clasificación de Titulares de Noticias
La clasificación automática de texto constituye una tarea fundamental en el procesamiento del lenguaje natural (NLP). Los titulares de noticias representan un desafío particular para esta clasificación debido a su brevedad y a la información altamente condensada que contienen. A diferencia de los artículos completos, los titulares ofrecen contexto limitado, lo que requiere modelos capaces de captar información esencial en pocas palabras.
Para este estudio se utilizó el "News Category Dataset" de Kaggle, una colección de aproximadamente 200,000 artículos de noticias con sus respectivos titulares, publicados por HuffPost entre 2012 y 2018. Este conjunto de datos contiene artículos en inglés clasificados en diversas categorías como "POLITICS", "WELLNESS", "ENTERTAINMENT", "TRAVEL", entre otras.
El dataset está estructurado en formato JSON, donde cada registro incluye varios campos como título, enlace, categoría, descripción corta y fecha de publicación. Para los propósitos de esta investigación, se utilizaron exclusivamente los campos "headline" (titular) y "category" (categoría), ya que el objetivo principal es predecir la categoría de una noticia basándose únicamente en su titular.


La elección de este dataset responde a tres criterios fundamentales:

Presenta una amplia diversidad de categorías, lo que incrementa la complejidad de la tarea de clasificación.
Los titulares son concisos pero informativos, lo que los convierte en candidatos ideales para evaluar la capacidad del modelo de extraer información relevante de textos breves.
Contiene una cantidad sustancial de datos, lo que contrasta favorablemente con experiencias previas en la clasificación de imágenes de mariposas mediante CNNs, un aspecto que se abordará posteriormente.

### Fundamento Teórico
Este trabajo se fundamenta en la investigación de Kim (2014) titulada "Convolutional Neural Networks for Sentence Classification". En dicho estudio, Kim demuestra que las redes neuronales convolucionales (CNN), tradicionalmente aplicadas al procesamiento de imágenes, pueden adaptarse eficazmente para clasificar textos con resultados notables, detectando patrones significativos en secuencias de palabras.

Según Kim, una CNN diseñada para clasificación textual debe incorporar:

Una capa de embedding que transforma palabras en vectores numéricos
Filtros convolucionales que identifican patrones locales (como frases o expresiones)
Una capa de pooling que selecciona las características más relevantes
Capas densas para la clasificación final

En el modelo implementado para este estudio se utilizan:

Embeddings de dimensión 128
Filtros convolucionales de tamaño 5
GlobalMaxPooling1D para reducir dimensionalidad, evitar sobreajuste y destacar la característica más relevante detectada por cada filtro, independientemente de su posición en la secuencia
Dropout (0.5) para prevenir el sobreajuste, siguiendo las recomendaciones de Kim

### Técnicas de Aumento de Datos 
Para enriquecer las oportunidades de aprendizaje del modelo, se implementaron técnicas de aumento de datos textuales basadas en el trabajo de Wei y Zou (2019) "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks". Estos investigadores proponen cuatro técnicas sencillas para la generación de datos adicionales: reemplazo por sinónimos, inserción aleatoria, intercambio aleatorio y eliminación aleatoria.
Para este estudio, se optó específicamente por la técnica de reemplazo por sinónimos utilizando WordNet, una exhaustiva base de datos léxica del inglés. Esta técnica sustituye determinadas palabras en un texto por sus sinónimos correspondientes. De acuerdo con Wei y Zou, el número de palabras a reemplazar se determina mediante un parámetro α multiplicado por la longitud del texto. Este método preserva el significado esencial del contenido mientras genera variaciones que contribuyen a mejorar la capacidad de generalización del modelo.
Wei y Zou demostraron que estas técnicas resultan particularmente beneficiosas en escenarios con datos limitados, mejorando significativamente la precisión en diversos conjuntos de datos. En el contexto de esta investigación, se estableció α = 0.1, lo que implica el reemplazo de aproximadamente el 10% de las palabras en cada titular.


## Metodología
El desarrollo de este proyecto siguió una secuencia estructurada de pasos que abarcan desde el preprocesamiento de datos hasta la evaluación del modelo. A continuación, se detalla cada fase del proceso metodológico implementado.
Preprocesamiento de Datos
Limpieza de Texto
Se aplicaron técnicas estándar de limpieza textual a todos los titulares:

Conversión a minúsculas para uniformidad
Eliminación de signos de puntuación
Normalización de espacios para eliminar duplicaciones

Codificación de Categorías
Las categorías textuales fueron transformadas a representaciones numéricas mediante un mapeo sistemático de cada categoría a un índice específico, facilitando su procesamiento por el modelo.
Aumento de Datos
Se implementó la técnica EDA (Easy Data Augmentation), específicamente el reemplazo por sinónimos utilizando WordNet, para enriquecer el conjunto de datos y mejorar la capacidad de generalización del modelo.
División de Datos
El conjunto de datos fue segmentado en tres subconjuntos:

Entrenamiento: 64% de los datos
Validación: 16% de los datos
Prueba: 20% de los datos

Se utilizó la estratificación (stratify) para preservar la distribución proporcional de las clases en cada subconjunto.
Tokenización y Padding
Los titulares fueron convertidos en secuencias numéricas mediante tokenización, y posteriormente se aplicó padding para asegurar que todas las secuencias tuvieran la misma longitud, requisito necesario para su procesamiento en la red neuronal.
Arquitectura del Modelo
Siguiendo el enfoque propuesto por Kim (2014), se implementó una red neuronal convolucional especializada para la clasificación de texto. La arquitectura del modelo incluye las siguientes capas:

Capa de Embedding: Transforma los índices de palabras en vectores densos de 128 dimensiones, creando representaciones numéricas significativas del texto.
Capa Convolucional 1D: Aplica 128 filtros de tamaño 5 para la detección de patrones lingüísticos en secuencias de palabras, identificando características relevantes.
GlobalMaxPooling1D: Extrae la característica más prominente detectada por cada filtro, reduciendo la dimensionalidad y preservando la información más relevante.
Dropout (0.5): Implementa una técnica de regularización que desactiva aleatoriamente el 50% de las neuronas durante el entrenamiento, previniendo el sobreajuste.
Capa Densa: Procesa las características extraídas mediante 128 neuronas con función de activación ReLU, permitiendo el modelado de relaciones no lineales complejas.
Capa de Salida: Presenta una neurona por cada categoría con función de activación softmax, generando distribuciones de probabilidad sobre las posibles clasificaciones.

Configuración del Entrenamiento
El proceso de entrenamiento del modelo fue configurado con los siguientes parámetros y técnicas:

Función de pérdida: Se utilizó sparse_categorical_crossentropy, una función apropiada para problemas de clasificación multiclase cuando las etiquetas están codificadas como enteros.
Optimizador: Se implementó Adam, un algoritmo de optimización que adapta automáticamente la tasa de aprendizaje durante el entrenamiento.
Métrica de evaluación: Se empleó Accuracy para cuantificar la proporción de predicciones correctas realizadas por el modelo.
Early Stopping: Se incorporó un mecanismo para detener automáticamente el entrenamiento cuando no se observa mejora en la pérdida de validación durante 3 épocas consecutivas, evitando el sobreajuste.
Model Checkpoint: Se configuró un sistema para guardar la versión del modelo que presenta la mejor precisión en el conjunto de validación, asegurando la conservación del modelo óptimo.
Batch size: Se definieron lotes de 32 muestras para actualizar los pesos del modelo durante el entrenamiento.
Épocas: Se entrenó por máximo de 400 episodios.

## Evaluación del Modelo
Métricas de Evaluación
Para evaluar exhaustivamente el desempeño del modelo, se implementó un conjunto diverso de métricas y visualizaciones que permiten analizar su comportamiento tanto a nivel global como por clase individual:
Matriz de Confusión
Se empleó una matriz de confusión para visualizar la distribución de predicciones correctas e incorrectas para cada categoría, proporcionando una representación gráfica intuitiva del rendimiento del clasificador.
Métricas por Clase
Para cada categoría del modelo, se calcularon las siguientes métricas:
Precision (Precisión)
La precisión mide qué tan exactas son las predicciones positivas del modelo. Representa la proporción de elementos correctamente clasificados en una categoría respecto al total de elementos asignados a dicha categoría. Esta métrica es particularmente relevante en contextos donde el costo de los falsos positivos es elevado.
$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$
Recall (Exhaustividad)
El recall mide qué tan completa es la detección de positivos del modelo. Indica la proporción de elementos correctamente identificados como pertenecientes a una categoría respecto al total de elementos que realmente pertenecen a dicha categoría. Esta métrica adquiere especial importancia cuando el costo asociado a los falsos negativos es alto.
$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$
F1-Score
El F1-Score representa la media armónica de precisión y recall, proporcionando un único valor que equilibra ambas métricas. Este indicador resulta especialmente útil cuando se busca un balance entre precisión y recall, particularmente en conjuntos de datos con distribuciones de clase desequilibradas, donde la exactitud global puede resultar engañosa.
$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
Se optó por utilizar el F1-Score como complemento a las métricas individuales para obtener una evaluación más robusta y equilibrada del desempeño del modelo.
Interpretación de Métricas Combinadas
La combinación de precision y recall permite interpretar el comportamiento del modelo según el siguiente esquema:

Alta precisión, bajo recall: El modelo adopta un enfoque conservador. Cuando predice una clase, suele acertar, pero omite muchos casos positivos.
Baja precisión, alto recall: El modelo adopta un enfoque liberal. Identifica la mayoría de los casos positivos, pero también clasifica erróneamente muchos negativos como positivos.
Alta precisión, alto recall: El modelo exhibe un excelente rendimiento. Identifica la mayoría de los positivos con pocos falsos positivos.
Baja precisión, bajo recall: El modelo presenta deficiencias significativas para esa clase específica.

Accuracy Global
Además de las métricas por clase, se monitorizó la exactitud (accuracy) global del modelo, que representa la proporción total de predicciones correctas respecto al total de predicciones realizadas.
Resultados de la Evaluación
Las visualizaciones generadas durante el proceso de evaluación revelaron información valiosa sobre el desempeño del modelo:
Mostrar imagen
Métricas por cada clase analizada, mostrando solo algunas clases representativas.
Mostrar imagen
Evolución del accuracy y loss global del modelo durante una ejecución de 5 episodios.
Mostrar imagen
Matriz de confusión implementada utilizando sklearn.
El análisis de los resultados indica que el modelo alcanzó una precisión media de 0.57, un recall medio de 0.48, y un F1-Score medio de 0.51. Estos valores sugieren que el modelo presenta un desempeño equilibrado, donde la precisión ligeramente superior al recall indica una tendencia moderadamente conservadora en sus predicciones, con un F1-Score que refleja un balance razonable entre ambas métricas.
Mostrar imagen
Resumen de clasificación con número de aciertos correctos e incorrectos.
Mostrar imagen
Matriz detallada de falsos positivos, falsos negativos, verdaderos positivos y verdaderos negativos.
El modelo alcanzó un accuracy global máximo de 0.8 en el conjunto de validación durante una ejecución de 30 epochs.
Funcionalidad de Predicción
Como complemento a la evaluación cuantitativa, se implementó una función interactiva que permite a los usuarios ingresar un titular de noticia o texto breve arbitrario, para que el modelo realice una predicción sobre la categoría a la que pertenecería. Esta funcionalidad proporciona una demostración práctica de la aplicabilidad del modelo en escenarios reales, además de ofrecer una forma intuitiva de evaluar cualitativamente su desempeño.

## # Resultados

## Evaluación del Desempeño General

El objetivo inicial de este proyecto no contemplaba alcanzar un umbral específico de precisión, sino más bien explorar si una implementación basada en la arquitectura propuesta por Kim (2014) podría resolver efectivamente la clasificación de titulares de noticias. En términos cuantitativos, se estableció como criterio mínimo de éxito superar un nivel de certeza del 50%.

Los resultados obtenidos confirman que el modelo implementado logró superar ampliamente este umbral, alcanzando un accuracy global máximo de 0.8 en el conjunto de validación durante una ejecución extendida a 30 epochs. Este nivel de desempeño demuestra que, incluso con una arquitectura relativamente simple, las redes neuronales convolucionales pueden capturar eficazmente los patrones lingüísticos presentes en textos breves como los titulares de noticias.

## Demostración Práctica: Predicción Interactiva

Para ilustrar la aplicabilidad práctica del modelo, se implementó una función de predicción interactiva que permite evaluar su desempeño con nuevos titulares. A modo de ejemplo, se probó el titular ficticio: "Korean beauty company causes acne", ante el cual el modelo asignó la categoría "Style & Beauty" con una confianza aproximada del 87%.

![Predicción interactiva](figura_prediccion.png)
*Interfaz de predicción interactiva mostrando la clasificación del titular ejemplo con su correspondiente nivel de confianza.*

Esta elevada confianza en la predicción sugiere que el modelo ha logrado identificar correctamente los elementos semánticos asociados a la categoría de belleza, demostrando su capacidad para generalizar a partir de los patrones aprendidos durante el entrenamiento.

## Cambio de Enfoque: Del Procesamiento de Imágenes al Procesamiento de Texto

### Dificultades con la Clasificación de Imágenes

Es importante destacar que la dirección actual del proyecto surgió como respuesta a los desafíos encontrados en un enfoque previo. Inicialmente, el objetivo era desarrollar un clasificador de imágenes de mariposas por especie utilizando redes neuronales convolucionales tradicionales.

Para este propósito, se utilizó el dataset "Butterfly Classification" de Kaggle. Sin embargo, desde las primeras etapas de implementación, se detectaron varios problemas fundamentales:

1. **Desbalance significativo del dataset**: El modelo mostraba una clara tendencia a favorecer las clases con mayor representación en el conjunto de datos.

2. **Limitaciones del aumento de datos**: A pesar de implementar técnicas de jitter para enriquecer la diversidad de las imágenes, estas intervenciones resultaron insuficientes para mitigar el problema del desbalance.

3. **Arquitecturas avanzadas sin mejora significativa**: La reimplementación del modelo utilizando la arquitectura InceptionV3 no produjo mejoras sustanciales en el rendimiento.

4. **Estancamiento en el aprendizaje**: A pesar de un entrenamiento prolongado con numerosos episodios, el accuracy global nunca superó el umbral del 0.2, lo que sugería que el modelo podría estar memorizando en lugar de aprendiendo patrones generalizables.

Esta situación es consistente con experiencias previas en aprendizaje por refuerzo, donde se ha observado que extender el entrenamiento no compensa las deficiencias fundamentales en la configuración del modelo o en la calidad de los datos. Por el contrario, un entrenamiento prolongado en estas circunstancias puede resultar contraproducente, llevando al modelo a aprender asociaciones incorrectas.

### Transición al Procesamiento de Texto

La transición al dataset de noticias y la implementación del enfoque actual representó un cambio estratégico que resultó en mejoras significativas en los resultados obtenidos. Esta experiencia refuerza la importancia de:

1. Seleccionar datasets con distribuciones balanceadas o implementar técnicas efectivas para compensar los desbalances.
2. Reconocer cuándo un enfoque no está produciendo resultados satisfactorios y considerar alternativas.
3. Aprovechar la transferencia de conocimiento entre dominios, adaptando arquitecturas probadas como la propuesta por Kim para resolver problemas específicos.

La evidencia del trabajo previo con imágenes de mariposas se encuentra documentada como referencia complementaria a este estudio, proporcionando un valioso contraste metodológico y contextual.

## Conclusiones
Este proyecto demuestra que una arquitectura CNN simple, como la propuesta por Kim (2014), sigue siendo efectiva para tareas de clasificación de texto, incluso con entradas tan breves como los titulares de noticias. Aplicando técnicas de limpieza adecuadas y un ligero aumento de datos, es posible alcanzar un rendimiento satisfactorio. En resumen, menos puede ser más cuando se aplica de forma correcta.

## Referencias

1. Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification*. arXiv preprint arXiv:1408.5882. [https://arxiv.org/abs/1408.5882](https://arxiv.org/abs/1408.5882)

2. Wei, J., & Zou, K. (2019). *EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks*. arXiv preprint arXiv:1901.11196. [https://arxiv.org/abs/1901.11196](https://arxiv.org/abs/1901.11196)

3. News Category Dataset on Kaggle: [https://www.kaggle.com/datasets/rmisra/news-category-dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)

4. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media. (Referencia para el uso de WordNet mediante NLTK)

5. Scikit-learn: Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830. [https://jmlr.org/papers/v12/pedregosa11a.html](https://jmlr.org/papers/v12/pedregosa11a.html)



# Archivado, se queda para evidencia de trabajo, por favor no leer a partir de aquí.
# Butterfly Classification
# V1 — Semana del 31 de marzo al 4 de abril
## Elección del Dataset

Elegí el dataset Butterfly Image Classification disponible en Kaggle:
🔗 https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification

La temática me pareció interesante y atractiva, asumí que las mariposas se diferenciaban principalmente por color, lo que podría facilitar la clasificación. Además, el dataset cuenta con buenas reseñas de usabilidad en la plataforma.

Decidí trabajar en Google Colab por la facilidad que ofrece en compatibilidad de librerías y por su mayor capacidad de cómputo en comparación con mi equipo local. Aunque tuve algunos problemas menores, como con ImageDataGenerator, en general fue más estable que trabajar en local.

Implementé una arquitectura similar a la vista en clase, pero yo usé:

- 2 capas convolucionales

- 2 capas de pooling

- 1 capa densa

- Data augmentation básico: zoom, horizontal_flip y rotation

- Herramientas utilizadas: TensorFlow, Keras, sklearn

Dividí el dataset inicialmente en 80% para entrenamiento y 20% para pruebas.

Durante esta semana, entrené el modelo por hasta 20 épocas, alcanzando un accuracy promedio de 0.5.
El modelo fue capaz de detectar al menos una clase correctamente, aunque el desempeño general aún era muy básico.


# V2 — Semana del 7 al 11 de abril

Implementé una versión personalizada de ColorJitter inspirada en (Ref2), con el objetivo de hacer el data augmentation más robusto.
Sin embargo, el accuracy cayó de 0.5 a un máximo de 0.02, con fluctuaciones muy abruptas y comportamiento inestable.

Reorganicé la distribución del dataset siguiendo una recomendación:

- 80% entrenamiento

- 10% validación

- 10% prueba

### Distribución actual del dataset:

* Total de etiquetas: 75 clases

* Imágenes de entrenamiento: 5199

* Imágenes de validación: 650

* Imágenes de prueba: 650

Encontré un paper relevante (Ref1) sobre clasificación de mariposas con más de 100 clases usando Inception-V3.
Aún estoy investigando cómo implementar esa arquitectura y los tres tipos de data augmentation que utilizaron.
Por ahora, el enfoque está en optimizar un solo tipo de aumento antes de añadir más complejidad.

Aumenté el número de épocas de entrenamiento de 10 a 100, lo cual representa un cambio importante en el ciclo de pruebas.

### Gráficos de precisión

Antes de modificar el data augmentation:

![image](https://github.com/user-attachments/assets/6af0b676-5c32-4dc0-b62a-5f6dc140547e)

Después de la modificación agresiva:

![image](https://github.com/user-attachments/assets/e1ca6bf7-e3be-4687-8d2c-aff2bf254c59)

Con ajustes menos agresivos:

![image](https://github.com/user-attachments/assets/20704f3d-f9d4-455a-b995-f69e89f9ac38)

Para entender el desempeño por clase, creé una matriz de confusión. La primera versión gráfica resultó poco legible por la cantidad de clases, así que opté por un resumen textual:

![image](https://github.com/user-attachments/assets/43d431b8-1623-454e-af57-d7576e82b318)

Actualmente el modelo solo reconoce correctamente 3 clases de mariposas. Esto sugiere posibles problemas de underfitting y que el data augmentation aún no está bien calibrado.

### Referencias

[Ref1] Paper sobre clasificación de mariposas con más de 100 clases:
https://dl.futuretechsci.org/id/eprint/73/1/9443-Article%20Text-30154-5-10-20240615.pdf

[Ref2] Paper sobre técnicas avanzadas de data augmentation:
https://doi.org/10.48550/arXiv.2502.18691
