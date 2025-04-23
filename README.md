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
